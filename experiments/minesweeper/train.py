"""Pool-based trainer for minesweeper.

Same dpll_step loop as sudoku / maze. Pool dynamics are identical to maze
(K=1 always — each board has a unique CVC5-verified solution, so there are
no alternative ground truths to OR over). Training signal per step:
  1. Sample batch from pool: (orig_x, state, solutions [P,1,S,C], depth).
  2. Forward → BCE + softmax + conflict heads.
  3. Compute losses against ground truth:
       - BCE target = state * orig_y                   (lattice ∩ truth)
       - Softmax CE target = orig_y.argmax(-1)          (GT digit per cell)
       - Conflict target = gt_conflict
  4. Backprop + step.
  5. dpll_step (no-grad) advances state.
  6. Discard policy: solved | (detected_conflict & gt_conflict_post) | age > max_age.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from experiments.minesweeper.data import MinesweeperConfig, MinesweeperDataset, N_CHANNELS
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.training.utils.checkpoint import save_checkpoint
from lattice_diffusion.training.utils.scheduler import make_cosine_scheduler

from experiments.sudoku.aug import aug_forward
from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import ModelEMA
from experiments.sudoku.solve import SolveConfig, solve
from experiments.sudoku.train import (
    _alpha_surviving, _gt_conflict, _given_mask, _losses,
)


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.1
    seed: int = 0

    bce_pos_mult: float = 4.0
    bce_neg_mult: float = 0.5
    softmax_loss_weight: float = 0.2
    conflict_loss_weight: float = 1.0

    step: StepConfig = field(default_factory=StepConfig)
    max_age: int = 100
    pool_size_mult: float = 1.0
    compile: bool = True

    use_ema: bool = False
    ema_decay: float = 0.999

    log_every: int = 20
    eval_every: int = 100
    eval_n_puzzles: int = 200
    eval_max_rounds: int = 5
    eval_n_chains: int = 64

    out_dir: str = "checkpoints/minesweeper"
    name: str = ""

    checkpoint_every: int = 1000
    resume_path: str | None = None
    commit_volume: bool = False

    model: LoopedTransformerConfig = field(default_factory=LoopedTransformerConfig)
    data: MinesweeperConfig = field(default_factory=MinesweeperConfig)


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = out_dir / f"{cfg.name}_{ts}.pt"

    model = PowersetModel(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}  steps={cfg.steps}  bs={cfg.batch_size}",
          flush=True)
    if cfg.compile and device.type == "cuda":
        print("torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)

    cfg.data.batch_size = cfg.batch_size
    dataset = MinesweeperDataset(cfg.data)

    # Held-out eval set for in-train mini-solve (from test split, no aug).
    eval_data_cfg = MinesweeperConfig(
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        split="test",
        n_puzzles=cfg.eval_n_puzzles,
        batch_size=cfg.eval_n_puzzles,
        seed=200,
        augment_dihedral=False,
    )
    eval_ds = MinesweeperDataset(eval_data_cfg)
    ex_x, ex_sols, ex_sat = eval_ds.next_batch()
    eval_ds.close()
    sat_mask = ex_sat.bool()
    eval_x = ex_x[sat_mask][:cfg.eval_n_puzzles].to(device).float()
    # solutions [B,1,S,C] → unwrap K dim → [B,S,C]
    eval_y = ex_sols[sat_mask, 0][:cfg.eval_n_puzzles].to(device).float()
    eval_given = (eval_x.sum(dim=-1) == 1)
    print(f"In-train eval set: {eval_x.shape[0]} puzzles "
          f"(every {cfg.eval_every} steps, K={cfg.eval_n_chains}, "
          f"max_rounds={cfg.eval_max_rounds})", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=cfg.betas,
    )
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_fraction))
    scheduler = make_cosine_scheduler(optimizer, cfg.steps, warmup_steps)

    ema: ModelEMA | None = None
    if cfg.use_ema:
        ema = ModelEMA(model, cfg.ema_decay)
        print(f"EMA enabled (decay={cfg.ema_decay})", flush=True)

    # Resume support.
    start_step = 1
    if cfg.resume_path and Path(cfg.resume_path).exists():
        print(f"[resume] loading snapshot from {cfg.resume_path}", flush=True)
        rckpt = torch.load(cfg.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        scheduler.load_state_dict(rckpt["scheduler_state_dict"])
        if ema is not None and "ema_state_dict" in rckpt:
            ema.shadow = {k: v.to(device) for k, v in rckpt["ema_state_dict"].items()}
        start_step = int(rckpt["step"]) + 1
        try:
            torch.set_rng_state(rckpt["torch_rng"])
            if torch.cuda.is_available() and rckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(rckpt["cuda_rng"])
        except (TypeError, RuntimeError) as e:
            print(f"[resume] RNG state restore skipped: {e}", flush=True)
        print(f"[resume] resumed at step {start_step}", flush=True)
    elif cfg.resume_path:
        print(f"[resume] no snapshot at {cfg.resume_path}; starting from step 1",
              flush=True)

    pool_size = max(cfg.batch_size, int(round(cfg.batch_size * cfg.pool_size_mult)))

    # Leftover buffer so backfill calls don't waste a full dataset batch.
    ds_buf: list[torch.Tensor | None] = [None, None]  # [x, solutions]

    def fresh_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x [n,S,C], solutions [n,1,S,C]) from the dataset buffer."""
        while ds_buf[0] is None or ds_buf[0].shape[0] < n:
            x_b, sols_b, _ = dataset.next_batch()
            if ds_buf[0] is None:
                ds_buf[0], ds_buf[1] = x_b, sols_b
            else:
                ds_buf[0] = torch.cat([ds_buf[0], x_b], dim=0)
                ds_buf[1] = torch.cat([ds_buf[1], sols_b], dim=0)
        x = ds_buf[0][:n].to(device).float()
        solutions = ds_buf[1][:n].to(device).float()
        ds_buf[0] = ds_buf[0][n:]
        ds_buf[1] = ds_buf[1][n:]
        return x, solutions

    # Initialize pool.
    pool_orig_x, pool_solutions = fresh_batch(pool_size)   # solutions: [P,1,S,C]
    pool_state = pool_orig_x.clone()
    pool_last_alpha = pool_solutions.bool().any(dim=1).float()  # [P,S,C] (K=1 → solutions[:,0])
    pool_age = torch.zeros(pool_size, dtype=torch.long, device=device)
    print(f"Pool size: {pool_size}  (bs={cfg.batch_size}, mult={cfg.pool_size_mult})  "
          f"augment={cfg.step.augment}", flush=True)

    n_solved_total = 0
    n_tp_conflict_total = 0
    n_examples_seen = 0

    for step in range(start_step, cfg.steps + 1):
        sample_idx = torch.randperm(pool_size, device=device)[:cfg.batch_size]
        state = pool_state[sample_idx]
        solutions = pool_solutions[sample_idx]   # [B,1,S,C]
        last_alpha = pool_last_alpha[sample_idx]
        orig_x = pool_orig_x[sample_idx]
        age = pool_age[sample_idx]
        given_mask = _given_mask(orig_x)
        orig_y = _alpha_surviving(state, solutions, last_alpha)
        gt_conflict_pre = _gt_conflict(state, orig_y)
        is_sat_pre = ~gt_conflict_pre

        # ---- forward + losses (grad-tracked) ----
        # permute_digits=False → aug_vocab_dim=0 (no channel shuffle).
        aug_vocab_dim = 0 if not cfg.step.permute_digits else cfg.step.vocab_dim
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out, aug_info = aug_forward(
            model, state, given_mask, orig_y=orig_y,
            augment=cfg.step.augment,
            augment_dihedral=cfg.step.augment_dihedral,
            cell_perms=None,
            vocab_dim=aug_vocab_dim,
            return_all=True,
        )
        loss = _losses(
            out,
            aug_info["aug_state"], aug_info["aug_orig_y"], aug_info["aug_given_mask"],
            is_sat_pre, gt_conflict_pre,
            cfg.bce_pos_mult, cfg.bce_neg_mult,
            cfg.softmax_loss_weight, cfg.conflict_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        n_examples_seen += cfg.batch_size

        # ---- advance state (no grad) ----
        model.eval()
        want_info_stats = (step % cfg.log_every == 0)
        with torch.no_grad():
            new_state, detected_conflict, solved, _, info = dpll_step(
                model, state, given_mask, cfg.step, orig_y=orig_y,
                want_stats=want_info_stats,
            )
        gt_conflict_post = _gt_conflict(new_state, orig_y)

        # ---- discard policy ----
        new_age = age + 1
        age_exceeded = new_age > cfg.max_age
        true_positive_conflict = detected_conflict & gt_conflict_post
        discard = solved | true_positive_conflict | age_exceeded

        n_solved_total += int(solved.sum().item())
        n_tp_conflict_total += int(true_positive_conflict.sum().item())

        # ---- backfill ----
        n_to_replace = int(discard.sum().item())
        new_last_alpha = orig_y.clone()
        if n_to_replace > 0:
            fx, f_solutions = fresh_batch(n_to_replace)
            new_state = new_state.clone()
            new_orig_x = orig_x.clone()
            new_solutions = solutions.clone()
            new_age = new_age.clone()
            new_state[discard] = fx
            new_orig_x[discard] = fx
            new_solutions[discard] = f_solutions
            new_age[discard] = 0
            new_last_alpha[discard] = f_solutions.bool().any(dim=1).float()
        else:
            new_orig_x = orig_x
            new_solutions = solutions

        pool_state[sample_idx] = new_state
        pool_orig_x[sample_idx] = new_orig_x
        pool_solutions[sample_idx] = new_solutions
        pool_last_alpha[sample_idx] = new_last_alpha
        pool_age[sample_idx] = new_age

        # ---- periodic resume snapshot ----
        if (cfg.resume_path
                and cfg.checkpoint_every > 0
                and step % cfg.checkpoint_every == 0):
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": (ema.shadow if ema is not None else None),
                "step": step,
                "model_cfg": asdict(cfg.model),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state()
                             if torch.cuda.is_available() else None),
            }
            torch.save(ckpt_payload, cfg.resume_path)
            unique_path = (Path(cfg.resume_path).with_suffix("")
                           .as_posix() + f".step{step:07d}.pt")
            torch.save(ckpt_payload, unique_path)
            if cfg.commit_volume:
                from lattice_diffusion.modal.image import checkpoint_volume
                checkpoint_volume.commit()
            print(f"  [resume ckpt] saved at step {step} → {cfg.resume_path}", flush=True)

        # ---- in-train eval ----
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            model.eval()
            ema_backup = ema.swap_in(model) if ema is not None else None
            with torch.no_grad():
                eval_solve_cfg = SolveConfig(
                    step=cfg.step,
                    max_rounds=cfg.eval_max_rounds,
                    n_chains=cfg.eval_n_chains,
                    batch_size=cfg.batch_size,
                )
                eval_res = solve(model, eval_x, eval_y, eval_given, eval_solve_cfg,
                                 verbose=False)
            if ema is not None and ema_backup is not None:
                ema.swap_out(model, ema_backup)
            n_e = eval_res.solved.shape[0]
            n_cor_e = int(eval_res.correct.sum().item())
            n_wr_e = int(eval_res.wrong.sum().item())
            n_to_e = int(eval_res.timeouts.sum().item())
            den = max(eval_res.diag_total_deduced, 1)
            unsound_e = eval_res.diag_total_unsound_deductions / den
            cls_p_e = eval_res.diag_conflict_tp / max(
                eval_res.diag_conflict_tp + eval_res.diag_conflict_fp, 1)
            cls_r_e = eval_res.diag_conflict_tp / max(
                eval_res.diag_conflict_tp + eval_res.diag_conflict_fn, 1)
            print(
                f"  [intrain-eval step={step}] "
                f"correct={n_cor_e}/{n_e}  wrong={n_wr_e}  timeout={n_to_e}  "
                f"calls={eval_res.model_calls}  "
                f"unsound={unsound_e:.3%}  "
                f"cls:P={cls_p_e:.2f}/R={cls_r_e:.2f}",
                flush=True,
            )

        # ---- logging ----
        if step % cfg.log_every == 0:
            cls_msg = ""
            cls_hi_msg = ""
            if cfg.conflict_loss_weight > 0 and "conflict" in out:
                with torch.no_grad():
                    c_logits = out["conflict"][-1].squeeze(-1)
                    pred_unsat = c_logits > 0
                    target_unsat = gt_conflict_pre
                    tp = int((pred_unsat & target_unsat).sum().item())
                    fp = int((pred_unsat & ~target_unsat).sum().item())
                    fn = int((~pred_unsat & target_unsat).sum().item())
                    tn = int((~pred_unsat & ~target_unsat).sum().item())
                    P = tp / max(tp + fp, 1)
                    R = tp / max(tp + fn, 1)
                    fill = (state.sum(dim=-1) == 1).float().mean(dim=-1)
                    hi = fill > 0.9
                    n_hi = int(hi.sum().item())
                    if n_hi > 0:
                        tp_hi = int((pred_unsat & target_unsat & hi).sum().item())
                        fp_hi = int((pred_unsat & ~target_unsat & hi).sum().item())
                        fn_hi = int((~pred_unsat & target_unsat & hi).sum().item())
                        tn_hi = int((~pred_unsat & ~target_unsat & hi).sum().item())
                        P_hi = tp_hi / max(tp_hi + fp_hi, 1)
                        R_hi = tp_hi / max(tp_hi + fn_hi, 1)
                        cls_hi_msg = (f"cls@fill>0.9:P={P_hi:.2f}/R={R_hi:.2f}"
                                      f"[tp={tp_hi}/fp={fp_hi}/tn={tn_hi}/fn={fn_hi}]"
                                      f"  n_hi={n_hi}")
                    else:
                        cls_hi_msg = "cls@fill>0.9:n_hi=0"
                cls_msg = f"cls:P={P:.2f}/R={R:.2f}[tp={tp}/fp={fp}/tn={tn}/fn={fn}]"
            with torch.no_grad():
                aug_state_l = aug_info["aug_state"]
                aug_orig_y_l = aug_info["aug_orig_y"]
                multi_alive = (aug_state_l.sum(dim=-1) > 1.5)
                masked = out["bce"][-1].masked_fill(
                    ~(aug_state_l > 0.5), float("-inf"),
                )
                pred = masked.argmax(dim=-1)
                gt_argmax = aug_orig_y_l.argmax(dim=-1)
                n_unc = int(multi_alive.sum().item())
                n_cor = int(((pred == gt_argmax) & multi_alive).sum().item())
                cell_acc = n_cor / max(n_unc, 1)
            min_d = int(pool_age.min().item())
            med_d = int(pool_age.median().item())
            avg_d = float(pool_age.float().mean().item())
            max_d = int(pool_age.max().item())
            pool_alpha = _alpha_surviving(pool_state, pool_solutions, pool_last_alpha)
            sat_frac = float((~_gt_conflict(pool_state, pool_alpha)).float().mean().item())
            print(
                f"step={step:5d}/{cfg.steps}  loss={loss.item():.4f}  "
                f"acc={cell_acc:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.6f}",
                flush=True,
            )
            print(
                f"    deduce={info['n_deduced']}  decide={info['n_decided']}  "
                f"solved={info['n_solved']}  "
                f"conflict={info['n_conflict']}(empty={info['n_conflict_empty']},"
                f"cls={info['n_conflict_cls']})  "
                f"age(min/med/avg/max)={min_d}/{med_d}/{avg_d:.1f}/{max_d}  "
                f"pool_sat={sat_frac:.2f}",
                flush=True,
            )
            if cls_msg or cls_hi_msg:
                print(f"    {cls_msg}  {cls_hi_msg}", flush=True)
            with torch.no_grad():
                new_fill = (new_state.sum(dim=-1) == 1).float().mean(dim=-1)
            depth_boundaries = torch.tensor([1, 2, 4, 8, 16, 32], device=device)
            depth_labels = ["d=0", "d=1", "d=2-3", "d=4-7", "d=8-15", "d=16-31", "d=32+"]
            depth_bins = torch.bucketize(age, depth_boundaries, right=False)
            parts: list[str] = []
            for b in range(len(depth_labels)):
                mask = (depth_bins == b)
                n_b = int(mask.sum().item())
                if n_b == 0:
                    continue
                n_s = int((solved & mask).sum().item())
                n_d = int((detected_conflict & mask).sum().item())
                n_g = int((gt_conflict_post & mask).sum().item())
                f_b = float(new_fill[mask].mean().item())
                parts.append(
                    f"{depth_labels[b]}={n_b}@f{f_b:.2f}({n_s}s/{n_d}c/{n_g}g)"
                )
            if parts:
                print(f"    depth: {' '.join(parts)}", flush=True)

    extra = {
        "n_params": n_params,
        "n_examples_seen": n_examples_seen,
        "n_solved_total": n_solved_total,
        "n_tp_conflict_total": n_tp_conflict_total,
    }
    if ema is not None:
        extra["ema_state_dict"] = ema.state_dict()
        extra["ema_decay"] = cfg.ema_decay
    save_checkpoint(
        path=ckpt_path, model=model, model_cfg=cfg.model, train_cfg=cfg,
        extra=extra,
    )
    print(f"Saved: {ckpt_path}", flush=True)
    dataset.close()
    return ckpt_path
