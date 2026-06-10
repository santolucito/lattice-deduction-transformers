"""Modal entry point for minesweeper: train then evaluate.

Data setup (run once locally before training):
  cd /path/to/minesweeper-sym
  python3 dataset.py -n 10000 --difficulty 1 --seed 42  -o train.jsonl
  python3 dataset.py -n 1000  --difficulty 1 --seed 9999 -o test.jsonl
  modal volume put data-vol train.jsonl minesweeper/train.jsonl
  modal volume put data-vol test.jsonl  minesweeper/test.jsonl

Usage:
    uv run modal run experiments/minesweeper/run.py
    uv run modal run experiments/minesweeper/run.py --steps 8000 --batch-size 512
"""

import dataclasses
import hashlib
import json
import time
from dataclasses import asdict

import modal
import torch
from torch import nn


_RUN_PARAMS: tuple[tuple[str, object], ...] = (
    ("steps",                   4000),
    ("batch_size",              512),
    ("n_eval_puzzles",          1000),
    ("seed",                    0),
    ("bce_pos_mult",            4.0),
    ("bce_neg_mult",            0.5),
    ("softmax_loss_weight",     0.2),
    ("conflict_loss_weight",    1.0),
    ("weight_decay",            0.01),
    ("lr",                      3e-3),
    ("max_age",                 100),
    ("warmup_fraction",         0.1),
    ("threshold",               0.10),
    ("temp_decide",             1.5),
    ("cls_threshold",           0.5),
    ("eval_cls_threshold",      0.6),
    ("eval_max_rounds",         1000),
    ("eval_n_chains",           64),
    ("eval_batch_size",         -1),
    ("augment",                 True),
    ("train_augment",           True),
    ("use_ema",                 False),
    ("ema_decay",               0.999),
    ("estimate_sequential",     False),
    ("seq_drain_max_rounds",    200),
    ("eval_dropout_p",          0.05),
    ("n_loops",                 16),
    ("pool_size_mult",          1.0),
    ("model_dim",               128),
    ("num_layers",              4),
    ("intrain_eval_every",      100),
    ("intrain_eval_max_rounds", 5),
    ("resume",                  False),
    ("checkpoint_every",        1000),
    ("use_rope",                False),
    ("n_train_puzzles",         None),
    ("n_test_puzzles",          None),
)


def _print_run_config(values: dict) -> None:
    BOLD, RESET = "\033[1m", "\033[0m"
    print("=" * 64, flush=True)
    print("RUN CONFIG  (* = non-default)", flush=True)
    print("=" * 64, flush=True)
    n_changed = 0
    for name, default in _RUN_PARAMS:
        val = values.get(name, "<unset>")
        is_changed = (val != default)
        if is_changed:
            n_changed += 1
            marker = "*"
            shown = f"{BOLD}{val!r}{RESET}"
            tail = f"   (default {default!r})"
        else:
            marker = " "
            shown = f"{val!r}"
            tail = ""
        print(f"  {marker} {name:<28} = {shown}{tail}", flush=True)
    print(f"  {n_changed}/{len(_RUN_PARAMS)} non-default", flush=True)
    print("=" * 64, flush=True)


from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.minesweeper.data import MinesweeperConfig, MinesweeperDataset, N_CHANNELS
from experiments.minesweeper.train import TrainConfig, train


app = modal.App("minesweeper")


@app.function(
    image=image,
    gpu="B200",
    timeout=3600 * 8,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    steps: int = 4000,
    batch_size: int = 512,
    n_eval_puzzles: int = 1000,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 1.0,
    weight_decay: float = 0.01,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = -1,
    augment: bool = True,
    train_augment: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    pool_size_mult: float = 1.0,
    model_dim: int = 128,
    num_layers: int = 4,
    intrain_eval_every: int = 100,
    intrain_eval_max_rounds: int = 5,
    resume: bool = False,
    checkpoint_every: int = 1000,
    use_rope: bool = False,
    n_train_puzzles: int | None = None,
    n_test_puzzles: int | None = None,
):
    if eval_batch_size < 0:
        eval_batch_size = batch_size

    _loc_snapshot = dict(locals())
    _arg_values = {name: _loc_snapshot[name] for name, _ in _RUN_PARAMS}
    _print_run_config(_arg_values)

    ts = time.strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Minesweeper: dihedral aug only (channels encode mine/safe-count semantics,
    # not interchangeable labels), no digit-perm.
    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=train_augment,
        augment_dihedral=True,
        permute_digits=False,
    )

    train_data_path = f"{DATA_MOUNT}/minesweeper/train.jsonl"
    test_data_path = f"{DATA_MOUNT}/minesweeper/test.jsonl"

    data_cfg = MinesweeperConfig(
        train_path=train_data_path,
        test_path=test_data_path,
        split="train",
        batch_size=batch_size,
        seed=42,
        n_puzzles=n_train_puzzles,
        augment_dihedral=train_augment,
    )

    # 9×9 beginner boards → seq_len=81, grid_rows=grid_cols=9
    model_cfg = LoopedTransformerConfig(
        n_channels=N_CHANNELS,
        seq_len=81,
        grid_rows=9,
        grid_cols=9,
        cls_token=(conflict_loss_weight > 0),
        n_loops=n_loops,
        dim=model_dim,
        num_layers=num_layers,
        use_rope=use_rope,
    )

    resume_spec = json.dumps(
        {k: _arg_values[k] for k, _ in _RUN_PARAMS if k != "resume"},
        sort_keys=True,
    )
    resume_hash = hashlib.sha256(resume_spec.encode()).hexdigest()[:12]
    resume_path = (f"{CHECKPOINT_MOUNT}/minesweeper/resume_{resume_hash}.pt"
                   if resume else None)
    if resume:
        print(f"Resume mode: snapshot at {resume_path} "
              f"(saved every {checkpoint_every} steps)", flush=True)

    ckpt_path = train(TrainConfig(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        bce_pos_mult=bce_pos_mult,
        bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        warmup_fraction=warmup_fraction,
        step=step_cfg,
        max_age=max_age,
        pool_size_mult=pool_size_mult,
        use_ema=use_ema,
        ema_decay=ema_decay,
        eval_every=intrain_eval_every,
        eval_max_rounds=intrain_eval_max_rounds,
        eval_n_chains=eval_n_chains,
        eval_n_puzzles=min(n_eval_puzzles, 200),
        checkpoint_every=checkpoint_every,
        resume_path=resume_path,
        commit_volume=resume,
        model=model_cfg,
        data=data_cfg,
        out_dir=f"{CHECKPOINT_MOUNT}/minesweeper",
        name=f"minesweeper_9x9_seed{seed}_{steps}s_bs{batch_size}_aug{int(train_augment)}_{ts}",
    ))
    checkpoint_volume.commit()

    print("\n" + "=" * 60, flush=True)
    print(f"Eval ({n_eval_puzzles} test puzzles)", flush=True)
    print("=" * 60, flush=True)

    ckpt = load_checkpoint(str(ckpt_path))
    cfg_loaded = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg_loaded)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)

    if eval_dropout_p > 0.0:
        n_drop = n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = eval_dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = eval_dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: {n_drop} Dropout + {n_mha} MHA → p={eval_dropout_p}",
              flush=True)

    eval_ds_cfg = MinesweeperConfig(
        train_path=train_data_path,
        test_path=test_data_path,
        split="test",
        n_puzzles=n_test_puzzles,
        batch_size=n_eval_puzzles,
        seed=200,
        augment_dihedral=False,
    )
    eval_ds = MinesweeperDataset(eval_ds_cfg)
    x, sols, sat = eval_ds.next_batch()
    eval_ds.close()
    # solutions: [B, 1, S, C] → [B, S, C]
    y = sols[:, 0]
    sat_mask = sat.bool()
    x = x[sat_mask].to(device).float()
    y = y[sat_mask].to(device).float()
    given_mask = (x.sum(dim=-1) == 1)
    n_sat = x.shape[0]
    print(f"  Loaded {n_sat}/{n_eval_puzzles} eval puzzles", flush=True)

    eval_step_cfg = dataclasses.replace(
        step_cfg, cls_threshold=eval_cls_threshold, augment=augment,
    )
    solve_cfg = SolveConfig(
        step=eval_step_cfg,
        max_rounds=eval_max_rounds,
        n_chains=eval_n_chains,
        batch_size=eval_batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
    )
    res = solve(model, x, y, given_mask, solve_cfg)

    n = res.solved.shape[0]
    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_rounds_solved = float(
        res.round_solved[res.solved].float().mean().item()
        if int(res.solved.sum().item()) > 0 else 0.0
    )
    avg_resets = float(res.n_resets.float().mean().item())

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT SUMMARY\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}  "
          f"n_chains={res.n_chains}", flush=True)
    print(f"  Total model calls: {res.model_calls}  "
          f"(amortized: {res.model_calls / max(n_correct, 1):.1f} calls/correct)",
          flush=True)
    print(f"  Avg rounds-to-solve: {avg_rounds_solved:.1f}  "
          f"Avg resets/puzzle: {avg_resets:.2f}", flush=True)
    print(f"  Deduction soundness: {res.diag_total_unsound_deductions} unsound / "
          f"{res.diag_total_deduced} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head: P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={res.diag_conflict_tp} fp={res.diag_conflict_fp} "
          f"fn={res.diag_conflict_fn} tn={res.diag_conflict_tn}]", flush=True)
    print(f"{'='*60}", flush=True)

    eval_json_path = ckpt_path.with_suffix(".eval.json")
    eval_json_path.write_text(json.dumps({
        "checkpoint": str(ckpt_path),
        "n_eval_puzzles": n,
        "n_chains": res.n_chains,
        "correct": n_correct,
        "wrong": n_wrong,
        "timeouts": n_timeout,
        "model_calls_total": res.model_calls,
        "avg_rounds_solved": avg_rounds_solved,
        "avg_resets": avg_resets,
        "step_cfg": asdict(step_cfg),
        "max_rounds": eval_max_rounds,
        "diag": {
            "total_deduced": res.diag_total_deduced,
            "total_unsound_deductions": res.diag_total_unsound_deductions,
            "unsound_rate": unsound_rate,
            "conflict_tp": res.diag_conflict_tp,
            "conflict_fp": res.diag_conflict_fp,
            "conflict_fn": res.diag_conflict_fn,
            "conflict_tn": res.diag_conflict_tn,
            "conflict_precision": cls_p,
            "conflict_recall": cls_r,
            "active_chain_rounds": res.diag_active_chain_rounds,
        },
    }, indent=2))

    eval_jsonl_path = ckpt_path.with_suffix(".eval.jsonl")
    with eval_jsonl_path.open("w") as fh:
        fh.write(json.dumps({
            "kind": "header",
            "checkpoint": str(ckpt_path),
            "n_eval_puzzles": n,
            "n_chains": res.n_chains,
            "max_rounds": eval_max_rounds,
            "step_cfg": asdict(step_cfg),
            "run_args": {name: _arg_values[name] for name, _ in _RUN_PARAMS},
            "summary": {
                "correct": n_correct,
                "wrong": n_wrong,
                "timeouts": n_timeout,
                "model_calls_total": res.model_calls,
                "avg_rounds_solved": avg_rounds_solved,
                "avg_resets": avg_resets,
                "unsound_rate": unsound_rate,
                "conflict_p": cls_p,
                "conflict_r": cls_r,
            },
        }) + "\n")
        for i in range(n):
            is_correct = bool(res.correct[i].item())
            is_wrong = bool(res.wrong[i].item())
            is_timeout = bool(res.timeouts[i].item())
            rs = int(res.round_solved[i].item())
            forwards_unbatched = (
                (rs + 1) * eval_n_chains if is_correct
                else eval_max_rounds * eval_n_chains
            )
            fh.write(json.dumps({
                "kind": "puzzle",
                "puzzle_idx": i,
                "correct": is_correct,
                "wrong": is_wrong,
                "timeout": is_timeout,
                "round_solved": rs,
                "n_resets": int(res.n_resets[i].item()),
                "forwards_unbatched": forwards_unbatched,
            }) + "\n")
    checkpoint_volume.commit()

    return {
        "steps": steps,
        "batch_size": batch_size,
        "correct": n_correct,
        "wrong": n_wrong,
        "timeouts": n_timeout,
        "n_chains": res.n_chains,
        "checkpoint": str(ckpt_path),
    }


@app.local_entrypoint()
def entrypoint(
    steps: int = 4000,
    batch_size: int = 512,
    n_eval_puzzles: int = 1000,
    seed: int = 0,
    bce_pos_mult: float = 4.0,
    bce_neg_mult: float = 0.5,
    softmax_loss_weight: float = 0.2,
    conflict_loss_weight: float = 1.0,
    weight_decay: float = 0.01,
    lr: float = 3e-3,
    max_age: int = 100,
    warmup_fraction: float = 0.1,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.5,
    eval_cls_threshold: float = 0.6,
    eval_max_rounds: int = 1000,
    eval_n_chains: int = 64,
    eval_batch_size: int = -1,
    augment: bool = True,
    train_augment: bool = True,
    use_ema: bool = False,
    ema_decay: float = 0.999,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    eval_dropout_p: float = 0.05,
    n_loops: int = 16,
    pool_size_mult: float = 1.0,
    model_dim: int = 128,
    num_layers: int = 4,
    intrain_eval_every: int = 100,
    intrain_eval_max_rounds: int = 5,
    resume: bool = False,
    checkpoint_every: int = 1000,
    use_rope: bool = False,
    n_train_puzzles: int | None = None,
    n_test_puzzles: int | None = None,
):
    result = run.remote(
        steps=steps, batch_size=batch_size,
        n_eval_puzzles=n_eval_puzzles, seed=seed,
        bce_pos_mult=bce_pos_mult, bce_neg_mult=bce_neg_mult,
        softmax_loss_weight=softmax_loss_weight,
        conflict_loss_weight=conflict_loss_weight,
        weight_decay=weight_decay, lr=lr, max_age=max_age,
        warmup_fraction=warmup_fraction,
        threshold=threshold, temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        eval_cls_threshold=eval_cls_threshold,
        eval_max_rounds=eval_max_rounds,
        eval_n_chains=eval_n_chains,
        eval_batch_size=eval_batch_size,
        augment=augment, train_augment=train_augment,
        use_ema=use_ema, ema_decay=ema_decay,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        eval_dropout_p=eval_dropout_p,
        n_loops=n_loops, pool_size_mult=pool_size_mult,
        model_dim=model_dim, num_layers=num_layers,
        intrain_eval_every=intrain_eval_every,
        intrain_eval_max_rounds=intrain_eval_max_rounds,
        resume=resume, checkpoint_every=checkpoint_every,
        use_rope=use_rope,
        n_train_puzzles=n_train_puzzles,
        n_test_puzzles=n_test_puzzles,
    )
    print(f"\nFinal: {result}", flush=True)
