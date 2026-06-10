"""Re-evaluate an existing minesweeper checkpoint with the hybrid-batched solver.

Usage:
    uv run modal run --detach experiments/minesweeper/eval_only.py \
      --checkpoint /checkpoints/minesweeper/minesweeper_9x9_seed0_4000s_bs512_aug1_<ts>.pt
"""

import json
import time

import modal
import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve
from experiments.minesweeper.data import MinesweeperConfig, MinesweeperDataset


app = modal.App("minesweeper-eval-only")


@app.function(
    image=image, gpu="B200", timeout=7200,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str,
    n_eval: int = 1000,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    out_suffix: str = ".eval.fixed.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Loading: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)

    if dropout_p > 0.0:
        n_drop = n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = dropout_p
                n_mha += 1
        model.train()
        print(f"  Dropout-noise eval: {n_drop} Dropout + {n_mha} MHA → p={dropout_p}",
              flush=True)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    eval_ds = MinesweeperDataset(MinesweeperConfig(
        train_path=f"{DATA_MOUNT}/minesweeper/train.jsonl",
        test_path=f"{DATA_MOUNT}/minesweeper/test.jsonl",
        split="test",
        n_puzzles=n_eval,
        batch_size=n_eval,
        seed=200,
        augment_dihedral=False,
    ))
    x_all, sols_all, sat = eval_ds.next_batch()
    eval_ds.close()
    # solutions: [B, 1, S, C] → [B, S, C]
    y_all = sols_all[:, 0]
    sat_mask = sat.bool()
    x = x_all[sat_mask].to(device).float()
    y = y_all[sat_mask].to(device).float()
    given_mask = (x.sum(dim=-1) == 1)
    n = x.shape[0]
    print(f"Loaded {n}/{n_eval} eval puzzles", flush=True)

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
        augment_dihedral=True,
        permute_digits=False,
    )
    solve_cfg = SolveConfig(
        step=step_cfg,
        max_rounds=max_rounds,
        n_chains=n_chains,
        batch_size=batch_size,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
    )
    print(f"Solving {n} puzzles | n_chains={n_chains} batch_size={batch_size} "
          f"max_rounds={max_rounds}", flush=True)

    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg)
    elapsed = time.time() - t0

    n_correct = int(res.correct.sum().item())
    n_wrong = int(res.wrong.sum().item())
    n_timeout = int(res.timeouts.sum().item())
    avg_calls = res.model_calls / max(n_correct, 1)

    den = max(res.diag_total_deduced, 1)
    unsound_rate = res.diag_total_unsound_deductions / den
    cls_p = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fp, 1)
    cls_r = res.diag_conflict_tp / max(res.diag_conflict_tp + res.diag_conflict_fn, 1)

    print(f"\n{'='*60}\nRESULT ({n} puzzles)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  wrong={n_wrong}  timeouts={n_timeout}", flush=True)
    print(f"  total_calls={res.model_calls}  avg/correct={avg_calls:.1f}", flush=True)
    print(f"  Deduction soundness: {res.diag_total_unsound_deductions} unsound / "
          f"{res.diag_total_deduced} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={res.diag_conflict_tp} fp={res.diag_conflict_fp} "
          f"fn={res.diag_conflict_fn} tn={res.diag_conflict_tn}]", flush=True)
    print(f"  wall: {elapsed:.0f}s", flush=True)

    out = {
        "checkpoint": checkpoint,
        "n_eval": n,
        "solver_config": {
            "threshold": threshold, "temp_decide": temp_decide,
            "cls_threshold": cls_threshold,
            "n_chains": n_chains, "batch_size": batch_size,
            "max_rounds": max_rounds, "augment": augment,
        },
        "summary": {
            "correct": n_correct, "wrong": n_wrong, "timeouts": n_timeout,
            "total_calls": res.model_calls,
            "avg_calls_per_correct": avg_calls,
            "avg_resets": float(res.n_resets.float().mean().item()),
        },
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
    }
    eval_path = checkpoint.replace(".pt", out_suffix)
    with open(eval_path, "w") as f:
        json.dump(out, f, indent=2)

    eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))
    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **out}) + "\n")
        for i in range(n):
            is_correct = bool(res.correct[i].item())
            rs = int(res.round_solved[i].item())
            forwards_unbatched = (
                (rs + 1) * n_chains if is_correct
                else max_rounds * n_chains
            )
            fh.write(json.dumps({
                "kind": "puzzle",
                "puzzle_idx": i,
                "correct": is_correct,
                "wrong": bool(res.wrong[i].item()),
                "timeout": bool(res.timeouts[i].item()),
                "round_solved": rs,
                "n_resets": int(res.n_resets[i].item()),
                "forwards_unbatched": forwards_unbatched,
            }) + "\n")
    checkpoint_volume.commit()
    print(f"Wrote {eval_path}", flush=True)
    return out


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    n_eval: int = 1000,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    out_suffix: str = ".eval.fixed.json",
):
    result = run.remote(
        checkpoint=checkpoint, n_eval=n_eval,
        threshold=threshold, temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        n_chains=n_chains, batch_size=batch_size,
        max_rounds=max_rounds, augment=augment,
        estimate_sequential=estimate_sequential,
        seq_drain_max_rounds=seq_drain_max_rounds,
        dropout_p=dropout_p,
        out_suffix=out_suffix,
    )
    print(f"\nFinal: {result}", flush=True)
