"""Eval an existing checkpoint with the hybrid-batched solver.

Eval data: seed=200, zero_hint_weight=1.0 (all SAT), n=200 (post-filter).

Optionally fans out the eval across N modal workers (each on its own B200);
each worker compiles the model, processes its puzzle slice, and returns
per-puzzle rows + diag aggregates that the driver merges.

Usage:
    uv run modal run --detach experiments/maze/eval_only.py \
      --checkpoint /checkpoints/maze/maze_hard_30x30_seed0_20000s_bs192_aug1_<ts>.pt \
      --workers 10 --batch-size 64 --dataset maze_hard --n-eval 1000
"""

import json
import time
from dataclasses import asdict, dataclass

import modal
import torch
from torch import nn

from experiments.maze.data import (
    MazeConfig, MazeDataset, grid_dims, make_maze_label_fn, rescore,
)
from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.sudoku.solve import SolveConfig, solve


app = modal.App("maze-eval-only")


@dataclass
class _WorkerArgs:
    """Self-contained eval-config bundle handed to each worker. All fields
    are scalars / strings so it ships cleanly across Modal RPC."""
    checkpoint: str
    n_eval: int
    threshold: float
    temp_decide: float
    cls_threshold: float
    n_chains: int
    batch_size: int
    max_rounds: int
    augment: bool
    estimate_sequential: bool
    seq_drain_max_rounds: int
    dropout_p: float
    log_per_round_fill: bool
    dataset: str
    grid_size: int | None
    indices: list[int]   # per-worker puzzle indices into the post-filter pool


def _summarize_solve_result(
    res, y_chunk, H: int, W: int, indices: list[int],
    n_chains: int, max_rounds: int,
    estimate_sequential: bool, log_per_round_fill: bool,
) -> dict:
    """Convert a SolveResult over `indices` puzzles into a dict that
    serializes cleanly across modal RPC.

    Calls `rescore()` on the result so we get maze 5-way labels per puzzle.
    Returns per-puzzle rows (with original puzzle_idx) + aggregate diag totals.
    """
    rescore(res, y_chunk, H, W)
    valid_c = res._maze_valid
    gt_exact_c = res._maze_gt_exact

    rows: list[dict] = []
    for j, orig_i in enumerate(indices):
        is_correct = bool(res.correct[j].item())
        is_wrong = bool(res.wrong[j].item())
        is_timeout = bool(res.timeouts[j].item())
        rs = int(res.round_solved[j].item())
        fwd = (rs + 1) * n_chains if is_correct else max_rounds * n_chains
        if is_timeout:
            label = "TIMEOUT"
        elif is_correct and bool(gt_exact_c[j].item()):
            label = "CORRECT_GT"
        elif is_correct:
            label = "CORRECT_ALT"
        elif bool(valid_c[j].item()):
            label = "WRONG_VALID"
        else:
            label = "WRONG_INVALID"
        row = {
            "kind": "puzzle",
            "puzzle_idx": orig_i,
            "correct": is_correct,
            "wrong": is_wrong,
            "timeout": is_timeout,
            "label": label,
            "round_solved": rs,
            "n_resets": int(res.n_resets[j].item()),
            "forwards_unbatched": fwd,
            "puzzle_calls": int(res.puzzle_calls[j].item()),
        }
        if estimate_sequential:
            seq_v = int(res.forwards_seq[j].item())
            w_idx = int(res.seq_winning_idx[j].item())
            done = int(res.seq_attempts_done[j].item())
            row["forwards_seq"] = seq_v
            row["seq_winning_idx"] = w_idx
            row["seq_attempts_done"] = done
            row["seq_avg_attempt_len"] = (
                (seq_v / max(w_idx + 1, 1)) if w_idx >= 0 else None
            )
        if log_per_round_fill and is_correct:
            row["n_givens"] = res.n_givens[j]
            row["deduction_fills_per_round"] = res.deduction_fills_per_round[j]
            row["decision_fills_per_round"] = res.decision_fills_per_round[j]
            row["deduction_bitflips_per_round"] = res.deduction_bitflips_per_round[j]
            row["decision_bitflips_per_round"] = res.decision_bitflips_per_round[j]
        rows.append(row)

    return {
        "rows": rows,
        "diag": {
            "model_calls": int(res.model_calls),
            "total_deduced": int(res.diag_total_deduced),
            "total_unsound_deductions": int(res.diag_total_unsound_deductions),
            "conflict_tp": int(res.diag_conflict_tp),
            "conflict_fp": int(res.diag_conflict_fp),
            "conflict_fn": int(res.diag_conflict_fn),
            "conflict_tn": int(res.diag_conflict_tn),
            "active_chain_rounds": int(res.diag_active_chain_rounds),
        },
    }


@app.function(
    image=image, gpu="B200", timeout=21600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def eval_chunk(args: _WorkerArgs) -> dict:
    """Run the solver over `args.indices` puzzles. Each worker loads the
    full eval pool from cache, then `index_select`s its slice — we pay the
    cache-load cost once per worker but compile the model fresh each time
    (Modal containers don't share state)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    print(f"[worker] loading {args.checkpoint}", flush=True)
    ckpt = load_checkpoint(args.checkpoint)
    cfg = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if args.dropout_p > 0.0:
        n_drop = 0
        n_mha = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = args.dropout_p
                n_drop += 1
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = args.dropout_p
                n_mha += 1
        model.train()
        print(f"[worker] dropout-noise: overrode {n_drop} Dropout + {n_mha} "
              f"MHA-attn-dropout to p={args.dropout_p}, model in train()", flush=True)
    if device.type == "cuda":
        print("[worker] torch.compile(dynamic=False) …", flush=True)
        model = torch.compile(model, dynamic=False)
    print(f"[worker] params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    eval_ds_cfg = MazeConfig(
        dataset=args.dataset,
        cache_dir=DATA_MOUNT,
        split="test" if args.dataset == "maze_hard" else "train",
        n_puzzles=args.n_eval,
        batch_size=args.n_eval,
        seed=200,
        grid_size=args.grid_size,
        augment_dihedral=False,
        augment_swap_endpoints=False,
    )
    H, W = grid_dims(eval_ds_cfg)
    eval_ds = MazeDataset(eval_ds_cfg)
    x_all, sols_all, sat = eval_ds.next_batch(); eval_ds.close()
    y_all = sols_all[:, 0]
    sat_mask = sat.bool()
    x_full = x_all[sat_mask].to(device).float()
    y_full = y_all[sat_mask].to(device).float()
    given_full = (x_full.sum(dim=-1) == 1)
    n_total = x_full.shape[0]

    idx_t = torch.tensor(args.indices, dtype=torch.long, device=device)
    x = x_full.index_select(0, idx_t)
    y = y_full.index_select(0, idx_t)
    given_mask = given_full.index_select(0, idx_t)
    print(f"[worker] eval pool loaded: {n_total} total, "
          f"this worker handles {len(args.indices)} indices "
          f"({args.indices[0]}..{args.indices[-1]})", flush=True)

    step_cfg = StepConfig(
        threshold=args.threshold,
        temp_decide=args.temp_decide,
        cls_threshold=args.cls_threshold,
        augment=args.augment,
        augment_dihedral=True,
        permute_digits=False,
    )
    solve_cfg = SolveConfig(
        step=step_cfg, max_rounds=args.max_rounds,
        n_chains=args.n_chains, batch_size=args.batch_size,
        estimate_sequential=args.estimate_sequential,
        seq_drain_max_rounds=args.seq_drain_max_rounds,
        log_per_round_fill=args.log_per_round_fill,
    )

    t0 = time.time()
    res = solve(model, x, y, given_mask, solve_cfg,
                label_fn=make_maze_label_fn(H, W))
    elapsed = time.time() - t0
    print(f"[worker] solve done in {elapsed:.0f}s", flush=True)

    summary = _summarize_solve_result(
        res, y, H, W, args.indices, args.n_chains, args.max_rounds,
        args.estimate_sequential, args.log_per_round_fill,
    )
    summary["wall_seconds"] = elapsed
    summary["n_total"] = n_total
    return summary


@app.function(
    image=image, gpu=None, timeout=600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def _count_puzzles(dataset: str, n_eval: int, grid_size: int | None) -> int:
    """Tiny helper that loads the eval pool and returns `n_total` so the
    driver can split indices without doing GPU work locally."""
    eval_ds_cfg = MazeConfig(
        dataset=dataset,
        cache_dir=DATA_MOUNT,
        split="test" if dataset == "maze_hard" else "train",
        n_puzzles=n_eval,
        batch_size=n_eval,
        seed=200,
        grid_size=grid_size,
        augment_dihedral=False,
        augment_swap_endpoints=False,
    )
    eval_ds = MazeDataset(eval_ds_cfg)
    x_all, _, sat = eval_ds.next_batch(); eval_ds.close()
    return int(sat.bool().sum().item())


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    n_eval: int = 200,
    threshold: float = 0.5,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    n_chains: int = 64,
    batch_size: int = 512,
    max_rounds: int = 1000,
    augment: bool = True,
    estimate_sequential: bool = False,
    seq_drain_max_rounds: int = 200,
    dropout_p: float = 0.05,
    log_per_round_fill: bool = False,
    dataset: str = "maze_hard",
    grid_size: int | None = None,
    out_suffix: str = ".eval.fixed.json",
    workers: int = 1,
):
    print(f"[driver] counting puzzles in eval pool …", flush=True)
    n_total = _count_puzzles.remote(dataset, n_eval, grid_size)
    print(f"[driver] {n_total} SAT eval puzzles", flush=True)

    # Round-robin index assignment so each worker sees an equally-mixed
    # difficulty distribution rather than e.g. all-easy-first.
    workers = max(1, min(workers, n_total))
    chunks: list[list[int]] = [[] for _ in range(workers)]
    for i in range(n_total):
        chunks[i % workers].append(i)
    chunks = [c for c in chunks if c]
    print(f"[driver] fanning out to {len(chunks)} workers "
          f"(min={min(len(c) for c in chunks)}, "
          f"max={max(len(c) for c in chunks)} puzzles each)", flush=True)

    worker_args = [
        _WorkerArgs(
            checkpoint=checkpoint, n_eval=n_eval,
            threshold=threshold, temp_decide=temp_decide,
            cls_threshold=cls_threshold,
            n_chains=n_chains, batch_size=batch_size, max_rounds=max_rounds,
            augment=augment,
            estimate_sequential=estimate_sequential,
            seq_drain_max_rounds=seq_drain_max_rounds,
            dropout_p=dropout_p, log_per_round_fill=log_per_round_fill,
            dataset=dataset, grid_size=grid_size,
            indices=ids,
        )
        for ids in chunks
    ]

    t0 = time.time()
    # `.map(args)` fans out each item to a fresh container, in parallel.
    parts = list(eval_chunk.map(worker_args))
    elapsed = time.time() - t0
    print(f"[driver] all workers done in {elapsed:.0f}s", flush=True)

    # Merge per-puzzle rows by puzzle_idx, sum diag aggregates.
    rows_by_idx: dict[int, dict] = {}
    accum = {
        "model_calls": 0,
        "total_deduced": 0,
        "total_unsound_deductions": 0,
        "conflict_tp": 0,
        "conflict_fp": 0,
        "conflict_fn": 0,
        "conflict_tn": 0,
        "active_chain_rounds": 0,
    }
    for part in parts:
        for row in part["rows"]:
            rows_by_idx[row["puzzle_idx"]] = row
        for k in accum:
            accum[k] += part["diag"][k]

    rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
    n = len(rows)

    # Aggregate counts.
    n_correct = sum(1 for r in rows if r["correct"])
    n_correct_gt = sum(1 for r in rows if r.get("label") == "CORRECT_GT")
    n_correct_alt = sum(1 for r in rows if r.get("label") == "CORRECT_ALT")
    n_wrong = sum(1 for r in rows if r["wrong"])
    n_wrong_valid = sum(1 for r in rows if r.get("label") == "WRONG_VALID")
    n_wrong_invalid = sum(1 for r in rows if r.get("label") == "WRONG_INVALID")
    n_timeout = sum(1 for r in rows if r["timeout"])
    avg_calls = accum["model_calls"] / max(n_correct, 1)
    den = max(accum["total_deduced"], 1)
    unsound_rate = accum["total_unsound_deductions"] / den
    cls_p = accum["conflict_tp"] / max(accum["conflict_tp"] + accum["conflict_fp"], 1)
    cls_r = accum["conflict_tp"] / max(accum["conflict_tp"] + accum["conflict_fn"], 1)

    print(f"\n{'='*60}\nRESULT ({n} puzzles, {len(parts)} workers)\n{'='*60}", flush=True)
    print(f"  correct={n_correct}/{n}  (gt={n_correct_gt} alt={n_correct_alt})  "
          f"wrong={n_wrong}  (valid={n_wrong_valid} invalid={n_wrong_invalid})  "
          f"timeouts={n_timeout}", flush=True)
    print(f"  total_calls={accum['model_calls']}  avg/correct={avg_calls:.1f}", flush=True)
    print(f"  Deduction soundness: {accum['total_unsound_deductions']} unsound / "
          f"{accum['total_deduced']} deduced  (rate={unsound_rate:.4%})", flush=True)
    print(f"  Conflict head P={cls_p:.3f} R={cls_r:.3f} "
          f"[tp={accum['conflict_tp']} fp={accum['conflict_fp']} "
          f"fn={accum['conflict_fn']} tn={accum['conflict_tn']}] "
          f"over {accum['active_chain_rounds']} active chain-rounds", flush=True)
    print(f"  driver wall: {elapsed:.0f}s "
          f"(slowest worker: {max(p['wall_seconds'] for p in parts):.0f}s)",
          flush=True)

    out = {
        "checkpoint": checkpoint,
        "n_eval": n,
        "solver": "hybrid_per_chunk_parallel",
        "n_workers": len(parts),
        "wall_seconds_driver": elapsed,
        "wall_seconds_max_worker": max(p["wall_seconds"] for p in parts),
        "solver_config": {
            "threshold": threshold, "temp_decide": temp_decide,
            "cls_threshold": cls_threshold,
            "n_chains": n_chains, "batch_size": batch_size,
            "max_rounds": max_rounds,
            "augment": augment, "dropout_p": dropout_p,
        },
        "summary": {
            "correct": n_correct,
            "correct_gt": n_correct_gt,
            "correct_alt": n_correct_alt,
            "wrong": n_wrong,
            "wrong_valid": n_wrong_valid,
            "wrong_invalid": n_wrong_invalid,
            "timeouts": n_timeout,
            "total_calls": accum["model_calls"],
            "avg_calls_per_correct": avg_calls,
        },
        "diag": {
            "total_deduced": accum["total_deduced"],
            "total_unsound_deductions": accum["total_unsound_deductions"],
            "unsound_rate": unsound_rate,
            "conflict_tp": accum["conflict_tp"],
            "conflict_fp": accum["conflict_fp"],
            "conflict_fn": accum["conflict_fn"],
            "conflict_tn": accum["conflict_tn"],
            "conflict_precision": cls_p,
            "conflict_recall": cls_r,
            "active_chain_rounds": accum["active_chain_rounds"],
        },
    }
    eval_path = checkpoint.replace(".pt", out_suffix)
    eval_jsonl_path = checkpoint.replace(".pt", out_suffix.replace(".json", ".jsonl"))

    # Write output via a tiny modal helper since the volume isn't mounted
    # locally.
    _write_eval_outputs.remote(eval_path, eval_jsonl_path, out, rows)
    print(f"Wrote {eval_path}", flush=True)
    print(f"Wrote {eval_jsonl_path}", flush=True)


@app.function(
    image=image, gpu=None, timeout=120,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def _write_eval_outputs(eval_path: str, eval_jsonl_path: str,
                        summary: dict, rows: list[dict]) -> None:
    with open(eval_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(eval_jsonl_path, "w") as fh:
        fh.write(json.dumps({"kind": "header", **summary}) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    checkpoint_volume.commit()
