"""Generate replay traces from a minesweeper LDT checkpoint.

Runs a single-chain solve on N test puzzles and records every cell commitment
(reveal/flag) per round plus reset events.  Output is a JSONL file compatible
with minesweeper-sym/replay.py.

Usage:
    uv run modal run experiments/minesweeper/trace_gen.py \
        --checkpoint /checkpoints/minesweeper/<ckpt>.pt \
        --n 20
"""

import json

import modal
import torch
from torch import nn

from lattice_diffusion.models.looped_transformer import LoopedTransformerConfig, PowersetModel
from lattice_diffusion.modal.image import (
    CHECKPOINT_MOUNT, DATA_MOUNT,
    checkpoint_volume, data_volume, hf_secret, image,
)
from lattice_diffusion.training.utils.checkpoint import load_checkpoint

from experiments.sudoku.dpll import StepConfig, dpll_step
from experiments.sudoku.ema import swap_in_ema_if_present
from experiments.minesweeper.data import MinesweeperConfig, MinesweeperDataset


app = modal.App("minesweeper-trace-gen")


def _trace_single_puzzle(model, x, given_mask, step_cfg, max_rounds, cols):
    """Run a K=1 solve on one puzzle and return a list of trace moves.

    Each move is {"type": "reveal"|"flag"|"round"|"reset", "pos": [r,c]|null}.
    """
    device = x.device
    state    = x.unsqueeze(0).clone()    # [1, S, C]
    original = x.unsqueeze(0).clone()
    gm       = given_mask.unsqueeze(0)   # [1, S]

    moves = []

    for _round in range(max_rounds):
        pre_state = state.clone()
        new_state, conflict, just_solved, _, _ = dpll_step(
            model, state, gm, step_cfg, want_stats=False,
        )

        # Cells that went from multi-alive → single-alive this round.
        pre_alive  = (pre_state[0] > 0.5).sum(-1)   # [S]
        post_alive = (new_state[0] > 0.5).sum(-1)   # [S]
        newly = ((pre_alive > 1) & (post_alive == 1)).nonzero(as_tuple=True)[0]

        for s in newly.tolist():
            r, c = s // cols, s % cols
            is_mine = bool(new_state[0, s, 0].item() > 0.5)
            moves.append({"type": "flag" if is_mine else "reveal", "pos": [r, c]})

        moves.append({"type": "round", "pos": None})

        if conflict[0]:
            moves.append({"type": "reset", "pos": None})
            state = original.clone()
        else:
            state = new_state

        if just_solved[0]:
            break

    return moves


@app.function(
    image=image, gpu="B200", timeout=3600,
    secrets=[hf_secret],
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def run(
    checkpoint: str,
    n: int = 20,
    max_rounds: int = 200,
    threshold: float = 0.10,
    temp_decide: float = 1.5,
    cls_threshold: float = 0.6,
    augment: bool = True,
    dropout_p: float = 0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    print(f"Loading: {checkpoint}", flush=True)
    ckpt = load_checkpoint(checkpoint)
    cfg  = LoopedTransformerConfig(**ckpt["model_cfg"])
    model = PowersetModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    swap_in_ema_if_present(model, ckpt)
    if dropout_p > 0:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p
        model.train()

    rows, cols = cfg.grid_rows, cfg.grid_cols

    eval_ds = MinesweeperDataset(MinesweeperConfig(
        train_path=f"{DATA_MOUNT}/minesweeper/train.jsonl",
        test_path=f"{DATA_MOUNT}/minesweeper/test.jsonl",
        split="test", n_puzzles=n, batch_size=n, seed=300, augment_dihedral=False,
    ))
    x_all, sols_all, sat = eval_ds.next_batch()
    eval_ds.close()
    y_all    = sols_all[:, 0]
    sat_mask = sat.bool()
    x_all    = x_all[sat_mask].to(device).float()
    y_all    = y_all[sat_mask].to(device).float()
    given_mask_all = (x_all.sum(-1) == 1)

    step_cfg = StepConfig(
        threshold=threshold,
        temp_decide=temp_decide,
        cls_threshold=cls_threshold,
        augment=augment,
        augment_dihedral=True,
        permute_digits=False,
    )

    traces = []
    for i in range(min(n, x_all.shape[0])):
        x  = x_all[i]
        gm = given_mask_all[i]
        y  = y_all[i]

        # Find start cell: the given cell nearest to board centre.
        given_cells = gm.nonzero(as_tuple=True)[0].tolist()
        centre = rows // 2 * cols + cols // 2
        start_s = min(given_cells, key=lambda s: abs(s - centre))
        start = [start_s // cols, start_s % cols]

        # All mines: ch0 is the GT mine channel.
        all_mines = []
        gt_argmax = y.argmax(-1)  # [S]
        for s in range(rows * cols):
            if int(gt_argmax[s].item()) == 0:
                all_mines.append([s // cols, s % cols])

        # Determine outcome (solve with K=1 and check)
        with torch.no_grad():
            trace_moves = _trace_single_puzzle(
                model, x, gm, step_cfg, max_rounds, cols,
            )

        # Check if correctly solved: final committed cells match GT
        # (just use the last state from the trace — re-run to get solution)
        state = x.unsqueeze(0).clone()
        original = x.unsqueeze(0).clone()
        gm_b = gm.unsqueeze(0)
        solved = False
        correct = False
        with torch.no_grad():
            for _r in range(max_rounds):
                new_state, conflict, just_solved, _, _ = dpll_step(
                    model, state, gm_b, step_cfg, want_stats=False,
                )
                if conflict[0]:
                    state = original.clone()
                else:
                    state = new_state
                if just_solved[0]:
                    solved = True
                    pred_idx = new_state[0].argmax(-1)
                    gt_idx   = y.argmax(-1)
                    correct  = bool((pred_idx == gt_idx).all().item())
                    break

        outcome = "correct" if correct else ("timeout" if not solved else "wrong")

        traces.append({
            "source": "ldt",
            "model": f"ldt-{checkpoint.split('/')[-1].replace('.pt', '')}",
            "puzzle_id": i,
            "rows": rows,
            "cols": cols,
            "mines": all_mines,
            "start": start,
            "moves": [{"type": "reveal", "pos": start}] + trace_moves,
            "outcome": outcome,
        })
        print(f"  puzzle {i}: {outcome}  moves={len(trace_moves)}", flush=True)

    out_path = checkpoint.replace(".pt", ".traces.jsonl")
    with open(out_path, "w") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")
    checkpoint_volume.commit()
    print(f"Wrote {len(traces)} traces → {out_path}", flush=True)
    return traces


@app.local_entrypoint()
def entrypoint(
    checkpoint: str,
    n: int = 20,
    max_rounds: int = 200,
    out: str = "/tmp/ldt_traces.jsonl",
):
    traces = run.remote(checkpoint=checkpoint, n=n, max_rounds=max_rounds)
    with open(out, "w") as fh:
        for t in traces:
            fh.write(json.dumps(t) + "\n")
    print(f"Saved {len(traces)} traces → {out}")
