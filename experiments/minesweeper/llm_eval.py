"""Evaluate an LLM via OpenRouter on Minesweeper deduction.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    uv run python experiments/minesweeper/llm_eval.py \
        --model google/gemini-2.0-flash-001 \
        --n 10 \
        --data /tmp/ms_test.jsonl

The script shows the LLM the initial revealed board and asks it to identify
all forced mines and safe cells (pure deduction, same task as LDT).
Scores: full-board correct / wrong / abstain.
"""

import argparse
import json
import re
import sys
from collections import deque

import requests


# ── board helpers ──────────────────────────────────────────────────────────────

def bfs_reveal(grid, start):
    rows, cols = len(grid), len(grid[0])
    revealed = set()
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) in revealed or not (0 <= r < rows and 0 <= c < cols):
            continue
        revealed.add((r, c))
        if grid[r][c] == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    queue.append((r + dr, c + dc))
    return revealed


def board_to_text(grid, revealed):
    rows, cols = len(grid), len(grid[0])
    lines = []
    header = "   " + " ".join(f"{c}" for c in range(cols))
    lines.append(header)
    lines.append("   " + "-" * (cols * 2 - 1))
    for r in range(rows):
        row_str = f"{r} | "
        for c in range(cols):
            if (r, c) in revealed:
                v = grid[r][c]
                row_str += ("." if v == 0 else str(v)) + " "
            else:
                row_str += "? "
        lines.append(row_str.rstrip())
    return "\n".join(lines)


def ground_truth(grid, revealed):
    """Return (mines, safes) — sets of (r,c) for unrevealed cells."""
    rows, cols = len(grid), len(grid[0])
    mines, safes = set(), set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in revealed:
                if grid[r][c] == -1:
                    mines.add((r, c))
                else:
                    safes.add((r, c))
    return mines, safes


# ── prompt ─────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert Minesweeper solver. You will be shown a partially-revealed \
9×9 board with 10 mines. Numbers show how many mines are adjacent (8 neighbours). \
A dot (.) means 0 adjacent mines. A question mark (?) means unrevealed.

Your job: using pure logical deduction, identify which unrevealed (?) cells are \
DEFINITELY mines and which are DEFINITELY safe. Do not guess — only report cells \
you can prove.

OUTPUT FORMAT — you MUST start your response with these two lines, \
before any explanation:
MINES: (r,c) (r,c) ...
SAFE: (r,c) (r,c) ...

Then optionally explain. Use 0-indexed row,col. \
If none provable write MINES: none and SAFE: none on the first two lines."""


def make_prompt(board_text):
    return f"Board (row | col 0-8):\n\n{board_text}\n\nSolve."


# ── OpenRouter call ────────────────────────────────────────────────────────────

def call_openrouter(api_key, model, system, user, max_tokens=1024):
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"], data.get("usage", {})


# ── response parsing ───────────────────────────────────────────────────────────

def parse_coords(text):
    return set(tuple(int(x) for x in m) for m in re.findall(r"\((\d+),\s*(\d+)\)", text))


def parse_response(response):
    mines_section = re.search(r"MINES:\s*(.*?)(?:SAFE:|$)", response, re.DOTALL | re.IGNORECASE)
    safe_section  = re.search(r"SAFE:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
    pred_mines = parse_coords(mines_section.group(1)) if mines_section else set()
    pred_safes = parse_coords(safe_section.group(1))  if safe_section  else set()
    return pred_mines, pred_safes


# ── scoring ────────────────────────────────────────────────────────────────────

def score(pred_mines, pred_safes, true_mines, true_safes):
    """Returns dict of per-puzzle stats."""
    all_pred    = pred_mines | pred_safes
    all_unknown = true_mines | true_safes

    # Cells the model claimed something about
    correct_mines = pred_mines & true_mines
    wrong_mines   = pred_mines & true_safes   # called mine but was safe
    correct_safes = pred_safes & true_safes
    wrong_safes   = pred_safes & true_mines   # called safe but was mine — fatal

    n_wrong  = len(wrong_mines) + len(wrong_safes)
    n_correct_cells = len(correct_mines) + len(correct_safes)
    n_claimed = len(all_pred)
    n_unknown = len(all_unknown)

    # Fully correct: got every unrevealed cell right and no errors
    fully_correct = (pred_mines == true_mines and pred_safes == true_safes)
    has_error     = n_wrong > 0

    return {
        "fully_correct": fully_correct,
        "has_error": has_error,
        "n_claimed": n_claimed,
        "n_unknown": n_unknown,
        "n_correct_cells": n_correct_cells,
        "n_wrong_cells": n_wrong,
        "wrong_mines": sorted(wrong_mines),    # called mine, was safe
        "wrong_safes": sorted(wrong_safes),    # called safe, was mine (fatal!)
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data", default="/tmp/ms_test.jsonl")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw model response for first puzzle")
    parser.add_argument("--save-traces", default=None, metavar="PATH",
                        help="Save replay traces as JSONL to this path")
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY or pass --api-key")

    with open(args.data) as f:
        puzzles = [json.loads(l) for l in f if l.strip()][:args.n]

    print(f"Model: {args.model}")
    print(f"Puzzles: {len(puzzles)}")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_puzzle(i, puzzle):
        grid     = puzzle["grid"]
        start    = tuple(puzzle["start"])
        revealed = bfs_reveal(grid, start)
        true_mines, true_safes = ground_truth(grid, revealed)
        board_text = board_to_text(grid, revealed)
        prompt     = make_prompt(board_text)
        try:
            response, usage = call_openrouter(api_key, args.model, SYSTEM, prompt)
        except Exception as e:
            print(f"[{i:2d}] API ERROR: {e}", flush=True)
            err_s = {"fully_correct": False, "has_error": True,
                     "n_wrong_cells": 0, "n_claimed": 0,
                     "n_unknown": len(true_mines | true_safes),
                     "wrong_safes": [], "wrong_mines": [], "n_correct_cells": 0}
            return i, None, {}, err_s, None
        if args.debug and i == 0:
            print(f"\n--- RAW RESPONSE [{i}] ---\n{response}\n--- END ---\n", flush=True)
        pred_mines, pred_safes = parse_response(response)
        # Filter out already-revealed cells from predictions
        pred_mines = pred_mines - revealed
        pred_safes = pred_safes - revealed
        s = score(pred_mines, pred_safes, true_mines, true_safes)

        # Build replay trace
        mines_list = [[r, c] for r, c in
                      sorted((r, c) for r, c in true_mines | true_safes
                             if grid[r][c] == -1)]
        # All mines on the board (not just unknown ones)
        all_mines = [[r, c] for r in range(len(grid))
                     for c in range(len(grid[0])) if grid[r][c] == -1]
        outcome = ("correct" if s["fully_correct"] else
                   ("wrong" if s["has_error"] else "partial"))
        trace_moves = [{"type": "reveal", "pos": list(start)}]
        for pos in sorted(pred_mines):
            trace_moves.append({"type": "flag", "pos": list(pos)})
        for pos in sorted(pred_safes):
            trace_moves.append({"type": "reveal", "pos": list(pos)})
        if s.get("wrong_safes"):
            trace_moves.append({"type": "reset", "pos": None})
        trace = {
            "source": "llm",
            "model": args.model,
            "puzzle_id": i,
            "rows": len(grid), "cols": len(grid[0]),
            "mines": all_mines,
            "start": list(start),
            "moves": trace_moves,
            "outcome": outcome,
        }
        return i, response, usage, s, trace

    results = [None] * len(puzzles)
    traces  = [None] * len(puzzles)
    total_input_tokens = total_output_tokens = 0

    with ThreadPoolExecutor(max_workers=len(puzzles)) as ex:
        futures = {ex.submit(run_puzzle, i, p): i for i, p in enumerate(puzzles)}
        for fut in as_completed(futures):
            i, response, usage, s, trace = fut.result()
            results[i] = s
            traces[i]  = trace
            if usage:
                total_input_tokens  += usage.get("prompt_tokens", 0)
                total_output_tokens += usage.get("completion_tokens", 0)
            status = "CORRECT" if s["fully_correct"] else ("WRONG" if s["has_error"] else "PARTIAL")
            print(f"[{i:2d}] {status:8s}  "
                  f"claimed={s['n_claimed']}/{s['n_unknown']}  "
                  f"correct_cells={s['n_correct_cells']}  "
                  f"wrong={s['n_wrong_cells']}")
            if s.get("wrong_safes"):
                print(f"       !! called safe but was mine: {s['wrong_safes']}")
            if s.get("wrong_mines"):
                print(f"       !! called mine but was safe: {s['wrong_mines']}")

    # Save traces
    if args.save_traces:
        with open(args.save_traces, "w") as fh:
            for t in traces:
                if t is not None:
                    fh.write(json.dumps(t) + "\n")
        print(f"Traces saved to {args.save_traces}")

    # Summary
    print("=" * 60)
    n = len(results)
    n_correct = sum(r["fully_correct"] for r in results)
    n_wrong   = sum(r["has_error"] for r in results)
    n_partial = n - n_correct - n_wrong
    print(f"Fully correct : {n_correct}/{n}")
    print(f"Has errors    : {n_wrong}/{n}")
    print(f"Partial/abstain: {n_partial}/{n}")
    print(f"Tokens used   : {total_input_tokens} in / {total_output_tokens} out")


if __name__ == "__main__":
    main()
