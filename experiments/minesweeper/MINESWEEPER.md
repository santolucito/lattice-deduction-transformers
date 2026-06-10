# Minesweeper — LDT Experiment Notes

Minesweeper was added as a fourth puzzle domain to test whether the LDT
architecture can learn constraint propagation over **cardinality constraints**
(`sum(neighbors) = N`), as distinct from the equality and uniqueness constraints
that drive Sudoku and Snowflake.

Boards come from [minesweeper-sym](https://github.com/santolucito/minesweeper-sym),
a CVC5-backed generator that only accepts boards solvable by pure logic — no
guessing required. This makes `correct_rate` a clean metric: a wrong answer
is a model error, not an ambiguous board.

---

## Encoding

**10 channels per cell:**

| ch | Meaning |
|---|---|
| 0 | `mine` |
| 1–9 | `safe_0` through `safe_8` (safe + N adjacent mines) |

| Cell state | Input `x` | Target `y` |
|---|---|---|
| Revealed safe, count N | one-hot at `ch(1+N)` | one-hot at `ch(1+N)` |
| Unrevealed | all-ones `[1,…,1]` | one-hot at `ch0` or `ch(1+N)` |

`given_mask = (x.sum(-1) == 1)` — same convention as every other domain.

The initial revealed region is the BFS flood-fill from the board centre
`(rows//2, cols//2)`, identical to the in-game first-click reveal.

---

## Data

**9×9 Beginner** (81 cells, 10 mines, centre start).

```bash
cd /path/to/minesweeper-sym
python3 dataset.py -n 10000 --difficulty 1 --seed 42   -o train.jsonl
python3 dataset.py -n 1000  --difficulty 1 --seed 9999 -o test.jsonl

uv run modal volume put lattice-diffusion-data train.jsonl minesweeper/train.jsonl
uv run modal volume put lattice-diffusion-data test.jsonl  minesweeper/test.jsonl
```

**Data quality note:** the raw test split contains ~146 duplicate mine
configurations out of 1,000 boards (same mines, same grid). Downstream evals
on small samples should de-duplicate or use the Modal-volume copy
(loaded with a fixed shuffle seed) to avoid testing the same board twice.

---

## Training

### Run 1 — 4,000 steps

```bash
uv run modal run --detach experiments/minesweeper/run.py --steps 4000
```

| Parameter | Value |
|---|---|
| Steps | 4,000 |
| Batch size | 512 |
| LR | 3e-3 |
| Model | dim=128, n\_loops=16, n\_heads=4, n\_layers=4 |
| Augmentation | dihedral D4 (no digit permutation) |
| `permute_digits` | False — mine vs. safe channels have distinct semantics |

**Checkpoint:**
`/checkpoints/minesweeper/minesweeper_9x9_seed0_4000s_bs512_aug1_20260606_163351_20260606_163352.pt`

**Eval on 1,000 test puzzles:**

| Metric | Value |
|---|---|
| Correct (full solve) | 149 / 1,000 (14.9%) |
| Wrong | 839 / 1,000 |
| Timeouts | 12 / 1,000 |
| Conflict head P / R | 0.974 / 0.207 |
| Avg resets per puzzle | 556 |
| Deduction soundness | 99.89% |

**Bottleneck:** conflict-head **recall is 0.207**. The model detects only
~1 in 5 actual contradictions after a wrong guess, so bad chains continue
to run rather than resetting. Most of the 14.9% success comes from the
restart-and-retry mechanism, not pure deduction.

### Run 2 — 20,000 steps (killed at ~12,100)

Upward trend through steps 4k–10k (floor rising from ~7 to ~17 per 200 mini-eval),
but no breakthrough; conflict-head recall oscillated 0.25–0.75 without converging.
Run stopped manually.

---

## Solve rate with and without restarts

On a 20-puzzle sample from the Modal test split, measuring only
**round-0 deductions** (before any decision or conflict reset) gives
a fairer picture of the model's raw deductive ability:

| Condition | Solved | Note |
|---|---|---|
| Full eval (restarts allowed, 1,000 puzzles) | 14.9% | includes stochastic lucky restarts |
| No restarts, 20-puzzle sample | 2 / 20 (10%) | only round-0 deductions committed |

Puzzle 14 had no resets, correctly identified all 10 mines and 19 safe cells,
but timed out before the model's `just_solved` signal fired at `max_rounds=100`.

---

## LLM comparison

To compare fairly, both the LDT and an LLM are given **one look at the
initial flood-fill board** with no feedback. The LLM sees the board once
and outputs all its claims in a single response; the LDT is scored on its
**round-0 deductions only** (before any decision or reset).

**Model:** `google/gemini-2.5-flash-lite` via OpenRouter

```bash
export OPENROUTER_API_KEY=<key>
uv run python experiments/minesweeper/llm_eval.py \
    --model google/gemini-2.5-flash-lite \
    --n 20 \
    --data /tmp/ldt_puzzles.jsonl \
    --save-traces /tmp/llm_traces.jsonl
```

### Fair head-to-head (same 20 unique puzzles, same initial information)

| | LLM (one shot) | LDT (round 0 only) |
|---|---|---|
| Cells claimed | 216 | 219 |
| Correct claims | 116 | 213 |
| Wrong claims | 100 | 6 |
| **Precision** | **53.7%** | **97.3%** |
| Fully correct puzzles | 0 / 20 | 0 / 20 |

The LDT claims roughly the same number of cells as the LLM but is nearly
sound: only 6 wrong deductions out of 219. The LLM's informal reasoning
produces ~100 errors on the same 20 boards.

Neither solver fully solves any puzzle from round-0 deductions alone — both
require iterative information gathering (more reveals) to make progress.

**Key takeaway:** the LDT has genuinely learned constraint propagation; its
deductions are trustworthy. The bottleneck is the guess-and-backtrack loop
once the initial constraints are exhausted, not the quality of the deductions
themselves.

---

## Theoretical note — why the powerset lattice struggles here

The per-cell powerset lattice that underlies the LDT's abstract domain is
**vacuous** for cardinality constraints.

A Minesweeper constraint has the form `x_a + x_b + x_c = 1` (one mine among
three neighbours). The only information the powerset lattice can extract
by bit-OR collapse is: at least one cell is a mine, and at least two cells
are safe — but only in the degenerate cases where a cell's entire 8-bit
possibility set collapses to a singleton. For mixed assignments, the
bit-OR of all valid assignments gives `⊤` (all bits alive), so the abstract
deduction is empty.

In contrast, Sudoku constraints (each digit appears exactly once per
row/col/box) are equality constraints over a partition, which the powerset
domain handles naturally. Minesweeper's counting constraints would be better
matched by a **quantitative lattice** (per-cell mine-probability marginals)
or a **relational domain** (Karr's affine-equality domain), both of which
would require architectural changes beyond the current token-per-cell design.

---

## Replay traces

Replay traces capture every move (reveal, flag, reset, round boundary) for
both the LDT and the LLM, in a format consumable by the
[minesweeper-sym](https://github.com/santolucito/minesweeper-sym) replay
viewer.

### Generate LDT traces (single-chain, records per-round deductions)

```bash
uv run modal run experiments/minesweeper/trace_gen.py \
    --checkpoint /checkpoints/minesweeper/<ckpt>.pt \
    --n 20 \
    --out /tmp/ldt_traces.jsonl
```

### Generate LLM traces

```bash
uv run python experiments/minesweeper/llm_eval.py \
    --model google/gemini-2.5-flash-lite \
    --n 20 \
    --data /tmp/ldt_puzzles.jsonl \
    --save-traces /tmp/llm_traces.jsonl
```

### Watch the replay

```bash
cd /path/to/minesweeper-sym
# Interleaved LLM + LDT for direct comparison (press N to step through)
python3 replay.py /tmp/merged_traces.jsonl --speed 4

# Controls: Space=pause  N/P=next/prev  R=restart  ↑↓=speed  Esc=quit
```

### Trace format (JSONL, one game per line)

```json
{
  "source": "ldt",
  "model": "ldt-minesweeper_9x9_seed0_4000s_...",
  "puzzle_id": 6,
  "rows": 9, "cols": 9,
  "mines": [[0,4], [2,5], ...],
  "start": [4, 4],
  "moves": [
    {"type": "reveal", "pos": [4, 4]},
    {"type": "flag",   "pos": [0, 4]},
    {"type": "reveal", "pos": [0, 5]},
    {"type": "round",  "pos": null},
    {"type": "reset",  "pos": null}
  ],
  "outcome": "correct"
}
```

Move types:
- `reveal` — reveal a hidden cell (may flood-fill if adjacency = 0)
- `flag` — mark a hidden cell as a mine
- `round` — LDT round boundary (visual pause only, no state change)
- `reset` — conflict detected; replay resets board to initial flood-fill state

---

## Files

| File | Purpose |
|---|---|
| `data.py` | JSONL dataset loader with BFS flood-fill reveal |
| `train.py` | Pool-based trainer (mirrors `maze/train.py`) |
| `run.py` | Modal training + inline eval entrypoint |
| `eval_only.py` | Re-evaluate a saved checkpoint on Modal |
| `llm_eval.py` | Evaluate an LLM via OpenRouter; `--save-traces` for replay |
| `trace_gen.py` | Single-chain LDT trace generation on Modal |

Replay viewer lives in the minesweeper-sym repo at `replay.py`.
