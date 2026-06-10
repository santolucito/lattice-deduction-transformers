"""Minesweeper dataset loader.

Reads pre-generated JSONL files produced by:
  python3 /path/to/minesweeper-sym/dataset.py -n N --difficulty 1 --seed S -o boards.jsonl

Each JSONL record:
  {id, rows, cols, num_mines, start, mines, grid, generation_attempts}
  - grid[r][c]: -1 = mine, 0-8 = adjacency count (full ground truth)
  - start: first-revealed cell (center by default, guaranteed safe)

All boards are CVC5-verified logically solvable: pure deduction reaches the
full solution without guessing.

Encoding: 10 channels per cell
  ch0:   mine
  ch1-9: safe_0 through safe_8 (safe cell with N adjacent mines)

Input x:  revealed cells → one-hot singleton; hidden → all-ones [1,…,1]
Target y: mine → ch0; safe-with-count-N → ch(1+N)

next_batch() returns (x, solutions, is_sat) where solutions: [B, 1, S, 10],
matching the maze interface (K=1 always for minesweeper).
"""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from queue import Queue

N_CHANNELS = 10  # mine + safe_0..safe_8


@dataclass
class MinesweeperConfig:
    train_path: str = "data/minesweeper/train.jsonl"
    test_path: str = "data/minesweeper/test.jsonl"
    split: str = "train"          # "train" or "test"
    batch_size: int = 512
    seed: int = 42
    n_puzzles: int | None = None  # None = use entire file; int = random subsample
    augment_dihedral: bool = True
    prefetch_batches: int = 2


def _bfs_reveal(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    """BFS flood-fill from start. Reveals start cell; if its value is 0,
    expands to all 8 neighbors and recurses on 0-neighbors. Returns set of
    revealed (r, c) positions (mines are never in this set because the JSONL
    start cell is always guaranteed safe by the generator).
    """
    rows, cols = grid.shape
    revealed: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) in revealed:
            continue
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        revealed.add((r, c))
        if grid[r, c] == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in revealed and 0 <= nr < rows and 0 <= nc < cols:
                        queue.append((nr, nc))
    return revealed


def _record_to_tensors(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert one JSONL record to (x, y) float32 arrays of shape [rows*cols, 10].

    x encodes the initial puzzle state (post BFS-reveal from start cell).
    y encodes the complete ground-truth solution.
    """
    rows: int = record["rows"]
    cols: int = record["cols"]
    grid = np.array(record["grid"], dtype=np.int8)  # [rows, cols]
    start = tuple(record["start"])                  # (row, col)

    revealed = _bfs_reveal(grid, start)

    n_cells = rows * cols
    x = np.ones((n_cells, N_CHANNELS), dtype=np.float32)   # default: all-ones (hidden)
    y = np.zeros((n_cells, N_CHANNELS), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            v = int(grid[r, c])
            if v == -1:
                y[i, 0] = 1.0       # mine → ch0
                # x stays all-ones (unrevealed mine)
            else:
                y[i, 1 + v] = 1.0  # safe-with-count-v → ch(1+v)
                if (r, c) in revealed:
                    x[i, :] = 0.0
                    x[i, 1 + v] = 1.0  # revealed → one-hot singleton
                # else: hidden safe → x stays all-ones

    return x, y


class MinesweeperDataset:
    """Finite-pool dataset from a pre-generated JSONL file.

    Loads all records into memory, encodes them as (x, y) arrays, then
    streams shuffled batches with optional D4 dihedral augmentation from a
    background thread.
    """

    def __init__(self, cfg: MinesweeperConfig):
        self.cfg = cfg
        path = cfg.train_path if cfg.split == "train" else cfg.test_path

        records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if cfg.n_puzzles is not None and cfg.n_puzzles < len(records):
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(len(records), size=cfg.n_puzzles, replace=False)
            records = [records[i] for i in sorted(idx)]

        self._N = len(records)
        print(f"  [minesweeper] loaded {self._N} {cfg.split} puzzles from {path}",
              flush=True)

        xs, ys = [], []
        for rec in records:
            x, y = _record_to_tensors(rec)
            xs.append(x)
            ys.append(y)
        self._pool_x = np.stack(xs, axis=0)   # [N, S, 10]
        self._pool_y = np.stack(ys, axis=0)   # [N, S, 10]
        self._S = self._pool_x.shape[1]       # seq_len = rows * cols

        # Infer grid side for dihedral aug (assumes square grid)
        self._side = int(round(self._S ** 0.5))
        assert self._side * self._side == self._S, (
            f"non-square grid (S={self._S}); dihedral aug requires square grid"
        )

        self._rng = np.random.default_rng(cfg.seed + 7)
        self._queue: Queue[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = Queue(
            maxsize=cfg.prefetch_batches
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cfg = self.cfg
        bs = cfg.batch_size
        H = W = self._side
        cursor = 0
        while not self._stop.is_set():
            idx = (np.arange(cursor, cursor + bs)) % self._N
            cursor = (cursor + bs) % self._N
            x_b = self._pool_x[idx].copy()   # [B, S, 10]
            y_b = self._pool_y[idx].copy()   # [B, S, 10]

            if cfg.augment_dihedral:
                for i in range(bs):
                    rot_k = int(self._rng.integers(0, 4))
                    flip = bool(self._rng.integers(0, 2))
                    if rot_k or flip:
                        xg = x_b[i].reshape(H, W, N_CHANNELS)
                        yg = y_b[i].reshape(H, W, N_CHANNELS)
                        if rot_k:
                            xg = np.rot90(xg, k=rot_k, axes=(0, 1))
                            yg = np.rot90(yg, k=rot_k, axes=(0, 1))
                        if flip:
                            xg = np.flip(xg, axis=1)
                            yg = np.flip(yg, axis=1)
                        x_b[i] = np.ascontiguousarray(xg.reshape(H * W, N_CHANNELS))
                        y_b[i] = np.ascontiguousarray(yg.reshape(H * W, N_CHANNELS))

            tx = torch.from_numpy(x_b.copy())
            # solutions: [B, 1, S, 10] — K=1, matches maze interface
            tsols = torch.from_numpy(y_b.copy()).unsqueeze(1)
            tsat = torch.ones(bs, dtype=torch.bool)
            try:
                self._queue.put((tx, tsols, tsat), timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    return

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x [B,S,10], solutions [B,1,S,10], is_sat [B])."""
        return self._queue.get()

    def close(self) -> None:
        self._stop.set()
