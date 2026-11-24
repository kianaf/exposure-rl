"""Cached exact Minesweeper oracle and certainty games, shared by the Minesweeper and baseline scripts.

``load_oracle`` caches the occupancy of the belief-optimal policy with exact ``Q`` and ``E[G^2]``;
``load_games`` caches exact Shapley values of ``v_mask`` and ``v_state`` for every decision observation
(all ``2^n`` coalitions, subset-sum transform). ``scored_indices`` fixes the observations on which
models and baselines are scored (all if at most 2000, else 300 drawn by occupancy weight, seed 0).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from exposure_rl import coalitions
from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle

SCORED = 300


def tag(g: Geometry) -> str:
    return f"{g.rows}x{g.columns}_{g.mines}"


def load_oracle(g: Geometry, cache_dir: Path) -> dict:
    path = cache_dir / f"minesweeper_{tag(g)}.npz"
    if path.exists():
        return dict(np.load(path))
    t = time.time()
    oracle = MinesweeperOracle(g)
    dist = oracle.occupancy()
    points = np.array([p.observation for p in dist.points], np.int8)
    moments = [oracle.action_moments(p) for p in points]
    data = {
        "points": points,
        "weights": np.array([p.weight for p in dist.points]),
        "q": np.stack([m[0] for m in moments]),
        "g2": np.stack([m[1] for m in moments]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"exact oracle: {len(points)} decision observations in {time.time() - t:.0f}s", flush=True)
    return data


def scored_indices(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(indices, scoring weights)``: every observation, or a fixed occupancy-weighted sample."""

    w = weights / weights.sum()
    if len(w) <= 2000:
        return np.arange(len(w)), w
    return np.sort(np.random.default_rng(0).choice(len(w), size=SCORED, p=w)), np.full(SCORED, 1 / SCORED)


def exact_games(oracle: dict, i: int):
    """``(q_mean, v_mask, v_state)`` over all coalitions for decision observation ``i`` (greedy action)."""

    points, weights, q, g2 = oracle["points"], oracle["weights"], oracle["q"], oracle["g2"]
    a = int(np.nanargmax(q[i]))
    legal = points[:, a] == HIDDEN
    return coalitions.certainty_games(
        points[legal],
        weights[legal],
        q[legal, a],
        g2[legal, a] - q[legal, a] ** 2,
        points[i],
        q[i, a],
        g2[i, a] - q[i, a] ** 2,
    )


_SHARED: dict = {}


def _games_for(i: int):
    s = _SHARED
    n, sizes = s["n"], s["sizes"]
    _, v_mask, v_state = exact_games(s["oracle"], i)
    return (
        coalitions.uncertainty_reduction_shapley(v_mask, n),
        coalitions.uncertainty_reduction_shapley(v_state, n),
        coalitions.leave_one_out(v_mask, n),
        np.array([v_mask[0], v_state[0], v_state[-1]]),
        np.array([v_mask[sizes == k].mean() for k in range(n + 1)]),
        v_mask if i in s["keep"] else None,
    )


def load_games(g: Geometry, oracle: dict, keep: np.ndarray, cache_dir: Path) -> dict:
    path = cache_dir / f"minesweeper_{tag(g)}_games.npz"
    if path.exists():
        return dict(np.load(path))
    t = time.time()
    n = g.cells
    _SHARED.update(oracle=oracle, n=n, sizes=coalitions.coalition_sizes(n), keep={int(i) for i in keep})
    with mp.get_context("fork").Pool(min(16, mp.cpu_count())) as pool:
        rows = pool.map(_games_for, range(len(oracle["points"])), chunksize=64)
    games = {
        "phi_mask": np.stack([r[0] for r in rows]),
        "phi_state": np.stack([r[1] for r in rows]),
        "loo_mask": np.stack([r[2] for r in rows]),
        "summary": np.stack([r[3] for r in rows]),  # v_mask(empty), v_state(empty), v_state(N)
        "v_mask_by_size": np.stack([r[4] for r in rows]),
        "keep": keep,
        "v_mask_keep": np.stack([rows[int(i)][5] for i in keep]),
    }
    np.savez_compressed(path, **games)
    print(f"exact games over 2^{n} coalitions for {len(rows)} observations in {time.time() - t:.0f}s", flush=True)
    return games
