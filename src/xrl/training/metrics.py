"""Shared, game-agnostic training-metric helpers.

Both training loops (the single-agent Minesweeper trainer and the AEC
experiment runner) derive the same families of signals: trailing-window means
over per-episode facts, filtered/grouped MLflow projections, Q-value health
statistics over a fixed probe set, and robust cross-seed aggregates. This
module is their single home so a new game wires metrics by calling these
helpers rather than re-implementing them.

MLflow key grouping convention used across all verticals: ``train/...`` for
per-episode series, ``eval/...`` for evaluation checkpoints, ``probe/...`` for
Q-value health, ``policy/...`` for policy checkpoints, and ``agg/...`` for
multi-seed aggregates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

import numpy as np

# Env seeds for probe-state collection live far from both the training stream
# (seed + 1 .. seed + episodes) and the evaluation stream (seed + 100_000 + ...,
# see CHECKPOINT_EVAL_SEED_OFFSET in runner.py) so the three never overlap.
PROBE_SEED_OFFSET = 200_000

# Column names under which probe statistics land in checkpoint CSV rows.
PROBE_STAT_FIELDS: tuple[str, ...] = ("probe_q_mean", "probe_q_max", "probe_q_mean_max")


def _row_value(row: Any, column: str) -> float:
    """Read a numeric column from a mapping or an attribute-style row."""

    value = row[column] if isinstance(row, Mapping) else getattr(row, column)
    return float(value)


def rolling_means(
    history: Sequence[Any],
    *,
    columns: Mapping[str, str],
    window: int,
) -> dict[str, float]:
    """Trailing-window means over the last ``window`` rows of ``history``.

    ``columns`` maps a source column to its output key, e.g.
    ``{"win": "rolling_win_rate"}`` — the mean of a 0/1 ``win`` column *is* the
    win rate. Rows may be mappings or dataclass-like objects.
    """

    if window < 1:
        raise ValueError(f"window must be positive, got {window}")
    recent = history[-window:]
    if not recent:
        return {output: 0.0 for output in columns.values()}
    return {output: sum(_row_value(row, source) for row in recent) / len(recent) for source, output in columns.items()}


def mlflow_metrics_projection(
    row: Mapping[str, Any],
    *,
    exclude: frozenset[str] = frozenset(),
    prefix: str = "",
) -> dict[str, float | int]:
    """Project a metrics row onto MLflow-worthy keys.

    Drops excluded keys (per-run constants, ids) and non-numeric values
    (strings, ``None``, booleans), and prefixes the survivors so charts group
    by family (``train/``, ``eval/``, ...).
    """

    return {
        f"{prefix}{key}": value
        for key, value in row.items()
        if key not in exclude and isinstance(value, Number) and not isinstance(value, bool)
    }


def probe_q_stats(q_values: np.ndarray, masks: np.ndarray) -> dict[str, float]:
    """Q-value health statistics over a fixed probe set of states.

    ``q_values`` has shape ``(states, actions)``; ``masks`` marks legal
    actions. Returns the mean and max Q over all legal state-action pairs plus
    ``probe_q_mean_max`` — the mean over states of the per-state legal maximum,
    the Nature-DQN "average maximum predicted Q" divergence signal.
    """

    values = np.asarray(q_values, dtype=np.float64)
    legal = np.asarray(masks).astype(bool)
    if values.shape != legal.shape:
        raise ValueError(f"q_values shape {values.shape} does not match masks shape {legal.shape}")
    if not legal.any():
        return {"probe_q_mean": 0.0, "probe_q_max": 0.0, "probe_q_mean_max": 0.0}
    legal_values = values[legal]
    states_with_actions = legal.any(axis=1)
    per_state_max = np.where(legal, values, -np.inf).max(axis=1)[states_with_actions]
    return {
        "probe_q_mean": float(legal_values.mean()),
        "probe_q_max": float(legal_values.max()),
        "probe_q_mean_max": float(per_state_max.mean()),
    }


def probe_mlflow_metrics(stats: Mapping[str, float]) -> dict[str, float]:
    """Rename probe stats to grouped MLflow keys (``probe_q_mean`` → ``probe/q_mean``)."""

    return {f"probe/{key.removeprefix('probe_')}": float(value) for key, value in stats.items()}


class ProbeSetBuilder:
    """Accumulate distinct probe states (with action masks) up to a limit.

    Games feed every visited state during random-policy rollouts; the builder
    dedupes by state bytes and stops accepting once ``limit`` distinct states
    are held, so tiny boards simply yield smaller probe sets.
    """

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        self.limit = limit
        self._seen: set[bytes] = set()
        self._states: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []

    @property
    def full(self) -> bool:
        return len(self._states) >= self.limit

    def add(self, state: np.ndarray, mask: np.ndarray) -> None:
        if self.full:
            return
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        key = state.tobytes()
        if key in self._seen:
            return
        self._seen.add(key)
        self._states.append(state)
        self._masks.append(np.asarray(mask).reshape(-1).astype(bool))

    def build(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(states, masks)`` arrays, or ``None`` when nothing was collected."""

        if not self._states:
            return None
        return np.stack(self._states), np.stack(self._masks)


def iqm(values: Sequence[float]) -> float:
    """Interquartile mean: the mean of the middle 50% of ``values``.

    Trims ``len(values) // 4`` entries from each end of the sorted sequence,
    so for fewer than four values it degenerates to the plain mean — fine for
    the small seed counts typical here, but not a robust estimate.
    """

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    trim = len(ordered) // 4
    kept = ordered[trim : len(ordered) - trim]
    return sum(kept) / len(kept)
