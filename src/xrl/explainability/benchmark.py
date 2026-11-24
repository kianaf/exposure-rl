"""Compute-matched comparison between training arms, read off FLOP curves.

Two questions, both asked along a FLOP axis rather than an episode axis because
the arms do different amounts of work per episode (see
:mod:`xrl.training.flops`):

* **Iso-FLOP quality** — given the same compute, which arm is better?
* **FLOPs to parity** — how much compute does each arm need to first reach a
  given evaluation performance, and hold it?

Both are answered from the ``training_flops`` column already written on every
DQN evaluation row, so neither requires training under a budget cap. Reading
them off the recorded curves is exact, where stopping training at a budget would
overshoot by up to one evaluation interval, and it yields the whole curve
instead of a single point. ``--flop-budget`` therefore selects where iso-FLOP
quality is *reported*, not where training stops.

Comparing totals measured different ways would be comparing two definitions of
the word, so an arm records its ``flops_source`` and
:func:`common_flop_budget` refuses a mismatch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrl.training.metrics_io import read_dict_rows

__all__ = [
    "ArmCurve",
    "ParityPoint",
    "common_flop_budget",
    "flops_to_parity",
    "load_arm_curve",
    "parity_rows",
    "quality_at_budget",
    "summarize_arms",
]


@dataclass(frozen=True, slots=True)
class ArmCurve:
    """One arm's evaluation history along its cumulative-FLOP axis."""

    name: str
    run_dir: Path
    flops: tuple[float, ...]
    values: tuple[float, ...]
    metric: str
    flops_source: str

    def __post_init__(self) -> None:
        if len(self.flops) != len(self.values):
            raise ValueError(f"{self.name}: {len(self.flops)} FLOP points but {len(self.values)} values")
        if not self.flops:
            raise ValueError(f"{self.name}: no evaluation checkpoints recorded training_flops")
        if list(self.flops) != sorted(self.flops):
            raise ValueError(f"{self.name}: cumulative FLOPs must be non-decreasing")

    @property
    def total_flops(self) -> float:
        return self.flops[-1]


@dataclass(frozen=True, slots=True)
class ParityPoint:
    """Where an arm first reached a target and held it, or that it never did."""

    name: str
    threshold: float
    flops: float | None
    episode: int | None

    @property
    def reached(self) -> bool:
        return self.flops is not None


def load_arm_curve(
    run_dir: str | Path,
    *,
    name: str,
    metric: str,
    opponent: str | None = None,
) -> ArmCurve:
    """Read one arm's ``eval_metrics.csv`` into a FLOP-indexed curve.

    ``opponent`` selects one evaluation stream on environments that have
    several; single-agent environments record no opponent column and must not
    be given one.
    """

    run_dir = Path(run_dir)
    rows = read_dict_rows(run_dir / "eval_metrics.csv")
    if not rows:
        raise ValueError(f"{run_dir}/eval_metrics.csv has no rows; train with --eval-interval > 0")
    if opponent is not None:
        if "opponent" not in rows[0]:
            raise ValueError(f"{run_dir} records no opponent column; it is a single-agent run, so pass opponent=None")
        rows = [row for row in rows if row.get("opponent") == opponent]
        if not rows:
            raise ValueError(f"{run_dir} has no evaluation rows for opponent {opponent!r}")
    missing = [column for column in ("training_flops", metric) if column not in rows[0]]
    if missing:
        raise ValueError(f"{run_dir}/eval_metrics.csv is missing {', '.join(missing)}")

    flops: list[float] = []
    values: list[float] = []
    for row in rows:
        flop = _as_float(row.get("training_flops"))
        value = _as_float(row.get(metric))
        if flop is None or value is None:
            continue
        flops.append(flop)
        values.append(value)
    sources = {str(row.get("flops_source", "")) for row in rows if row.get("flops_source")}
    if len(sources) > 1:
        raise ValueError(f"{run_dir} mixes FLOPs sources {sorted(sources)} within one run")
    return ArmCurve(
        name=name,
        run_dir=run_dir,
        flops=tuple(flops),
        values=tuple(values),
        metric=metric,
        flops_source=next(iter(sources), "unknown"),
    )


def common_flop_budget(curves: Sequence[ArmCurve]) -> float:
    """The largest compute every arm actually reached.

    Comparing beyond it would extrapolate one arm past where it was measured.
    """

    if not curves:
        raise ValueError("at least one curve is required")
    sources = {curve.flops_source for curve in curves}
    if len(sources) > 1:
        raise ValueError(
            f"cannot compare arms measured different ways: {sorted(sources)}. "
            "Retrain on one backend, or compare only arms sharing a flops_source."
        )
    return min(curve.total_flops for curve in curves)


def quality_at_budget(curve: ArmCurve, budget: float) -> float:
    """The arm's metric at ``budget`` FLOPs, linearly interpolated.

    Checkpoints are sparse, and two arms land on different FLOP values, so
    reading either one's nearest checkpoint would silently compare them at
    different compute. Interpolation keeps the budget honest.
    """

    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    if budget > curve.total_flops:
        raise ValueError(
            f"{curve.name} was only measured to {curve.total_flops:.3e} FLOPs; "
            f"{budget:.3e} would extrapolate past the run"
        )
    if budget <= curve.flops[0]:
        return curve.values[0]
    for index in range(1, len(curve.flops)):
        left, right = curve.flops[index - 1], curve.flops[index]
        if budget <= right:
            if right == left:
                return curve.values[index]
            weight = (budget - left) / (right - left)
            return curve.values[index - 1] + weight * (curve.values[index] - curve.values[index - 1])
    return curve.values[-1]


def flops_to_parity(
    curve: ArmCurve,
    threshold: float,
    *,
    hold: int = 2,
    episodes: Sequence[int] | None = None,
) -> ParityPoint:
    """FLOPs at the first checkpoint that reaches ``threshold`` and holds it.

    ``hold`` consecutive checkpoints must stay at or above the threshold, so a
    single lucky evaluation does not count as having arrived. An arm that never
    holds it returns ``flops=None`` — which is a result, not a failure, and is
    reported as "not reached" rather than as an infinite cost.
    """

    if hold < 1:
        raise ValueError(f"hold must be positive, got {hold}")
    for index in range(len(curve.values)):
        window = curve.values[index : index + hold]
        if len(window) < hold:
            break
        if all(point >= threshold for point in window):
            episode = int(episodes[index]) if episodes is not None and index < len(episodes) else None
            return ParityPoint(name=curve.name, threshold=threshold, flops=curve.flops[index], episode=episode)
    return ParityPoint(name=curve.name, threshold=threshold, flops=None, episode=None)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_arms(
    curves: Sequence[ArmCurve],
    *,
    threshold: float,
    hold: int = 2,
) -> dict[str, Any]:
    """Both comparisons over a set of arms, ready to serialize."""

    budget = common_flop_budget(curves)
    return {
        "metric": curves[0].metric,
        "flops_source": curves[0].flops_source,
        "iso_flop_budget": budget,
        "parity_threshold": threshold,
        "parity_hold_checkpoints": hold,
        "arms": {
            curve.name: {
                "run_dir": str(curve.run_dir),
                "total_flops": curve.total_flops,
                "quality_at_iso_flop_budget": quality_at_budget(curve, budget),
                "final_value": curve.values[-1],
                "flops_to_parity": flops_to_parity(curve, threshold, hold=hold).flops,
                "checkpoints": len(curve.flops),
            }
            for curve in curves
        },
    }


def parity_rows(curves: Sequence[ArmCurve], thresholds: Sequence[float], *, hold: int = 2) -> list[Mapping[str, Any]]:
    """One row per (arm, threshold) for the FLOPs-to-parity table."""

    rows: list[Mapping[str, Any]] = []
    for threshold in thresholds:
        for curve in curves:
            point = flops_to_parity(curve, threshold, hold=hold)
            rows.append(
                {
                    "arm": curve.name,
                    "threshold": threshold,
                    "flops_to_parity": point.flops,
                    "reached": point.reached,
                }
            )
    return rows
