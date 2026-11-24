"""Figures for a compute-matched benchmark, rendered from its on-disk artifacts.

Re-runnable against any directory the benchmark driver produced, so a study can
be replotted without repeating the training. Every entry point returns ``[]``
when matplotlib — the optional ``monitoring`` extra — is unavailable, matching
:func:`xrl.explainability.comparison_plots.plot_comparison`.

Each figure answers one question:

``quality_vs_flops``   which arm is ahead at any given compute
``flops_to_parity``    how much compute each arm needs to reach a target
``iso_flop_quality``   where the arms stand at equal compute, on both the policy
                       metric and attribution quality

The x axis is cumulative FLOPs, not episodes: the arms do different amounts of
work per episode, so an episode axis would silently hand one of them more
compute. See :mod:`xrl.explainability.benchmark`.

There is deliberately no attribution-agreement figure. The arms are
independently trained networks scored in separate studies against their own
policies, so no agreement pair exists within a study; comparing them needs a
shared probe set and a shared action, which the Tic Tac Toe walkthrough does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from xrl.explainability.benchmark import ArmCurve
from xrl.training.progress import announce

__all__ = ["benchmark_plot", "plot_benchmark"]

# Colourblind-safe qualitative palette (Okabe-Ito), matching comparison_plots.
_PALETTE = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
_GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}


def plot_benchmark(out_dir: str | Path) -> list[Path]:
    """Render every figure the benchmark's artifacts support."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    root = Path(out_dir)
    summary_path = root / "benchmark_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for stale in plots_dir.glob("*.png"):
        stale.unlink()

    curves = _curves_from_summary(summary)
    written = [
        _quality_vs_flops(plt, curves, summary, plots_dir / "quality_vs_flops.png"),
        _flops_to_parity(plt, summary, plots_dir / "flops_to_parity.png"),
        _iso_flop_quality(plt, summary, plots_dir / "iso_flop_quality.png"),
    ]
    plt.close("all")
    return [path for path in written if path is not None]


def benchmark_plot(
    directory: Annotated[Path, typer.Argument(help="A directory written by the compute-matched benchmark driver.")],
) -> None:
    """Plot a compute-matched benchmark directory.

    Reads only the on-disk artifacts, so a benchmark can be replotted long after
    the training it summarises.
    """

    plots = plot_benchmark(directory)
    if not plots:
        announce(
            f"no plots written for {directory}; install the monitoring extra "
            f"or check that benchmark_summary.json exists"
        )
    print(json.dumps({"out_dir": str(directory), "plots": [str(path) for path in plots]}, indent=2))


def _curves_from_summary(summary: Mapping[str, Any]) -> list[ArmCurve]:
    curves: list[ArmCurve] = []
    for name, arm in sorted(summary.get("curves", {}).items()):
        flops = tuple(float(value) for value in arm.get("flops", ()))
        values = tuple(float(value) for value in arm.get("values", ()))
        if not flops or len(flops) != len(values):
            continue
        curves.append(
            ArmCurve(
                name=name,
                run_dir=Path(str(arm.get("run_dir", ""))),
                flops=flops,
                values=values,
                metric=str(summary.get("metric", "value")),
                flops_source=str(summary.get("flops_source", "unknown")),
            )
        )
    return curves


def _quality_vs_flops(
    plt: Any,
    curves: Sequence[ArmCurve],
    summary: Mapping[str, Any],
    path: Path,
) -> Path | None:
    if not curves:
        return None

    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for index, curve in enumerate(curves):
        axes.plot(
            curve.flops,
            curve.values,
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=_PALETTE[index % len(_PALETTE)],
            label=curve.name,
        )
    budget = summary.get("iso_flop_budget")
    if budget:
        axes.axvline(
            float(budget),
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            label="iso-FLOP budget",
        )
    threshold = summary.get("parity_threshold")
    if threshold is not None:
        axes.axhline(float(threshold), color="#999999", linestyle=":", linewidth=1.0, label="parity target")
    axes.set_xscale("log")
    axes.set_xlabel("cumulative training FLOPs")
    axes.set_ylabel(str(summary.get("metric", "value")))
    axes.set_title(f"Quality vs compute ({summary.get('env_id', 'run')})")
    axes.grid(True, **_GRID_KWARGS)
    axes.legend(fontsize="small")
    return _save(figure, path)


def _flops_to_parity(plt: Any, summary: Mapping[str, Any], path: Path) -> Path | None:
    arms = summary.get("arms", {})
    reached = {name: arm["flops_to_parity"] for name, arm in arms.items() if arm.get("flops_to_parity")}
    if not reached:
        return None

    names = sorted(reached)
    figure, axes = plt.subplots(figsize=(6.5, 4.0))
    axes.bar(
        names,
        [reached[name] for name in names],
        color=[_PALETTE[index % len(_PALETTE)] for index in range(len(names))],
        width=0.55,
    )
    unreached = [name for name, arm in arms.items() if not arm.get("flops_to_parity")]
    axes.set_ylabel("FLOPs to reach and hold the target")
    title = f"Compute to parity ({summary.get('metric', 'value')} >= {summary.get('parity_threshold')})"
    if unreached:
        # An arm that never reached the target is a result, not a missing bar.
        title += f"\nnot reached by: {', '.join(sorted(unreached))}"
    axes.set_title(title, fontsize="medium")
    axes.grid(True, axis="y", **_GRID_KWARGS)
    return _save(figure, path)


def _iso_flop_quality(plt: Any, summary: Mapping[str, Any], path: Path) -> Path | None:
    arms = summary.get("arms", {})
    if not arms:
        return None

    names = sorted(arms)
    coalition = [arms[name].get("quality_at_iso_flop_budget") for name in names]
    secondary = [arms[name].get("secondary_value") for name in names]
    figure, axes = plt.subplots(figsize=(7.0, 4.0))
    positions = range(len(names))
    width = 0.38
    axes.bar(
        [position - width / 2 for position in positions],
        [value if value is not None else 0.0 for value in coalition],
        width=width,
        color=_PALETTE[0],
        label=str(summary.get("metric", "value")),
    )
    if any(value is not None for value in secondary):
        # ADR 0015 requires reporting both sides: the auxiliary masked loss can
        # buy coalition quality at the deployed policy's expense.
        axes.bar(
            [position + width / 2 for position in positions],
            [value if value is not None else 0.0 for value in secondary],
            width=width,
            color=_PALETTE[1],
            label=str(summary.get("secondary_metric_name", "attribution quality")),
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(names)
    budget = summary.get("iso_flop_budget")
    axes.set_title(f"At equal compute ({float(budget):.3e} FLOPs)" if budget else "At equal compute")
    axes.grid(True, axis="y", **_GRID_KWARGS)
    axes.legend(fontsize="small")
    return _save(figure, path)


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _save(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    return path
