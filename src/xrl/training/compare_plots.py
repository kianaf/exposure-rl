"""Plot the artifacts written by ``xrl compare-runs``.

Kept separate from :mod:`xrl.training.compare_runs` on purpose. The comparison
itself is CSV in, CSV out and must work with no optional extras installed; this
module is the only part that needs matplotlib, and it reads nothing but the
files already on disk. That makes the artifacts the contract — a comparison can
be re-plotted later, or plotted somewhere else entirely — and keeps the
comparison command's stdout a single JSON object.

One series per method: the cross-seed mean as a line, ±1 standard deviation as
a band, in the same black-and-white marker/linestyle scheme ``xrl compare``
already uses, so figures from either command sit together in one document.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Annotated, Any

import typer

from xrl.training.comparison import linestyle_for_group, marker_for_group
from xrl.training.metrics_io import read_dict_rows
from xrl.training.progress import announce

__all__ = ["compare_plot", "plot_compare_runs"]

_METRICS_FILE = "aggregate_metrics.csv"
_EVAL_FILE = "aggregate_eval_metrics.csv"


def plot_compare_runs(directory: Path) -> list[Path]:
    """Write one PNG per aggregated column into ``<directory>/plots``.

    Returns ``[]`` when matplotlib (the optional ``monitoring`` extra) is
    unavailable, matching ``plot_comparison`` and ``plot_seed_sweep``.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    directory = Path(directory)
    metric_rows = read_dict_rows(directory / _METRICS_FILE)
    eval_rows = read_dict_rows(directory / _EVAL_FILE)
    if not metric_rows and not eval_rows:
        return []

    plots_dir = directory / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path | None] = []

    for column in _columns(metric_rows):
        written.append(_plot_column(metric_rows, column, plots_dir / f"{column}.png", title=column))

    for group, rows in _split_by_opponent(eval_rows).items():
        stem = f"eval_{_safe_name(group)}" if group else "eval"
        for column in _columns(rows):
            written.append(
                _plot_column(
                    rows,
                    column,
                    plots_dir / f"{stem}_{column}.png",
                    title=f"{stem} {column}".replace("_", " "),
                )
            )

    plt.close("all")
    return [path for path in written if path is not None]


def compare_plot(
    directory: Annotated[
        Path | None,
        typer.Argument(help="A directory written by xrl compare-runs. Read from piped JSON when omitted."),
    ] = None,
) -> None:
    """Plot a comparison written by `xrl compare-runs`.

    Reads only the aggregate CSVs in the directory, so a comparison can be
    plotted long after it was computed, or piped straight from the command:
    `xrl compare-runs runs/sweep | xrl compare-plot`.
    """

    target = directory if directory is not None else _directory_from_stdin()
    plots = plot_compare_runs(target)
    if not plots:
        announce(f"no plots written for {target}; install the monitoring extra or check the aggregate CSVs")
    print(json.dumps({"out_dir": str(target), "plots": [str(path) for path in plots]}, indent=2))


def _directory_from_stdin() -> Path:
    """Take ``out_dir`` from a piped ``xrl compare-runs`` result."""

    if sys.stdin.isatty():
        raise typer.BadParameter("name a comparison directory, or pipe the output of xrl compare-runs")
    try:
        piped = json.loads(sys.stdin.read())
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"piped input is not JSON from xrl compare-runs: {error}") from error
    out_dir = piped.get("out_dir") if isinstance(piped, Mapping) else None
    if not out_dir:
        raise typer.BadParameter("piped JSON has no out_dir; expected the output of xrl compare-runs")
    return Path(str(out_dir))


def _columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Recover the aggregated column names from the ``<column>_mean`` headers."""

    if not rows:
        return []
    return [key.removesuffix("_mean") for key in rows[0] if key.endswith("_mean")]


def _split_by_opponent(rows: Sequence[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    if not rows or "opponent" not in rows[0]:
        return {"": list(rows)}
    split: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        split.setdefault(str(row.get("opponent", "")), []).append(row)
    return dict(sorted(split.items()))


def _plot_column(
    rows: Sequence[Mapping[str, str]],
    column: str,
    path: Path,
    *,
    title: str,
) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np

    series = {method: _downsample(points) for method, points in _series(rows, column).items()}
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (method, points) in enumerate(sorted(series.items())):
        episodes = np.asarray([episode for episode, _, _ in points], dtype=np.float64)
        means = np.asarray([mean for _, mean, _ in points], dtype=np.float64)
        deviations = np.asarray([deviation for _, _, deviation in points], dtype=np.float64)
        ax.plot(
            episodes,
            means,
            color="black",
            linestyle=linestyle_for_group(method, index),
            linewidth=1.3,
            label=method,
        )
        if episodes.size <= _MARKED_POINTS:
            ax.scatter(episodes, means, marker=marker_for_group(method, index), color="black", s=48)
        if np.any(deviations > 0):
            ax.fill_between(episodes, means - deviations, means + deviations, color="black", alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# Above this many points per series, markers become a smear; the linestyle
# alone still tells the series apart.
_MARKED_POINTS = 40

# Points drawn per series after binning. Enough to keep the shape of a learning
# curve, few enough that three series stay distinguishable.
_MAX_POINTS = 80


def _series(rows: Sequence[Mapping[str, str]], column: str) -> dict[str, list[tuple[float, float, float]]]:
    """Group ``(episode, mean, std)`` triples per method, ordered by episode."""

    series: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        episode = _number(row.get("episode"))
        mean = _number(row.get(f"{column}_mean"))
        if episode is None or mean is None:
            continue
        deviation = _number(row.get(f"{column}_std")) or 0.0
        series.setdefault(str(row.get("method", "")), []).append((episode, mean, deviation))
    return {method: sorted(points) for method, points in series.items() if points}


def _downsample(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Average a long curve into at most :data:`_MAX_POINTS` evenly spaced bins.

    A 10 000-episode curve drawn per episode is a solid band in which no series
    can be told from another, however smooth the underlying trend. Binning
    keeps both the trend and the cross-seed spread readable. Only the figure is
    reduced — the aggregate CSV stays lossless, so anyone re-plotting elsewhere
    still has every episode.
    """

    if len(points) <= _MAX_POINTS:
        return points
    size = math.ceil(len(points) / _MAX_POINTS)
    return [
        (
            chunk[len(chunk) // 2][0],
            fmean(mean for _, mean, _ in chunk),
            fmean(deviation for _, _, deviation in chunk),
        )
        for chunk in (points[start : start + size] for start in range(0, len(points), size))
    ]


def _number(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")
