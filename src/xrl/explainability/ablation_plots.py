"""Figures for an ablation sweep, rendered from its on-disk study artifacts.

Every other figure in this package reads one study. These read many, keyed by
the ``ablation_manifest.json`` that :mod:`examples.ttt_ablation` writes, and plot
a metric against the knob that was varied. Re-runnable against any sweep
directory, and like :func:`xrl.explainability.comparison_plots.plot_comparison`
every entry point returns ``[]`` when matplotlib is unavailable.

Each figure answers one question:

``fidelity_vs_knob``   does the knob move the *game* — how far each learned
                       coalition game sits from the exact one it targets
``agreement_vs_knob``  does the knob move the *attributions* — rank and linear
                       agreement with the exact on-manifold reference, plotted
                       together because their gap is itself a diagnostic
``estimator_vs_knob``  does the estimator converge to its own exact answer,
                       the one curve with a known-correct shape and so the
                       sweep's check on itself

The baseline point appears on every axis, since it is what each axis is varied
around.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xrl.explainability.comparison_plots import (
    _GRID_KWARGS,
    _PALETTE,
    _as_float,
    _grouped,
    _nanmean,
    _read_csv,
    _save,
)

__all__ = ["plot_ablation"]

# The reference every head-to-head method is scored against. Named here rather
# than imported from the harness so a sweep directory stays plottable on its own.
_REFERENCE = "conditional-occupancy:exact"
_BASELINE = "baseline"


def plot_ablation(sweep_dir: str | Path) -> list[Path]:
    """Render every figure a sweep's artifacts support; return the paths written."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    root = Path(sweep_dir)
    manifest_path = root / "ablation_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points = _load_points(manifest)
    if not points:
        return []

    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _fidelity_vs_knob(plt, points, plots_dir / "fidelity_vs_knob.png"),
        _agreement_vs_knob(plt, points, plots_dir / "agreement_vs_knob.png"),
        _estimator_vs_knob(plt, points, plots_dir / "estimator_vs_knob.png"),
    ]
    return [path for path in written if path is not None]


def _load_points(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Attach each point's study tables to its manifest entry, dropping unrun points."""

    baseline_settings = manifest.get("baseline", {})
    loaded = []
    for entry in manifest.get("points", ()):
        study = Path(entry["study_dir"])
        if not (study / "summary.json").exists():
            continue
        loaded.append(
            {
                "axis": entry["axis"],
                # The baseline's knob value is whatever the mode table set, and
                # differs per axis, so it is resolved when the axis is known.
                "value": entry["value"],
                "settings": entry.get("settings", {}),
                "baseline_settings": baseline_settings,
                "agreement": _read_csv(study / "agreement.csv"),
                "fidelity": _read_csv(study / "coalition_fidelity.csv"),
                "reference_error": _read_csv(study / "reference_error.csv"),
            }
        )
    return loaded


def _axis_series(points: Sequence[Mapping[str, Any]], axis: str, field: str) -> list[tuple[float, Mapping[str, Any]]]:
    """Points on one axis, sorted by knob value, with the baseline folded in.

    The baseline belongs on every axis: it is the point each axis is varied
    around, and omitting it would draw curves that start at the first *changed*
    setting rather than at the default.
    """

    series = []
    for point in points:
        if point["axis"] not in (axis, _BASELINE):
            continue
        settings = point["settings"] if point["axis"] == axis else point["baseline_settings"]
        value = settings.get(field)
        if value is None:
            continue
        try:
            series.append((float(value), point))
        except (TypeError, ValueError):
            # A non-numeric knob (the dataset contrast) has no position on a
            # numeric axis; it is reported in the manifest rather than plotted.
            continue
    return sorted(series, key=lambda item: item[0])


def _axes_in(points: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """The ``(axis, settings field)`` pairs the sweep actually varied."""

    found = []
    for point in points:
        if point["axis"] == _BASELINE:
            continue
        changed = [
            field
            for field, value in point["settings"].items()
            if value != point["baseline_settings"].get(field) and _is_numeric(value)
        ]
        for field in changed:
            pair = (point["axis"], field)
            if pair not in found:
                found.append(pair)
    return found


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _panels(plt: Any, axes: Sequence[tuple[str, str]], height: float) -> tuple[Any, Any]:
    """One panel per axis, sharing a y-axis so the panels can be read against each other.

    The point of the figure is which knob moves a metric and which leaves it
    flat; per-panel autoscaling would magnify noise on a flat axis until it
    looked like an effect.
    """

    figure, grid = plt.subplots(
        1,
        len(axes),
        figsize=(1.0 + 4.6 * len(axes), height),
        squeeze=False,
        sharey=True,
    )
    return figure, grid[0]


def _fidelity_vs_knob(plt: Any, points: Sequence[Mapping[str, Any]], path: Path) -> Path | None:
    """Distance from each learned game to the exact one, against each knob.

    This is the model-error term. An axis that moves it is changing the *game*;
    an axis that leaves it flat is not, whatever it does to the attributions.
    """

    axes = _axes_in(points)
    if not axes:
        return None
    figure, grid = _panels(plt, axes, 4.4)
    drawn = False
    for position, (axis, field) in enumerate(axes):
        panel = grid[position]
        series = _axis_series(points, axis, field)
        games = sorted({row["game"] for _, point in series for row in point["fidelity"]})
        for index, game in enumerate(games):
            values = [
                (knob, _nanmean(_grouped(point["fidelity"], ("game",), "normalized_mae").get((game,), [])))
                for knob, point in series
            ]
            values = [(knob, value) for knob, value in values if math.isfinite(value)]
            if not values:
                continue
            drawn = True
            panel.plot(
                [knob for knob, _ in values],
                [value for _, value in values],
                marker="o",
                markersize=4,
                color=_PALETTE[index % len(_PALETTE)],
                label=game,
            )
        panel.axhline(1.0, color="gray", linewidth=1.0, linestyle="--")
        panel.set_xlabel(axis)
        panel.grid(**_GRID_KWARGS)
        if position == 0:
            panel.set_ylabel("mean |v − v_reference| / sd(v_reference)")
            panel.legend(fontsize="x-small")
    if not drawn:
        plt.close(figure)
        return None
    # A ratio to the reference's own spread, so parity is 1.0 and a halved gap
    # is as far from it as a doubled one. A linear axis also buries the parity
    # line entirely once an undertrained game runs to many times the signal.
    if _all_positive(points, "normalized_mae"):
        grid[0].set_yscale("log")
    else:
        grid[0].set_ylim(bottom=0.0)
    figure.suptitle("Model error: does the knob move the game? (dashed line = gap is the whole signal)")
    return _save(figure, path)


def _all_positive(points: Sequence[Mapping[str, Any]], column: str) -> bool:
    values = [_as_float(row[column]) for point in points for row in point["fidelity"] if column in row]
    finite = [value for value in values if math.isfinite(value)]
    return bool(finite) and all(value > 0.0 for value in finite)


def _agreement_vs_knob(plt: Any, points: Sequence[Mapping[str, Any]], path: Path) -> Path | None:
    """Rank and linear agreement with the exact reference, against each knob.

    Both correlations are drawn because their *gap* carries information no
    single one does: Spearman is invariant to any monotone transform of the
    attributions and Pearson only to affine ones, so a knob that closes one and
    not the other is changing calibration rather than ordering.
    """

    axes = _axes_in(points)
    if not axes:
        return None
    figure, grid = _panels(plt, axes, 4.6)
    drawn = False
    for position, (axis, field) in enumerate(axes):
        panel = grid[position]
        series = _axis_series(points, axis, field)
        methods = sorted(
            {method for _, point in series for method in _methods_against_reference(point["agreement"])},
        )
        for index, method in enumerate(methods):
            color = _PALETTE[index % len(_PALETTE)]
            for statistic, style in (("spearman", "-"), ("pearson", ":")):
                curve = [
                    (knob, _reference_correlation(point["agreement"], method, statistic)) for knob, point in series
                ]
                curve = [(knob, value) for knob, value in curve if math.isfinite(value)]
                if not curve:
                    continue
                drawn = True
                panel.plot(
                    [knob for knob, _ in curve],
                    [value for _, value in curve],
                    marker="o" if statistic == "spearman" else "s",
                    markersize=4,
                    linestyle=style,
                    color=color,
                    label=f"{method} ({statistic})",
                )
        panel.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        panel.set_xlabel(axis)
        panel.set_ylim(-1.05, 1.05)
        panel.grid(**_GRID_KWARGS)
        if position == 0:
            panel.set_ylabel(f"correlation with {_REFERENCE}")
    if not drawn:
        plt.close(figure)
        return None
    handles, labels = grid[0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize="xx-small", loc="lower center", ncol=3)
    figure.suptitle("Attribution agreement with the exact on-manifold reference (solid Spearman, dotted Pearson)")
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _estimator_vs_knob(plt: Any, points: Sequence[Mapping[str, Any]], path: Path) -> Path | None:
    """Approximation error against each method's *own* exact answer.

    The sweep's check on itself: a budget axis has a known-correct shape here —
    more coalitions cannot make KernelSHAP worse at recovering the exact
    Shapley values of the same game — so a curve that rises is a defect rather
    than a finding.
    """

    axes = _axes_in(points)
    if not axes:
        return None
    figure, grid = _panels(plt, axes, 4.4)
    drawn = False
    for position, (axis, field) in enumerate(axes):
        panel = grid[position]
        series = _axis_series(points, axis, field)
        methods = sorted({row["method"] for _, point in series for row in point["reference_error"]})
        for index, method in enumerate(methods):
            curve = [
                (
                    knob,
                    _nanmean(_grouped(point["reference_error"], ("method",), "mean_absolute_error").get((method,), [])),
                )
                for knob, point in series
            ]
            curve = [(knob, value) for knob, value in curve if math.isfinite(value)]
            if not curve:
                continue
            drawn = True
            panel.plot(
                [knob for knob, _ in curve],
                [value for _, value in curve],
                marker="o",
                markersize=4,
                color=_PALETTE[index % len(_PALETTE)],
                label=method,
            )
        panel.set_xlabel(axis)
        panel.set_ylim(bottom=0.0)
        panel.grid(**_GRID_KWARGS)
        if position == 0:
            panel.set_ylabel("mean absolute error vs its own exact arm")
            panel.legend(fontsize="x-small")
    if not drawn:
        plt.close(figure)
        return None
    figure.suptitle("Estimator error against the exact answer to the same game")
    return _save(figure, path)


def _methods_against_reference(rows: Sequence[Mapping[str, str]]) -> set[str]:
    methods = set()
    for row in rows:
        pair = (row.get("method_a", ""), row.get("method_b", ""))
        if _REFERENCE in pair:
            methods.add(pair[1] if pair[0] == _REFERENCE else pair[0])
    return methods


def _reference_correlation(rows: Sequence[Mapping[str, str]], method: str, statistic: str) -> float:
    """Mean correlation between ``method`` and the reference, over states.

    ``agreement.csv`` stores each pair once under a sorted ordering, so which
    of ``method_a``/``method_b`` holds the reference is not fixed and both
    orientations have to be accepted.
    """

    values = [
        _as_float(row[statistic])
        for row in rows
        if statistic in row and {row.get("method_a"), row.get("method_b")} == {method, _REFERENCE}
    ]
    return _nanmean(values) if values else math.nan
