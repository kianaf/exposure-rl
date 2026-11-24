"""Figures for a comparison study, rendered from its on-disk CSV artifacts.

Re-runnable against any directory :meth:`ComparisonReport.write` produced, so
a study can be replotted without repeating the (dominant) explanation compute.
Every entry point returns ``[]`` when matplotlib — the optional ``monitoring``
extra — is unavailable, matching :func:`xrl.training.seed_sweep.plot_seed_sweep`.

Each figure answers one question:

``agreement_heatmap``    who agrees with whom by *rank*, averaged over states
``agreement_heatmap_pearson``
                         the same pairs by *linear* agreement, which a monotone
                         but nonlinear disagreement separates from the above
``spearman_vs_pearson``  the gap between those two, where the distance from the
                         diagonal says whether a disagreement is about ordering
                         or about spacing
``agreement_spread``     whether that mean hides a bimodal distribution
``attribution_scatter``  whether two methods are on the same *scale*, not just
                         the same ordering — the pooled fit, whose slope no
                         correlation reports
``faithfulness_curves``  which method the model actually supports
``reference_error``      how close each approximation is to its exact answer
``coalition_fidelity``   how far each *game* is from the exact one, one level
                         below the attributions — the only figure that separates
                         model error from a difference in semantics
``cost_vs_faithfulness`` what the faithfulness costs in wall-clock seconds
``efficiency_residual``  whether any estimator failed to converge
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["plot_comparison"]

# Colourblind-safe qualitative palette (Okabe-Ito).
_PALETTE = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
_GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}

# Display text for ``coalition_fidelity.csv``'s ``relation`` column. Read from
# the CSV rather than imported, so the figures stay renderable against any
# artifact directory without the harness that wrote it.
_RELATION_LABELS = {
    "targets_reference": "targets the reference — model error, lower is better",
    "distinct_game": "differs by construction — the semantics gap, no target value",
}


def plot_comparison(out_dir: str | Path) -> list[Path]:
    """Render every figure a study's artifacts support; return the paths written."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    root = Path(out_dir)
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agreement = _read_csv(root / "agreement.csv")
    faithfulness = _read_csv(root / "faithfulness.csv")
    curves = _read_csv(root / "curves.csv")
    references = _read_csv(root / "reference_error.csv")
    attributions = _read_csv(root / "attributions.csv")
    fidelity = _read_csv(root / "coalition_fidelity.csv")

    written = [
        _agreement_heatmap(plt, agreement, plots_dir / "agreement_heatmap.png"),
        _agreement_heatmap(
            plt,
            agreement,
            plots_dir / "agreement_heatmap_pearson.png",
            column="pearson",
            title="Linear agreement (mean Pearson over states)",
        ),
        _spearman_vs_pearson(plt, agreement, plots_dir / "spearman_vs_pearson.png"),
        _agreement_spread(plt, agreement, plots_dir / "agreement_spread.png"),
        _attribution_scatter(
            plt,
            attributions,
            _read_csv(root / "agreement_pooled.csv"),
            plots_dir / "attribution_scatter.png",
        ),
        _faithfulness_curves(plt, curves, plots_dir / "faithfulness_curves.png"),
        _reference_error(plt, references, plots_dir / "reference_error.png"),
        _coalition_fidelity(plt, fidelity, plots_dir / "coalition_fidelity.png"),
        _cost_vs_faithfulness(plt, faithfulness, attributions, plots_dir / "cost_vs_faithfulness.png"),
        _efficiency_residual(plt, attributions, plots_dir / "efficiency_residual.png"),
    ]
    return [path for path in written if path is not None]


def _agreement_heatmap(
    plt: Any,
    rows: Sequence[Mapping[str, str]],
    path: Path,
    *,
    column: str = "spearman",
    title: str = "Rank agreement (mean Spearman over states)",
) -> Path | None:
    """Mean pairwise correlation for every method pair, under one statistic."""

    if not rows or not any(column in row for row in rows):
        return None
    methods = sorted({row["method_a"] for row in rows} | {row["method_b"] for row in rows})
    index = {method: position for position, method in enumerate(methods)}
    matrix = np.full((len(methods), len(methods)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for (first, second), values in _grouped(rows, ("method_a", "method_b"), column).items():
        mean = _nanmean(values)
        matrix[index[first], index[second]] = mean
        matrix[index[second], index[first]] = mean

    figure, axes = plt.subplots(figsize=(1.6 + 1.1 * len(methods), 1.4 + 1.0 * len(methods)))
    image = axes.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
    axes.set_yticks(range(len(methods)), methods)
    for row_position in range(len(methods)):
        for column_position in range(len(methods)):
            value = matrix[row_position, column_position]
            if math.isnan(value):
                continue
            axes.text(
                column_position,
                row_position,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 else "black",
            )
    axes.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label=column.capitalize())
    return _save(figure, path)


def _spearman_vs_pearson(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """Rank agreement against linear agreement, one point per method pair.

    The two heatmaps answer neighbouring questions and the gap between them is
    itself the finding, which neither shows on its own. Spearman is invariant
    to any monotone transform of the attributions, Pearson only to affine ones,
    so distance from the diagonal reads directly:

    ``below`` (Pearson < Spearman)
        the ordering survives but the spacing does not — nonlinear
        miscalibration between the two methods.
    ``above`` (Pearson > Spearman)
        a few high-magnitude cells carry the agreement while the near-zero tail
        contributes rank noise, which on a nine-cell board with most cells
        close to zero is the expected way to land here.
    """

    if not rows:
        return None
    ranked = _grouped(rows, ("method_a", "method_b"), "spearman")
    linear = _grouped(rows, ("method_a", "method_b"), "pearson")
    points = [
        (_nanmean(ranked[pair]), _nanmean(linear[pair]), f"{pair[0]}\nvs {pair[1]}")
        for pair in ranked
        if pair in linear
    ]
    points = [point for point in points if math.isfinite(point[0]) and math.isfinite(point[1])]
    if not points:
        return None
    points.sort(key=lambda point: point[1] - point[0])

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 1.6 + 0.42 * len(points)))
    left.plot((-1.0, 1.0), (-1.0, 1.0), color="#999999", linestyle=":", linewidth=1.0, label="equal")
    left.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=70,
        color=_PALETTE[0],
        alpha=0.75,
        edgecolors="none",
        zorder=3,
    )
    left.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    left.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    left.set_xlim(-1.05, 1.05)
    left.set_ylim(-1.05, 1.05)
    left.set_aspect("equal")
    left.set_xlabel("Spearman (rank agreement)")
    left.set_ylabel("Pearson (linear agreement)")
    left.set_title("Rank against linear agreement")
    left.grid(**_GRID_KWARGS)
    left.legend(fontsize="x-small", loc="upper left")

    # The pairs pile onto the diagonal whenever the disagreement is purely about
    # ordering, and there the scatter cannot be read. The signed gap can, and it
    # doubles as the scatter's legend — 13 pairs outnumber any palette.
    positions = np.arange(len(points), dtype=np.float64)
    gaps = [linear_value - rank_value for rank_value, linear_value, _ in points]
    right.barh(positions, gaps, color=[_PALETTE[1] if gap < 0 else _PALETTE[2] for gap in gaps])
    right.axvline(0.0, color="gray", linewidth=0.8)
    right.set_yticks(positions, [label for _, _, label in points], fontsize=6)
    right.set_xlabel("Pearson − Spearman")
    right.set_title("Positive: magnitudes agree more than ranks")
    right.grid(axis="x", **_GRID_KWARGS)
    return _save(figure, path)


def _attribution_scatter(
    plt: Any,
    attributions: Sequence[Mapping[str, str]],
    pooled: Sequence[Mapping[str, str]],
    path: Path,
) -> Path | None:
    """Pooled cell-by-cell scatter with the fitted line, one panel per pair.

    A correlation near one still hides a method that attributes half as much
    everywhere; the slope is what shows it, and a slope far from the identity
    line is visible here at a glance.
    """

    if not attributions or not pooled:
        return None
    by_method: dict[str, dict[tuple[str, str], float]] = {}
    for row in attributions:
        by_method.setdefault(row["method"], {})[(row["state_index"], row["feature"])] = _as_float(row["attribution"])

    panels = []
    for row in pooled:
        first, second = row["method_a"], row["method_b"]
        if first not in by_method or second not in by_method:
            continue
        shared = sorted(set(by_method[first]) & set(by_method[second]))
        if not shared:
            continue
        panels.append(
            (
                first,
                second,
                row,
                np.asarray([by_method[first][key] for key in shared]),
                np.asarray([by_method[second][key] for key in shared]),
            )
        )
    if not panels:
        return None

    columns = min(3, len(panels))
    rows_count = math.ceil(len(panels) / columns)
    figure, axes_grid = plt.subplots(rows_count, columns, figsize=(4.0 * columns, 3.6 * rows_count), squeeze=False)
    for index, (first, second, row, x, y) in enumerate(panels):
        axes = axes_grid[index // columns][index % columns]
        axes.scatter(x, y, s=8, alpha=0.4, color=_PALETTE[index % len(_PALETTE)], edgecolors="none")
        span = np.asarray([min(x.min(), y.min()), max(x.max(), y.max())], dtype=np.float64)
        axes.plot(span, span, color="#999999", linestyle=":", linewidth=1.0, label="identity")
        slope, intercept = _as_float(row["slope"]), _as_float(row["intercept"])
        if math.isfinite(slope) and math.isfinite(intercept):
            axes.plot(span, intercept + slope * span, color="#000000", linewidth=1.2, label=f"slope {slope:.2f}")
        axes.set_xlabel(first, fontsize="small")
        axes.set_ylabel(second, fontsize="small")
        axes.grid(True, **_GRID_KWARGS)
        axes.legend(fontsize="x-small")
    for index in range(len(panels), rows_count * columns):
        axes_grid[index // columns][index % columns].axis("off")
    figure.suptitle("Pooled attributions per cell (slope, not fit, is the point)")
    return _save(figure, path)


def _agreement_spread(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """Per-pair distribution of agreement, so a mean cannot hide bimodality."""

    if not rows:
        return None
    grouped = _grouped(rows, ("method_a", "method_b"), "spearman")
    labels = [f"{first}\nvs {second}" for first, second in grouped]
    samples = [[value for value in values if math.isfinite(value)] for values in grouped.values()]
    keep = [position for position, values in enumerate(samples) if values]
    if not keep:
        return None

    figure, axes = plt.subplots(figsize=(2.0 + 1.3 * len(keep), 4.5))
    axes.boxplot([samples[position] for position in keep], tick_labels=[labels[position] for position in keep])
    axes.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes.set_ylabel("Spearman per state")
    axes.set_ylim(-1.05, 1.05)
    axes.set_title("Agreement varies by state")
    axes.grid(axis="y", **_GRID_KWARGS)
    return _save(figure, path)


def _faithfulness_curves(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """Deletion and insertion curves, mean over states, one panel per evaluator."""

    if not rows:
        return None
    evaluators = sorted({row["evaluator"] for row in rows})
    modes = ("deletion", "insertion")
    figure, grid = plt.subplots(
        len(evaluators),
        len(modes),
        figsize=(5.2 * len(modes), 3.8 * len(evaluators)),
        squeeze=False,
    )
    methods = sorted({row["method"] for row in rows})
    colors = dict(zip(methods, _PALETTE * (1 + len(methods) // len(_PALETTE)), strict=False))

    for row_position, evaluator in enumerate(evaluators):
        for column_position, mode in enumerate(modes):
            axes = grid[row_position][column_position]
            selected = [row for row in rows if row["evaluator"] == evaluator and row["mode"] == mode]
            for method in methods:
                steps, means = _mean_curve([row for row in selected if row["method"] == method])
                if not steps:
                    continue
                axes.plot(steps, means, marker="o", markersize=3, color=colors[method], label=method)
            axes.set_title(f"{mode} — {evaluator} evaluator")
            axes.set_xlabel("features removed" if mode == "deletion" else "features revealed")
            axes.set_ylabel("mean Q")
            axes.grid(**_GRID_KWARGS)
    grid[0][0].legend(fontsize=8)
    figure.suptitle("Faithfulness: moving further from the starting value means a better ranking")
    return _save(figure, path)


def _reference_error(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """Within-family approximation error against each method's exact reference."""

    if not rows:
        return None
    grouped = _grouped(rows, ("method", "reference"), "mean_absolute_error")
    labels = [f"{method}\nvs {reference}" for method, reference in grouped]
    samples = [[value for value in values if math.isfinite(value)] for values in grouped.values()]
    keep = [position for position, values in enumerate(samples) if values]
    if not keep:
        return None

    figure, axes = plt.subplots(figsize=(2.0 + 1.5 * len(keep), 4.2))
    axes.boxplot([samples[position] for position in keep], tick_labels=[labels[position] for position in keep])
    axes.set_ylabel("mean absolute error")
    axes.set_title("Approximation error against the exact answer to the same game")
    axes.grid(axis="y", **_GRID_KWARGS)
    return _save(figure, path)


def _coalition_fidelity(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """How far each game's ``v`` sits from the exact reference, and where.

    Every other figure reads attributions, where a game that differs by
    construction and a game that fails to reproduce its own target look
    identical. This one reads the characteristic functions directly, before any
    Shapley transform, which is what tells the two apart — so colour carries
    ``relation`` and the legend says which reading applies.

    The left panel is normalised by the reference's own spread because a raw
    error cannot be read without it: 0.1 is negligible against a game ranging
    over units and total against one ranging over tenths. The dashed line marks
    where the gap equals the entire signal.

    The right panel splits the same gap by endpoint. A game can reproduce
    ``v(N)`` exactly — the direct arm does, being the base Q under a full mask —
    while drifting at ``v(∅)``, and one pooled mean hides precisely that.
    """

    if not rows:
        return None
    relations = {row["game"]: row.get("relation", "") for row in rows}
    distances = _grouped(rows, ("game",), "normalized_mae")
    samples = {game: [value for value in values if math.isfinite(value)] for (game,), values in distances.items()}
    games = [game for game, values in samples.items() if values]
    if not games:
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(5.4 + 1.7 * len(games), 4.8))
    ordered_relations = sorted({relations.get(game, "") for game in games})
    colors = {relation: _PALETTE[position % len(_PALETTE)] for position, relation in enumerate(ordered_relations)}

    boxes = left.boxplot([samples[game] for game in games], tick_labels=games, patch_artist=True)
    for patch, game in zip(boxes["boxes"], games, strict=False):
        patch.set_facecolor(colors[relations.get(game, "")])
        patch.set_alpha(0.6)
    for median in boxes["medians"]:
        median.set_color("black")
    left.axhline(1.0, color="gray", linewidth=1.0, linestyle="--")
    left.annotate(
        "gap = the whole signal",
        xy=(0.99, 1.0),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    # The scale is a ratio, so parity sits at 1.0 and a half-sized gap is as far
    # from it as a doubled one. A linear axis buries that under the outlying
    # states, where a distinct game runs to several times the signal.
    if all(value > 0.0 for values in samples.values() for value in values):
        left.set_yscale("log")
    else:
        left.set_ylim(bottom=0.0)
    left.set_ylabel("mean |v − v_reference| / sd(v_reference)")
    left.set_title("Distance from the exact game")
    left.grid(axis="y", **_GRID_KWARGS)
    left.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, facecolor=colors[relation], alpha=0.6) for relation in ordered_relations],
        labels=[_RELATION_LABELS.get(relation, relation) for relation in ordered_relations],
        fontsize="x-small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )

    endpoints = (("empty_endpoint_error", "v(∅) — nothing observed"), ("full_endpoint_error", "v(N) — all observed"))
    positions = np.arange(len(games), dtype=np.float64)
    width = 0.38
    for offset, (column, label) in enumerate(endpoints):
        grouped = _grouped(rows, ("game",), column)
        means = [_nanmean(grouped.get((game,), [])) for game in games]
        bars = right.bar(
            positions + (offset - 0.5) * width,
            means,
            width,
            color=_PALETTE[2 + offset],
            label=label,
        )
        # An exactly-zero endpoint is a structural result, not a missing bar.
        right.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    right.axhline(0.0, color="gray", linewidth=0.8)
    right.set_xticks(positions, games)
    right.set_ylabel("mean signed error")
    right.set_title("Where the gap sits")
    right.grid(axis="y", **_GRID_KWARGS)
    right.legend(fontsize="x-small")

    figure.suptitle("Coalition fidelity: the games differ before any Shapley value is taken")
    return _save(figure, path)


def _cost_vs_faithfulness(
    plt: Any,
    faithfulness: Sequence[Mapping[str, str]],
    attributions: Sequence[Mapping[str, str]],
    path: Path,
) -> Path | None:
    """Median wall-clock against deletion lift — the practical tradeoff."""

    if not faithfulness or not attributions:
        return None
    marginal = [row for row in faithfulness if row["evaluator"] == "marginal"] or list(faithfulness)
    lift = {method: _nanmean(values) for (method,), values in _grouped(marginal, ("method",), "deletion_lift").items()}
    durations = _method_durations(attributions)
    methods = sorted(set(lift) & set(durations))
    if not methods:
        return None

    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    for position, method in enumerate(methods):
        axes.scatter(
            durations[method],
            lift[method],
            s=90,
            color=_PALETTE[position % len(_PALETTE)],
            zorder=3,
            label=method,
        )
        axes.annotate(method, (durations[method], lift[method]), textcoords="offset points", xytext=(7, 4), fontsize=8)
    axes.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    axes.set_xscale("log")
    axes.set_xlabel("median seconds per explanation (log scale)")
    axes.set_ylabel("deletion lift over a random ordering")
    axes.set_title("Faithfulness bought per second of compute")
    axes.grid(**_GRID_KWARGS)
    return _save(figure, path)


def _efficiency_residual(plt: Any, rows: Sequence[Mapping[str, str]], path: Path) -> Path | None:
    """Per-method efficiency residuals; a non-zero bar means non-convergence."""

    if not rows:
        return None
    seen: set[tuple[str, str]] = set()
    residuals: dict[str, list[float]] = {}
    for row in rows:
        key = (row["state_index"], row["method"])
        if key in seen:
            continue
        seen.add(key)
        residuals.setdefault(row["method"], []).append(abs(_as_float(row["efficiency_residual"])))
    methods = sorted(residuals)
    values = [_nanmean(residuals[method]) for method in methods]
    if not any(math.isfinite(value) for value in values):
        return None

    figure, axes = plt.subplots(figsize=(1.6 + 1.1 * len(methods), 4.0))
    axes.bar(methods, values, color=_PALETTE[0])
    axes.set_ylabel("mean |sum(attributions) - (full - baseline)|")
    axes.set_title("Efficiency residual (near zero is correct)")
    axes.tick_params(axis="x", rotation=30)
    axes.grid(axis="y", **_GRID_KWARGS)
    return _save(figure, path)


def _mean_curve(rows: Sequence[Mapping[str, str]]) -> tuple[list[int], list[float]]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(_as_float(row["value"]))
    steps = sorted(by_step)
    return steps, [_nanmean(by_step[step]) for step in steps]


def _method_durations(attributions: Sequence[Mapping[str, str]]) -> dict[str, float]:
    seen: set[tuple[str, str]] = set()
    durations: dict[str, list[float]] = {}
    for row in attributions:
        key = (row["state_index"], row["method"])
        if key in seen:
            continue
        seen.add(key)
        durations.setdefault(row["method"], []).append(_as_float(row["duration_seconds"]))
    return {
        method: float(np.median([value for value in values if math.isfinite(value) and value > 0] or [math.nan]))
        for method, values in durations.items()
    }


def _grouped(
    rows: Iterable[Mapping[str, str]],
    keys: Sequence[str],
    column: str,
) -> dict[tuple[str, ...], list[float]]:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        if column not in row:
            continue
        grouped.setdefault(tuple(row[key] for key in keys), []).append(_as_float(row[column]))
    return dict(sorted(grouped.items()))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nanmean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _save(figure: Any, path: Path) -> Path:
    import matplotlib.pyplot as plt

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
