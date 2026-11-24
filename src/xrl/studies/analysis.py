"""Seed-level paired analysis and artifact-only plots for controlled studies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from xrl._optional import pa, pq, require_storage_dependencies
from xrl.explainability.agreement import (
    bootstrap_mean_ci,
    coalition_error,
    compare_attributions,
    reference_error,
)

# zstd over the default snappy: study tables are tall and narrow with heavily
# repeated key columns, where the extra compression is worth far more than the
# write time it costs. Row groups are sized so a concatenation streams in
# bounded memory regardless of how large the destination table becomes.
PARQUET_COMPRESSION = "zstd"
PARQUET_ROW_GROUP_ROWS = 65_536

_PALETTE = ("#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#000000")
# A study's series count is the product of its condition grid, which outruns any
# palette. Cycling a second channel keeps series distinguishable past seven.
_LINESTYLES = ("-", "--", ":", "-.")
_MARKERS = ("o", "s", "^", "D")
_GRID_KWARGS = {"alpha": 0.3, "linewidth": 0.6}


def _series_style(index: int) -> dict[str, Any]:
    return {
        "color": _PALETTE[index % len(_PALETTE)],
        "linestyle": _LINESTYLES[(index // len(_PALETTE)) % len(_LINESTYLES)],
        "marker": _MARKERS[(index // len(_PALETTE)) % len(_MARKERS)],
    }


class StudyAnalysisError(ValueError):
    """Raised when rows violate the predeclared analysis unit or pairing."""


@dataclass(frozen=True, slots=True)
class FidelityScores:
    """Expected-Q game and attribution errors for one state/action pair."""

    attribution_mae: float
    attribution_max_error: float
    attribution_rmse: float
    relative_attribution_error: float
    attribution_spearman: float
    attribution_kendall_tau_b: float
    attribution_sign_agreement: float
    attribution_top_3_overlap: float
    coalition_mae: float
    coalition_max_error: float
    coalition_rmse: float
    coalition_normalized_mae: float
    coalition_reference_spread: float
    empty_endpoint_error: float
    full_endpoint_error: float

    def to_row(self) -> dict[str, float]:
        return asdict(self)


def score_fidelity(
    approximate_attributions: Sequence[float] | np.ndarray,
    exact_attributions: Sequence[float] | np.ndarray,
    approximate_coalitions: Sequence[float] | np.ndarray,
    exact_coalitions: Sequence[float] | np.ndarray,
) -> FidelityScores:
    """Score one approximation against the exact game it claims to target."""

    attribution = reference_error(approximate_attributions, exact_attributions)
    agreement = compare_attributions(approximate_attributions, exact_attributions, top_k=3)
    coalition = coalition_error(approximate_coalitions, exact_coalitions)
    return FidelityScores(
        attribution_mae=attribution.mean_absolute_error,
        attribution_max_error=attribution.max_absolute_error,
        attribution_rmse=attribution.root_mean_squared_error,
        relative_attribution_error=attribution.relative_error,
        attribution_spearman=agreement.spearman,
        attribution_kendall_tau_b=agreement.kendall_tau_b,
        attribution_sign_agreement=agreement.sign_agreement,
        attribution_top_3_overlap=agreement.top_k_jaccard,
        coalition_mae=coalition.mean_absolute_error,
        coalition_max_error=coalition.max_absolute_error,
        coalition_rmse=coalition.root_mean_squared_error,
        coalition_normalized_mae=coalition.normalized_mae,
        coalition_reference_spread=coalition.reference_spread,
        empty_endpoint_error=coalition.empty_endpoint_error,
        full_endpoint_error=coalition.full_endpoint_error,
    )


def aggregate_within_seed(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
    seed_field: str = "seed",
) -> tuple[dict[str, Any], ...]:
    """Average actions/states/coalitions before treating a seed as a replicate."""

    groups = tuple(group_fields)
    metrics = tuple(metric_fields)
    if seed_field in groups:
        raise StudyAnalysisError(f"{seed_field!r} is implicit and must not appear in group_fields")
    if not groups or not metrics:
        raise StudyAnalysisError("group_fields and metric_fields must be non-empty")
    if len(set((*groups, *metrics, seed_field))) != len((*groups, *metrics, seed_field)):
        raise StudyAnalysisError("group, metric, and seed fields must be distinct")

    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_fields(row, (*groups, seed_field, *metrics))
        buckets[tuple(row[field] for field in (*groups, seed_field))].append(row)
    if not buckets:
        raise StudyAnalysisError("cannot aggregate an empty table")

    aggregated: list[dict[str, Any]] = []
    for key in sorted(buckets, key=_stable_key):
        bucket = buckets[key]
        result = {field: value for field, value in zip((*groups, seed_field), key, strict=True)}
        for metric in metrics:
            finite = [_as_float(row[metric], field=metric) for row in bucket]
            finite = [value for value in finite if math.isfinite(value)]
            result[metric] = float(np.mean(finite)) if finite else math.nan
        result["within_seed_rows"] = len(bucket)
        aggregated.append(result)
    return tuple(aggregated)


def paired_treatment_effects(
    seed_rows: Iterable[Mapping[str, Any]],
    *,
    treatment_field: str,
    baseline: Any,
    strata_fields: Sequence[str],
    metric_fields: Sequence[str],
    seed_field: str = "seed",
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Compute treatment-minus-baseline effects by resampling paired seeds."""

    strata = tuple(strata_fields)
    metrics = tuple(metric_fields)
    if treatment_field in strata or seed_field in strata:
        raise StudyAnalysisError("treatment_field and seed_field must not appear in strata_fields")
    if len(set((*strata, treatment_field, seed_field, *metrics))) != len(
        (*strata, treatment_field, seed_field, *metrics)
    ):
        raise StudyAnalysisError("strata, treatment, seed, and metric fields must be distinct")
    if not metrics:
        raise StudyAnalysisError("metric_fields must be non-empty")

    indexed: dict[tuple[Any, ...], dict[Any, dict[Any, Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in seed_rows:
        _require_fields(row, (*strata, treatment_field, seed_field, *metrics))
        stratum = tuple(row[field] for field in strata)
        treatment = row[treatment_field]
        seed = row[seed_field]
        if seed in indexed[stratum][treatment]:
            raise StudyAnalysisError(
                f"duplicate seed-level row for stratum={stratum}, {treatment_field}={treatment!r}, seed={seed!r}"
            )
        indexed[stratum][treatment][seed] = row
    if not indexed:
        raise StudyAnalysisError("cannot compare an empty table")

    effects: list[dict[str, Any]] = []
    for stratum in sorted(indexed, key=_stable_key):
        treatments = indexed[stratum]
        if baseline not in treatments:
            raise StudyAnalysisError(f"stratum={stratum} has no baseline {treatment_field}={baseline!r}")
        baseline_seeds = set(treatments[baseline])
        for treatment in sorted((value for value in treatments if value != baseline), key=_stable_key):
            treatment_seeds = set(treatments[treatment])
            if treatment_seeds != baseline_seeds:
                missing_treatment = sorted(baseline_seeds - treatment_seeds, key=_stable_key)
                missing_baseline = sorted(treatment_seeds - baseline_seeds, key=_stable_key)
                raise StudyAnalysisError(
                    f"unpaired seeds for stratum={stratum}, {treatment_field}={treatment!r}; "
                    f"missing treatment={missing_treatment}, missing baseline={missing_baseline}"
                )
            paired_seeds = sorted(baseline_seeds, key=_stable_key)
            for metric in metrics:
                baseline_values: list[float] = []
                treatment_values: list[float] = []
                differences: list[float] = []
                for seed in paired_seeds:
                    baseline_value = _as_float(treatments[baseline][seed][metric], field=metric)
                    treatment_value = _as_float(treatments[treatment][seed][metric], field=metric)
                    if math.isfinite(baseline_value) and math.isfinite(treatment_value):
                        baseline_values.append(baseline_value)
                        treatment_values.append(treatment_value)
                        differences.append(treatment_value - baseline_value)
                derived_seed = _derived_seed(bootstrap_seed, stratum, treatment, metric)
                mean, low, high = bootstrap_mean_ci(
                    differences,
                    confidence=confidence,
                    resamples=bootstrap_resamples,
                    seed=derived_seed,
                )
                effects.append(
                    {
                        **{field: value for field, value in zip(strata, stratum, strict=True)},
                        treatment_field: treatment,
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_treatment": float(np.mean(treatment_values)) if treatment_values else math.nan,
                        "mean_baseline": float(np.mean(baseline_values)) if baseline_values else math.nan,
                        "mean_paired_difference": mean,
                        "confidence": confidence,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_resamples": bootstrap_resamples,
                        "bootstrap_seed": derived_seed,
                        "exact_sign_flip_p_value": exact_sign_flip_p_value(differences),
                    }
                )
    if not effects:
        raise StudyAnalysisError("no non-baseline treatments were found")
    return tuple(effects)


def exact_sign_flip_p_value(differences: Sequence[float] | np.ndarray) -> float:
    """Two-sided exact paired randomisation p-value for a mean difference."""

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if values.size > 20:
        raise StudyAnalysisError("exact sign-flip inference is limited to at most 20 paired seeds")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 1 << int(values.size)
    for signs in product((-1.0, 1.0), repeat=int(values.size)):
        statistic = abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        extreme += int(statistic >= observed - 1e-15)
    return extreme / total


def match_curves_at_common_budget(
    rows: Iterable[Mapping[str, Any]],
    *,
    budget_field: str,
    treatment_field: str,
    strata_fields: Sequence[str],
    metric_fields: Sequence[str],
    seed_field: str = "seed",
    target_budget: float | None = None,
) -> tuple[dict[str, Any], ...]:
    """Linearly interpolate every paired curve at its stratum's largest common budget."""

    strata = tuple(strata_fields)
    metrics = tuple(metric_fields)
    if budget_field in {*strata, treatment_field, seed_field}:
        raise StudyAnalysisError("budget_field must be distinct from strata, treatment, and seed fields")
    curves: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _require_fields(row, (*strata, treatment_field, seed_field, budget_field, *metrics))
        key = tuple(row[field] for field in (*strata, treatment_field, seed_field))
        curves[key].append(row)
    if not curves:
        raise StudyAnalysisError("cannot match an empty curve table")

    by_stratum: dict[tuple[Any, ...], dict[tuple[Any, Any], list[Mapping[str, Any]]]] = defaultdict(dict)
    for key, curve in curves.items():
        stratum = key[: len(strata)]
        treatment, seed = key[-2:]
        by_stratum[stratum][(treatment, seed)] = curve

    matched: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum, key=_stable_key):
        cell_curves = by_stratum[stratum]
        treatments = {treatment for treatment, _seed in cell_curves}
        seeds = {seed for _treatment, seed in cell_curves}
        expected = {(treatment, seed) for treatment in treatments for seed in seeds}
        missing = expected - set(cell_curves)
        if missing:
            raise StudyAnalysisError(f"missing treatment/seed curves for stratum={stratum}: {sorted(missing)}")

        prepared: dict[tuple[Any, Any], tuple[np.ndarray, list[Mapping[str, Any]]]] = {}
        for key, curve in cell_curves.items():
            ordered = sorted(curve, key=lambda row: _as_float(row[budget_field], field=budget_field))
            budgets = np.asarray([_as_float(row[budget_field], field=budget_field) for row in ordered])
            if not np.all(np.isfinite(budgets)) or np.any(np.diff(budgets) <= 0):
                raise StudyAnalysisError(f"{budget_field!r} must be finite and strictly increasing within each curve")
            prepared[key] = budgets, ordered
        common_low = max(float(budgets[0]) for budgets, _ordered in prepared.values())
        common_high = min(float(budgets[-1]) for budgets, _ordered in prepared.values())
        budget = common_high if target_budget is None else float(target_budget)
        if not math.isfinite(budget) or not common_low <= budget <= common_high:
            raise StudyAnalysisError(
                f"target {budget_field}={budget} lies outside common interval [{common_low}, {common_high}] "
                f"for stratum={stratum}"
            )

        for treatment, seed in sorted(prepared, key=_stable_key):
            budgets, ordered = prepared[(treatment, seed)]
            row = {
                **{field: value for field, value in zip(strata, stratum, strict=True)},
                treatment_field: treatment,
                seed_field: seed,
                budget_field: budget,
                "matching_budget": budget_field,
            }
            for metric in metrics:
                values = np.asarray([_as_float(item[metric], field=metric) for item in ordered])
                if not np.all(np.isfinite(values)):
                    raise StudyAnalysisError(f"cannot interpolate non-finite metric {metric!r}")
                row[metric] = float(np.interp(budget, budgets, values))
            row["interpolated"] = bool(not np.any(np.isclose(budgets, budget, rtol=0.0, atol=1e-12)))
            matched.append(row)
    return tuple(matched)


def arrow_table(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]], *, source: object) -> Any:
    """Build one Arrow table with a fixed column order, inferring each column's type.

    A field missing from a row becomes null rather than an error, which is what
    makes the union schema above work. Arrow's per-column inference is reported
    as a study error naming the column, because the raw message says only that
    some value did not convert and gives no way to find it.
    """

    require_storage_dependencies("Writing study tables")
    columns = {}
    for name in fieldnames:
        values = [row.get(name) for row in rows]
        try:
            columns[name] = pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise StudyAnalysisError(f"column {name!r} of {source} holds values of more than one type: {exc}") from exc
    return pa.table(columns)


def write_table(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write a deterministic union-schema Parquet table.

    Column order is the first row's key order followed by any further keys
    sorted, so a table's columns stay stable across runs. Unlike the CSV tables
    this replaced, values keep their type: consumers read a float as a float
    instead of re-parsing a string, and a column's type is now checked once here
    rather than at every read.
    """

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise StudyAnalysisError(f"refusing to write empty table {path}")
    fieldnames = list(materialized[0])
    fieldnames.extend(sorted({key for row in materialized for key in row} - set(fieldnames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow_table(fieldnames, materialized, source=path), path, compression=PARQUET_COMPRESSION)
    return path


def read_table(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a persisted study table with its column types intact."""

    require_storage_dependencies("Reading study tables")
    return tuple(pq.read_table(path).to_pylist())


def reduce_long_table(
    table_path: Path,
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
    filters: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Filter and average a long Parquet table in bounded memory.

    Partial sums are computed per row group and merged by key, so the result is
    exact while memory scales with the number of groups rather than the number
    of coalition rows. Filtering happens in Arrow before conversion to Python;
    Phase 1 uses this to keep the policy-law greedy-action diagnostic separate
    from uniform-law and all-legal-action rows.
    """

    require_storage_dependencies("Reducing study tables")
    import pyarrow.compute as pc

    groups = tuple(group_fields)
    metrics = tuple(metric_fields)
    predicates = dict(filters or {})
    if not groups or not metrics:
        raise StudyAnalysisError("group_fields and metric_fields must be non-empty")
    if len(set((*groups, *metrics))) != len((*groups, *metrics)):
        raise StudyAnalysisError("group and metric fields must be distinct")

    reader = pq.ParquetFile(table_path)
    available = set(reader.schema_arrow.names)
    requested = (*groups, *metrics, *predicates)
    missing = [name for name in requested if name not in available]
    if missing:
        raise StudyAnalysisError(f"{table_path} has no column(s) {missing}")

    sums: dict[tuple[Any, ...], list[float]] = defaultdict(lambda: [0.0] * len(metrics))
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    aggregation = [(metric, "sum") for metric in metrics] + [(metrics[0], "count")]
    columns = tuple(dict.fromkeys(requested))
    for batch in reader.iter_batches(batch_size=PARQUET_ROW_GROUP_ROWS, columns=list(columns)):
        table = pa.Table.from_batches([batch])
        if predicates:
            selected = None
            for name, expected in predicates.items():
                predicate = pc.equal(table[name], pa.scalar(expected, type=table.schema.field(name).type))
                selected = predicate if selected is None else pc.and_(selected, predicate)
            table = table.filter(selected)
        if table.num_rows == 0:
            continue
        partial = table.select([*groups, *metrics]).group_by(list(groups)).aggregate(aggregation)
        for row in partial.to_pylist():
            key = tuple(row[name] for name in groups)
            totals = sums[key]
            for index, metric in enumerate(metrics):
                totals[index] += row[f"{metric}_sum"] or 0.0
            counts[key] += row[f"{metrics[0]}_count"]

    reduced: list[dict[str, Any]] = []
    for key in sorted(sums, key=_stable_key):
        rows = counts[key]
        result: dict[str, Any] = dict(zip(groups, key, strict=True))
        for index, metric in enumerate(metrics):
            result[metric] = sums[key][index] / rows if rows else math.nan
        result["reduced_rows"] = rows
        reduced.append(result)
    if not reduced:
        detail = f" after filters {predicates}" if predicates else ""
        raise StudyAnalysisError(f"cannot reduce an empty table {table_path}{detail}")
    return tuple(reduced)


def concatenate_tables(paths: Iterable[Path], out_path: Path) -> Path:
    """Atomically concatenate identically shaped Parquet artifacts without materializing them.

    Row groups stream from each source straight into the destination writer, so
    a table far larger than memory concatenates in bounded space -- the property
    the CSV version guaranteed by never holding more than one line, and the one
    that matters most for the coalition table.
    """

    require_storage_dependencies("Concatenating study tables")
    sources = tuple(Path(path) for path in paths)
    if not sources:
        raise StudyAnalysisError(f"cannot concatenate an empty file set into {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer: Any = None
        expected: Any = None
        try:
            for source in sources:
                reader = pq.ParquetFile(source)
                schema = reader.schema_arrow
                if writer is None:
                    expected = schema
                    writer = pq.ParquetWriter(staging, schema, compression=PARQUET_COMPRESSION)
                elif not schema.equals(expected):
                    raise StudyAnalysisError(f"table schema mismatch for {source}; expected {expected}, got {schema}")
                for batch in reader.iter_batches(batch_size=PARQUET_ROW_GROUP_ROWS):
                    writer.write_batch(batch)
        finally:
            # Closing writes the footer, so it must happen before the rename; a
            # renamed file whose footer never landed is an unreadable table.
            if writer is not None:
                writer.close()
        os.replace(staging, out_path)
    finally:
        if staging.exists():
            staging.unlink()
    return out_path


def plot_metric_curves(
    table_path: Path,
    out_path: Path,
    *,
    x_field: str,
    y_field: str,
    series_fields: Sequence[str],
    seed_field: str = "seed",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    x_scale: str = "linear",
    facet_field: str | None = None,
    bootstrap_resamples: int = 2_000,
    show_seed_lines: bool = False,
    simultaneous_band: bool = False,
) -> Path | None:
    """Plot seed-level mean curves with deterministic 95% bootstrap bands.

    ``facet_field`` splits one of ``series_fields`` across subplots instead of
    encoding it in the legend, which is what keeps the legend readable once the
    condition grid grows.
    """

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    groups = _independent_groups(
        rows,
        group_fields=(*series_fields, x_field),
        seed_field=seed_field,
        metric_fields=(y_field,),
    )
    series: dict[tuple[Any, ...], list[tuple[float, float, float, float]]] = defaultdict(list)
    seed_curves: dict[tuple[Any, ...], dict[Any, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        _require_fields(row, (*series_fields, x_field, y_field, seed_field))
        key = tuple(row[field] for field in series_fields)
        seed_curves[key][row[seed_field]].append(
            (_as_float(row[x_field], field=x_field), _as_float(row[y_field], field=y_field))
        )
    if simultaneous_band:
        for key, by_seed in seed_curves.items():
            ordered = {seed: sorted(points) for seed, points in by_seed.items()}
            x_values = tuple(point[0] for point in next(iter(ordered.values())))
            if any(tuple(point[0] for point in points) != x_values for points in ordered.values()):
                raise StudyAnalysisError(f"simultaneous curve series {key} has unequal seed checkpoints")
            matrix = np.asarray(
                [[point[1] for point in ordered[seed]] for seed in sorted(ordered, key=_stable_key)],
                dtype=np.float64,
            )
            mean, low, high = simultaneous_mean_band(
                matrix,
                resamples=bootstrap_resamples,
                seed=_derived_seed(0, key, y_field, "simultaneous"),
            )
            series[key] = [
                (x, float(mean[index]), float(low[index]), float(high[index])) for index, x in enumerate(x_values)
            ]
    else:
        for key, values in groups.items():
            x = _as_float(key[-1], field=x_field)
            y_values = [_as_float(row[y_field], field=y_field) for row in values]
            mean, low, high = bootstrap_mean_ci(
                y_values, resamples=bootstrap_resamples, seed=_derived_seed(0, key, y_field)
            )
            series[key[:-1]].append((x, mean, low, high))

    facet_index, legend_fields = _split_facets(series_fields, facet_field)
    facets = sorted({_facet_of(key, facet_index) for key in series}, key=_stable_key)
    legend_order = sorted({_legend_key(key, facet_index) for key in series}, key=_stable_key)
    styles = {key: _series_style(i) for i, key in enumerate(legend_order)}

    figure, axes_row = plt.subplots(
        1,
        len(facets),
        figsize=(max(7.5, 4.6 * len(facets)), 4.5),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes_row[0])
    for axes, facet in zip(axes_list, facets, strict=True):
        for key in sorted(series, key=_stable_key):
            if _facet_of(key, facet_index) != facet:
                continue
            points = sorted(series[key])
            legend_key = _legend_key(key, facet_index)
            style = styles[legend_key]
            if show_seed_lines:
                for seed_points in seed_curves[key].values():
                    ordered_seed = sorted(seed_points)
                    axes.plot(
                        [point[0] for point in ordered_seed],
                        [point[1] for point in ordered_seed],
                        color=style["color"],
                        linewidth=0.6,
                        alpha=0.18,
                        zorder=1,
                    )
            axes.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                markersize=3,
                linewidth=1.5,
                label=_series_label(legend_fields, legend_key) if legend_fields else "_nolegend_",
                **style,
            )
            axes.fill_between(
                [point[0] for point in points],
                [point[2] for point in points],
                [point[3] for point in points],
                color=style["color"],
                alpha=0.15,
            )
        axes.set_xscale(x_scale)
        axes.set_xlabel(x_label or x_field)
        axes.grid(True, **_GRID_KWARGS)
        if facet is not None:
            axes.set_title(f"{facet_field}={facet}", fontsize="medium")
    axes_list[0].set_ylabel(y_label or y_field)
    figure.suptitle(title or f"{y_field} vs {x_field}")
    _place_legend(figure, axes_list, legend_fields)
    return _save_figure(plt, figure, out_path)


def simultaneous_mean_band(
    seed_curves: Sequence[Sequence[float]] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed-cluster bootstrap sup-norm band for an entire mean curve."""

    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    curves = np.asarray(seed_curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[0] < 1 or curves.shape[1] < 1:
        raise StudyAnalysisError("seed_curves must be a non-empty seed-by-checkpoint matrix")
    if not np.all(np.isfinite(curves)):
        raise StudyAnalysisError("simultaneous bands require finite seed curves")
    mean = curves.mean(axis=0)
    if curves.shape[0] == 1:
        return mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, curves.shape[0], size=(resamples, curves.shape[0]))
    bootstrap_means = curves[draws].mean(axis=1)
    radius = float(np.quantile(np.max(np.abs(bootstrap_means - mean[None, :]), axis=1), confidence))
    return mean, mean - radius, mean + radius


def plot_categorical_metric(
    table_path: Path,
    out_path: Path,
    *,
    category_field: str,
    y_field: str,
    series_fields: Sequence[str],
    seed_field: str = "seed",
    title: str | None = None,
    facet_field: str | None = None,
    category_order: Sequence[Any] | None = None,
) -> Path | None:
    """Plot a role/opening-style categorical comparison from seed rows."""

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    groups = _independent_groups(
        rows,
        group_fields=(*series_fields, category_field),
        seed_field=seed_field,
        metric_fields=(y_field,),
    )
    observed_categories = {key[-1] for key in groups}
    categories = sorted(observed_categories, key=_stable_key) if category_order is None else list(category_order)
    if len(categories) != len(observed_categories) or set(categories) != observed_categories:
        raise StudyAnalysisError("category_order must contain every observed category exactly once")
    series_keys = sorted({key[:-1] for key in groups}, key=_stable_key)
    facet_index, legend_fields = _split_facets(series_fields, facet_field)
    facets = sorted({_facet_of(key, facet_index) for key in series_keys}, key=_stable_key)
    legend_order = sorted({_legend_key(key, facet_index) for key in series_keys}, key=_stable_key)
    styles = {key: _series_style(i) for i, key in enumerate(legend_order)}

    figure, axes_row = plt.subplots(
        1,
        len(facets),
        figsize=(max(7.0, 1.1 * len(categories)) * len(facets), 4.5),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes_row[0])
    positions = np.arange(len(categories), dtype=np.float64)
    for axes, facet in zip(axes_list, facets, strict=True):
        panel_keys = [key for key in series_keys if _facet_of(key, facet_index) == facet]
        width = 0.8 / max(1, len(panel_keys))
        for index, series_key in enumerate(panel_keys):
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for category in categories:
                values = groups.get((*series_key, category), ())
                samples = [_as_float(row[y_field], field=y_field) for row in values]
                mean, low, high = bootstrap_mean_ci(samples, seed=_derived_seed(0, series_key, category, y_field))
                means.append(mean)
                lows.append(mean - low)
                highs.append(high - mean)
            offset = (index - (len(panel_keys) - 1) / 2.0) * width
            legend_key = _legend_key(series_key, facet_index)
            axes.errorbar(
                positions + offset,
                means,
                yerr=np.asarray([lows, highs]),
                fmt=styles[legend_key]["marker"],
                capsize=3,
                color=styles[legend_key]["color"],
                label=_series_label(legend_fields, legend_key) if legend_fields else "_nolegend_",
            )
        axes.set_xticks(positions)
        axes.set_xticklabels([str(category) for category in categories])
        axes.set_xlabel(category_field)
        axes.grid(True, axis="y", **_GRID_KWARGS)
        if facet is not None:
            axes.set_title(f"{facet_field}={facet}", fontsize="medium")
    axes_list[0].set_ylabel(y_field)
    figure.suptitle(title or f"{y_field} by {category_field}")
    _place_legend(figure, axes_list, legend_fields)
    return _save_figure(plt, figure, out_path)


def plot_paired_seed_slopes(
    table_path: Path,
    out_path: Path,
    *,
    category_field: str,
    y_field: str,
    seed_field: str = "seed",
    title: str | None = None,
) -> Path | None:
    """Plot each paired seed as a faint slope and overlay category means."""

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    _independent_groups(
        rows,
        group_fields=(category_field,),
        seed_field=seed_field,
        metric_fields=(y_field,),
    )
    categories = sorted({row[category_field] for row in rows}, key=_stable_key)
    by_seed: dict[Any, dict[Any, float]] = defaultdict(dict)
    for row in rows:
        seed = row[seed_field]
        category = row[category_field]
        if category in by_seed[seed]:
            raise StudyAnalysisError(
                f"duplicate seed-level row for {seed_field}={seed!r}, {category_field}={category!r}"
            )
        by_seed[seed][category] = _as_float(row[y_field], field=y_field)
    expected = set(categories)
    incomplete = [seed for seed, values in by_seed.items() if set(values) != expected]
    if incomplete:
        raise StudyAnalysisError(f"paired slope plot has incomplete categories for seeds {incomplete}")

    positions = np.arange(len(categories), dtype=np.float64)
    figure, axes = plt.subplots(figsize=(max(6.5, 1.4 * len(categories)), 4.5))
    for seed in sorted(by_seed, key=_stable_key):
        axes.plot(
            positions,
            [by_seed[seed][category] for category in categories],
            color="#777777",
            linewidth=0.7,
            alpha=0.28,
            marker="o",
            markersize=2.5,
        )
    means = [float(np.mean([by_seed[seed][category] for seed in by_seed])) for category in categories]
    axes.plot(positions, means, color="#000000", linewidth=2.0, marker="D", label="mean")
    axes.set_xticks(positions)
    axes.set_xticklabels([str(category) for category in categories])
    axes.set_xlabel(category_field)
    axes.set_ylabel(y_field)
    axes.grid(True, axis="y", **_GRID_KWARGS)
    axes.set_title(title or f"Paired {y_field}")
    return _save_figure(plt, figure, out_path)


def plot_policy_fidelity_tradeoff(
    table_path: Path,
    out_path: Path,
    *,
    policy_field: str,
    fidelity_field: str,
    series_fields: Sequence[str],
    seed_field: str = "seed",
    title: str | None = None,
    facet_field: str | None = None,
    show_seed_points: bool = False,
) -> Path | None:
    """Plot seed-level policy quality against explanation error."""

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    groups = _independent_groups(
        rows,
        group_fields=tuple(series_fields),
        seed_field=seed_field,
        metric_fields=(policy_field, fidelity_field),
    )
    facet_index, legend_fields = _split_facets(series_fields, facet_field)
    facets = sorted({_facet_of(key, facet_index) for key in groups}, key=_stable_key)
    legend_order = sorted({_legend_key(key, facet_index) for key in groups}, key=_stable_key)
    styles = {key: _series_style(i) for i, key in enumerate(legend_order)}

    figure, axes_row = plt.subplots(
        1,
        len(facets),
        figsize=(max(7.0, 4.4 * len(facets)), 4.8),
        sharey=True,
        squeeze=False,
    )
    axes_list = list(axes_row[0])
    for axes, facet in zip(axes_list, facets, strict=True):
        for key in sorted(groups, key=_stable_key):
            if _facet_of(key, facet_index) != facet:
                continue
            values = groups[key]
            policy = [_as_float(row[policy_field], field=policy_field) for row in values]
            fidelity = [_as_float(row[fidelity_field], field=fidelity_field) for row in values]
            x_mean, x_low, x_high = bootstrap_mean_ci(policy, seed=_derived_seed(0, key, policy_field))
            y_mean, y_low, y_high = bootstrap_mean_ci(fidelity, seed=_derived_seed(0, key, fidelity_field))
            legend_key = _legend_key(key, facet_index)
            if show_seed_points:
                axes.scatter(
                    policy,
                    fidelity,
                    s=14,
                    alpha=0.25,
                    color=styles[legend_key]["color"],
                )
            axes.errorbar(
                x_mean,
                y_mean,
                xerr=[[x_mean - x_low], [x_high - x_mean]],
                yerr=[[y_mean - y_low], [y_high - y_mean]],
                fmt=styles[legend_key]["marker"],
                capsize=3,
                color=styles[legend_key]["color"],
                label=_series_label(legend_fields, legend_key),
            )
        axes.set_xlabel(policy_field)
        axes.grid(True, **_GRID_KWARGS)
        if facet is not None:
            axes.set_title(f"{facet_field}={facet}", fontsize="medium")
    axes_list[0].set_ylabel(fidelity_field)
    figure.suptitle(title or "Policy quality vs explanation error")
    _place_legend(figure, axes_list, legend_fields)
    return _save_figure(plt, figure, out_path)


def plot_calibration_residuals(
    table_path: Path,
    out_path: Path,
    *,
    exact_field: str,
    approximate_field: str,
    facet_field: str,
    color_field: str,
    weight_field: str | None = None,
    title: str | None = None,
    exact_label: str | None = None,
    approximate_label: str | None = None,
) -> Path | None:
    """Plot pointwise parity and residuals without treating points as replicates."""

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    required = [exact_field, approximate_field, facet_field, color_field]
    if weight_field is not None:
        required.append(weight_field)
    for row in rows:
        _require_fields(row, required)

    facets = sorted({row[facet_field] for row in rows}, key=_stable_key)
    colors = sorted({row[color_field] for row in rows}, key=_stable_key)
    color_map = {value: _PALETTE[index % len(_PALETTE)] for index, value in enumerate(colors)}
    all_exact = np.asarray([_as_float(row[exact_field], field=exact_field) for row in rows])
    all_approximate = np.asarray([_as_float(row[approximate_field], field=approximate_field) for row in rows])
    lower = float(min(all_exact.min(), all_approximate.min()))
    upper = float(max(all_exact.max(), all_approximate.max()))
    span = max(upper - lower, 1e-9)
    bounds = (lower - 0.04 * span, upper + 0.04 * span)
    residual_limit = 1.05 * float(np.max(np.abs(all_approximate - all_exact)))
    residual_limit = max(residual_limit, 1e-9)

    figure, axes = plt.subplots(
        2,
        len(facets),
        figsize=(max(8.0, 4.2 * len(facets)), 7.4),
        sharex="col",
        squeeze=False,
    )
    for column, facet in enumerate(facets):
        facet_rows = [row for row in rows if row[facet_field] == facet]
        top = axes[0, column]
        bottom = axes[1, column]
        for color_value in colors:
            selected = [row for row in facet_rows if row[color_field] == color_value]
            if not selected:
                continue
            exact = np.asarray([_as_float(row[exact_field], field=exact_field) for row in selected])
            approximate = np.asarray([_as_float(row[approximate_field], field=approximate_field) for row in selected])
            if weight_field is None:
                sizes: float | np.ndarray = 7.0
            else:
                weights = np.asarray([_as_float(row[weight_field], field=weight_field) for row in selected])
                maximum = max(float(weights.max()), 1e-12)
                sizes = 5.0 + 22.0 * np.sqrt(np.maximum(weights, 0.0) / maximum)
            top.scatter(
                exact,
                approximate,
                s=sizes,
                color=color_map[color_value],
                alpha=0.20,
                linewidths=0,
                rasterized=True,
                label=str(color_value),
            )
            bottom.scatter(
                exact,
                approximate - exact,
                s=sizes,
                color=color_map[color_value],
                alpha=0.20,
                linewidths=0,
                rasterized=True,
            )
        weights = (
            np.ones(len(facet_rows), dtype=np.float64)
            if weight_field is None
            else np.asarray([_as_float(row[weight_field], field=weight_field) for row in facet_rows])
        )
        exact = np.asarray([_as_float(row[exact_field], field=exact_field) for row in facet_rows])
        approximate = np.asarray([_as_float(row[approximate_field], field=approximate_field) for row in facet_rows])
        mae = float(np.average(np.abs(approximate - exact), weights=weights))
        top.plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1.0)
        top.set_xlim(bounds)
        top.set_ylim(bounds)
        top.set_aspect("equal", adjustable="box")
        try:
            facet_label = f"r={float(facet):g}"
        except (TypeError, ValueError):
            facet_label = str(facet)
        top.set_title(f"{facet_label}\nweighted MAE={mae:.4f}", fontsize="medium")
        top.grid(True, **_GRID_KWARGS)
        bottom.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        bottom.set_xlim(bounds)
        bottom.set_ylim((-residual_limit, residual_limit))
        bottom.set_xlabel(exact_label or exact_field)
        bottom.grid(True, **_GRID_KWARGS)
    axes[0, 0].set_ylabel(approximate_label or approximate_field)
    axes[1, 0].set_ylabel("approximate − exact")
    figure.suptitle(title or f"{approximate_field} calibration")
    _place_legend(figure, tuple(axes[0]), (color_field,))
    return _save_figure(plt, figure, out_path)


def plot_correlation_scatter(
    table_path: Path,
    out_path: Path,
    *,
    exact_field: str,
    approximate_field: str,
    selection_field: str,
    selection_value: Any,
    color_field: str,
    weight_field: str | None = None,
    title: str | None = None,
    exact_label: str | None = None,
    approximate_label: str | None = None,
) -> Path | None:
    """Plot approximate against exact values for one selected method."""

    plt = _load_pyplot()
    if plt is None:
        return None
    rows = read_table(table_path)
    required = [exact_field, approximate_field, selection_field, color_field]
    if weight_field is not None:
        required.append(weight_field)
    for row in rows:
        _require_fields(row, required)
    selected = [row for row in rows if row[selection_field] == selection_value]
    if not selected:
        raise StudyAnalysisError(f"no rows match {selection_field}={selection_value!r}")

    all_exact = np.asarray([_as_float(row[exact_field], field=exact_field) for row in rows])
    all_approximate = np.asarray([_as_float(row[approximate_field], field=approximate_field) for row in rows])
    lower = float(min(all_exact.min(), all_approximate.min()))
    upper = float(max(all_exact.max(), all_approximate.max()))
    span = max(upper - lower, 1e-9)
    bounds = (lower - 0.04 * span, upper + 0.04 * span)

    colors = sorted({row[color_field] for row in selected}, key=_stable_key)
    color_map = {value: _PALETTE[index % len(_PALETTE)] for index, value in enumerate(colors)}
    figure, axes = plt.subplots(figsize=(6.4, 5.8))
    for color_value in colors:
        subset = [row for row in selected if row[color_field] == color_value]
        exact = np.asarray([_as_float(row[exact_field], field=exact_field) for row in subset])
        approximate = np.asarray([_as_float(row[approximate_field], field=approximate_field) for row in subset])
        if weight_field is None:
            sizes: float | np.ndarray = 7.0
        else:
            weights = np.asarray([_as_float(row[weight_field], field=weight_field) for row in subset])
            maximum = max(float(weights.max()), 1e-12)
            sizes = 0.1 + 36.0 * np.sqrt(np.maximum(weights, 0.0) / maximum)
        axes.scatter(
            exact,
            approximate,
            s=sizes,
            color=color_map[color_value],
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            label=str(color_value),
        )
    axes.plot(bounds, bounds, color="#333333", linestyle="--", linewidth=1.0, label="exact agreement")
    axes.set_xlim(bounds)
    axes.set_ylim(bounds)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(exact_label or exact_field)
    axes.set_ylabel(approximate_label or approximate_field)
    axes.set_title(title or f"{selection_value}: approximate vs exact")
    axes.grid(True, **_GRID_KWARGS)
    axes.legend(frameon=True)
    return _save_figure(plt, figure, out_path)


def _independent_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    seed_field: str,
    metric_fields: Sequence[str],
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        _require_fields(row, (*group_fields, seed_field, *metric_fields))
        key = tuple(row[field] for field in group_fields)
        identity = (*key, row[seed_field])
        if identity in seen:
            raise StudyAnalysisError(
                f"plot table is not seed-level: duplicate group={key}, {seed_field}={row[seed_field]!r}"
            )
        seen.add(identity)
        groups[key].append(row)
    if not groups:
        raise StudyAnalysisError("cannot plot an empty table")
    return groups


def _load_pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def _save_figure(plt: Any, figure: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", dpi=150, metadata={"Software": "xrl"})
    plt.close(figure)
    return path


def _series_label(fields: Sequence[str], values: Sequence[Any]) -> str:
    # One field is named once in the legend title, so repeating it on every
    # entry only costs width; several fields still need naming to stay legible.
    if len(fields) == 1:
        return str(values[0])
    return ", ".join(f"{field}={value}" for field, value in zip(fields, values, strict=True)) or "all"


def _split_facets(
    series_fields: Sequence[str],
    facet_field: str | None,
) -> tuple[int | None, tuple[str, ...]]:
    """Return the facet position and the fields that remain in the legend."""

    if facet_field is None:
        return None, tuple(series_fields)
    if facet_field not in series_fields:
        raise ValueError(f"facet_field {facet_field!r} must be one of {tuple(series_fields)}")
    index = tuple(series_fields).index(facet_field)
    return index, tuple(f for f in series_fields if f != facet_field)


def _facet_of(key: Sequence[Any], index: int | None) -> Any:
    return None if index is None else key[index]


def _legend_key(key: Sequence[Any], index: int | None) -> tuple[Any, ...]:
    if index is None:
        return tuple(key)
    return tuple(value for position, value in enumerate(key) if position != index)


def _place_legend(figure: Any, axes_list: Sequence[Any], legend_fields: Sequence[str]) -> None:
    """Put one shared legend outside the axes.

    A legend drawn inside the axes is unreadable once the series count grows
    past a handful, and a study's series count is the product of its condition
    grid. Collecting handles once and anchoring the legend below the figure
    keeps the plotting area clear however many series there are.

    Handles are collected from every facet, not just the first. Most panels
    repeat the same series in each facet, where the dedup below is a no-op; a
    facet that partitions the series instead -- one panel per method -- would
    otherwise lose every series that does not appear in the leftmost panel.
    """

    handles: list[Any] = []
    labels: list[str] = []
    for axes in axes_list:
        for handle, label in zip(*axes.get_legend_handles_labels(), strict=True):
            if label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    if not handles:
        return
    columns = max(1, min(3, (len(handles) + 7) // 8))
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=columns,
        fontsize="x-small",
        frameon=False,
        title=", ".join(legend_fields) or None,
        title_fontsize="x-small",
    )


def _derived_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps((base_seed, *parts), sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _stable_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise StudyAnalysisError(f"row is missing required fields {missing}: {dict(row)}")


def _as_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StudyAnalysisError(f"field {field!r} must be numeric, got {value!r}") from exc
