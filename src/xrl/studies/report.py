"""Engineering gate and derived report for a completed controlled-Q study."""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from xrl._optional import jax
from xrl.explainability.adapters import load_explanation_run
from xrl.studies.analysis import (
    StudyAnalysisError,
    aggregate_within_seed,
    concatenate_tables,
    match_curves_at_common_budget,
    paired_treatment_effects,
    plot_calibration_residuals,
    plot_categorical_metric,
    plot_correlation_scatter,
    plot_metric_curves,
    plot_paired_seed_slopes,
    plot_policy_fidelity_tradeoff,
    read_table,
    reduce_long_table,
    write_table,
)
from xrl.studies.core import StudyCellResult, StudyCondition, StudyProfile
from xrl.training.metrics_io import read_dict_rows
from xrl.training.neural_q import NeuralQAgent

PRIMARY_POLICY_METRICS = (
    "expected_return",
    "exploitability",
    "optimal_action_mass",
    "mean_one_step_regret",
)

_POLICY_METRICS = (
    "expected_return",
    "win_probability",
    "draw_probability",
    "loss_probability",
    "best_response_return",
    "exploitability",
    "optimal_action_mass",
    "mean_one_step_regret",
    "expected_decisions",
    "decision_support_states",
)
_ACCOUNTING_METRICS = (
    "replay_transition_count",
    "gradient_update_count",
    "full_training_example_count",
    "masked_training_example_count",
    "masked_legal_target_count",
    "predictor_training_example_count",
    "training_flops",
)
_FIDELITY_METRICS = (
    "attribution_mae",
    "attribution_max_error",
    "attribution_rmse",
    "relative_attribution_error",
    "attribution_spearman",
    "attribution_kendall_tau_b",
    "attribution_sign_agreement",
    "attribution_top_3_overlap",
    "coalition_mae",
    "coalition_max_error",
    "coalition_rmse",
    "coalition_normalized_mae",
    "coalition_reference_spread",
    "empty_endpoint_error",
    "full_endpoint_error",
    "return_q_attribution_mae",
    "base_q_attribution_mae",
    "return_q_coalition_mae",
    "base_q_coalition_mae",
)


class EngineeringGateError(RuntimeError):
    """Raised after a gate report records one or more failed checks."""


@dataclass(frozen=True, slots=True)
class EngineeringGateCheck:
    name: str
    passed: bool
    detail: str
    condition_key: str | None = None
    seed: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EngineeringGateReport:
    passed: bool
    checks: tuple[EngineeringGateCheck, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [check.to_json_dict() for check in self.checks]}


@dataclass(frozen=True, slots=True)
class ControlledQReportResult:
    primary_metric: str
    tables: tuple[Path, ...]
    plots: tuple[Path, ...]
    gate_path: Path

    def to_json_dict(self, *, root: Path) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "tables": [path.relative_to(root).as_posix() for path in self.tables],
            "plots": [path.relative_to(root).as_posix() for path in self.plots],
            "engineering_gate": self.gate_path.relative_to(root).as_posix(),
        }


def validate_controlled_q_cell(
    cell: StudyCellResult,
    *,
    condition: StudyCondition,
    profile: StudyProfile,
) -> tuple[tuple[EngineeringGateCheck, ...], tuple[tuple[int, ...], ...] | None]:
    """Validate one immutable cell and return its parameter-shape signature."""

    path = cell.path
    parameters = condition.parameters
    settings = profile.settings
    checks: list[EngineeringGateCheck] = []

    def add(name: str, check: Callable[[], Any]) -> Any | None:
        try:
            detail = check()
            checks.append(
                EngineeringGateCheck(
                    name=name,
                    passed=True,
                    detail="ok" if detail is None else str(detail),
                    condition_key=condition.key,
                    seed=cell.seed,
                )
            )
            return detail
        except Exception as exc:  # noqa: BLE001 - every failed invariant belongs in the gate artifact
            checks.append(
                EngineeringGateCheck(
                    name=name,
                    passed=False,
                    detail=f"{type(exc).__name__}: {exc}",
                    condition_key=condition.key,
                    seed=cell.seed,
                )
            )
            return None

    required = (
        "config.json",
        "model.msgpack",
        "metrics.csv",
        "summary.json",
        "checkpoint_index.parquet",
        "exact_policy_metrics.parquet",
        "exact_role_metrics.parquet",
        "exact_opening_metrics.parquet",
    )
    add("required_artifacts", lambda: _require_files(path, required))
    checkpoints = read_table(path / "checkpoint_index.parquet")
    policy_rows = read_table(path / "exact_policy_metrics.parquet")

    def checkpoint_schedule() -> str:
        expected = tuple(range(0, int(settings["episodes"]) + 1, int(settings["checkpoint_interval"])))
        actual = tuple(int(row["episode"]) for row in checkpoints)
        if actual != expected:
            raise ValueError(f"expected checkpoint episodes {expected}, got {actual}")
        return f"{len(actual)} checkpoints"

    add("checkpoint_schedule", checkpoint_schedule)

    def exact_counters() -> str:
        batch_size = int(settings["batch_size"])
        ratio = float(parameters["masked_batch_ratio"])
        for row in checkpoints:
            updates = int(row["gradient_update_count"])
            expected_full = updates * batch_size
            expected_masked = updates * round(ratio * batch_size)
            full = int(row["full_training_example_count"])
            masked = int(row["masked_training_example_count"])
            legal_targets = int(row["masked_legal_target_count"])
            predictor = int(row["predictor_training_example_count"])
            if (full, masked, predictor) != (expected_full, expected_masked, expected_full + expected_masked):
                raise ValueError(
                    f"episode {row['episode']} counters {(full, masked, predictor)} != "
                    f"{(expected_full, expected_masked, expected_full + expected_masked)}"
                )
            target = str(settings["masked_target"])
            if target == "td-action" and legal_targets != masked:
                raise ValueError(f"episode {row['episode']} legal targets {legal_targets} != masked rows {masked}")
            if target == "self-distill" and not masked <= legal_targets <= 9 * masked:
                raise ValueError(
                    f"episode {row['episode']} legal targets {legal_targets} outside "
                    f"the valid [{masked}, {9 * masked}] range"
                )
        return f"final updates={checkpoints[-1]['gradient_update_count']}"

    add("exact_example_counters", exact_counters)

    def monotone_accounting() -> str:
        for field in (*_ACCOUNTING_METRICS[:-1], "training_flops"):
            values = [float(row[field]) for row in checkpoints]
            if any(right < left for left, right in zip(values, values[1:], strict=False)):
                raise ValueError(f"{field} is not monotone: {values}")
        sources = {row["flops_source"] for row in checkpoints if row["flops_source"] != "unmeasured"}
        if len(sources) > 1:
            raise ValueError(f"mixed FLOP sources {sources}")
        return f"flops_source={next(iter(sources), 'unmeasured')}"

    add("monotone_accounting", monotone_accounting)

    def finite_metrics() -> str:
        for row in policy_rows:
            for field in _POLICY_METRICS:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite exact metric {field} at episode {row['episode']}")
        for row in read_dict_rows(path / "metrics.csv"):
            for field in ("epsilon", "learner_return", "td_error_mean", "td_error_max"):
                if row.get(field, "") and not math.isfinite(float(row[field])):
                    raise ValueError(f"non-finite training metric {field} at episode {row.get('episode')}")
        return f"{len(policy_rows)} exact checkpoints"

    add("finite_losses_and_metrics", finite_metrics)
    add(
        "final_checkpoint_matches_model",
        lambda: _require_equal_bytes(path / "model.msgpack", path / checkpoints[-1]["checkpoint"]),
    )

    agent: NeuralQAgent | None = None
    shape_signature: tuple[tuple[int, ...], ...] | None = None
    try:
        agent = cast(NeuralQAgent, load_explanation_run(path).policy)
        shape_signature = tuple(
            tuple(int(size) for size in np.asarray(value).shape) for value in jax.tree_util.tree_leaves(agent.params)
        )
        checks.append(
            EngineeringGateCheck(
                "checkpoint_reload",
                True,
                f"parameter arrays={len(shape_signature)}",
                condition.key,
                cell.seed,
            )
        )
        for row in checkpoints:
            agent.load_params(path / row["checkpoint"])
    except Exception as exc:  # noqa: BLE001
        checks.append(
            EngineeringGateCheck(
                "checkpoint_reload",
                False,
                f"{type(exc).__name__}: {exc}",
                condition.key,
                cell.seed,
            )
        )

    if bool(parameters["feature_mask_input"]) and agent is not None:

        def full_mask_equivalence() -> str:
            observation = np.zeros(agent.observation_size, dtype=np.float32)
            known_mask = np.ones(agent.observation_size, dtype=np.int8)
            full = agent.q_values(observation)
            masked = agent.masked_q_values(
                observation,
                known_mask,
                allow_untrained=float(parameters["masked_batch_ratio"]) == 0.0,
            )
            error = float(np.max(np.abs(np.asarray(full) - np.asarray(masked))))
            if error > 1e-6:
                raise ValueError(f"full-mask Q differs by {error}")
            return f"max_abs_error={error:.3g}"

        add("full_mask_equivalence", full_mask_equivalence)

    explanation_expected = parameters["arm"] == "primary" and cell.seed in {
        int(seed) for seed in settings["explanation_seeds"]
    }

    def posthoc_contract() -> str:
        output = cell.output.get("posthoc", {})
        if bool(output.get("executed")) != explanation_expected:
            raise ValueError(f"posthoc executed={output.get('executed')} but expected {explanation_expected}")
        summary_path = path / "posthoc_summary.json"
        if not explanation_expected:
            if summary_path.exists():
                raise ValueError("unselected cell contains posthoc artifacts")
            return "not selected"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protocol = str(settings.get("posthoc_protocol", "legacy"))
        # The requested count is an upper bound, not an equality: a converged
        # policy's occupancy support can be smaller, in which case every support
        # state is explained and reference_coverage.parquet records the shortfall.
        requested_states = int(settings["shared_explanation_states"]) + int(settings["policy_explanation_states"])
        if summary["selected_states"] <= 0:
            raise ValueError("post-hoc comparison selected no states")
        if protocol == "legacy" and summary["selected_states"] > requested_states:
            raise ValueError(f"wrong selected-state count {summary['selected_states']} for at most {requested_states}")
        detail_comparisons = int(summary.get("coalition_detail_comparisons", summary["model_comparisons"]))
        if summary["coalition_rows"] != 512 * detail_comparisons:
            raise ValueError("coalition row count does not match 512 coalitions per detailed comparison")
        if summary["attribution_rows"] != 9 * summary["model_comparisons"]:
            raise ValueError("attribution row count does not match nine features per comparison")
        if summary["fidelity_rows"] != summary["model_comparisons"]:
            raise ValueError("fidelity row count does not match comparisons")
        surrogate_rows = read_table(path / "surrogate_runs.parquet")
        if protocol == "phase-one":
            expected_runs = 9 if math.isclose(float(parameters["masked_batch_ratio"]), 1.0) else 1
            if summary["surrogate_runs"] != expected_runs or len(surrogate_rows) != expected_runs:
                raise ValueError(
                    f"Phase 1 expected {expected_runs} surrogate fits, got "
                    f"summary={summary['surrogate_runs']}, rows={len(surrogate_rows)}"
                )
            _require_files(path, ("state_buffer_occurrences.parquet", "state_buffer_counts.parquet"))
            surrogate_keys = tuple(str(row["fit_key"]) for row in surrogate_rows)
        else:
            surrogate_keys = ("shared", "policy")
        for reference in surrogate_keys:
            surrogate_summary = json.loads(
                (path / "surrogates" / reference / "surrogate_summary.json").read_text(encoding="utf-8")
            )
            if not math.isfinite(float(surrogate_summary["final_loss"])):
                raise ValueError(f"non-finite {reference} surrogate loss")
        coverage = read_table(path / "reference_coverage.parquet")
        if len(coverage) != 4:
            raise ValueError(f"expected four reference/role coverage rows, got {len(coverage)}")
        if protocol == "phase-one" and {str(row["sample_reference"]) for row in coverage} != {
            "policy",
            "uniform",
        }:
            raise ValueError("Phase 1 coverage must contain separate policy and uniform laws")
        return f"comparisons={summary['model_comparisons']}"

    add("posthoc_contract", posthoc_contract)
    return tuple(checks), shape_signature


def validate_controlled_q_study(plan: Any, result: Any) -> EngineeringGateReport:
    """Validate every cell plus the within-head architecture invariant."""

    checks: list[EngineeringGateCheck] = []
    shapes: dict[str, set[tuple[tuple[int, ...], ...]]] = defaultdict(set)
    specs_and_results = [(plan.primary, result.primary), (plan.raw_input, result.raw_input)]
    scalar_spec = getattr(plan, "scalar_learning", None)
    scalar_result = getattr(result, "scalar_learning", None)
    checks.append(
        EngineeringGateCheck(
            name="scalar_learning_presence",
            passed=(scalar_spec is None) == (scalar_result is None),
            detail=f"planned={scalar_spec is not None}, completed={scalar_result is not None}",
        )
    )
    if scalar_spec is not None and scalar_result is not None:
        specs_and_results.append((scalar_spec, scalar_result))
    for spec, study_result in specs_and_results:
        conditions = {condition.key: condition for condition in spec.conditions}
        expected_ids = {(condition.key, seed) for condition in spec.conditions for seed in spec.seeds}
        actual_ids = {(cell.condition_key, cell.seed) for cell in study_result.cells}
        checks.append(
            EngineeringGateCheck(
                name=f"{spec.name}_matrix_complete",
                passed=actual_ids == expected_ids,
                detail=f"expected={len(expected_ids)}, actual={len(actual_ids)}",
            )
        )
        for cell in study_result.cells:
            condition = conditions[cell.condition_key]
            cell_checks, shape = validate_controlled_q_cell(cell, condition=condition, profile=spec.profile)
            checks.extend(cell_checks)
            if condition.parameters["arm"] == "primary" and shape is not None:
                shapes[str(condition.parameters["value_head"])].add(shape)
    for head, signatures in sorted(shapes.items()):
        checks.append(
            EngineeringGateCheck(
                name=f"matched_parameter_shapes_{head}",
                passed=len(signatures) == 1,
                detail=f"distinct_signatures={len(signatures)}",
            )
        )
    return EngineeringGateReport(passed=all(check.passed for check in checks), checks=tuple(checks))


def build_controlled_q_report(
    plan: Any,
    result: Any,
    *,
    primary_metric: str = "expected_return",
) -> ControlledQReportResult:
    """Validate, aggregate, and plot a completed immutable study."""

    report_started = time.perf_counter()
    if primary_metric not in PRIMARY_POLICY_METRICS:
        raise ValueError(f"primary_metric must be one of {PRIMARY_POLICY_METRICS}, got {primary_metric!r}")
    root = Path(plan.root)
    gate = validate_controlled_q_study(plan, result)
    gate_path = root / "engineering_gate.json"
    _write_json_atomic(gate_path, gate.to_json_dict())
    if not gate.passed:
        failed = [check.name for check in gate.checks if not check.passed]
        raise EngineeringGateError(f"controlled study engineering gate failed: {', '.join(failed[:8])}")

    tables_dir = root / "tables"
    plots_dir = root / "plots"
    scalar_result = getattr(result, "scalar_learning", None)
    scalar_cells = scalar_result.cells if scalar_result is not None else ()
    all_cells = (*result.primary.cells, *result.raw_input.cells, *scalar_cells)
    selected_cells = [cell for cell in result.primary.cells if bool(cell.output.get("posthoc", {}).get("executed"))]
    tables: list[Path] = []

    def concatenate(name: str, cells: Sequence[StudyCellResult], source_name: str) -> Path:
        path = concatenate_tables((cell.path / source_name for cell in cells), tables_dir / name)
        tables.append(path)
        return path

    policy_table = concatenate("policy_checkpoints.parquet", all_cells, "exact_policy_metrics.parquet")
    role_table = concatenate("role_checkpoints.parquet", all_cells, "exact_role_metrics.parquet")
    opening_table = concatenate("opening_checkpoints.parquet", all_cells, "exact_opening_metrics.parquet")
    concatenate("checkpoint_index.parquet", all_cells, "checkpoint_index.parquet")
    fidelity_table: Path | None = None
    attribution_table: Path | None = None
    coalition_table: Path | None = None
    surrogate_runs_table: Path | None = None
    if selected_cells:
        surrogate_runs_table = concatenate(
            "surrogate_runs.parquet",
            selected_cells,
            "surrogate_runs.parquet",
        )
        concatenate("reference_coverage.parquet", selected_cells, "reference_coverage.parquet")
        concatenate("state_selection.parquet", selected_cells, "state_selection.parquet")
        fidelity_table = concatenate("fidelity.parquet", selected_cells, "fidelity.parquet")
        attribution_table = concatenate(
            "attribution_fidelity.parquet",
            selected_cells,
            "attribution_fidelity.parquet",
        )
        coalition_table = concatenate(
            "coalition_fidelity.parquet",
            selected_cells,
            "coalition_fidelity.parquet",
        )
        if getattr(plan.profile, "posthoc_protocol", "legacy") == "phase-one":
            concatenate("state_buffer_occurrences.parquet", selected_cells, "state_buffer_occurrences.parquet")
            concatenate("state_buffer_counts.parquet", selected_cells, "state_buffer_counts.parquet")

    policy_rows = read_table(policy_table)
    primary_policy_rows = [row for row in policy_rows if row["arm"] == "primary"]
    policy_seed = aggregate_within_seed(
        primary_policy_rows,
        group_fields=("condition_key", "arm", "value_head", "masked_batch_ratio", "episode"),
        metric_fields=(*_POLICY_METRICS, *_ACCOUNTING_METRICS),
    )
    policy_seed_path = write_table(tables_dir / "policy_seed.parquet", policy_seed)
    tables.append(policy_seed_path)
    all_policy_seed = aggregate_within_seed(
        policy_rows,
        group_fields=("condition_key", "arm", "value_head", "masked_batch_ratio", "episode"),
        metric_fields=(*_POLICY_METRICS, *_ACCOUNTING_METRICS),
    )
    phase_one = getattr(plan.profile, "posthoc_protocol", "legacy") == "phase-one"
    declared_primary_heads = getattr(plan.profile, "primary_value_heads", ())
    phase_one_primary_heads = {str(head) for head in declared_primary_heads}
    if not phase_one_primary_heads:
        phase_one_primary_heads = {str(row["value_head"]) for row in all_policy_seed if str(row["arm"]) == "primary"}
    performance_curve_path: Path | None = None
    if phase_one:
        performance_curve = [
            {**row, "performance_method": _phase_one_performance_method(row)}
            for row in all_policy_seed
            if str(row["value_head"]) in phase_one_primary_heads
            and str(row["arm"]) in {"primary", "raw-input-diagnostic"}
        ]
        performance_curve_path = write_table(
            tables_dir / "performance_checkpoints_seed.parquet",
            performance_curve,
        )
        tables.append(performance_curve_path)

    matched_policy: list[dict[str, Any]] = []
    for budget_field in ("episode", "full_training_example_count", "training_flops"):
        matched_policy.extend(
            match_curves_at_common_budget(
                policy_seed,
                budget_field=budget_field,
                treatment_field="masked_batch_ratio",
                strata_fields=("value_head",),
                metric_fields=_POLICY_METRICS,
            )
        )
    matched_policy_path = write_table(tables_dir / "matched_policy_seed.parquet", matched_policy)
    tables.append(matched_policy_path)
    # A within-root paired effect is measured against the ratio-zero control. A
    # profile that declares no such ratio -- the distilled arm, where r = 0 would
    # train no masked rows and so reproduce the baseline objective exactly -- has
    # no control to pair against, and its comparison is a cross-root one. The
    # seed-level tables are written either way; only the paired effect is absent.
    has_untrained_control = 0.0 in plan.profile.mask_ratios
    if has_untrained_control:
        policy_effects = paired_treatment_effects(
            matched_policy,
            treatment_field="masked_batch_ratio",
            baseline=0.0,
            strata_fields=("value_head", "matching_budget"),
            metric_fields=PRIMARY_POLICY_METRICS,
            bootstrap_resamples=int(plan.profile.bootstrap_resamples),
        )
        tables.append(write_table(tables_dir / "policy_paired_effects.parquet", policy_effects))

    role_seed = aggregate_within_seed(
        [row for row in read_table(role_table) if row["arm"] == "primary"],
        group_fields=(
            "condition_key",
            "arm",
            "value_head",
            "masked_batch_ratio",
            "episode",
            "learner_role",
        ),
        metric_fields=_POLICY_METRICS,
    )
    role_seed_path = write_table(tables_dir / "role_seed.parquet", role_seed)
    tables.append(role_seed_path)
    opening_seed = aggregate_within_seed(
        [row for row in read_table(opening_table) if row["arm"] == "primary"],
        group_fields=(
            "condition_key",
            "arm",
            "value_head",
            "masked_batch_ratio",
            "episode",
            "learner_role",
            "opening_class",
        ),
        metric_fields=_POLICY_METRICS[:8],
    )
    opening_seed_path = write_table(tables_dir / "opening_seed.parquet", opening_seed)
    tables.append(opening_seed_path)

    fidelity_seed: tuple[dict[str, Any], ...] = ()
    tradeoff_path: Path | None = None
    shapley_primary_path: Path | None = None
    location_plot_paths: dict[str, Path] = {}
    error_components_path: Path | None = None
    shapley_calibration_path: Path | None = None
    state_return_calibration_path: Path | None = None
    diagnostic_plot_paths: dict[str, Path] = {}
    if fidelity_table is not None:
        raw_fidelity = read_table(fidelity_table)
        raw_fidelity = tuple(_normalize_fidelity_row(row) for row in raw_fidelity)
        fidelity_groups = (
            "condition_key",
            "arm",
            "value_head",
            "masked_batch_ratio",
            "evaluation_reference",
            "model_method",
            "model_reference",
            "comparison_kind",
            "fit_key",
            "training_law",
            "buffer_occurrences_per_role",
            "surrogate_hidden_sizes",
            "surrogate_updates",
            "target_representation",
            "diagnostic",
        )
        all_legal = [
            {**row, "action_subset": "all_legal"}
            for row in _aggregate_fidelity_within_seed(
                raw_fidelity,
                group_fields=fidelity_groups,
                metric_fields=_FIDELITY_METRICS,
                action_subset="all_legal",
            )
        ]
        greedy = [
            {**row, "action_subset": "greedy"}
            for row in _aggregate_fidelity_within_seed(
                raw_fidelity,
                group_fields=fidelity_groups,
                metric_fields=_FIDELITY_METRICS,
                action_subset="greedy",
            )
        ]
        fidelity_seed = tuple((*all_legal, *greedy))
        fidelity_seed_path = write_table(tables_dir / "fidelity_seed.parquet", fidelity_seed)
        tables.append(fidelity_seed_path)
        transfer = [row for row in fidelity_seed if row["comparison_kind"] == "reference_transfer"]
        if transfer:
            tables.append(write_table(tables_dir / "reference_transfer_seed.parquet", transfer))
        if has_untrained_control:
            paired_fidelity_rows = fidelity_seed
            if getattr(plan.profile, "posthoc_protocol", "legacy") == "phase-one":
                maximum_buffer = max(plan.profile.policy_buffer_sizes)
                paired_fidelity_rows = tuple(
                    row
                    for row in fidelity_seed
                    if row["model_method"] == "direct"
                    or (
                        row["training_law"] == "policy"
                        and int(row["buffer_occurrences_per_role"]) == maximum_buffer
                        and row["surrogate_hidden_sizes"] == "64x64"
                        and row["target_representation"] == "native"
                    )
                )
            fidelity_effects = paired_treatment_effects(
                paired_fidelity_rows,
                treatment_field="masked_batch_ratio",
                baseline=0.0,
                strata_fields=(
                    "value_head",
                    "evaluation_reference",
                    "model_method",
                    "model_reference",
                    "comparison_kind",
                    "fit_key",
                    "training_law",
                    "buffer_occurrences_per_role",
                    "surrogate_hidden_sizes",
                    "surrogate_updates",
                    "target_representation",
                    "diagnostic",
                    "action_subset",
                ),
                metric_fields=_FIDELITY_METRICS,
                bootstrap_resamples=int(plan.profile.bootstrap_resamples),
            )
            tables.append(write_table(tables_dir / "fidelity_paired_effects.parquet", fidelity_effects))

        if phase_one:
            maximum_buffer = max(plan.profile.policy_buffer_sizes)
            primary_shapley = []
            for row in fidelity_seed:
                if (
                    row["evaluation_reference"] != "policy"
                    or row["action_subset"] != "greedy"
                    or not math.isclose(float(row["masked_batch_ratio"]), 1.0)
                ):
                    continue
                method = _phase_one_primary_method(row, maximum_buffer=maximum_buffer)
                if method is not None:
                    primary_shapley.append({**row, "comparison_method": method})
            shapley_primary_path = write_table(
                tables_dir / "shapley_primary_seed.parquet",
                primary_shapley,
            )
            tables.append(shapley_primary_path)
            shapley_effects = paired_treatment_effects(
                primary_shapley,
                treatment_field="comparison_method",
                baseline="post-hoc",
                strata_fields=("value_head", "masked_batch_ratio", "action_subset"),
                metric_fields=("attribution_mae",),
                bootstrap_resamples=int(plan.profile.bootstrap_resamples),
            )
            tables.append(write_table(tables_dir / "shapley_primary_paired_effects.parquet", shapley_effects))
            final_ratio_one = {
                int(row["seed"]): row
                for row in _final_rows(policy_seed)
                if math.isclose(float(row["masked_batch_ratio"]), 1.0)
            }
            win_fidelity_rows = [
                {
                    "comparison_method": row["comparison_method"],
                    "seed": row["seed"],
                    "win_probability": final_ratio_one[int(row["seed"])]["win_probability"],
                    "attribution_mae": row["attribution_mae"],
                }
                for row in primary_shapley
            ]
            win_fidelity_path = write_table(
                tables_dir / "win_fidelity_seed.parquet",
                win_fidelity_rows,
            )
            tables.append(win_fidelity_path)
            correlation_rows = []
            for method in sorted({str(row["comparison_method"]) for row in win_fidelity_rows}):
                method_rows = [row for row in win_fidelity_rows if row["comparison_method"] == method]
                wins = np.asarray([float(row["win_probability"]) for row in method_rows])
                errors = np.asarray([float(row["attribution_mae"]) for row in method_rows])
                correlation = (
                    float(np.corrcoef(wins, errors)[0, 1])
                    if len(method_rows) >= 2 and np.std(wins) > 0 and np.std(errors) > 0
                    else math.nan
                )
                correlation_rows.append(
                    {
                        "comparison_method": method,
                        "n_policy_seeds": len(method_rows),
                        "pearson_win_fidelity": correlation,
                    }
                )
            tables.append(
                write_table(
                    tables_dir / "win_fidelity_correlation.parquet",
                    correlation_rows,
                )
            )
            diagnostic_plot_paths["win_fidelity"] = win_fidelity_path

            ratio_one_greedy = [
                row
                for row in fidelity_seed
                if row["action_subset"] == "greedy"
                and math.isclose(float(row["masked_batch_ratio"]), 1.0)
                and row["model_method"] == "surrogate"
            ]
            buffer_rows = [
                row
                for row in ratio_one_greedy
                if row["evaluation_reference"] == "policy"
                and row["training_law"] == "policy"
                and row["surrogate_hidden_sizes"] == "64x64"
                and row["target_representation"] == "native"
            ]
            capacity_rows = [
                row
                for row in ratio_one_greedy
                if row["evaluation_reference"] == "policy"
                and row["training_law"] == "policy"
                and int(row["buffer_occurrences_per_role"]) == maximum_buffer
                and row["target_representation"] == "native"
            ]
            target_rows = [
                row
                for row in ratio_one_greedy
                if row["evaluation_reference"] == "policy"
                and row["training_law"] == "policy"
                and int(row["buffer_occurrences_per_role"]) == maximum_buffer
                and row["surrogate_hidden_sizes"] == "64x64"
            ]
            law_rows = [
                row
                for row in ratio_one_greedy
                if int(row["buffer_occurrences_per_role"]) == maximum_buffer
                and row["surrogate_hidden_sizes"] == "64x64"
                and row["target_representation"] == "native"
            ]
            direct_dose_rows = [
                row
                for row in fidelity_seed
                if row["evaluation_reference"] == "policy"
                and row["action_subset"] == "greedy"
                and row["model_method"] == "direct"
            ]
            for name, rows in (
                ("buffer", buffer_rows),
                ("capacity", capacity_rows),
                ("target", target_rows),
                ("law_transfer", law_rows),
                ("direct_dose", direct_dose_rows),
            ):
                path = write_table(tables_dir / f"{name}_diagnostic_seed.parquet", rows)
                tables.append(path)
                diagnostic_plot_paths[name] = path
            if surrogate_runs_table is None:
                raise StudyAnalysisError("Phase 1 has no surrogate accounting table")
            surrogate_accounting = {
                (str(row["condition_key"]), int(row["seed"]), str(row["fit_key"])): row
                for row in read_table(surrogate_runs_table)
            }
            compute_rows = []
            for row in ratio_one_greedy:
                if row["evaluation_reference"] != "policy":
                    continue
                accounting = surrogate_accounting[(str(row["condition_key"]), int(row["seed"]), str(row["fit_key"]))]
                compute_rows.append(
                    {
                        **row,
                        "surrogate_training_flops": accounting["training_flops"],
                        "surrogate_duration_seconds": accounting["duration_seconds"],
                        "surrogate_training_points": accounting["training_points"],
                    }
                )
            compute_diagnostic_path = write_table(
                tables_dir / "surrogate_compute_diagnostic_seed.parquet",
                compute_rows,
            )
            tables.append(compute_diagnostic_path)
            diagnostic_plot_paths["surrogate_compute"] = compute_diagnostic_path

            loss_rows = []
            for cell in selected_cells:
                for summary in read_table(cell.path / "surrogate_runs.parquet"):
                    fit_key = str(summary["fit_key"])
                    if (
                        str(summary["training_law"]) != "policy"
                        or str(summary["surrogate_hidden_sizes"]) != "64x64"
                        or str(summary["target_representation"]) != "native"
                        or int(summary["buffer_occurrences_per_role"]) != maximum_buffer
                    ):
                        continue
                    for metric in read_table(cell.path / "surrogates" / fit_key / "surrogate_metrics.parquet"):
                        loss_rows.append(
                            {
                                "condition_key": cell.condition_key,
                                "seed": cell.seed,
                                "masked_batch_ratio": summary["masked_batch_ratio"],
                                "fit_key": fit_key,
                                "update": metric["update"],
                                "loss": metric["loss"],
                            }
                        )
            loss_path = write_table(
                tables_dir / "surrogate_loss_checkpoints.parquet",
                loss_rows,
            )
            tables.append(loss_path)
            diagnostic_plot_paths["surrogate_loss"] = loss_path
        final_policy = _final_rows(policy_seed)
        policy_by_key = {(row["value_head"], row["masked_batch_ratio"], row["seed"]): row for row in final_policy}
        tradeoff = []
        for row in fidelity_seed:
            policy = policy_by_key[(row["value_head"], row["masked_batch_ratio"], row["seed"])]
            tradeoff.append(
                {
                    **row,
                    "expected_return": policy["expected_return"],
                    "exploitability": policy["exploitability"],
                }
            )
        tradeoff_path = write_table(tables_dir / "policy_fidelity_tradeoff_seed.parquet", tradeoff)
        tables.append(tradeoff_path)

    if (
        getattr(plan.profile, "posthoc_protocol", "legacy") == "phase-one"
        and coalition_table is not None
        and attribution_table is not None
    ):
        maximum_buffer = max(plan.profile.policy_buffer_sizes)
        location_model_fields = (
            "condition_key",
            "arm",
            "value_head",
            "masked_batch_ratio",
            "evaluation_reference",
            "model_method",
            "model_reference",
            "comparison_kind",
            "fit_key",
            "training_law",
            "buffer_occurrences_per_role",
            "surrogate_hidden_sizes",
            "surrogate_updates",
            "target_representation",
            "diagnostic",
        )
        location_metrics = (
            "coalition_error",
            "absolute_coalition_error",
            "base_q_coalition_error",
            "absolute_base_q_coalition_error",
            "total_return_coalition_error",
            "absolute_total_return_coalition_error",
        )
        for dimension, name in (("coalition_size", "coalition_size"), ("move_count", "state_depth")):
            state_location = reduce_long_table(
                coalition_table,
                group_fields=(
                    *location_model_fields,
                    "seed",
                    "learner_role",
                    "state_id",
                    "sample_weight",
                    dimension,
                ),
                metric_fields=location_metrics,
                filters={"evaluation_reference": "policy", "greedy_action": True},
            )
            if dimension == "move_count":
                # X and O learner decisions have different move-count parity.
                # Retain role-specific depth curves instead of fabricating an
                # equal-role average at depths where one role cannot occur.
                location = _aggregate_location_within_role(
                    state_location,
                    group_fields=(*location_model_fields, dimension),
                    metric_fields=location_metrics,
                )
            else:
                location = _aggregate_fidelity_within_seed(
                    state_location,
                    group_fields=(*location_model_fields, dimension),
                    metric_fields=location_metrics,
                    action_subset="all_legal",
                )
            tables.append(write_table(tables_dir / f"{name}_seed.parquet", location))
            primary_location = []
            for row in location:
                method = _phase_one_primary_method(row, maximum_buffer=maximum_buffer)
                if method is None or not math.isclose(float(row["masked_batch_ratio"]), 1.0):
                    continue
                primary_location.append({**row, "comparison_method": method})
            primary_path = write_table(tables_dir / f"{name}_primary_seed.parquet", primary_location)
            tables.append(primary_path)
            location_plot_paths[name] = primary_path

        attribution_state = reduce_long_table(
            attribution_table,
            group_fields=(
                *location_model_fields,
                "seed",
                "learner_role",
                "state_id",
                "sample_weight",
            ),
            metric_fields=(
                "attribution_error",
                "absolute_attribution_error",
                "base_q_attribution_error",
                "absolute_base_q_attribution_error",
                "total_return_attribution_error",
                "absolute_total_return_attribution_error",
            ),
            filters={
                "evaluation_reference": "policy",
                "greedy_action": True,
                "masked_batch_ratio": 1.0,
            },
        )
        attribution_components = _aggregate_fidelity_within_seed(
            attribution_state,
            group_fields=location_model_fields,
            metric_fields=(
                "attribution_error",
                "absolute_attribution_error",
                "base_q_attribution_error",
                "absolute_base_q_attribution_error",
                "total_return_attribution_error",
                "absolute_total_return_attribution_error",
            ),
            action_subset="all_legal",
        )
        primary_components = []
        for row in attribution_components:
            method = _phase_one_primary_method(row, maximum_buffer=maximum_buffer)
            if method is None:
                continue
            if not np.isclose(
                float(row["total_return_attribution_error"]),
                float(row["base_q_attribution_error"]) + float(row["attribution_error"]),
                atol=1e-10,
            ):
                raise StudyAnalysisError("weighted return-Q attribution decomposition does not close")
            primary_components.append({**row, "comparison_method": method})
        tables.append(
            write_table(
                tables_dir / "attribution_error_components_seed.parquet",
                primary_components,
            )
        )
        component_rows = []
        for row in primary_components:
            if row["comparison_method"] == "Direct-Q":
                component_rows.append(
                    {
                        "error_component": "base-Q",
                        "seed": row["seed"],
                        "signed_error": row["base_q_attribution_error"],
                        "absolute_error": row["absolute_base_q_attribution_error"],
                    }
                )
            component_rows.append(
                {
                    "error_component": f"{row['comparison_method']} explanation",
                    "seed": row["seed"],
                    "signed_error": row["attribution_error"],
                    "absolute_error": row["absolute_attribution_error"],
                }
            )
        error_components_path = write_table(
            tables_dir / "error_modes_seed.parquet",
            component_rows,
        )
        tables.append(error_components_path)

        calibration_reduced = reduce_long_table(
            attribution_table,
            group_fields=(
                *location_model_fields,
                "seed",
                "learner_role",
                "state_id",
                "move_count",
                "sample_weight",
                "action",
                "feature_index",
            ),
            metric_fields=(
                "exact_attribution",
                "approximate_attribution",
                "exact_return_attribution",
            ),
            filters={
                "evaluation_reference": "policy",
                "greedy_action": True,
                "masked_batch_ratio": 1.0,
                "diagnostic": False,
            },
        )
        calibration_rows = []
        for row in calibration_reduced:
            method = _phase_one_primary_method(row, maximum_buffer=maximum_buffer)
            if method is not None:
                calibration_rows.append({**row, "comparison_method": method})
        shapley_calibration_path = write_table(
            tables_dir / "shapley_calibration.parquet",
            calibration_rows,
        )
        tables.append(shapley_calibration_path)

        state_return_reduced = reduce_long_table(
            coalition_table,
            group_fields=(
                "condition_key",
                "masked_batch_ratio",
                "seed",
                "learner_role",
                "state_id",
                "move_count",
                "sample_weight",
                "action",
            ),
            metric_fields=("exact_value", "exact_return_value"),
            filters={
                "evaluation_reference": "policy",
                "model_method": "direct",
                "greedy_action": True,
                "coalition_size": 9,
                "diagnostic": False,
            },
        )
        state_return_calibration_path = write_table(
            tables_dir / "state_return_calibration.parquet",
            (
                {
                    **row,
                    "model_q_value": row["exact_value"],
                }
                for row in state_return_reduced
            ),
        )
        tables.append(state_return_calibration_path)

    plots: list[Path] = []

    def keep_plot(path: Path | None) -> None:
        if path is not None:
            plots.append(path)

    if phase_one:
        if performance_curve_path is None:
            raise StudyAnalysisError("Phase 1 has no performance checkpoint table")
        keep_plot(
            plot_metric_curves(
                performance_curve_path,
                plots_dir / "performance_learning.png",
                x_field="episode",
                y_field="expected_return",
                series_fields=("performance_method",),
                title="Exact expected return over training",
                show_seed_lines=True,
                simultaneous_band=True,
                bootstrap_resamples=int(plan.profile.bootstrap_resamples),
            )
        )
        if shapley_calibration_path is None or state_return_calibration_path is None:
            raise StudyAnalysisError("Phase 1 calibration tables are missing")
        keep_plot(
            plot_calibration_residuals(
                shapley_calibration_path,
                plots_dir / "shapley_model_q_calibration.png",
                exact_field="exact_attribution",
                approximate_field="approximate_attribution",
                facet_field="comparison_method",
                color_field="learner_role",
                weight_field="sample_weight",
                title="Fixed-action learned-Q SVERL calibration (r=1)",
                exact_label="Exact learned-Q Shapley value",
                approximate_label="Approximate Shapley value",
            )
        )
        keep_plot(
            plot_calibration_residuals(
                shapley_calibration_path,
                plots_dir / "shapley_return_calibration.png",
                exact_field="exact_return_attribution",
                approximate_field="approximate_attribution",
                facet_field="comparison_method",
                color_field="learner_role",
                weight_field="sample_weight",
                title="Fixed-action policy-return SVERL calibration (r=1)",
                exact_label="Exact policy-return Shapley value",
                approximate_label="Approximate Shapley value",
            )
        )
        keep_plot(
            plot_calibration_residuals(
                state_return_calibration_path,
                plots_dir / "state_return_calibration.png",
                exact_field="exact_return_value",
                approximate_field="model_q_value",
                facet_field="masked_batch_ratio",
                color_field="learner_role",
                weight_field="sample_weight",
                title="Greedy fixed-action continuation-return calibration",
                exact_label="Exact continuation return",
                approximate_label="Categorical model-Q estimate",
            )
        )
        keep_plot(
            plot_policy_fidelity_tradeoff(
                diagnostic_plot_paths["win_fidelity"],
                plots_dir / "diagnostic_win_fidelity.png",
                policy_field="win_probability",
                fidelity_field="attribution_mae",
                series_fields=("comparison_method",),
                title="Exploratory: final win probability vs Shapley MAE",
                show_seed_points=True,
            )
        )
        keep_plot(
            plot_policy_fidelity_tradeoff(
                diagnostic_plot_paths["surrogate_compute"],
                plots_dir / "diagnostic_surrogate_compute.png",
                policy_field="surrogate_training_flops",
                fidelity_field="attribution_mae",
                series_fields=("fit_key",),
                title="Diagnostic: post-hoc compute vs final Shapley MAE",
                show_seed_points=True,
            )
        )
        keep_plot(
            plot_metric_curves(
                diagnostic_plot_paths["surrogate_loss"],
                plots_dir / "diagnostic_surrogate_loss.png",
                x_field="update",
                y_field="loss",
                series_fields=("masked_batch_ratio",),
                x_scale="log",
                title="Diagnostic: matched post-hoc training loss",
            )
        )
    else:
        keep_plot(
            plot_metric_curves(
                policy_seed_path,
                plots_dir / "learning.png",
                x_field="episode",
                y_field=primary_metric,
                series_fields=("value_head", "masked_batch_ratio"),
                title=f"Policy learning: {primary_metric}",
                facet_field="value_head",
            )
        )
    compute_rows = [row for row in policy_seed if float(row["training_flops"]) > 0]
    compute_path = write_table(tables_dir / "policy_compute_seed.parquet", compute_rows)
    tables.append(compute_path)
    keep_plot(
        plot_metric_curves(
            compute_path,
            plots_dir / "compute.png",
            x_field="training_flops",
            y_field=primary_metric,
            series_fields=("value_head", "masked_batch_ratio"),
            title=f"Policy quality vs compute: {primary_metric}",
            x_scale="log",
            facet_field="value_head",
        )
    )
    final_roles = [row for row in _final_rows(role_seed) if row["learner_role"] != "balanced"]
    final_roles_path = write_table(tables_dir / "role_final_seed.parquet", final_roles)
    tables.append(final_roles_path)
    keep_plot(
        plot_categorical_metric(
            final_roles_path,
            plots_dir / "roles.png",
            category_field="learner_role",
            y_field=primary_metric,
            series_fields=("value_head", "masked_batch_ratio"),
            facet_field="value_head",
        )
    )
    final_openings_path = write_table(tables_dir / "opening_final_seed.parquet", _final_rows(opening_seed))
    tables.append(final_openings_path)
    keep_plot(
        plot_categorical_metric(
            final_openings_path,
            plots_dir / "openings.png",
            category_field="opening_class",
            y_field=primary_metric,
            series_fields=("value_head", "masked_batch_ratio", "learner_role"),
            facet_field="value_head",
        )
    )
    if fidelity_seed:
        keep_plot(
            plot_metric_curves(
                tables_dir / "fidelity_seed.parquet",
                plots_dir / "fidelity.png",
                x_field="masked_batch_ratio",
                y_field="attribution_mae",
                series_fields=(
                    "value_head",
                    "evaluation_reference",
                    "model_method",
                    "model_reference",
                    "action_subset",
                ),
                facet_field="value_head",
            )
        )
    for name, path in location_plot_paths.items():
        dimension = "coalition_size" if name == "coalition_size" else "move_count"
        series_fields = ("comparison_method", "learner_role") if name == "state_depth" else ("comparison_method",)
        for metric, suffix in (
            ("coalition_error", "signed"),
            ("absolute_coalition_error", "absolute"),
        ):
            keep_plot(
                plot_metric_curves(
                    path,
                    plots_dir / f"{name}_{suffix}.png",
                    x_field=dimension,
                    y_field=metric,
                    series_fields=series_fields,
                    title=f"{name.replace('_', ' ').title()}: {suffix} characteristic error",
                    x_label=("occupied cells before learner decision" if name == "state_depth" else None),
                )
            )
    if error_components_path is not None:
        keep_plot(
            plot_categorical_metric(
                error_components_path,
                plots_dir / "error_modes.png",
                category_field="error_component",
                y_field="absolute_error",
                series_fields=(),
                title="Base-Q and explanation error modes",
            )
        )
    if diagnostic_plot_paths:
        keep_plot(
            plot_metric_curves(
                diagnostic_plot_paths["buffer"],
                plots_dir / "diagnostic_buffer_size.png",
                x_field="buffer_occurrences_per_role",
                y_field="attribution_mae",
                series_fields=(),
                x_scale="log",
                title="Diagnostic: policy-buffer size",
            )
        )
        keep_plot(
            plot_categorical_metric(
                diagnostic_plot_paths["capacity"],
                plots_dir / "diagnostic_capacity.png",
                category_field="surrogate_hidden_sizes",
                y_field="attribution_mae",
                series_fields=(),
                category_order=("64x64", "128x128", "256x256"),
                title="Diagnostic: surrogate capacity",
            )
        )
        keep_plot(
            plot_categorical_metric(
                diagnostic_plot_paths["target"],
                plots_dir / "diagnostic_scalar_baseline.png",
                category_field="target_representation",
                y_field="attribution_mae",
                series_fields=(),
                title="Diagnostic: categorical target versus scalar-Q baseline",
            )
        )
        keep_plot(
            plot_categorical_metric(
                diagnostic_plot_paths["law_transfer"],
                plots_dir / "diagnostic_law_transfer.png",
                category_field="evaluation_reference",
                y_field="attribution_mae",
                series_fields=("training_law",),
                title="Diagnostic only: policy/replay/uniform law transfer",
            )
        )
        keep_plot(
            plot_categorical_metric(
                diagnostic_plot_paths["direct_dose"],
                plots_dir / "diagnostic_direct_q_dose.png",
                category_field="masked_batch_ratio",
                y_field="attribution_mae",
                series_fields=(),
                title="Diagnostic: Direct-Q explanation error by masked ratio",
            )
        )
    if tradeoff_path is not None:
        keep_plot(
            plot_policy_fidelity_tradeoff(
                tradeoff_path,
                plots_dir / "policy_fidelity_tradeoff.png",
                policy_field=primary_metric,
                fidelity_field="attribution_mae",
                series_fields=(
                    "value_head",
                    "masked_batch_ratio",
                    "evaluation_reference",
                    "model_method",
                    "model_reference",
                    "action_subset",
                ),
                facet_field="value_head",
            )
        )

    adapter_checkpoint_rows = []
    for row in all_policy_seed:
        adapter = _phase_one_adapter(row)
        if adapter is not None:
            adapter_checkpoint_rows.append(
                {
                    **row,
                    "adapter": adapter,
                }
            )
    adapter_checkpoint_path = write_table(
        tables_dir / "adapter_checkpoints_seed.parquet",
        adapter_checkpoint_rows,
    )
    tables.append(adapter_checkpoint_path)
    adapter_rows = _final_rows(adapter_checkpoint_rows)
    adapter_path = write_table(tables_dir / "adapter_diagnostic_seed.parquet", adapter_rows)
    tables.append(adapter_path)
    keep_plot(
        plot_categorical_metric(
            adapter_path,
            plots_dir / "adapter_diagnostic.png",
            category_field="value_head",
            y_field=primary_metric,
            series_fields=("adapter",),
            title="Diagnostic: final raw-input versus mask-input adapter",
        )
    )
    if phase_one:
        keep_plot(
            plot_metric_curves(
                adapter_checkpoint_path,
                plots_dir / "diagnostic_adapter_learning.png",
                x_field="episode",
                y_field="expected_return",
                series_fields=("value_head", "adapter"),
                facet_field="value_head",
                title="Diagnostic: adapter effect over categorical and scalar-Q learning",
                show_seed_lines=True,
                simultaneous_band=True,
                bootstrap_resamples=int(plan.profile.bootstrap_resamples),
            )
        )

    if phase_one:
        final_performance = []
        for row in _final_rows(all_policy_seed):
            if str(row["value_head"]) not in phase_one_primary_heads or str(row["arm"]) not in {
                "primary",
                "raw-input-diagnostic",
            }:
                continue
            final_performance.append(
                {
                    **row,
                    "performance_method": _phase_one_performance_method(row),
                }
            )
        performance_path = write_table(tables_dir / "performance_primary_seed.parquet", final_performance)
        tables.append(performance_path)
        performance_effects = paired_treatment_effects(
            final_performance,
            treatment_field="performance_method",
            baseline="raw-input",
            strata_fields=("value_head",),
            metric_fields=("expected_return", "win_probability", "draw_probability", "loss_probability"),
            bootstrap_resamples=int(plan.profile.bootstrap_resamples),
        )
        tables.append(write_table(tables_dir / "performance_primary_paired_effects.parquet", performance_effects))
        outcome_rows = [
            {
                "performance_method": row["performance_method"],
                "seed": row["seed"],
                "outcome": outcome,
                "probability": row[field],
            }
            for row in final_performance
            for outcome, field in (
                ("win", "win_probability"),
                ("draw", "draw_probability"),
                ("loss", "loss_probability"),
            )
        ]
        outcome_path = write_table(tables_dir / "performance_outcomes_seed.parquet", outcome_rows)
        tables.append(outcome_path)
        keep_plot(
            plot_categorical_metric(
                outcome_path,
                plots_dir / "performance_final_outcomes.png",
                category_field="outcome",
                y_field="probability",
                series_fields=("performance_method",),
                title="Final exact win/draw/loss probabilities",
            )
        )
        keep_plot(
            plot_paired_seed_slopes(
                performance_path,
                plots_dir / "performance_final_paired.png",
                category_field="performance_method",
                y_field="expected_return",
                title="Paired final exact expected return",
            )
        )

        report_specs = [plan.primary, plan.raw_input]
        if getattr(plan, "scalar_learning", None) is not None:
            report_specs.append(plan.scalar_learning)
        condition_parameters = {
            condition.key: condition.parameters for spec in report_specs for condition in spec.conditions
        }
        runtime_rows = []
        for cell in all_cells:
            timing = cell.output.get("timing")
            if not isinstance(timing, Mapping):
                raise EngineeringGateError(
                    f"Phase 1 cell {cell.condition_key}/seed={cell.seed} has no timing breakdown"
                )
            parameters = condition_parameters[cell.condition_key]
            runtime_rows.append(
                {
                    "condition_key": cell.condition_key,
                    "arm": parameters["arm"],
                    "masked_batch_ratio": parameters["masked_batch_ratio"],
                    "seed": cell.seed,
                    **{key: float(value) for key, value in timing.items()},
                    "artifact_bytes": _directory_size(cell.path),
                }
            )
        runtime_path = write_table(tables_dir / "runtime_cells.parquet", runtime_rows)
        tables.append(runtime_path)
        if shapley_primary_path is None:
            raise StudyAnalysisError("Phase 1 has no primary exact-Shapley table")
        keep_plot(
            plot_categorical_metric(
                shapley_primary_path,
                plots_dir / "shapley_primary.png",
                category_field="comparison_method",
                y_field="attribution_mae",
                series_fields=(),
                title="Marginal exact-model-Q Shapley MAE (95% seed CI)",
            )
        )
        if shapley_calibration_path is None:
            raise StudyAnalysisError("Phase 1 has no exact-vs-approximate Shapley table")
        for method, slug, label in (
            ("Direct-Q", "direct_q", "Direct-Q"),
            ("post-hoc", "post_hoc", "Post-hoc"),
        ):
            keep_plot(
                plot_correlation_scatter(
                    shapley_calibration_path,
                    plots_dir / f"shapley_correlation_{slug}.png",
                    exact_field="exact_attribution",
                    approximate_field="approximate_attribution",
                    selection_field="comparison_method",
                    selection_value=method,
                    color_field="learner_role",
                    weight_field="sample_weight",
                    title=f"{label}: approximate vs exact learned-Q Shapley values",
                    exact_label="Exact learned-Q Shapley value",
                    approximate_label="Approximate Shapley value",
                )
            )
        keep_plot(
            plot_paired_seed_slopes(
                shapley_primary_path,
                plots_dir / "shapley_primary_paired.png",
                category_field="comparison_method",
                y_field="attribution_mae",
                title="Within-seed exact-model-Q Shapley MAE",
            )
        )
        report_seconds = time.perf_counter() - report_started
        runtime_gate = _phase_one_runtime_projection(
            runtime_rows,
            observed_bytes=_directory_size(root),
            report_seconds=report_seconds,
            target_seeds=10,
        )
        _write_json_atomic(root / "runtime_gate.json", runtime_gate)

    report = ControlledQReportResult(
        primary_metric=primary_metric,
        tables=tuple(tables),
        plots=tuple(plots),
        gate_path=gate_path,
    )
    _write_json_atomic(root / "report_summary.json", report.to_json_dict(root=root))
    return report


def _final_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    maxima: dict[tuple[Any, ...], float] = {}
    identity_fields = ("condition_key", "seed")
    for row in rows:
        identity = tuple(row[field] for field in identity_fields)
        maxima[identity] = max(maxima.get(identity, -math.inf), float(row["episode"]))
    return [
        dict(row) for row in rows if float(row["episode"]) == maxima[tuple(row[field] for field in identity_fields)]
    ]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _normalize_fidelity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    model_method = str(row["model_method"])
    model_reference = str(row["model_reference"])
    normalized.setdefault("fit_key", "direct" if model_method == "direct" else model_reference)
    normalized.setdefault("training_law", "replay" if model_method == "direct" else model_reference)
    normalized.setdefault("buffer_occurrences_per_role", 0)
    normalized.setdefault("surrogate_hidden_sizes", "direct" if model_method == "direct" else "legacy")
    normalized.setdefault("surrogate_updates", 0)
    normalized.setdefault("target_representation", "native")
    normalized.setdefault("diagnostic", False)
    normalized.setdefault("return_q_attribution_mae", math.nan)
    normalized.setdefault("base_q_attribution_mae", math.nan)
    normalized.setdefault("return_q_coalition_mae", math.nan)
    normalized.setdefault("base_q_coalition_mae", math.nan)
    return normalized


def _phase_one_primary_method(
    row: Mapping[str, Any],
    *,
    maximum_buffer: int,
) -> str | None:
    if str(row["model_method"]) == "direct":
        return "Direct-Q"
    if (
        str(row["training_law"]) == "policy"
        and int(row["buffer_occurrences_per_role"]) == maximum_buffer
        and str(row["surrogate_hidden_sizes"]) == "64x64"
        and str(row["target_representation"]) == "native"
    ):
        return "post-hoc"
    return None


def _phase_one_adapter(row: Mapping[str, Any]) -> str | None:
    arm = str(row["arm"])
    if arm in {"raw-input-diagnostic", "scalar-learning-raw-input"}:
        return "raw_input"
    if arm in {"primary", "scalar-learning-mask-input", "learning-baseline"} and math.isclose(
        float(row["masked_batch_ratio"]),
        0.0,
    ):
        return "matched_mask_input"
    return None


def _phase_one_performance_method(row: Mapping[str, Any]) -> str:
    if str(row["arm"]) == "raw-input-diagnostic":
        return "raw-input"
    return f"Direct-Q r={float(row['masked_batch_ratio']):g}"


def _phase_one_runtime_projection(
    runtime_rows: Sequence[Mapping[str, Any]],
    *,
    observed_bytes: int,
    report_seconds: float,
    target_seeds: int,
) -> dict[str, Any]:
    """Project a ten-seed sequential run and enforce the preregistered gate."""

    observed_seeds = {int(row["seed"]) for row in runtime_rows}
    if not runtime_rows or not observed_seeds:
        raise StudyAnalysisError("Phase 1 runtime projection requires measured cell rows")
    if target_seeds < len(observed_seeds):
        raise StudyAnalysisError("target_seeds cannot be smaller than the observed pilot")
    factor = target_seeds / len(observed_seeds)
    cell_seconds = sum(float(row["cell_total_seconds"]) for row in runtime_rows)
    projected_seconds = factor * (cell_seconds + float(report_seconds))
    projected_bytes = int(math.ceil(factor * int(observed_bytes)))
    cell_sizes = np.asarray([int(row["artifact_bytes"]) for row in runtime_rows], dtype=np.float64)
    median_size = float(np.median(cell_sizes))
    largest_to_median = float(cell_sizes.max() / median_size) if median_size > 0 else math.inf
    checks = {
        "projected_sequential_wall_under_12h": projected_seconds <= 12 * 60 * 60,
        "projected_storage_under_25gb": projected_bytes <= 25 * 1024**3,
        "cell_storage_not_superlinear": largest_to_median <= 2.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_seeds": len(observed_seeds),
        "target_seeds": target_seeds,
        "projection_factor": factor,
        "observed_cell_seconds": cell_seconds,
        "observed_report_seconds": float(report_seconds),
        "projected_sequential_seconds": projected_seconds,
        "observed_bytes": int(observed_bytes),
        "projected_bytes": projected_bytes,
        "largest_to_median_cell_bytes": largest_to_median,
    }


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _aggregate_fidelity_within_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
    action_subset: str,
) -> tuple[dict[str, Any], ...]:
    """Average actions within state, occupancy-weight within role, then mix roles equally."""

    if action_subset not in {"all_legal", "greedy"}:
        raise StudyAnalysisError(f"unknown action_subset {action_subset!r}")
    selected = [row for row in rows if action_subset == "all_legal" or _as_bool(row["greedy_action"])]
    state_fields = (*group_fields, "seed", "learner_role", "state_id")
    state_buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        state_buckets[tuple(row[field] for field in state_fields)].append(row)
    if not state_buckets:
        raise StudyAnalysisError("cannot aggregate an empty fidelity table")

    state_rows: list[dict[str, Any]] = []
    for key, bucket in state_buckets.items():
        weights = {float(row["sample_weight"]) for row in bucket}
        if len(weights) != 1:
            raise StudyAnalysisError(f"state {key[-1]!r} has inconsistent occupancy weights")
        if action_subset == "greedy" and len(bucket) != 1:
            raise StudyAnalysisError(f"state {key[-1]!r} has {len(bucket)} greedy actions; expected one")
        state_rows.append(
            {
                **{field: value for field, value in zip(state_fields, key, strict=True)},
                "sample_weight": weights.pop(),
                **{metric: float(np.mean([float(row[metric]) for row in bucket])) for metric in metric_fields},
            }
        )

    role_fields = (*group_fields, "seed", "learner_role")
    role_buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in state_rows:
        role_buckets[tuple(row[field] for field in role_fields)].append(row)
    role_rows: list[dict[str, Any]] = []
    for key, bucket in role_buckets.items():
        role_row = {field: value for field, value in zip(role_fields, key, strict=True)}
        for metric in metric_fields:
            finite = [
                (float(row["sample_weight"]), float(row[metric])) for row in bucket if math.isfinite(float(row[metric]))
            ]
            total = sum(weight for weight, _value in finite)
            role_row[metric] = sum(weight * value for weight, value in finite) / total if total > 0 else math.nan
        role_row["within_role_states"] = len(bucket)
        role_rows.append(role_row)

    seed_fields = (*group_fields, "seed")
    seed_buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in role_rows:
        seed_buckets[tuple(row[field] for field in seed_fields)].append(row)
    aggregated = []
    for key in sorted(seed_buckets, key=lambda item: tuple(str(value) for value in item)):
        bucket = seed_buckets[key]
        if {str(row["learner_role"]) for row in bucket} != {"X", "O"}:
            raise StudyAnalysisError(f"seed-level fidelity group {key} does not contain both roles")
        aggregated.append(
            {
                **{field: value for field, value in zip(seed_fields, key, strict=True)},
                **{metric: float(np.mean([float(row[metric]) for row in bucket])) for metric in metric_fields},
                "within_seed_rows": sum(int(row["within_role_states"]) for row in bucket),
            }
        )
    return tuple(aggregated)


def _aggregate_location_within_role(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Occupancy-weight states while retaining structurally distinct roles."""

    fields = (*group_fields, "seed", "learner_role")
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[field] for field in fields)].append(row)
    if not buckets:
        raise StudyAnalysisError("cannot aggregate an empty role-specific location table")
    aggregated = []
    for key in sorted(buckets, key=lambda item: tuple(str(value) for value in item)):
        bucket = buckets[key]
        result = {field: value for field, value in zip(fields, key, strict=True)}
        for metric in metric_fields:
            finite = [
                (float(row["sample_weight"]), float(row[metric])) for row in bucket if math.isfinite(float(row[metric]))
            ]
            total = sum(weight for weight, _value in finite)
            result[metric] = sum(weight * value for weight, value in finite) / total if total > 0 else math.nan
        result["within_role_states"] = len(bucket)
        aggregated.append(result)
    return tuple(aggregated)


def _require_files(root: Path, names: Sequence[str]) -> str:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts {missing}")
    return f"{len(names)} files"


def _require_equal_bytes(first: Path, second: Path) -> str:
    if first.read_bytes() != second.read_bytes():
        raise ValueError(f"{first.name} differs from {second.relative_to(first.parent)}")
    return "byte-identical"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


__all__ = [
    "PRIMARY_POLICY_METRICS",
    "ControlledQReportResult",
    "EngineeringGateCheck",
    "EngineeringGateError",
    "EngineeringGateReport",
    "build_controlled_q_report",
    "validate_controlled_q_cell",
    "validate_controlled_q_study",
]
