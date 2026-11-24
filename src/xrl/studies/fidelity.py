"""Dual-reference masked-Q fidelity artifacts for controlled study cells."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from xrl._optional import pq, require_storage_dependencies
from xrl.explainability.core import PolicyView, all_coalitions, shapley_operator
from xrl.explainability.tictactoe import TicTacToeDistributionPoint, TicTacToeSVERLExplainer
from xrl.studies.analysis import (
    PARQUET_COMPRESSION,
    PARQUET_ROW_GROUP_ROWS,
    StudyAnalysisError,
    arrow_table,
    score_fidelity,
    write_table,
)
from xrl.studies.core import StudyCellContext
from xrl.studies.surrogate import (
    MaskedQSurrogateConfig,
    MaskedQSurrogateDataset,
    MaskedQSurrogateTrainer,
    build_masked_q_surrogate_dataset_from_arrays,
)
from xrl.studies.tictactoe import (
    DistributionSample,
    ExactReturnQOracle,
    ReferenceOccurrenceBuffer,
    RoleCoverage,
    build_nested_reference_occurrence_buffers,
    build_policy_reference_distribution,
    build_shared_reference_distribution,
    sample_reference_distribution,
    sample_stratified_reference_distribution,
)
from xrl.tictactoe.core import O, X
from xrl.training._neural.replay import ReplayTransition
from xrl.training.neural_q import NeuralQAgent
from xrl.training.policies import opponent_random_move_probability

_COALITIONS = all_coalitions(9)
_COALITION_MASKS = np.asarray(
    [[int(feature in coalition) for feature in range(9)] for coalition in _COALITIONS],
    dtype=np.int8,
)
# Both the coalition game and the Shapley step are linear in the values, so
# each explanation here is two matrix products rather than two Python loops
# over the 512 coalitions. See ADR 0042.
_SHAPLEY_OPERATOR = shapley_operator(9)
_ROLE_NAME = {X: "X", O: "O"}

# Knobs a study may leave unset. The mask-ratio study predeclares greedy
# occupancy, so its reference law is unchanged by this default.
_SETTING_DEFAULTS: dict[str, Any] = {"policy_reference_softening": 0.0}

_COMMON_FIELDS = (
    "condition_key",
    "arm",
    "value_head",
    "masked_batch_ratio",
    "seed",
    "evaluation_reference",
    "state_sample",
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
    "ood_direct_query",
    "state_id",
    "state_board",
    "observation",
    "learner_role",
    "move_count",
    "sample_weight",
    "action",
    "greedy_action",
)
_FIDELITY_FIELDS = (
    *_COMMON_FIELDS,
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
_ATTRIBUTION_FIELDS = (
    *_COMMON_FIELDS,
    "feature_index",
    "feature_name",
    "exact_attribution",
    "approximate_attribution",
    "attribution_error",
    "absolute_attribution_error",
    "exact_return_attribution",
    "base_q_attribution_error",
    "absolute_base_q_attribution_error",
    "total_return_attribution_error",
    "absolute_total_return_attribution_error",
)
_COALITION_FIELDS = (
    *_COMMON_FIELDS,
    "coalition_index",
    "coalition_size",
    "coalition_features",
    "exact_value",
    "approximate_value",
    "coalition_error",
    "absolute_coalition_error",
    "exact_return_value",
    "base_q_coalition_error",
    "absolute_base_q_coalition_error",
    "total_return_coalition_error",
    "absolute_total_return_coalition_error",
)


class BatchedMaskedQModel(Protocol):
    num_actions: int
    observation_size: int

    def q_values_batch(self, observations: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        """Return expected Q for each observation/mask row."""


class FidelityRowSink(Protocol):
    def write(self, row: Mapping[str, Any]) -> None:
        """Consume one long-format study row."""


@dataclass(frozen=True, slots=True)
class ControlledQPosthocResult:
    executed: bool
    surrogate_runs: int = 0
    selected_states: int = 0
    state_actions: int = 0
    model_comparisons: int = 0
    coalition_detail_comparisons: int = 0
    coalition_rows: int = 0
    attribution_rows: int = 0
    fidelity_rows: int = 0
    law_seconds: float = 0.0
    surrogate_seconds: float = 0.0
    inference_seconds: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "surrogate_runs": self.surrogate_runs,
            "selected_states": self.selected_states,
            "state_actions": self.state_actions,
            "model_comparisons": self.model_comparisons,
            "coalition_detail_comparisons": self.coalition_detail_comparisons,
            "coalition_rows": self.coalition_rows,
            "attribution_rows": self.attribution_rows,
            "fidelity_rows": self.fidelity_rows,
            "law_seconds": self.law_seconds,
            "surrogate_seconds": self.surrogate_seconds,
            "inference_seconds": self.inference_seconds,
        }


@dataclass(frozen=True, slots=True)
class PhaseOneSurrogateFit:
    """One factorised post-hoc fit in the preregistered Phase 1 matrix."""

    key: str
    training_law: str
    buffer_occurrences_per_role: int
    hidden_sizes: tuple[int, int]
    target_representation: str = "native"
    diagnostic: bool = False


def phase_one_surrogate_fits(
    masked_batch_ratio: float,
    policy_buffer_sizes: Sequence[int],
) -> tuple[PhaseOneSurrogateFit, ...]:
    """Return one matched fit per ratio and the six-plus-two ratio-1 checks."""

    sizes = tuple(int(size) for size in policy_buffer_sizes)
    if len(sizes) != 4 or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("Phase 1 requires four strictly increasing policy buffer sizes")
    maximum = sizes[-1]
    fits = [PhaseOneSurrogateFit(f"policy-n{maximum}-cat-h64", "policy", maximum, (64, 64))]
    if not np.isclose(masked_batch_ratio, 1.0):
        return tuple(fits)
    fits.extend(
        PhaseOneSurrogateFit(f"policy-n{size}-cat-h64", "policy", size, (64, 64), diagnostic=True)
        for size in sizes[:-1]
    )
    fits.extend(
        (
            PhaseOneSurrogateFit(
                f"policy-n{maximum}-scalar-h64",
                "policy",
                maximum,
                (64, 64),
                target_representation="scalar",
                diagnostic=True,
            ),
            PhaseOneSurrogateFit(f"policy-n{maximum}-cat-h128", "policy", maximum, (128, 128), diagnostic=True),
            PhaseOneSurrogateFit(f"policy-n{maximum}-cat-h256", "policy", maximum, (256, 256), diagnostic=True),
            PhaseOneSurrogateFit(f"uniform-n{maximum}-cat-h64", "uniform", maximum, (64, 64), diagnostic=True),
            PhaseOneSurrogateFit(f"replay-n{maximum}-cat-h64", "replay", maximum, (64, 64), diagnostic=True),
        )
    )
    return tuple(fits)


def _phase_one_policy_sample_per_role(settings: Mapping[str, Any]) -> int:
    """Resolve full-support versus finite role-balanced policy evaluation."""

    if bool(settings["exact_policy_full_support"]):
        return 10_000
    return int(settings["policy_explanation_states"]) // 2


class DirectBatchedMaskedQModel:
    """Batch adapter that makes the ratio-zero OOD choice explicit."""

    def __init__(self, agent: NeuralQAgent, *, allow_untrained: bool) -> None:
        self.agent = agent
        self.allow_untrained = bool(allow_untrained)
        self.num_actions = agent.num_actions
        self.observation_size = agent.observation_size

    def q_values_batch(self, observations: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        return self.agent.masked_q_values_batch(
            observations,
            known_masks,
            allow_untrained=self.allow_untrained,
        )


class _ParquetSink:
    """Stream long-format study rows into one Parquet table.

    A coalition table is over a million rows per cell, so rows are buffered and
    flushed a row group at a time rather than materialized. The column types are
    inferred once, from the first flushed batch, and every later batch is cast to
    that schema: the rows come from a single construction site with explicit
    ``float``/``int``/``bool``/``str`` conversions, so a column that changes type
    partway through a cell is a bug, and casting turns it into a loud one.
    """

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        require_storage_dependencies("Writing study cell tables")
        self.path = path
        self.fieldnames = tuple(fieldnames)
        self._writer: Any | None = None
        self._schema: Any | None = None
        self._buffer: list[Mapping[str, Any]] = []
        self.rows = 0

    def __enter__(self) -> _ParquetSink:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(row)
        self.rows += 1
        if len(self._buffer) >= PARQUET_ROW_GROUP_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = arrow_table(self.fieldnames, self._buffer, source=self.path)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema, compression=PARQUET_COMPRESSION)
        elif not table.schema.equals(self._schema):
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc, traceback
        try:
            # A failed cell is discarded with its staging directory, so there is
            # nothing to salvage by flushing a partial table on the way out.
            if exc_type is None:
                self._flush()
        finally:
            if self._writer is not None:
                self._writer.close()
        if exc_type is None and self.rows == 0:
            # Matching write_table: an empty table is a silent loss of the rows a
            # selected cell was supposed to produce, not a valid artifact.
            raise StudyAnalysisError(f"refusing to write empty table {self.path}")


def validate_explanation_sample_sizes(
    selected: Mapping[str, int],
    requested: Mapping[str, int],
) -> None:
    """Accept a fixed explanation sample no larger than the reference allows.

    A converged policy's occupancy support can be smaller than the requested
    fixed sample -- the X-role support shrinks as the policy sharpens. When it
    is, every support state is explained, which is the largest sample the
    reference admits, and ``reference_coverage.parquet`` records support, selection,
    coverage, and captured mass so the shortfall is reported rather than hidden.

    A reference with no support at all is still refused: there is no
    unconditional fallback when the conditioning event has zero mass.
    """

    for reference, total in requested.items():
        count = selected[reference]
        if count == 0:
            raise ValueError(f"{reference} reference has empty support; expected up to {total} fixed states")
        if count > total:
            raise ValueError(f"{reference} reference produced {count} of at most {total} fixed states")


def run_controlled_q_posthoc(
    agent: NeuralQAgent,
    context: StudyCellContext,
    *,
    replay_transitions: Sequence[ReplayTransition] = (),
) -> ControlledQPosthocResult:
    """Train both surrogates and score direct/native/crossed games in one cell."""

    parameters = context.condition.parameters
    settings = context.profile.settings
    selected_seeds = {int(seed) for seed in settings["explanation_seeds"]}
    if parameters["arm"] != "primary" or context.seed not in selected_seeds:
        return ControlledQPosthocResult(executed=False)
    if str(settings.get("posthoc_protocol", "legacy")) == "phase-one":
        return _run_phase_one_posthoc(agent, context, replay_transitions=replay_transitions)

    shared_distribution = build_shared_reference_distribution()
    policy_distribution = build_policy_reference_distribution(
        cast(PolicyView, agent),
        softening=float(resolved_setting(context, "policy_reference_softening")),
        opponent_random_move_probability=opponent_random_move_probability(
            str(settings["training_opponent"]),
            discount=float(settings["perturbed_minimax_discount"]),
            episode=int(settings["episodes"]),
        ),
    )
    distributions = {"shared": shared_distribution, "policy": policy_distribution}
    surrogates: dict[str, Any] = {}
    surrogate_rows: list[dict[str, Any]] = []
    for reference, distribution in distributions.items():
        config = MaskedQSurrogateConfig(
            reference=cast(Any, reference),
            seed=context.seed,
            updates=int(resolved_setting(context, "surrogate_updates")),
            batch_size=int(resolved_setting(context, "surrogate_batch_size")),
            hidden_sizes=tuple(int(value) for value in resolved_setting(context, "surrogate_hidden_sizes")),
            full_mask_example_interval=int(resolved_setting(context, "surrogate_full_mask_example_interval")),
            empty_mask_example_interval=int(resolved_setting(context, "surrogate_empty_mask_example_interval")),
            shapley_probability=float(resolved_setting(context, "mask_shapley_probability")),
            uniform_probability=float(resolved_setting(context, "mask_uniform_probability")),
        )
        trainer = MaskedQSurrogateTrainer(config)
        result = trainer.train(agent, distribution)
        trainer.write_artifacts(result, context.out_dir / "surrogates" / reference)
        surrogates[reference] = result.surrogate
        surrogate_rows.append({**_condition_fields(context), **result.summary.to_dict()})
    write_table(context.out_dir / "surrogate_runs.parquet", surrogate_rows)

    shared_total = int(settings["shared_explanation_states"])
    policy_total = int(settings["policy_explanation_states"])
    if shared_total % 2 or policy_total % 2:
        raise ValueError("shared and policy explanation state counts must be even")
    samples = {
        "shared": sample_stratified_reference_distribution(shared_distribution, total=shared_total, seed=0),
        "policy": sample_reference_distribution(policy_distribution, per_role=policy_total // 2, seed=context.seed),
    }
    validate_explanation_sample_sizes(
        {reference: len(sample.points) for reference, sample in samples.items()},
        {"shared": shared_total, "policy": policy_total},
    )
    _write_selection_artifacts(context, samples)

    direct = DirectBatchedMaskedQModel(
        agent,
        allow_untrained=float(parameters["masked_batch_ratio"]) == 0.0,
    )
    exact_explainers = {
        reference: TicTacToeSVERLExplainer(
            cast(PolicyView, agent),
            distribution=distribution,
            parallel=False,
        )
        for reference, distribution in distributions.items()
    }
    state_actions = sum(int(np.sum(point.action_mask_array)) for sample in samples.values() for point in sample.points)
    model_comparisons = 0
    with (
        _ParquetSink(context.out_dir / "coalition_fidelity.parquet", _COALITION_FIELDS) as coalition_sink,
        _ParquetSink(context.out_dir / "attribution_fidelity.parquet", _ATTRIBUTION_FIELDS) as attribution_sink,
        _ParquetSink(context.out_dir / "fidelity.parquet", _FIDELITY_FIELDS) as fidelity_sink,
    ):
        for evaluation_reference, sample in samples.items():
            models = (
                ("direct", "direct", direct),
                ("surrogate", "shared", surrogates["shared"]),
                ("surrogate", "policy", surrogates["policy"]),
            )
            for model_method, model_reference, model in models:
                model_comparisons += evaluate_masked_model_fidelity(
                    context=context,
                    agent=agent,
                    exact_explainer=exact_explainers[evaluation_reference],
                    sample=sample,
                    evaluation_reference=evaluation_reference,
                    model_method=model_method,
                    model_reference=model_reference,
                    model=model,
                    model_metadata=None,
                    coalition_sink=coalition_sink,
                    attribution_sink=attribution_sink,
                    fidelity_sink=fidelity_sink,
                )
        posthoc_result = ControlledQPosthocResult(
            executed=True,
            surrogate_runs=2,
            selected_states=sum(len(sample.points) for sample in samples.values()),
            state_actions=state_actions,
            model_comparisons=model_comparisons,
            coalition_detail_comparisons=model_comparisons,
            coalition_rows=coalition_sink.rows,
            attribution_rows=attribution_sink.rows,
            fidelity_rows=fidelity_sink.rows,
        )
    (context.out_dir / "posthoc_summary.json").write_text(
        json.dumps(posthoc_result.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return posthoc_result


def _run_phase_one_posthoc(
    agent: NeuralQAgent,
    context: StudyCellContext,
    *,
    replay_transitions: Sequence[ReplayTransition],
) -> ControlledQPosthocResult:
    """Run the exact-policy headline comparison and factorised diagnostics."""

    settings = context.profile.settings
    parameters = context.condition.parameters
    law_started = time.perf_counter()
    opponent_probability = opponent_random_move_probability(
        str(settings["training_opponent"]),
        discount=float(settings["perturbed_minimax_discount"]),
        episode=int(settings["episodes"]),
    )
    policy_distribution = build_policy_reference_distribution(
        cast(PolicyView, agent),
        softening=float(resolved_setting(context, "policy_reference_softening")),
        opponent_random_move_probability=opponent_probability,
    )
    uniform_distribution = build_shared_reference_distribution()
    sizes = tuple(int(size) for size in settings["policy_buffer_sizes"])
    policy_buffers = build_nested_reference_occurrence_buffers(
        policy_distribution,
        per_role_sizes=sizes,
        seed=context.seed,
        source="policy",
    )
    policy_by_size = {buffer.per_role: buffer for buffer in policy_buffers}
    uniform_buffers = build_nested_reference_occurrence_buffers(
        uniform_distribution,
        per_role_sizes=(int(settings["shared_explanation_states"]) // 2, sizes[-1]),
        seed=context.seed,
        source="uniform",
    )
    uniform_evaluation_buffer, uniform_buffer = uniform_buffers
    _write_buffer_artifacts(context, (*policy_buffers, *uniform_buffers), replay_transitions)
    law_seconds = time.perf_counter() - law_started

    fits = phase_one_surrogate_fits(float(parameters["masked_batch_ratio"]), sizes)
    replay_dataset = None
    if any(fit.training_law == "replay" for fit in fits):
        replay_dataset = _replay_surrogate_dataset(agent, replay_transitions, maximum=sizes[-1])
    models: dict[str, Any] = {}
    surrogate_rows: list[dict[str, Any]] = []
    surrogate_seconds = 0.0
    for fit in fits:
        reference = cast(Any, fit.training_law)
        config = MaskedQSurrogateConfig(
            reference=reference,
            seed=context.seed,
            updates=int(resolved_setting(context, "surrogate_updates")),
            batch_size=int(resolved_setting(context, "surrogate_batch_size")),
            hidden_sizes=fit.hidden_sizes,
            full_mask_example_interval=int(resolved_setting(context, "surrogate_full_mask_example_interval")),
            empty_mask_example_interval=int(resolved_setting(context, "surrogate_empty_mask_example_interval")),
            shapley_probability=float(resolved_setting(context, "mask_shapley_probability")),
            uniform_probability=float(resolved_setting(context, "mask_uniform_probability")),
            target_representation=cast(Any, fit.target_representation),
        )
        trainer = MaskedQSurrogateTrainer(config)
        if fit.training_law == "policy":
            result = trainer.train(agent, policy_by_size[fit.buffer_occurrences_per_role].distribution)
        elif fit.training_law == "uniform":
            result = trainer.train(agent, uniform_buffer.distribution)
        else:
            if replay_dataset is None:
                raise ValueError("Phase 1 replay fit has no replay dataset")
            result = trainer.train_dataset(replay_dataset, source_value_head=str(agent.config.value_head))
        trainer.write_artifacts(result, context.out_dir / "surrogates" / fit.key)
        models[fit.key] = result.surrogate
        metadata = _fit_metadata(fit, updates=config.updates)
        surrogate_rows.append({**_condition_fields(context), **metadata, **result.summary.to_dict()})
        surrogate_seconds += result.summary.duration_seconds
    write_table(context.out_dir / "surrogate_runs.parquet", surrogate_rows)

    policy_per_role = _phase_one_policy_sample_per_role(settings)
    policy_sample = sample_reference_distribution(policy_distribution, per_role=policy_per_role, seed=context.seed)
    uniform_sample = _sample_from_occurrence_buffer(
        uniform_evaluation_buffer,
        exact_distribution=uniform_distribution,
    )
    samples = {"policy": policy_sample, "uniform": uniform_sample}
    _write_selection_artifacts(context, samples)
    exact_explainers = {
        "policy": TicTacToeSVERLExplainer(cast(PolicyView, agent), distribution=policy_distribution, parallel=False),
        "uniform": TicTacToeSVERLExplainer(cast(PolicyView, agent), distribution=uniform_distribution, parallel=False),
    }
    return_q_oracles = {
        "policy": ExactReturnQOracle(
            cast(PolicyView, agent),
            policy_distribution,
            opponent_random_move_probability=opponent_probability,
        ),
        "uniform": ExactReturnQOracle(
            cast(PolicyView, agent),
            uniform_distribution,
            opponent_random_move_probability=opponent_probability,
        ),
    }
    direct = DirectBatchedMaskedQModel(
        agent,
        allow_untrained=float(parameters["masked_batch_ratio"]) == 0.0,
    )
    baseline_key = fits[0].key
    inference_started = time.perf_counter()
    state_actions = sum(int(np.sum(point.action_mask_array)) for sample in samples.values() for point in sample.points)
    model_comparisons = 0
    coalition_detail_comparisons = 0
    with (
        _ParquetSink(context.out_dir / "coalition_fidelity.parquet", _COALITION_FIELDS) as coalition_sink,
        _ParquetSink(context.out_dir / "attribution_fidelity.parquet", _ATTRIBUTION_FIELDS) as attribution_sink,
        _ParquetSink(context.out_dir / "fidelity.parquet", _FIDELITY_FIELDS) as fidelity_sink,
    ):
        for evaluation_reference, sample in samples.items():
            selected_fits = fits
            if evaluation_reference == "uniform":
                selected_fits = tuple(
                    fit for fit in fits if fit.key == baseline_key or fit.training_law in {"uniform", "replay"}
                )
            candidates = [("direct", "direct", direct, None)]
            candidates.extend(
                ("surrogate", fit.key, models[fit.key], _fit_metadata(fit, updates=int(settings["surrogate_updates"])))
                for fit in selected_fits
            )
            for model_method, model_reference, model, metadata in candidates:
                persist_coalitions = evaluation_reference == "policy" and (
                    model_method == "direct" or model_reference == baseline_key
                )
                persist_coalitions = persist_coalitions or (
                    evaluation_reference == "uniform"
                    and np.isclose(float(parameters["masked_batch_ratio"]), 1.0)
                    and (
                        model_method == "direct"
                        or model_reference == baseline_key
                        or (metadata is not None and metadata["training_law"] == "uniform")
                    )
                )
                comparisons = evaluate_masked_model_fidelity(
                    context=context,
                    agent=agent,
                    exact_explainer=exact_explainers[evaluation_reference],
                    exact_return_oracle=return_q_oracles[evaluation_reference],
                    sample=sample,
                    evaluation_reference=evaluation_reference,
                    model_method=model_method,
                    model_reference=model_reference,
                    model=model,
                    model_metadata=metadata,
                    persist_coalition_rows=persist_coalitions,
                    coalition_sink=coalition_sink,
                    attribution_sink=attribution_sink,
                    fidelity_sink=fidelity_sink,
                )
                model_comparisons += comparisons
                if persist_coalitions:
                    coalition_detail_comparisons += comparisons
        inference_seconds = time.perf_counter() - inference_started
        posthoc_result = ControlledQPosthocResult(
            executed=True,
            surrogate_runs=len(fits),
            selected_states=sum(len(sample.points) for sample in samples.values()),
            state_actions=state_actions,
            model_comparisons=model_comparisons,
            coalition_detail_comparisons=coalition_detail_comparisons,
            coalition_rows=coalition_sink.rows,
            attribution_rows=attribution_sink.rows,
            fidelity_rows=fidelity_sink.rows,
            law_seconds=law_seconds,
            surrogate_seconds=surrogate_seconds,
            inference_seconds=inference_seconds,
        )
    (context.out_dir / "posthoc_summary.json").write_text(
        json.dumps(posthoc_result.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return posthoc_result


def evaluate_masked_model_fidelity(
    *,
    context: StudyCellContext,
    agent: NeuralQAgent,
    exact_explainer: TicTacToeSVERLExplainer,
    exact_return_oracle: ExactReturnQOracle | None = None,
    sample: DistributionSample,
    evaluation_reference: str,
    model_method: str,
    model_reference: str,
    model: BatchedMaskedQModel,
    model_metadata: Mapping[str, Any] | None = None,
    persist_coalition_rows: bool = True,
    coalition_sink: FidelityRowSink,
    attribution_sink: FidelityRowSink,
    fidelity_sink: FidelityRowSink,
) -> int:
    comparisons = 0
    feature_names = exact_explainer.features.names
    for point in sample.points:
        observation = point.observation_array.astype(np.float32)
        observations = np.repeat(observation[None, :], len(_COALITIONS), axis=0)
        approximate_matrix = np.asarray(model.q_values_batch(observations, _COALITION_MASKS), dtype=np.float64)
        if approximate_matrix.shape != (len(_COALITIONS), agent.num_actions):
            raise ValueError(
                f"masked model returned {approximate_matrix.shape}, expected {(len(_COALITIONS), agent.num_actions)}"
            )
        if not np.all(np.isfinite(approximate_matrix)):
            raise ValueError("masked model returned non-finite expected Q")
        full_q = np.asarray(agent.q_values(observation), dtype=np.float64)
        legal = np.flatnonzero(point.action_mask_array)
        greedy_action = int(legal[np.argmax(full_q[legal])])
        for action_value in legal:
            action = int(action_value)
            exact_values = exact_explainer.q_value_coalition_values(
                point.state,
                action=action,
                learner_mark=point.learner_mark,
            )
            approximate_values = np.asarray(approximate_matrix[:, action], dtype=np.float64)
            exact_attributions = _SHAPLEY_OPERATOR @ exact_values
            approximate_attributions = _SHAPLEY_OPERATOR @ approximate_values
            if exact_return_oracle is None:
                exact_return_values = np.full_like(exact_values, np.nan)
                exact_return_attributions = np.full_like(exact_attributions, np.nan)
                return_scores = None
                base_q_scores = None
            else:
                exact_return_values = exact_return_oracle.coalition_values(
                    point.state,
                    action=action,
                    learner_mark=point.learner_mark,
                )
                exact_return_attributions = _SHAPLEY_OPERATOR @ exact_return_values
                return_scores = score_fidelity(
                    approximate_attributions,
                    exact_return_attributions,
                    approximate_values,
                    exact_return_values,
                )
                base_q_scores = score_fidelity(
                    exact_attributions,
                    exact_return_attributions,
                    exact_values,
                    exact_return_values,
                )
            scores = score_fidelity(
                approximate_attributions,
                exact_attributions,
                approximate_values,
                exact_values,
            )
            metadata = (
                _default_model_metadata(model_method, model_reference)
                if model_metadata is None
                else dict(model_metadata)
            )
            common = {
                **_condition_fields(context),
                "evaluation_reference": evaluation_reference,
                "state_sample": evaluation_reference,
                "model_method": model_method,
                "model_reference": model_reference,
                "comparison_kind": _comparison_kind(
                    model_method,
                    model_reference,
                    evaluation_reference,
                    training_law=str(metadata["training_law"]),
                ),
                **metadata,
                "ood_direct_query": bool(
                    model_method == "direct" and float(context.condition.parameters["masked_batch_ratio"]) == 0.0
                ),
                **_point_fields(point),
                "action": action,
                "greedy_action": action == greedy_action,
            }
            fidelity_sink.write(
                {
                    **common,
                    **scores.to_row(),
                    "return_q_attribution_mae": (math.nan if return_scores is None else return_scores.attribution_mae),
                    "base_q_attribution_mae": (math.nan if base_q_scores is None else base_q_scores.attribution_mae),
                    "return_q_coalition_mae": math.nan if return_scores is None else return_scores.coalition_mae,
                    "base_q_coalition_mae": math.nan if base_q_scores is None else base_q_scores.coalition_mae,
                }
            )
            for index, (exact_value, approximate_value, exact_return_value) in enumerate(
                zip(
                    exact_attributions,
                    approximate_attributions,
                    exact_return_attributions,
                    strict=True,
                )
            ):
                error = float(approximate_value - exact_value)
                base_q_error = float(exact_value - exact_return_value)
                total_error = float(approximate_value - exact_return_value)
                if exact_return_oracle is not None and not np.isclose(
                    total_error,
                    base_q_error + error,
                    atol=1e-12,
                ):
                    raise ValueError("return-Q attribution decomposition does not close")
                attribution_sink.write(
                    {
                        **common,
                        "feature_index": index,
                        "feature_name": feature_names[index],
                        "exact_attribution": exact_value,
                        "approximate_attribution": approximate_value,
                        "attribution_error": error,
                        "absolute_attribution_error": abs(error),
                        "exact_return_attribution": exact_return_value,
                        "base_q_attribution_error": base_q_error,
                        "absolute_base_q_attribution_error": abs(base_q_error),
                        "total_return_attribution_error": total_error,
                        "absolute_total_return_attribution_error": abs(total_error),
                    }
                )
            if persist_coalition_rows:
                for index, coalition in enumerate(_COALITIONS):
                    error = float(approximate_values[index] - exact_values[index])
                    base_q_error = float(exact_values[index] - exact_return_values[index])
                    total_error = float(approximate_values[index] - exact_return_values[index])
                    if exact_return_oracle is not None and not np.isclose(
                        total_error,
                        base_q_error + error,
                        atol=1e-12,
                    ):
                        raise ValueError("return-Q coalition decomposition does not close")
                    coalition_sink.write(
                        {
                            **common,
                            "coalition_index": index,
                            "coalition_size": len(coalition),
                            "coalition_features": "|".join(str(feature) for feature in sorted(coalition)),
                            "exact_value": exact_values[index],
                            "approximate_value": approximate_values[index],
                            "coalition_error": error,
                            "absolute_coalition_error": abs(error),
                            "exact_return_value": exact_return_values[index],
                            "base_q_coalition_error": base_q_error,
                            "absolute_base_q_coalition_error": abs(base_q_error),
                            "total_return_coalition_error": total_error,
                            "absolute_total_return_coalition_error": abs(total_error),
                        }
                    )
            comparisons += 1
    return comparisons


def _write_selection_artifacts(context: StudyCellContext, samples: Mapping[str, DistributionSample]) -> None:
    selection_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for reference, sample in samples.items():
        for index, point in enumerate(sample.points):
            selection_rows.append(
                {
                    **_condition_fields(context),
                    "sample_reference": reference,
                    "selected_index": index,
                    **_point_fields(point),
                }
            )
        coverage_rows.extend(
            {
                **_condition_fields(context),
                "sample_reference": reference,
                **coverage.to_row(),
            }
            for coverage in sample.coverage
        )
    write_table(context.out_dir / "state_selection.parquet", selection_rows)
    write_table(context.out_dir / "reference_coverage.parquet", coverage_rows)


def resolved_setting(context: StudyCellContext, key: str) -> Any:
    """Read one knob, letting a condition override the profile default.

    The mask-ratio study fixes surrogate construction in its profile, so every
    condition resolves identically and its behaviour is unchanged. An ablation
    that varies construction per condition needs the same post-hoc pipeline
    rather than a forked copy of it, which is what this override provides.
    """

    parameters = context.condition.parameters
    if key in parameters:
        return parameters[key]
    settings = context.profile.settings
    if key in settings:
        return settings[key]
    return _SETTING_DEFAULTS[key]


def _condition_fields(context: StudyCellContext) -> dict[str, Any]:
    parameters = context.condition.parameters
    return {
        "condition_key": context.condition.key,
        "arm": parameters["arm"],
        "value_head": parameters["value_head"],
        "masked_batch_ratio": parameters["masked_batch_ratio"],
        "seed": context.seed,
    }


def _point_fields(point: TicTacToeDistributionPoint) -> dict[str, Any]:
    board = "".join("X" if value == X else "O" if value == O else "." for value in point.state.board)
    observation = "|".join(str(value) for value in point.observation)
    return {
        "state_id": f"{_ROLE_NAME[point.learner_mark]}-{board}",
        "state_board": board,
        "observation": observation,
        "learner_role": _ROLE_NAME[point.learner_mark],
        "move_count": point.state.move_count,
        "sample_weight": point.weight,
    }


def _comparison_kind(
    model_method: str,
    model_reference: str,
    evaluation_reference: str,
    *,
    training_law: str | None = None,
) -> str:
    if model_method == "direct":
        return "direct"
    law = model_reference if training_law is None else training_law
    return "native_fidelity" if law == evaluation_reference else "reference_transfer"


def _fit_metadata(fit: PhaseOneSurrogateFit, *, updates: int) -> dict[str, Any]:
    return {
        "fit_key": fit.key,
        "training_law": fit.training_law,
        "buffer_occurrences_per_role": fit.buffer_occurrences_per_role,
        "surrogate_hidden_sizes": "x".join(str(size) for size in fit.hidden_sizes),
        "surrogate_updates": updates,
        "target_representation": fit.target_representation,
        "diagnostic": fit.diagnostic,
    }


def _default_model_metadata(model_method: str, model_reference: str) -> dict[str, Any]:
    if model_method != "direct":
        return {
            "fit_key": model_reference,
            "training_law": model_reference,
            "buffer_occurrences_per_role": 0,
            "surrogate_hidden_sizes": "legacy",
            "surrogate_updates": 0,
            "target_representation": "native",
            "diagnostic": False,
        }
    return {
        "fit_key": "direct",
        "training_law": "replay",
        "buffer_occurrences_per_role": 0,
        "surrogate_hidden_sizes": "direct",
        "surrogate_updates": 0,
        "target_representation": "native",
        "diagnostic": False,
    }


def _replay_surrogate_dataset(
    agent: NeuralQAgent,
    transitions: Sequence[ReplayTransition],
    *,
    maximum: int,
) -> MaskedQSurrogateDataset:
    retained = tuple(transitions[-maximum:])
    if not retained:
        raise ValueError("Phase 1 replay diagnostic requires retained replay transitions")
    counts = Counter(
        (tuple(np.asarray(item.state, dtype=np.float32)), tuple(np.asarray(item.action_mask, dtype=np.int8)))
        for item in retained
    )
    keys = sorted(counts)
    observations = np.asarray([key[0] for key in keys], dtype=np.float32)
    action_masks = np.asarray([key[1] for key in keys], dtype=np.int8)
    weights = np.asarray([counts[key] for key in keys], dtype=np.float32)
    return build_masked_q_surrogate_dataset_from_arrays(
        agent,
        observations,
        action_masks,
        weights,
        reference="replay",
    )


def _sample_from_occurrence_buffer(
    buffer: ReferenceOccurrenceBuffer,
    *,
    exact_distribution: Any,
) -> DistributionSample:
    coverage = []
    for mark in (X, O):
        exact_points = tuple(point for point in exact_distribution.points if point.learner_mark == mark)
        selected = tuple(point for point in buffer.distribution.points if point.learner_mark == mark)
        selected_identities = {(point.state, point.learner_mark) for point in selected}
        coverage.append(
            RoleCoverage(
                learner_role=_ROLE_NAME[mark],
                support_states=len(exact_points),
                selected_states=len(selected),
                total_mass=sum(point.weight for point in exact_points),
                captured_mass=sum(
                    point.weight for point in exact_points if (point.state, point.learner_mark) in selected_identities
                ),
            )
        )
    return DistributionSample(
        points=buffer.distribution.points,
        coverage=(coverage[0], coverage[1]),
    )


def _write_buffer_artifacts(
    context: StudyCellContext,
    buffers: Sequence[ReferenceOccurrenceBuffer],
    replay_transitions: Sequence[ReplayTransition],
) -> None:
    occurrence_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    maximum_by_source = {
        source: max(buffer.per_role for buffer in buffers if buffer.source == source)
        for source in {buffer.source for buffer in buffers}
    }
    for buffer in buffers:
        if buffer.per_role == maximum_by_source[buffer.source]:
            role_index = {X: 0, O: 0}
            for point in buffer.occurrences:
                occurrence_rows.append(
                    {
                        **_condition_fields(context),
                        "source": buffer.source,
                        "learner_role": _ROLE_NAME[point.learner_mark],
                        "occurrence_index": role_index[point.learner_mark],
                        "observation": "|".join(str(value) for value in point.observation),
                    }
                )
                role_index[point.learner_mark] += 1
        for point in buffer.distribution.points:
            count_rows.append(
                {
                    **_condition_fields(context),
                    "source": buffer.source,
                    "buffer_occurrences_per_role": buffer.per_role,
                    "learner_role": _ROLE_NAME[point.learner_mark],
                    "observation": "|".join(str(value) for value in point.observation),
                    "count": round(point.weight * 2 * buffer.per_role),
                    "weight": point.weight,
                }
            )
    for index, transition in enumerate(tuple(replay_transitions)[-max(maximum_by_source.values()) :]):
        occurrence_rows.append(
            {
                **_condition_fields(context),
                "source": "replay",
                "learner_role": "learner-relative",
                "occurrence_index": index,
                "observation": "|".join(str(value) for value in np.asarray(transition.state)),
            }
        )
    write_table(context.out_dir / "state_buffer_occurrences.parquet", occurrence_rows)
    write_table(context.out_dir / "state_buffer_counts.parquet", count_rows)


__all__ = [
    "BatchedMaskedQModel",
    "ControlledQPosthocResult",
    "PhaseOneSurrogateFit",
    "DirectBatchedMaskedQModel",
    "FidelityRowSink",
    "evaluate_masked_model_fidelity",
    "phase_one_surrogate_fits",
    "resolved_setting",
    "run_controlled_q_posthoc",
    "validate_explanation_sample_sizes",
]
