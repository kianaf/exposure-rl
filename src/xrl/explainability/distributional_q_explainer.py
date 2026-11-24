"""Exact and sampled Shapley explainers for masked Q-value models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from xrl._provenance import require_artifact_matches_run
from xrl.explainability.core import (
    CachedCharacteristicFunction,
    ExactShapleyEstimator,
    ExplanationResult,
    ExplanationSemantics,
    KernelShapleyEstimator,
    SampledShapleyEstimator,
    ShapleyEstimator,
    validate_action,
)
from xrl.explainability.distributional_q_masks import MaskedQValueModel

# Exhaustive 2^n coalition enumeration stays tractable up to this feature count;
# larger observations (e.g. 5x5 Minesweeper) need a SampledShapleyEstimator.
EXACT_SHAPLEY_MAX_FEATURES = 12

# How to extract phi from a masked coalition game. The game is fixed by the
# model being queried; these vary only the numerical method applied to it.
MASKED_Q_ESTIMATORS = ("auto", "exact", "permutation", "kernel")

# Estimator name -> the label that travels in ExplanationSemantics.
_ESTIMATOR_LABELS = {
    "exact": "exact",
    "permutation": "sampled_permutation",
    "kernel": "kernel_shap",
}


def select_masked_q_estimator(
    feature_count: int,
    estimator: str = "auto",
    *,
    num_permutations: int = 128,
    kernel_coalitions: int = 512,
    seed: int = 0,
) -> tuple[ShapleyEstimator, str]:
    """Return an estimator for a masked-Q game and the label it reports.

    ``auto`` enumerates exactly while that is tractable and falls back to
    permutation sampling beyond, which is the behaviour that predates an
    explicit choice.
    """

    if estimator not in MASKED_Q_ESTIMATORS:
        raise ValueError(f"estimator must be one of {MASKED_Q_ESTIMATORS}, got {estimator!r}")
    if estimator == "auto":
        estimator = "exact" if feature_count <= EXACT_SHAPLEY_MAX_FEATURES else "permutation"
    if estimator == "exact":
        return ExactShapleyEstimator(), _ESTIMATOR_LABELS["exact"]
    if estimator == "permutation":
        return (
            SampledShapleyEstimator(num_permutations=num_permutations, seed=seed),
            _ESTIMATOR_LABELS["permutation"],
        )
    return KernelShapleyEstimator(num_coalitions=kernel_coalitions, seed=seed), _ESTIMATOR_LABELS["kernel"]


SURROGATE_MASKED_Q_ASSUMPTION = "unknown features are represented by a separately trained masked Q surrogate"
DIRECT_MASKED_Q_ASSUMPTION = "unknown features are represented by the jointly trained direct masked Q-function"
GENERIC_MASKED_Q_ASSUMPTION = "unknown features are represented by the supplied masked Q-value model"


@dataclass(frozen=True, slots=True)
class DistributionalQExplanationResult:
    """Expected-Q Shapley explanation produced by a masked Q-value model."""

    action: int
    baseline_value: float
    full_value: float
    attributions: dict[str, float]
    feature_names: tuple[str, ...]
    support: tuple[float, ...] = ()
    characteristic_assumption: str = GENERIC_MASKED_Q_ASSUMPTION


class DistributionalQExplainer:
    """Expected-Q Shapley explanations over a masked Q-value model."""

    def __init__(
        self,
        surrogate: MaskedQValueModel,
        *,
        feature_names: Sequence[str] | None = None,
        estimator: ShapleyEstimator | None = None,
        characteristic_assumption: str | None = None,
    ) -> None:
        self.surrogate = surrogate
        if feature_names is None:
            observation_size = int(getattr(surrogate, "observation_size", 0)) or 9
            feature_names = tuple(f"cell_{index}" for index in range(observation_size))
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("at least one feature name is required")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names must be unique")
        self.estimator = estimator or ExactShapleyEstimator()
        self.characteristic_assumption = characteristic_assumption or str(
            getattr(surrogate, "characteristic_assumption", GENERIC_MASKED_Q_ASSUMPTION)
        )

    def explain_action(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        action: int,
    ) -> DistributionalQExplanationResult:
        """Explain the masked model's expected Q-value for one action."""

        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if len(observation_array) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(observation_array)}")
        self._validate_action(action, action_mask)
        characteristic = CachedCharacteristicFunction(
            len(self.feature_names),
            lambda coalition: self._expected_q(observation_array, coalition, action),
        )
        values = self.estimator.shapley_values(characteristic)
        return DistributionalQExplanationResult(
            action=action,
            baseline_value=characteristic.value(frozenset()),
            full_value=characteristic.value(frozenset(range(len(self.feature_names)))),
            attributions={name: float(value) for name, value in zip(self.feature_names, values, strict=True)},
            feature_names=self.feature_names,
            support=tuple(float(value) for value in getattr(self.surrogate, "support", ())),
            characteristic_assumption=self.characteristic_assumption,
        )

    def support_bin_attributions(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        action: int,
    ) -> dict[float, dict[str, float]]:
        """Return optional per-support-bin Shapley values over probability mass."""

        if not callable(getattr(self.surrogate, "q_distributions", None)) or not tuple(
            getattr(self.surrogate, "support", ())
        ):
            raise TypeError("support_bin_attributions requires a categorical masked Q model")
        observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if len(observation_array) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(observation_array)}")
        self._validate_action(action, action_mask)
        result: dict[float, dict[str, float]] = {}
        for bin_index, support_value in enumerate(self.surrogate.support):

            def probability_for(coalition: Iterable[int], *, index: int = bin_index) -> float:
                return self._support_probability(observation_array, coalition, action, index)

            characteristic = CachedCharacteristicFunction(
                len(self.feature_names),
                probability_for,
            )
            values = self.estimator.shapley_values(characteristic)
            result[float(support_value)] = {
                name: float(value) for name, value in zip(self.feature_names, values, strict=True)
            }
        return result

    def _expected_q(self, observation: np.ndarray, coalition: Iterable[int], action: int) -> float:
        mask = self._mask(coalition)
        values = np.asarray(self.surrogate.q_values(observation, mask), dtype=np.float64)
        return float(values[action])

    def _support_probability(
        self, observation: np.ndarray, coalition: Iterable[int], action: int, bin_index: int
    ) -> float:
        mask = self._mask(coalition)
        distributions = np.asarray(self.surrogate.q_distributions(observation, mask), dtype=np.float64)
        return float(distributions[action, bin_index])

    def _mask(self, coalition: Iterable[int]) -> np.ndarray:
        mask: np.ndarray = np.zeros(len(self.feature_names), dtype=np.int8)
        for index in coalition:
            mask[int(index)] = 1
        return mask

    def _validate_action(self, action: int, action_mask: np.ndarray | Iterable[int]) -> int:
        return validate_action(action, action_mask, num_actions=int(self.surrogate.num_actions))


def surrogate_explanation(
    surrogate_dir: str | Path,
    observation: np.ndarray | Iterable[int | float],
    action: int,
    *,
    action_mask: np.ndarray | Iterable[int],
    base_run_dir: str | Path,
    feature_names: Sequence[str],
    num_permutations: int = 128,
    seed: int = 0,
    estimator: str = "auto",
    kernel_coalitions: int = 512,
) -> ExplanationResult:
    """Explain one action's expected Q with a saved masked surrogate run.

    Defaults to exact enumeration up to :data:`EXACT_SHAPLEY_MAX_FEATURES`
    features and sampled permutations beyond, and converts the
    surrogate-specific result to the shared :class:`ExplanationResult`. Shared
    by the CLI and the browser UI.
    """

    return surrogate_masked_q_explanation(
        load_validated_surrogate(surrogate_dir, base_run_dir),
        observation,
        action,
        action_mask=action_mask,
        feature_names=feature_names,
        num_permutations=num_permutations,
        seed=seed,
        estimator=estimator,
        kernel_coalitions=kernel_coalitions,
    )


def load_validated_surrogate(surrogate_dir: str | Path, base_run_dir: str | Path) -> MaskedQValueModel:
    """Check a surrogate belongs to the selected run, then load it.

    Split out of :func:`surrogate_explanation` so a caller explaining many
    states pays the validation and the model rebuild once instead of per state.
    """

    from xrl.explainability.distributional_q_trainer import load_distributional_q_surrogate

    validate_surrogate_base_run(surrogate_dir, base_run_dir)
    return load_distributional_q_surrogate(Path(surrogate_dir))


def surrogate_masked_q_explanation(
    surrogate: MaskedQValueModel,
    observation: np.ndarray | Iterable[int | float],
    action: int,
    *,
    action_mask: np.ndarray | Iterable[int],
    feature_names: Sequence[str],
    num_permutations: int = 128,
    seed: int = 0,
    estimator: str = "auto",
    kernel_coalitions: int = 512,
) -> ExplanationResult:
    """Explain one action with an already-loaded masked surrogate."""

    return masked_q_explanation(
        surrogate,
        observation,
        action,
        action_mask=action_mask,
        feature_names=feature_names,
        num_permutations=num_permutations,
        seed=seed,
        estimator=estimator,
        kernel_coalitions=kernel_coalitions,
        coalition_semantics="surrogate_masked_q",
        value_model_source="masked_q_surrogate",
        characteristic_assumption=SURROGATE_MASKED_Q_ASSUMPTION,
    )


class DirectMaskedQAdapter:
    """Adapt a mask-aware DQN to the coalition-value model protocol."""

    def __init__(self, policy: object, *, allow_untrained: bool = False) -> None:
        masked_values = getattr(policy, "masked_q_values", None)
        if not callable(masked_values):
            raise TypeError("direct masked-Q explanations require policy.masked_q_values")
        self.policy = cast(Any, policy)
        self.allow_untrained = bool(allow_untrained)
        self.characteristic_assumption = DIRECT_MASKED_Q_ASSUMPTION
        self.num_actions = int(self.policy.num_actions)
        self.observation_size = int(self.policy.observation_size)
        config = getattr(self.policy, "config", None)
        self.support = (
            tuple(float(value) for value in np.asarray(self.policy.support, dtype=np.float32))
            if getattr(config, "value_head", None) in {"two-hot", "hl-gauss"}
            else ()
        )

    def q_values(
        self,
        observation: np.ndarray | Iterable[int | float],
        known_mask: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        arguments = (np.asarray(observation), np.asarray(known_mask))
        returned = (
            self.policy.masked_q_values(*arguments, allow_untrained=True)
            if self.allow_untrained
            else self.policy.masked_q_values(*arguments)
        )
        values = np.asarray(returned, dtype=np.float32)
        if values.shape != (self.num_actions,):
            raise ValueError(f"masked_q_values returned shape {values.shape}, expected ({self.num_actions},)")
        return values

    def q_distributions(
        self,
        observation: np.ndarray | Iterable[int | float],
        known_mask: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        method = getattr(self.policy, "masked_q_distributions", None)
        if not callable(method):
            raise TypeError("support-bin attributions require masked_q_distributions")
        arguments = (np.asarray(observation), np.asarray(known_mask))
        returned_support, probabilities = (
            method(*arguments, allow_untrained=True) if self.allow_untrained else method(*arguments)
        )
        support = np.asarray(returned_support, dtype=np.float32).reshape(-1)
        expected_support = np.asarray(self.support, dtype=np.float32)
        if support.shape != expected_support.shape or not np.allclose(support, expected_support):
            raise ValueError("masked_q_distributions returned support inconsistent with policy.support")
        matrix = np.asarray(probabilities, dtype=np.float32)
        expected_shape = (self.num_actions, len(self.support))
        if matrix.shape != expected_shape:
            raise ValueError(f"masked_q_distributions returned shape {matrix.shape}, expected {expected_shape}")
        return matrix


def direct_q_explanation(
    policy: object,
    observation: np.ndarray | Iterable[int | float],
    action: int,
    *,
    action_mask: np.ndarray | Iterable[int],
    feature_names: Sequence[str],
    num_permutations: int = 128,
    seed: int = 0,
    estimator: str = "auto",
    kernel_coalitions: int = 512,
    allow_untrained: bool = False,
) -> ExplanationResult:
    """Explain one action using a jointly trained mask-aware Q-function."""

    return masked_q_explanation(
        DirectMaskedQAdapter(policy, allow_untrained=allow_untrained),
        observation,
        action,
        action_mask=action_mask,
        feature_names=feature_names,
        num_permutations=num_permutations,
        seed=seed,
        estimator=estimator,
        kernel_coalitions=kernel_coalitions,
        coalition_semantics="direct_masked_q",
        value_model_source="base_q_function",
        characteristic_assumption=DIRECT_MASKED_Q_ASSUMPTION,
    )


def masked_q_explanation(
    model: MaskedQValueModel,
    observation: np.ndarray | Iterable[int | float],
    action: int,
    *,
    action_mask: np.ndarray | Iterable[int],
    feature_names: Sequence[str],
    num_permutations: int = 128,
    seed: int = 0,
    coalition_semantics: str,
    value_model_source: str,
    characteristic_assumption: str,
    estimator: str = "auto",
    kernel_coalitions: int = 512,
) -> ExplanationResult:
    """Apply a Shapley transform to masked coalition values.

    The game is whatever ``model`` answers; ``estimator`` chooses only how phi
    is extracted from it, so two calls differing only in ``estimator`` produce
    comparable numbers rather than answers to different questions.
    """

    names = tuple(feature_names)
    shapley, estimator_label = select_masked_q_estimator(
        len(names),
        estimator,
        num_permutations=num_permutations,
        kernel_coalitions=kernel_coalitions,
        seed=seed,
    )
    explainer = DistributionalQExplainer(
        model,
        feature_names=names,
        estimator=shapley,
        characteristic_assumption=characteristic_assumption,
    )
    explanation = explainer.explain_action(observation, action_mask, action=action)
    observation_array = np.asarray(observation, dtype=np.float32).reshape(-1)
    return ExplanationResult(
        target="q_value",
        state=tuple(float(value) for value in observation_array),
        action=explanation.action,
        baseline_value=explanation.baseline_value,
        full_value=explanation.full_value,
        attributions=explanation.attributions,
        feature_names=explanation.feature_names,
        characteristic_assumption=explanation.characteristic_assumption,
        semantics=ExplanationSemantics(
            target_quantity="fixed_action_q",
            coalition_semantics=coalition_semantics,
            value_model_source=value_model_source,
            shapley_estimator=estimator_label,
        ),
    )


def validate_surrogate_base_run(surrogate_dir: str | Path, base_run_dir: str | Path) -> None:
    """Reject a surrogate that cannot be tied to the selected base run."""

    require_artifact_matches_run(
        surrogate_dir,
        base_run_dir,
        config_name="surrogate_config.json",
        label="surrogate",
    )
