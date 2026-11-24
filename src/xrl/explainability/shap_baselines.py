"""Marginal SHAP baselines explaining Q(s, a) over rollout backgrounds.

Every explainer here attributes one action's Q-value to observation features
under the interventional (marginal) assumption: off-coalition features are
imputed from rollout-collected background states, so composed observations may
be game-invalid — the same assumption the masked distributional surrogate
makes. Results share :class:`ExplanationResult`, so they are directly
comparable with SVERL and masked-surrogate attributions.

The ``shap`` library (optional ``explain`` extra) is imported lazily inside
the kernel and tree paths; the exact path is pure in-repo code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from xrl._optional import require_explain_dependencies
from xrl.explainability.backgrounds import BackgroundDataset
from xrl.explainability.cohorts import CohortKeyFn, cohort_background
from xrl.explainability.core import (
    CachedCharacteristicFunction,
    ExactShapleyEstimator,
    ExplanationResult,
    ExplanationSemantics,
    FeatureSpec,
    PolicyView,
)
from xrl.explainability.core import validate_action as shared_validate_action

# Exhaustive coalition enumeration doubles per feature; beyond this use kernel.
EXACT_MARGINAL_MAX_FEATURES = 16

MARGINAL_ASSUMPTION = "off-coalition features imputed from rollout background states (interventional/marginal)"

MARGINAL_SHAP_METHODS = ("exact", "kernel", "cohort-kernel")


@dataclass(frozen=True, slots=True)
class ShapBaselineConfig:
    """Shared knobs for the marginal SHAP baseline explainers."""

    max_background: int = 128
    kernel_nsamples: int | str = "auto"
    cohort_min_size: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_background < 1:
            raise ValueError("max_background must be positive")
        if self.cohort_min_size < 1:
            raise ValueError("cohort_min_size must be positive")
        if isinstance(self.kernel_nsamples, str) and self.kernel_nsamples != "auto":
            raise ValueError("kernel_nsamples must be an integer or 'auto'")


def greedy_action(policy: PolicyView, observation: np.ndarray, action_mask: np.ndarray) -> int:
    """Return the highest-Q legal action for one observation."""

    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise ValueError("no legal action available to explain")
    values = np.asarray(policy.q_values(observation), dtype=np.float64)
    return int(legal[np.argmax(values[legal])])


def validate_action(policy: PolicyView, action_mask: np.ndarray, action: int) -> int:
    """Return one in-range legal action or raise a stable validation error."""

    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    num_actions = int(getattr(policy, "num_actions", mask.shape[0]))
    return shared_validate_action(action, mask, num_actions=num_actions)


def q_value_function(policy: PolicyView, action: int) -> Callable[[Any], np.ndarray]:
    """Return a batched ``f(X) -> Q(X, action)`` suitable for shap explainers.

    The per-row ``policy.q_values`` path is correct for every agent, including
    tabular agents and symmetry encoders (encoding happens inside the policy).
    Identity-encoder agents exposing ``q_values_batch`` get a vectorized fast
    path. Inputs are coerced to float32 arrays (shap may pass DataFrames) and
    the output is a plain 1-D float64 ndarray.
    """

    batch = getattr(policy, "q_values_batch", None)
    encoder = getattr(getattr(policy, "config", None), "encoder", None)

    def evaluate(matrix: Any) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if batch is not None and encoder == "identity":
            return np.asarray(batch(array), dtype=np.float64)[:, action]
        return np.asarray([float(policy.q_values(row)[action]) for row in array], dtype=np.float64)

    return evaluate


def marginal_characteristic(
    policy: PolicyView,
    observation: np.ndarray,
    action: int,
    background: BackgroundDataset,
    *,
    max_background: int = 128,
    seed: int = 0,
) -> CachedCharacteristicFunction:
    """Build ``v(C) = mean_b Q(compose(x, b, C), action)`` over background rows."""

    x = np.asarray(observation, dtype=np.float32).reshape(-1)
    rows = _subsample_rows(background.observations, max_background, seed)
    value_fn = q_value_function(policy, action)

    def evaluate(coalition) -> float:
        composed = rows.copy()
        indices = [int(index) for index in coalition]
        if indices:
            composed[:, indices] = x[indices]
        return float(np.mean(value_fn(composed)))

    return CachedCharacteristicFunction(x.shape[0], evaluate)


class MarginalShapExplainer:
    """Exact, kernel, and cohort-kernel marginal Shapley baselines for Q(s, a)."""

    def __init__(
        self,
        policy: PolicyView,
        feature_spec: FeatureSpec,
        background: BackgroundDataset,
        *,
        method: str = "kernel",
        cohort_key: CohortKeyFn | None = None,
        config: ShapBaselineConfig | None = None,
    ) -> None:
        if method not in MARGINAL_SHAP_METHODS:
            raise ValueError(f"method must be one of {MARGINAL_SHAP_METHODS}, got {method!r}")
        if method == "cohort-kernel" and cohort_key is None:
            raise ValueError("method 'cohort-kernel' requires a cohort_key function")
        self.policy = policy
        self.feature_spec = feature_spec
        self.background = background
        self.method = method
        self.cohort_key = cohort_key
        self.config = config or ShapBaselineConfig()

    def explain_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        action: int | None = None,
    ) -> ExplanationResult:
        """Attribute Q(observation, action) to features; greedy action by default."""

        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.feature_spec.feature_count:
            raise ValueError(f"expected {self.feature_spec.feature_count} features, got {x.shape[0]}")
        target_action = (
            greedy_action(self.policy, x, action_mask)
            if action is None
            else validate_action(self.policy, action_mask, action)
        )
        if self.method == "exact":
            return self._explain_exact(x, target_action)
        return self._explain_kernel(x, target_action)

    def _explain_exact(self, x: np.ndarray, action: int) -> ExplanationResult:
        feature_count = self.feature_spec.feature_count
        if feature_count > EXACT_MARGINAL_MAX_FEATURES:
            raise ValueError(
                f"exact marginal Shapley enumerates 2^{feature_count} coalitions; "
                f"use method 'kernel' beyond {EXACT_MARGINAL_MAX_FEATURES} features"
            )
        characteristic = marginal_characteristic(
            self.policy,
            x,
            action,
            self.background,
            max_background=self.config.max_background,
            seed=self.config.seed,
        )
        values = ExactShapleyEstimator().shapley_values(characteristic)
        return self._result(
            x,
            action,
            values,
            baseline=characteristic.value(frozenset()),
            full=characteristic.value(frozenset(range(feature_count))),
            assumption=f"{MARGINAL_ASSUMPTION}; exact Shapley enumeration",
        )

    def _explain_kernel(self, x: np.ndarray, action: int) -> ExplanationResult:
        require_explain_dependencies("KernelSHAP baselines")
        import shap

        background = self.background
        assumption = f"{MARGINAL_ASSUMPTION}; KernelSHAP weighted least-squares approximation"
        if self.method == "cohort-kernel":
            assert self.cohort_key is not None
            background = cohort_background(
                self.background,
                x,
                self.cohort_key,
                min_size=self.config.cohort_min_size,
            )
            assumption += f"; cohort of {background.size} background states with key {int(self.cohort_key(x))}"
        rows = _subsample_rows(background.observations, self.config.max_background, self.config.seed)
        value_fn = q_value_function(self.policy, action)
        explainer = shap.KernelExplainer(value_fn, rows)
        shap_values = explainer.shap_values(x[None, :], nsamples=self.config.kernel_nsamples, silent=True)
        values = [float(value) for value in np.asarray(shap_values, dtype=np.float64).reshape(-1)]
        baseline = float(np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)[0])
        return self._result(
            x,
            action,
            values,
            baseline=baseline,
            full=float(value_fn(x[None, :])[0]),
            assumption=assumption,
        )

    def _result(
        self,
        x: np.ndarray,
        action: int,
        values: list[float],
        *,
        baseline: float,
        full: float,
        assumption: str,
    ) -> ExplanationResult:
        names = self.feature_spec.names
        return ExplanationResult(
            target="q_value",
            state=tuple(float(value) for value in x),
            action=action,
            baseline_value=float(baseline),
            full_value=float(full),
            attributions={name: float(value) for name, value in zip(names, values, strict=True)},
            feature_names=names,
            characteristic_assumption=assumption,
            semantics=ExplanationSemantics(
                target_quantity="fixed_action_q",
                coalition_semantics="cohort_background" if self.method == "cohort-kernel" else "marginal_background",
                value_model_source="base_q_function",
                shapley_estimator="exact" if self.method == "exact" else "kernel_shap",
            ),
        )


@dataclass(frozen=True, slots=True)
class TreeSurrogateFit:
    """Fidelity report for one fitted tree surrogate."""

    action: int
    fidelity_r2: float
    train_states: int
    test_states: int


class TreeSurrogateShapExplainer:
    """Fit a gradient-boosted surrogate of Q(., action) on rollouts, then TreeSHAP.

    Attributions explain the *surrogate's* prediction (efficiency holds against
    the surrogate output), so the held-out fidelity R^2 reported in the result
    qualifies how faithful they are to the true policy.
    """

    def __init__(
        self,
        policy: PolicyView,
        feature_spec: FeatureSpec,
        background: BackgroundDataset,
        *,
        config: ShapBaselineConfig | None = None,
        n_estimators: int = 200,
        max_depth: int = 3,
    ) -> None:
        require_explain_dependencies("Tree-surrogate SHAP")
        self.policy = policy
        self.feature_spec = feature_spec
        self.background = background
        self.config = config or ShapBaselineConfig()
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self._fits: dict[int, tuple[Any, TreeSurrogateFit]] = {}

    def explain_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        action: int | None = None,
    ) -> ExplanationResult:
        """Attribute the surrogate's Q(observation, action); greedy action by default."""

        import shap

        x = np.asarray(observation, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.feature_spec.feature_count:
            raise ValueError(f"expected {self.feature_spec.feature_count} features, got {x.shape[0]}")
        target_action = (
            greedy_action(self.policy, x, action_mask)
            if action is None
            else validate_action(self.policy, action_mask, action)
        )
        model, fit = self._surrogate_for(target_action)
        rows = _subsample_rows(self.background.observations, self.config.max_background, self.config.seed)
        explainer = shap.TreeExplainer(model, data=rows, feature_perturbation="interventional")
        shap_values = np.asarray(explainer.shap_values(x[None, :]), dtype=np.float64).reshape(-1)
        baseline = float(np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)[0])
        names = self.feature_spec.names
        return ExplanationResult(
            target="q_value",
            state=tuple(float(value) for value in x),
            action=target_action,
            baseline_value=baseline,
            full_value=float(model.predict(x[None, :])[0]),
            attributions={name: float(value) for name, value in zip(names, shap_values, strict=True)},
            feature_names=names,
            characteristic_assumption=(
                f"{MARGINAL_ASSUMPTION}; TreeSHAP over a gradient-boosted surrogate "
                f"(held-out fidelity R^2={fit.fidelity_r2:.3f} on {fit.test_states} states)"
            ),
            semantics=ExplanationSemantics(
                target_quantity="fixed_action_q",
                coalition_semantics="marginal_background",
                value_model_source="tree_surrogate",
                shapley_estimator="tree_shap",
            ),
        )

    def surrogate_fit(self, action: int) -> TreeSurrogateFit:
        """Return the fidelity report for one action's surrogate, fitting if needed."""

        return self._surrogate_for(int(action))[1]

    def _surrogate_for(self, action: int) -> tuple[Any, TreeSurrogateFit]:
        cached = self._fits.get(action)
        if cached is not None:
            return cached
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split

        features = np.asarray(self.background.observations, dtype=np.float32)
        targets = q_value_function(self.policy, action)(features)
        x_train, x_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=self.config.seed
        )
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.config.seed,
        )
        model.fit(x_train, y_train)
        fidelity = float(r2_score(y_test, model.predict(x_test))) if len(y_test) >= 2 else float("nan")
        fit = TreeSurrogateFit(
            action=action,
            fidelity_r2=fidelity,
            train_states=int(len(y_train)),
            test_states=int(len(y_test)),
        )
        self._fits[action] = (model, fit)
        return model, fit


def _subsample_rows(rows: np.ndarray, limit: int, seed: int) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    if array.shape[0] <= limit:
        return array.copy()
    rng = np.random.default_rng(seed)
    indices = rng.choice(array.shape[0], size=limit, replace=False)
    return array[indices]
