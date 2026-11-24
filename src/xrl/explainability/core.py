"""Core SVERL attribution interfaces."""

from __future__ import annotations

import math
import sys
import sysconfig
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations
from threading import RLock
from typing import Any, Protocol

import numpy as np

from xrl.training.interfaces import PolicyView  # noqa: F401 - re-exported from xrl.explainability.core

Coalition = frozenset[int]


@cache
def all_coalitions(feature_count: int) -> tuple[Coalition, ...]:
    """Every coalition over ``feature_count`` features, empty first, grand last.

    The order is part of the contract: callers that compare two characteristic
    functions position-wise rely on index 0 being ``v(∅)`` and index -1 being
    ``v(N)``, the two endpoints that diagnose different failures.
    """

    if feature_count < 0:
        raise ValueError(f"feature_count must be non-negative, got {feature_count}")
    return tuple(
        frozenset(index for index in range(feature_count) if bits & (1 << index)) for bits in range(1 << feature_count)
    )


@cache
def shapley_operator(feature_count: int) -> np.ndarray:
    """The exact Shapley map as a fixed ``(feature_count, 2**feature_count)`` matrix.

    Shapley values are a *linear* functional of the characteristic function:

        phi_i = sum over C not containing i of w(|C|) [v(C u {i}) - v(C)],
        w(k) = k! (n - k - 1)! / n!

    The weights depend only on sizes, so the map is the same matrix for every
    game over the same feature count. Building it once turns each explanation
    into one matrix-vector product instead of an O(n 2^n) Python loop over
    coalition pairs. Columns follow :func:`all_coalitions` order, so index 0 is
    ``v(the empty coalition)`` and index -1 is ``v(N)``.
    """

    if feature_count < 0:
        raise ValueError(f"feature_count must be non-negative, got {feature_count}")
    size = 1 << feature_count
    operator = np.zeros((feature_count, size), dtype=np.float64)
    denominator = math.factorial(feature_count)
    for feature in range(feature_count):
        bit = 1 << feature
        for bits in range(size):
            if bits & bit:
                continue
            weight = math.factorial(bits.bit_count()) * math.factorial(feature_count - bits.bit_count() - 1)
            weight /= denominator
            operator[feature, bits | bit] += weight
            operator[feature, bits] -= weight
    # The matrix is cached and shared by every caller, so it is handed out
    # read-only: one in-place edit would silently corrupt every later
    # explanation over the same feature count.
    operator.flags.writeable = False
    return operator


def free_threading_enabled() -> bool:
    """Return whether this CPython runtime is running without the GIL."""

    if not bool(sysconfig.get_config_var("Py_GIL_DISABLED")):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if callable(is_gil_enabled):
        return not bool(is_gil_enabled())
    return True


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Feature metadata and extraction for attribution."""

    names: tuple[str, ...]
    extractor: Callable[[Any], Sequence[Any]] | None = None
    display_metadata: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        if self.display_metadata and len(self.display_metadata) != len(self.names):
            raise ValueError("display metadata must match the number of features")

    @property
    def feature_count(self) -> int:
        return len(self.names)

    def values(self, item: Any) -> tuple[Any, ...]:
        values = np.asarray(item).reshape(-1).tolist() if self.extractor is None else list(self.extractor(item))
        if len(values) != len(self.names):
            raise ValueError(f"expected {len(self.names)} feature values, got {len(values)}")
        return tuple(values)

    def matches(self, left: Sequence[Any], right: Sequence[Any], coalition: Iterable[int]) -> bool:
        return all(left[index] == right[index] for index in coalition)


class CharacteristicFunction(Protocol):
    """A coalitional game over state features."""

    feature_count: int

    def value(self, coalition: Coalition | Iterable[int]) -> float:
        """Return the characteristic value for a coalition."""


class StateDistribution(Protocol):
    """Distribution over states used to represent unknown features."""

    def conditional_expectation(
        self,
        target_features: Sequence[Any],
        coalition: Coalition,
        value_fn: Callable[[Any], float],
    ) -> float:
        """Return E[value_fn(S) | S_C = target_C]."""


@dataclass(slots=True)
class CachedCharacteristicFunction:
    """Characteristic function wrapper with coalition-value caching."""

    feature_count: int
    evaluator: Callable[[Coalition], float]
    _cache: dict[Coalition, float] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def value(self, coalition: Coalition | Iterable[int]) -> float:
        key = frozenset(coalition)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        value = float(self.evaluator(key))
        with self._lock:
            return self._cache.setdefault(key, value)


@dataclass(frozen=True, slots=True)
class ExplanationSemantics:
    """Machine-readable meaning of an explanation and its estimator."""

    target_quantity: str = "unspecified"
    coalition_semantics: str = "unspecified"
    value_model_source: str = "unspecified"
    shapley_estimator: str = "unspecified"


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    """Local feature-attribution result for one explanation target."""

    target: str
    state: Any
    action: int | None
    baseline_value: float
    full_value: float
    attributions: dict[str, float]
    feature_names: tuple[str, ...]
    characteristic_assumption: str
    semantics: ExplanationSemantics = field(default_factory=ExplanationSemantics)


class ShapleyEstimator(Protocol):
    """Anything that maps a characteristic function to per-feature Shapley values."""

    def shapley_values(self, characteristic: CharacteristicFunction) -> list[float]:
        """Return one Shapley value per feature."""


class ExactShapleyEstimator:
    """Exact Shapley value estimator over all feature coalitions."""

    def __init__(self, *, parallel: bool | None = None, max_workers: int | None = None) -> None:
        self.parallel = free_threading_enabled() if parallel is None else bool(parallel)
        self.max_workers = max_workers

    def shapley_values(self, characteristic: CharacteristicFunction) -> list[float]:
        """Return one Shapley value per feature.

        The characteristic function is evaluated once per coalition and the
        Shapley map is then applied as a matrix (see :func:`shapley_operator`).
        The work that is worth parallelising is the evaluation -- a coalition
        value may cost a network query or a game-tree expansion -- not the
        weighted sum, which is now a single matrix-vector product.
        """

        n = characteristic.feature_count
        if n < 1:
            return []

        coalitions = all_coalitions(n)
        if self.parallel and n > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                values = list(executor.map(characteristic.value, coalitions))
        else:
            values = [characteristic.value(coalition) for coalition in coalitions]
        return [float(value) for value in shapley_operator(n) @ np.asarray(values, dtype=np.float64)]


class SampledShapleyEstimator:
    """Monte-Carlo permutation Shapley estimator for larger feature counts.

    Samples random feature permutations and accumulates each feature's
    marginal contribution as the coalition grows along the permutation. Every
    sampled permutation is also evaluated reversed (antithetic pairing), which
    cancels much of the positional variance. Per-permutation contributions
    telescope, so the efficiency identity ``sum(values) == v(full) - v(empty)``
    holds exactly. Wrap expensive characteristics in
    :class:`CachedCharacteristicFunction` — permutations revisit coalitions.
    """

    def __init__(self, *, num_permutations: int = 128, seed: int = 0) -> None:
        if num_permutations < 1:
            raise ValueError(f"num_permutations must be positive, got {num_permutations}")
        self.num_permutations = int(num_permutations)
        self.seed = int(seed)

    def shapley_values(self, characteristic: CharacteristicFunction) -> list[float]:
        n = characteristic.feature_count
        if n < 1:
            return []
        rng = np.random.default_rng(self.seed)
        totals: np.ndarray = np.zeros(n, dtype=np.float64)
        for _ in range(self.num_permutations):
            permutation = rng.permutation(n)
            for ordering in (permutation, permutation[::-1]):
                previous = characteristic.value(frozenset())
                members: list[int] = []
                for feature in ordering:
                    members.append(int(feature))
                    current = characteristic.value(frozenset(members))
                    totals[int(feature)] += current - previous
                    previous = current
        return [float(value) for value in totals / (2 * self.num_permutations)]


class KernelShapleyEstimator:
    """KernelSHAP over any characteristic function, in pure NumPy.

    Shapley values are the solution of a weighted least-squares problem
    (Lundberg & Lee, 2017)::

        min_phi  sum_S pi(S) [v(S) - v(empty) - sum_{i in S} phi_i]^2
        s.t.     sum_i phi_i = v(full) - v(empty)

    with the Shapley kernel ``pi(S) = (n-1) / (C(n,|S|) * |S| * (n-|S|))``.
    Nothing in that program refers to how ``v`` is defined — it consumes ``v``
    as an oracle — which is why this belongs beside the exact and permutation
    estimators rather than inside any one explainer. ``shap.KernelExplainer``
    performs its own background substitution and so is bound to the marginal
    coalition semantics; this is not.

    The efficiency constraint is *substituted*, not penalised: one coefficient
    is eliminated and recovered from the constraint, so
    ``sum(values) == v(full) - v(empty)`` holds exactly at every budget.

    When the budget covers every proper non-empty coalition the estimator
    enumerates them with their exact kernel weights, which makes it exact
    rather than merely consistent. Otherwise coalitions are sampled from the
    kernel-induced size distribution ``p(s) proportional to 1/(s(n-s))``, whose
    symmetry under ``s -> n-s`` is what lets each draw be paired with its
    complement at no bias: complementary pairs are the standard variance
    reduction for KernelSHAP (Covert & Lee, 2021).

    Wrap expensive characteristics in :class:`CachedCharacteristicFunction` —
    sampling revisits coalitions.
    """

    def __init__(self, *, num_coalitions: int = 512, seed: int = 0, paired: bool = True) -> None:
        if num_coalitions < 1:
            raise ValueError(f"num_coalitions must be positive, got {num_coalitions}")
        self.num_coalitions = int(num_coalitions)
        self.seed = int(seed)
        self.paired = bool(paired)

    def shapley_values(self, characteristic: CharacteristicFunction) -> list[float]:
        n = characteristic.feature_count
        if n < 1:
            return []
        empty = characteristic.value(frozenset())
        total = characteristic.value(frozenset(range(n))) - empty
        if n == 1:
            # The constraint alone determines the single value.
            return [float(total)]

        masks, weights = self._coalitions(n)
        targets = np.array(
            [characteristic.value(frozenset(np.flatnonzero(mask).tolist())) - empty for mask in masks],
            dtype=np.float64,
        )
        return self._solve(masks, weights, targets, total)

    def _coalitions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return 0/1 coalition rows and their regression weights."""

        if 2**n - 2 <= self.num_coalitions:
            return _enumerated_coalitions(n)
        return self._sampled_coalitions(n)

    def _sampled_coalitions(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        sizes = np.arange(1, n)
        # p(s) is the total kernel mass on coalitions of size s, normalised.
        size_probabilities = 1.0 / (sizes * (n - sizes))
        size_probabilities /= size_probabilities.sum()

        draws = self.num_coalitions // 2 if self.paired else self.num_coalitions
        rows: list[tuple[int, ...]] = []
        for size in rng.choice(sizes, size=max(draws, 1), p=size_probabilities):
            members = rng.choice(n, size=int(size), replace=False)
            mask = np.zeros(n, dtype=np.int8)
            mask[members] = 1
            rows.append(tuple(int(value) for value in mask))
            if self.paired:
                rows.append(tuple(1 - int(value) for value in mask))

        # Sampling was already proportional to the kernel, so each draw carries
        # equal weight; identical draws collapse into one row with their count.
        counts: dict[tuple[int, ...], int] = {}
        for row in rows:
            counts[row] = counts.get(row, 0) + 1
        ordered = sorted(counts)
        masks = np.array(ordered, dtype=np.float64)
        weights = np.array([counts[row] for row in ordered], dtype=np.float64)
        return masks, weights

    @staticmethod
    def _solve(masks: np.ndarray, weights: np.ndarray, targets: np.ndarray, total: float) -> list[float]:
        """Weighted least squares with the efficiency constraint substituted in."""

        last = masks[:, -1]
        design = masks[:, :-1] - last[:, None]
        adjusted = targets - last * total
        scale = np.sqrt(weights)[:, None]
        solution, *_ = np.linalg.lstsq(design * scale, adjusted * scale[:, 0], rcond=None)
        values = np.empty(masks.shape[1], dtype=np.float64)
        values[:-1] = solution
        values[-1] = total - solution.sum()
        return [float(value) for value in values]


def _enumerated_coalitions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every proper non-empty coalition with its exact Shapley-kernel weight.

    With this basis the weighted least-squares solution *is* the Shapley value
    vector, so the estimator is exact rather than consistent.
    """

    rows: list[np.ndarray] = []
    weights: list[float] = []
    features = tuple(range(n))
    for size in range(1, n):
        weight = (n - 1) / (math.comb(n, size) * size * (n - size))
        for subset in combinations(features, size):
            mask = np.zeros(n, dtype=np.float64)
            mask[list(subset)] = 1.0
            rows.append(mask)
            weights.append(weight)
    return np.array(rows, dtype=np.float64), np.array(weights, dtype=np.float64)


class LocalExplainer(Protocol):
    """Behaviour, outcome, and prediction explanation surface."""

    features: FeatureSpec

    def explain_behavior(self, state: Any, *, action: int) -> ExplanationResult:
        """Explain an action probability."""

    def explain_outcome(self, state: Any) -> ExplanationResult:
        """Explain expected return through behaviour."""

    def explain_prediction(self, state: Any) -> ExplanationResult:
        """Explain a value estimate."""


class SVERLExplainer(LocalExplainer, Protocol):
    """Backward-compatible name for SVERL-style local explainers."""


def validate_action(
    action: int,
    action_mask: np.ndarray | Iterable[int] | None = None,
    *,
    num_actions: int,
    label: str = "action",
) -> int:
    """Return one in-range, legal action, or raise a stable validation error.

    The fixed-action game ``v_a(S) = E[Q(x_S, a)]`` is only well posed for an
    action the state permits: no transition is defined for an illegal one and no
    TD target ever visits its Q-value, so every coalition difference would read
    an unconstrained network output. See ADR 0024.

    The three checks are ordered by what each needs. The range check types an
    index into the value model's output and is answerable without the state; the
    mask length binds the mask's index set to the model's; legality is a fact
    about the state. ``action_mask`` may be ``None`` for callers that hold no
    mask, which skips the last two.
    """

    if not isinstance(action, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, got {action!r}")
    target = int(action)
    if not 0 <= target < int(num_actions):
        raise ValueError(f"{label} must be in [0, {int(num_actions)}), got {target}")
    if action_mask is None:
        return target
    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    if mask.shape[0] != int(num_actions):
        raise ValueError(f"expected an action mask of length {int(num_actions)}, got {mask.shape[0]}")
    if not bool(mask[target]):
        raise ValueError(f"{label} {target} is illegal in the explained state")
    return target
