"""Agreement statistics between two feature-attribution vectors.

Attribution vectors are sparse: most cells of a board contribute nothing, so
exact ties are the common case rather than an edge case. Every rank statistic
here therefore uses *average ranks* — breaking ties by array position (what a
bare ``argsort`` does) invents an ordering among tied features and inflates
apparent agreement.

Deliberately pure NumPy. SciPy is only present via the ``explain`` extra
(as a transitive dependency of scikit-learn), and re-analysing saved
explanation documents must work on a base install.

Three different questions live here and should not be conflated. The first two
are about attributions; the third is about the game they are computed from.

* :func:`compare_attributions` — *do two methods agree?* Symmetric, scale-free,
  valid across method families that answer different questions.
* :func:`reference_error` — *how close is an approximation to its own exact
  answer?* Asymmetric and in value units, valid only within a family that
  shares one characteristic function (``kernel`` against ``exact``,
  ``fastshap`` against an exactly-estimated ``masked-q``).
* :func:`coalition_error` — *how close is one characteristic function to
  another?* Compares ``v`` itself over coalitions, before any Shapley transform
  is applied. This is the only statistic here that can separate a disagreement
  caused by two games being different from one caused by a learned game failing
  to reproduce the game it targets; the attribution-level statistics show both
  as "the vectors differ".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = [
    "DEFAULT_TOP_K_VALUES",
    "AgreementScores",
    "AttributionRegression",
    "CoalitionError",
    "FeatureAgreement",
    "ReferenceError",
    "attribution_regression",
    "average_ranks",
    "bootstrap_mean_ci",
    "coalition_error",
    "compare_attributions",
    "cosine_similarity",
    "efficiency_residual",
    "kendall_tau_b",
    "pairwise_agreement",
    "pearson",
    "per_feature_agreement",
    "reference_error",
    "sign_agreement",
    "spearman",
    "top_1_agreement",
    "top_k_jaccard",
]

# A single k is arbitrary. Sweeping a few keeps the top-weighted view honest:
# k=1 is what a reader takes off a heatmap, k=5 is a working shortlist.
DEFAULT_TOP_K_VALUES = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class AgreementScores:
    """Symmetric agreement between two attribution vectors over one state."""

    spearman: float
    kendall_tau_b: float
    top_k_jaccard: float
    sign_agreement: float
    cosine: float
    top_k: int
    pearson: float = math.nan
    top_1_agreement: float = math.nan
    # Jaccard at each of several k, keyed by k. A single k hides whether two
    # methods agree on the one strongest feature but not the shortlist, or the
    # reverse.
    top_k_jaccards: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttributionRegression:
    """Least-squares fit of one method's attributions on another's.

    Pooled over every (state, feature) cell, so this answers a different
    question from the per-state statistics in :class:`AgreementScores`: the
    ``slope`` says whether two methods are on the same *scale*, which
    correlation cannot. Estimators of one game should give ``slope ~ 1`` and
    ``intercept ~ 0``.

    ``r_squared`` is *not* a substitute for the mean per-state correlation.
    Writing ``phi_tj = mu_t + e_tj``, the pooled covariance is
    ``Cov(mu_A, mu_B) + E_t[Cov_j(e_A, e_B)]``; the first term is shared
    between-state scale that both methods track, and it inflates the pooled
    coefficient above the within-state agreement actually being asked about.
    Two methods disagreeing about every cell in every state can still pool to
    ``r ~ 0.95``.

    The slope is also *directional*, and not simply invertible: it regresses
    ``right`` on ``left``, so noise in ``left`` attenuates it toward zero
    (errors-in-variables). Read a slope below one as "``right`` is smaller than
    ``left`` **or** ``left`` is noisy", and disambiguate with ``r_squared``: a
    high ``r_squared`` with a slope of 0.5 is genuine shrinkage, a low one is
    mostly dilution.
    """

    slope: float
    intercept: float
    r_squared: float
    n: int


@dataclass(frozen=True, slots=True)
class FeatureAgreement:
    """Correlation of one feature's attributions across states.

    ``n_effective`` counts the states where the feature was not constant under
    both methods; a board cell that is occupied in most probes contributes
    little and says so rather than silently weakening the mean.
    """

    feature: str
    pearson: float
    spearman: float
    n_effective: int


@dataclass(frozen=True, slots=True)
class ReferenceError:
    """Error of an approximate attribution vector against an exact reference.

    Only meaningful when both vectors estimate the *same* characteristic
    function; comparing across families measures the gap between two different
    questions, not approximation quality.
    """

    mean_absolute_error: float
    max_absolute_error: float
    root_mean_squared_error: float
    relative_error: float
    spearman: float


@dataclass(frozen=True, slots=True)
class CoalitionError:
    """Distance between two characteristic functions over the same coalitions.

    Measured before any Shapley transform, so it answers a question the
    attribution statistics cannot: whether two methods disagree because their
    games differ or because one game fails to compute what it targets.

    ``normalized_mae`` is the field that carries the meaning. A raw error is
    uninterpretable without the spread of the thing being approximated — an MAE
    of 0.1 is negligible against a reference ranging over units and total
    against one ranging over tenths — so the error is divided by the
    reference's own standard deviation across coalitions. At 1.0 the error is
    the size of the entire signal.

    The two endpoints are kept separately because they diagnose different
    failures: ``empty_endpoint_error`` is the error at ``v(∅)``, where a learned
    game that has collapsed towards a constant shows up first, and
    ``full_endpoint_error`` is the error at ``v(N)``, which should be near zero
    for any game that reproduces the unmasked Q-value.
    """

    coalitions: int
    mean_absolute_error: float
    max_absolute_error: float
    root_mean_squared_error: float
    normalized_mae: float
    reference_spread: float
    empty_endpoint_error: float
    full_endpoint_error: float


def coalition_error(
    approximate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> CoalitionError:
    """Compare two coalition-value vectors evaluated over the same coalitions.

    Both arguments must be ordered identically, with the empty coalition first
    and the grand coalition last — the ordering :func:`itertools.combinations`
    produces when swept over increasing sizes.
    """

    estimate = _as_finite_vector(approximate, name="approximate")
    truth = _as_finite_vector(_match_length(approximate, reference, kind="coalition"), name="reference")
    if estimate.size == 0:
        raise ValueError("coalition vectors must be non-empty")
    difference = estimate - truth
    spread = float(truth.std())
    return CoalitionError(
        coalitions=int(estimate.size),
        mean_absolute_error=float(np.abs(difference).mean()),
        max_absolute_error=float(np.abs(difference).max()),
        root_mean_squared_error=float(np.sqrt(np.mean(difference**2))),
        # A constant reference carries no signal to be a fraction of, so the
        # normalisation is undefined rather than infinite. Callers report the
        # raw error in that case.
        normalized_mae=float(np.abs(difference).mean() / spread) if spread > 0 else math.nan,
        reference_spread=spread,
        empty_endpoint_error=float(difference[0]),
        full_endpoint_error=float(difference[-1]),
    )


def average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return 1-based ranks of ``values``, tied entries sharing their mean rank.

    Ties get the average of the positions they span, so a vector of exact
    zeros contributes no ordering information instead of a spurious one.
    """

    array = _as_finite_vector(values, name="values")
    count = array.size
    order = np.argsort(array, kind="stable")
    ordered = array[order]
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties.

    Returns NaN when either vector is entirely tied — the correlation is
    genuinely undefined there, and NaN keeps it out of downstream means
    instead of pulling them toward a fabricated zero.
    """

    left_ranks = average_ranks(left)
    right_ranks = average_ranks(_match_length(left, right))
    return _pearson(left_ranks, right_ranks)


def pearson(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Linear correlation of two attribution vectors.

    The coefficient of the least-squares fit of one on the other: ``r**2`` is
    that regression's coefficient of determination. Unlike :func:`spearman` it
    is sensitive to magnitude, so it separates two methods that order features
    identically but scale them differently.

    Returns NaN when either vector is constant, matching :func:`spearman`.
    """

    return _pearson(_as_finite_vector(left, name="left"), _as_finite_vector(_match_length(left, right), name="right"))


def kendall_tau_b(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Kendall's tau-b, the tie-corrected variant.

    More conservative than :func:`spearman` on sparse attributions because
    tied pairs are removed from the normaliser rather than ranked.
    """

    first = _as_finite_vector(left, name="left")
    second = _as_finite_vector(_match_length(left, right), name="right")
    if first.size < 2:
        return math.nan

    concordant_minus_discordant = 0
    for i, j in combinations(range(first.size), 2):
        sign = np.sign(first[i] - first[j]) * np.sign(second[i] - second[j])
        concordant_minus_discordant += int(sign)

    total_pairs = first.size * (first.size - 1) / 2
    denominator = math.sqrt((total_pairs - _tied_pairs(first)) * (total_pairs - _tied_pairs(second)))
    if denominator == 0:
        return math.nan
    return float(concordant_minus_discordant / denominator)


def top_k_jaccard(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
) -> float:
    """Jaccard overlap of the strongest-magnitude features of each vector.

    Selection is by magnitude *threshold* rather than by position, so features
    tied with the k-th strongest are all included. This keeps the statistic
    from depending on arbitrary ordering among equal attributions, at the cost
    of occasionally selecting more than ``top_k`` features.
    """

    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    first = _as_finite_vector(left, name="left")
    second = _as_finite_vector(_match_length(left, right), name="right")
    left_set = _top_magnitude_indices(first, top_k)
    right_set = _top_magnitude_indices(second, top_k)
    union = left_set | right_set
    if not union:
        return math.nan
    return len(left_set & right_set) / len(union)


def top_1_agreement(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Whether both vectors name the same strongest-magnitude feature.

    The strictest useful ranking statistic, and the one a reader applies when
    they look at a board heatmap and ask which cell mattered most. Ties are
    respected: agreement means the two top sets intersect, so a vector with two
    equal maxima is not forced to pick one arbitrarily.
    """

    if not _has_signal(left, right):
        return math.nan
    # Deliberately a float, not a bool: the study aggregator averages numeric
    # columns and excludes booleans, so a bool here would silently vanish.
    return float(top_k_jaccard(left, right, top_k=1) > 0.0)


def _has_signal(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> bool:
    first = _as_finite_vector(left, name="left")
    second = _as_finite_vector(_match_length(left, right), name="right")
    return bool(np.any(first != 0.0) and np.any(second != 0.0))


def sign_agreement(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 0.0,
) -> float:
    """Fraction of features on which both vectors agree about direction.

    Magnitudes within ``tolerance`` of zero count as "no effect", so a feature
    both methods consider irrelevant counts as agreement.
    """

    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    first = _as_finite_vector(left, name="left")
    second = _as_finite_vector(_match_length(left, right), name="right")
    if first.size == 0:
        return math.nan
    left_sign = np.where(np.abs(first) <= tolerance, 0.0, np.sign(first))
    right_sign = np.where(np.abs(second) <= tolerance, 0.0, np.sign(second))
    return float(np.mean(left_sign == right_sign))


def cosine_similarity(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """Cosine similarity, sensitive to magnitude profile but not to overall scale."""

    first = _as_finite_vector(left, name="left")
    second = _as_finite_vector(_match_length(left, right), name="right")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return math.nan
    return float(first @ second / denominator)


def compare_attributions(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
    sign_tolerance: float = 0.0,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
) -> AgreementScores:
    """Bundle every symmetric agreement statistic for one pair of vectors.

    Computed *within* one state. Aggregating these over states is the primary
    reading of "do these methods agree": it conditions away the between-state
    variation in attribution scale that a pooled statistic would absorb (see
    :class:`AttributionRegression`).
    """

    return AgreementScores(
        spearman=spearman(left, right),
        pearson=pearson(left, right),
        kendall_tau_b=kendall_tau_b(left, right),
        top_k_jaccard=top_k_jaccard(left, right, top_k=top_k),
        top_k_jaccards={int(k): top_k_jaccard(left, right, top_k=int(k)) for k in top_k_values},
        top_1_agreement=top_1_agreement(left, right),
        sign_agreement=sign_agreement(left, right, tolerance=sign_tolerance),
        cosine=cosine_similarity(left, right),
        top_k=int(top_k),
    )


def attribution_regression(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> AttributionRegression:
    """Fit ``right = intercept + slope * left`` over pooled attribution cells.

    Read the slope, not the fit quality: see :class:`AttributionRegression` for
    why ``r_squared`` pooled across states overstates within-state agreement.
    """

    x = _as_finite_vector(left, name="left")
    y = _as_finite_vector(_match_length(left, right), name="right")
    if x.size < 2:
        return AttributionRegression(math.nan, math.nan, math.nan, int(x.size))
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(x_centered @ x_centered)
    if denominator == 0:
        # A constant predictor supports no slope; the intercept is the mean.
        return AttributionRegression(math.nan, float(y.mean()), math.nan, int(x.size))
    slope = float(x_centered @ y_centered / denominator)
    intercept = float(y.mean() - slope * x.mean())
    correlation = _pearson(x, y)
    r_squared = math.nan if math.isnan(correlation) else float(correlation**2)
    return AttributionRegression(slope, intercept, r_squared, int(x.size))


def per_feature_agreement(
    left: Sequence[Mapping[str, float]],
    right: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
) -> list[FeatureAgreement]:
    """Correlate each feature's attributions across states.

    Answers a third question, distinct from both the per-state and the pooled
    readings: does *this* feature play the same role across positions? On a
    board that is one coefficient per cell.
    """

    if len(left) != len(right):
        raise ValueError(f"attribution sequences must have equal length, got {len(left)} and {len(right)}")
    scores: list[FeatureAgreement] = []
    for name in feature_names:
        first = np.asarray([float(state[name]) for state in left], dtype=np.float64)
        second = np.asarray([float(state[name]) for state in right], dtype=np.float64)
        effective = int(np.count_nonzero((first != 0.0) | (second != 0.0)))
        scores.append(
            FeatureAgreement(
                feature=name,
                pearson=_pearson(first, second),
                spearman=spearman(first, second) if first.size >= 2 else math.nan,
                n_effective=effective,
            )
        )
    return scores


def pairwise_agreement(
    attributions: Mapping[str, Mapping[str, float]],
    feature_names: Sequence[str],
    *,
    top_k: int = 5,
    sign_tolerance: float = 0.0,
) -> dict[tuple[str, str], AgreementScores]:
    """Compare every unordered pair of methods over a shared feature ordering.

    ``attributions`` maps method name to that method's ``feature -> value``
    mapping; the shared ``feature_names`` ordering is what makes the vectors
    comparable.
    """

    names = tuple(feature_names)
    vectors = {
        method: np.asarray([float(values[name]) for name in names], dtype=np.float64)
        for method, values in attributions.items()
    }
    return {
        (first, second): compare_attributions(
            vectors[first], vectors[second], top_k=top_k, sign_tolerance=sign_tolerance
        )
        for first, second in combinations(sorted(vectors), 2)
    }


def reference_error(
    approximate: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> ReferenceError:
    """Error of ``approximate`` against an exact ``reference`` for the same game.

    ``relative_error`` normalises by the reference's total attribution mass, so
    it stays comparable across states whose Q-values differ in scale.
    """

    estimate = _as_finite_vector(approximate, name="approximate")
    truth = _as_finite_vector(_match_length(approximate, reference), name="reference")
    difference = estimate - truth
    mass = float(np.abs(truth).sum())
    return ReferenceError(
        mean_absolute_error=float(np.abs(difference).mean()) if difference.size else math.nan,
        max_absolute_error=float(np.abs(difference).max()) if difference.size else math.nan,
        root_mean_squared_error=float(np.sqrt(np.mean(difference**2))) if difference.size else math.nan,
        relative_error=float(np.abs(difference).sum() / mass) if mass > 0 else math.nan,
        spearman=spearman(estimate, truth),
    )


def efficiency_residual(
    attributions: Sequence[float] | Mapping[str, float],
    *,
    baseline_value: float,
    full_value: float,
) -> float:
    """Return ``sum(attributions) - (full_value - baseline_value)``.

    Shapley values satisfy this identity exactly; a large residual means the
    estimator did not converge, and is the cheapest sanity check available.
    """

    values = list(attributions.values()) if isinstance(attributions, Mapping) else list(attributions)
    total = float(np.asarray(values, dtype=np.float64).sum())
    return total - (float(full_value) - float(baseline_value))


def bootstrap_mean_ci(
    samples: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean, returning ``(mean, low, high)``.

    NaN samples — degenerate correlations from fully tied vectors — are
    dropped rather than propagated, so the interval reflects the states where
    the statistic was actually defined.
    """

    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    array = np.asarray(list(samples), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    if array.size == 1:
        single = float(array[0])
        return single, single, single

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(array.mean()), float(low), float(high)


def _as_finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; got non-finite attribution values")
    return array


def _match_length(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    kind: str = "attribution",
) -> Sequence[float] | np.ndarray:
    left_size = np.asarray(left).size
    right_size = np.asarray(right).size
    if left_size != right_size:
        raise ValueError(f"{kind} vectors must have equal length, got {left_size} and {right_size}")
    return right


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return math.nan
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0:
        return math.nan
    return float(left_centered @ right_centered / denominator)


def _tied_pairs(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    return float(np.sum(counts * (counts - 1) / 2))


def _top_magnitude_indices(values: np.ndarray, top_k: int) -> set[int]:
    magnitudes = np.abs(values)
    positive = magnitudes[magnitudes > 0]
    if positive.size == 0:
        return set()
    keep = min(top_k, positive.size)
    threshold = np.sort(positive)[::-1][keep - 1]
    return {int(index) for index in np.flatnonzero(magnitudes >= threshold)}
