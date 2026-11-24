"""Deletion and insertion curves measuring how faithful an attribution is.

Agreement between methods says nothing about correctness — two methods can
agree and both be wrong. These curves test an attribution against the model
itself: rank features by attributed importance, then perturb them in that
order and watch the value move. An attribution that identifies genuinely
important features makes the value collapse quickly when they are deleted and
recover quickly when they are inserted.

Both curves are computed against a :class:`CharacteristicFunction` — the same
``v(C)`` object the Shapley estimators consume, so the value semantics are
exactly the ones the explanation was produced under.

An important caveat when reading the results: the *choice* of characteristic
function is itself a modelling assumption. Scoring every method with the
marginal ``v(C)`` measures faithfulness to the interventional game, which is
the game the marginal family optimises — that comparison is biased in their
favour. Score each family under both evaluators (see
:func:`aopc_summary`) and treat a method that wins under both as the only
robust conclusion.

Ranking ties are broken by feature index. A curve needs a total order, and
tied features are by definition interchangeable in attributed importance, so
the choice only reorders steps inside a tie group.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from xrl.explainability.core import CharacteristicFunction

__all__ = [
    "DELETION",
    "INSERTION",
    "PerturbationCurve",
    "aopc_summary",
    "attribution_ranking",
    "control_curves",
    "deletion_curve",
    "insertion_curve",
    "perturbation_curves",
    "random_ranking",
    "summarize_curves",
]

DELETION = "deletion"
INSERTION = "insertion"

# Below this, the full-vs-empty value span is noise and normalising by it
# would manufacture huge ratios out of rounding error.
_MIN_SPAN = 1e-9


@dataclass(frozen=True, slots=True)
class PerturbationCurve:
    """One deletion or insertion curve over a single explained state.

    ``values`` is the signed trajectory of ``v`` — plot that. ``aopc`` reduces
    it to *absolute* deviation from the starting point: deletion measures how
    far the value moves away from ``v(full)``, insertion how far it moves away
    from ``v(empty)``. Larger is better under both modes.

    The absolute value is a deliberate departure from the textbook AOPC, which
    assumes a positive target (a class probability) and sums signed drops. A
    Q-value attribution is signed, and features are ranked by ``|φ|``, so a
    feature that strongly pushes Q *down* is correctly ranked important yet
    *raises* the value when deleted. Summing signed drops would score a method
    as worse than random for finding it. Faithfulness asks whether the model
    depends on those features, and dependence has no direction.

    ``normalized_aopc`` divides by ``|v(full) - v(empty)|``, making states with
    different Q-scales comparable; it is NaN when that span is ~0, since a
    state whose value never moves cannot rank anything.
    """

    mode: str
    steps: tuple[int, ...]
    values: tuple[float, ...]
    aopc: float
    normalized_aopc: float
    reference_span: float

    @property
    def feature_count(self) -> int:
        return max(self.steps)


def attribution_ranking(
    attributions: Sequence[float] | Mapping[str, float],
    feature_names: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Order feature indices by descending attribution magnitude.

    Ties are broken by ascending feature index so the ordering is
    reproducible; pass ``feature_names`` to fix the column order when
    ``attributions`` is a mapping.
    """

    values = _attribution_vector(attributions, feature_names)
    magnitudes = np.abs(values)
    # Sort by magnitude descending, index ascending; lexsort applies the last
    # key first, so index is the tie-breaker.
    order = np.lexsort((np.arange(magnitudes.size), -magnitudes))
    return tuple(int(index) for index in order)


def random_ranking(feature_count: int, *, seed: int = 0) -> tuple[int, ...]:
    """Return a random feature ordering, the control an explainer must beat.

    A method whose deletion curve is no steeper than this one has not
    identified anything about the model.
    """

    if feature_count < 0:
        raise ValueError(f"feature_count must be non-negative, got {feature_count}")
    rng = np.random.default_rng(seed)
    return tuple(int(index) for index in rng.permutation(feature_count))


def deletion_curve(
    characteristic: CharacteristicFunction,
    ranking: Sequence[int],
    *,
    steps: Sequence[int] | None = None,
) -> PerturbationCurve:
    """Remove features in ``ranking`` order, recording ``v`` after each removal.

    Step ``k`` evaluates the coalition holding every feature *except* the top
    ``k`` ranked ones, so the curve starts at ``v(full)`` and ends at
    ``v(empty)``.
    """

    order, ladder = _prepare(characteristic, ranking, steps)
    everything = frozenset(range(characteristic.feature_count))
    values = tuple(float(characteristic.value(everything - frozenset(order[:k]))) for k in ladder)
    full = float(characteristic.value(everything))
    empty = float(characteristic.value(frozenset()))
    return _curve(DELETION, ladder, values, [abs(full - value) for value in values], full - empty)


def insertion_curve(
    characteristic: CharacteristicFunction,
    ranking: Sequence[int],
    *,
    steps: Sequence[int] | None = None,
) -> PerturbationCurve:
    """Reveal features in ``ranking`` order, recording ``v`` after each addition.

    Step ``k`` evaluates the coalition holding only the top ``k`` ranked
    features, so the curve starts at ``v(empty)`` and ends at ``v(full)``.
    """

    order, ladder = _prepare(characteristic, ranking, steps)
    values = tuple(float(characteristic.value(frozenset(order[:k]))) for k in ladder)
    full = float(characteristic.value(frozenset(range(characteristic.feature_count))))
    empty = float(characteristic.value(frozenset()))
    return _curve(INSERTION, ladder, values, [abs(value - empty) for value in values], full - empty)


def perturbation_curves(
    characteristic: CharacteristicFunction,
    attributions: Sequence[float] | Mapping[str, float],
    *,
    feature_names: Sequence[str] | None = None,
    steps: Sequence[int] | None = None,
) -> dict[str, PerturbationCurve]:
    """Compute both curves for one attribution vector, keyed by mode."""

    ranking = attribution_ranking(attributions, feature_names)
    return {
        DELETION: deletion_curve(characteristic, ranking, steps=steps),
        INSERTION: insertion_curve(characteristic, ranking, steps=steps),
    }


def control_curves(
    characteristic: CharacteristicFunction,
    *,
    steps: Sequence[int] | None = None,
    random_seed: int = 0,
) -> dict[str, PerturbationCurve]:
    """Curves for a random feature ordering — the baseline every method must beat.

    The control depends only on the state and the seed, not on any method, so
    a caller scoring several methods over one state should compute it once and
    pass it to :func:`summarize_curves` rather than paying for it per method.
    """

    ranking = random_ranking(characteristic.feature_count, seed=random_seed)
    return {
        DELETION: deletion_curve(characteristic, ranking, steps=steps),
        INSERTION: insertion_curve(characteristic, ranking, steps=steps),
    }


def summarize_curves(
    curves: Mapping[str, PerturbationCurve],
    control: Mapping[str, PerturbationCurve],
) -> dict[str, float]:
    """Reduce a method's curves and the random control to scalar AOPC metrics.

    ``deletion_lift`` and ``insertion_lift`` are the margins over the control.
    They are the numbers worth reporting: a bare AOPC mixes the explainer's
    quality with how much signal the state carries at all, while the lift
    isolates the explainer.
    """

    return {
        "deletion_aopc": curves[DELETION].aopc,
        "insertion_aopc": curves[INSERTION].aopc,
        "deletion_aopc_normalized": curves[DELETION].normalized_aopc,
        "insertion_aopc_normalized": curves[INSERTION].normalized_aopc,
        "random_deletion_aopc": control[DELETION].aopc,
        "random_insertion_aopc": control[INSERTION].aopc,
        "deletion_lift": curves[DELETION].aopc - control[DELETION].aopc,
        "insertion_lift": curves[INSERTION].aopc - control[INSERTION].aopc,
        "reference_span": curves[DELETION].reference_span,
    }


def aopc_summary(
    characteristic: CharacteristicFunction,
    attributions: Sequence[float] | Mapping[str, float],
    *,
    feature_names: Sequence[str] | None = None,
    steps: Sequence[int] | None = None,
    random_seed: int = 0,
) -> dict[str, float]:
    """Summarise faithfulness as AOPC values plus a random-ordering control."""

    curves = perturbation_curves(characteristic, attributions, feature_names=feature_names, steps=steps)
    control = control_curves(characteristic, steps=steps, random_seed=random_seed)
    return summarize_curves(curves, control)


def _prepare(
    characteristic: CharacteristicFunction,
    ranking: Sequence[int],
    steps: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    count = int(characteristic.feature_count)
    order = tuple(int(index) for index in ranking)
    if len(order) != count or set(order) != set(range(count)):
        raise ValueError(f"ranking must be a permutation of the {count} feature indices")
    ladder = tuple(range(count + 1)) if steps is None else tuple(int(step) for step in steps)
    if any(step < 0 or step > count for step in ladder):
        raise ValueError(f"steps must lie within [0, {count}]")
    if not ladder:
        raise ValueError("steps must not be empty")
    return order, ladder


def _curve(
    mode: str,
    ladder: tuple[int, ...],
    values: tuple[float, ...],
    deviations: Iterable[float],
    span: float,
) -> PerturbationCurve:
    aopc = float(np.mean(list(deviations)))
    normalized = aopc / abs(span) if abs(span) > _MIN_SPAN else math.nan
    return PerturbationCurve(
        mode=mode,
        steps=ladder,
        values=values,
        aopc=aopc,
        normalized_aopc=float(normalized),
        reference_span=float(span),
    )


def _attribution_vector(
    attributions: Sequence[float] | Mapping[str, float],
    feature_names: Sequence[str] | None,
) -> np.ndarray:
    if isinstance(attributions, Mapping):
        if feature_names is None:
            raise ValueError("feature_names is required when attributions is a mapping")
        values = [float(attributions[name]) for name in feature_names]
    else:
        values = [float(value) for value in attributions]
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("attributions must be finite")
    return array
