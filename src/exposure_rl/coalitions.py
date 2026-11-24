"""Exact coalition games for any number of features, without the dense ``2^n x points`` operator.

For an observation ``o`` and a coalition ``C``, conditioning on ``O_C = o_C`` keeps the support points
``p`` whose mismatch mask ``m_p = {i : p_i != o_i}`` avoids ``C``. Grouping point weights by mismatch
mask, every conditional sum over all ``2^n`` coalitions follows from one subset-sum (zeta) transform:

    A[m] = sum_{p : m_p = m} w_p f_p,     B[T] = sum_{m subset T} A[m],     S(C) = B[N \\ C],

which costs ``O(points + n 2^n)`` per observation instead of ``O(points 2^n)``. Used for Minesweeper 4x4
(16 cells, 65536 coalitions); for 9 cells it reproduces ``oracle.certainty_games`` exactly.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def subset_sums(a: np.ndarray, n: int) -> np.ndarray:
    """Zeta transform over the first axis (length ``2^n``): ``B[T] = sum_{m subset T} A[m]``."""

    b = a.copy()
    for i in range(n):
        view = b.reshape(1 << (n - 1 - i), 2, 1 << i, *b.shape[1:])
        view[:, 1] += view[:, 0]
    return b


def conditional_moments(points: np.ndarray, weights: np.ndarray, values: np.ndarray, observation) -> np.ndarray:
    """``E[f | O_C = o_C]`` for every coalition ``C`` (bitmask order) and every column of ``values``.

    ``points``: ``(P, n)`` support observations; ``weights``: ``(P,)``; ``values``: ``(P, k)``.
    Returns ``(2^n, k)``; coalitions with no consistent support point are NaN.
    """

    n = points.shape[1]
    mismatch = ((points != np.asarray(observation)[None]) << np.arange(n)).sum(1)
    full = (1 << n) - 1
    cols = [np.bincount(mismatch, weights=weights, minlength=1 << n)]
    cols += [np.bincount(mismatch, weights=weights * v, minlength=1 << n) for v in values.T]
    sums = subset_sums(np.stack(cols, 1), n)[full - np.arange(1 << n)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums[:, 1:] / sums[:, :1]


def certainty_games(points, weights, q, var, observation, own_q: float, own_var: float):
    """Exact ``(q_mean, v_mask, v_state)`` over all coalitions for one observation and action.

    ``q``/``var``: per support point, ``Q(p, a)`` and ``Var(G | p, a)``; the support must already be
    restricted to points where ``a`` is legal (action-legal view, ADR 0005). The grand coalition is
    the observation itself: ``v_mask(N) = 0``, ``v_state(N)`` = the agent's own outcome variance.
    """

    m = conditional_moments(points, weights, np.stack([q, q**2, var], 1), observation)
    q_mean = m[:, 0]
    v_mask = np.maximum(m[:, 1] - q_mean**2, 0.0)
    v_state = m[:, 2]
    q_mean[-1], v_mask[-1], v_state[-1] = own_q, 0.0, own_var
    return q_mean, v_mask, v_state


def shapley_weights(n: int) -> np.ndarray:
    s = np.arange(n)
    return np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in s])


def shapley(values: np.ndarray, n: int) -> np.ndarray:
    """Exact Shapley values of a game ``(..., 2^n)`` in bitmask order → ``(..., n)``."""

    coalitions = np.arange(1 << n)
    sizes = coalition_sizes(n)
    w = shapley_weights(n)
    out = []
    for i in range(n):
        without = coalitions[(coalitions >> i) & 1 == 0]
        out.append(np.sum(w[sizes[without]] * (values[..., without | (1 << i)] - values[..., without]), axis=-1))
    return np.stack(out, -1)


def uncertainty_reduction_shapley(values: np.ndarray, n: int) -> np.ndarray:
    """Shapley values of ``u(C) = v(empty) - v(C)``: positive = the feature makes the agent more certain."""

    return -shapley(values, n)


def leave_one_out(values: np.ndarray, n: int) -> np.ndarray:
    grand = (1 << n) - 1
    return np.stack([values[..., grand & ~(1 << i)] - values[..., grand] for i in range(n)], -1)


def coalition_sizes(n: int) -> np.ndarray:
    return np.array([c.bit_count() for c in range(1 << n)])


def completion_entropy(points: np.ndarray, weights: np.ndarray, observation) -> np.ndarray:
    """Exact ``H[O | O_C = o_C]`` (nats) over the support points consistent with ``o_C``, all coalitions.

    ``H = log W - E[log w]`` with ``W`` the consistent weight mass; the ground truth for the
    representation-entropy game (which completions of the visible cells remain possible).
    """

    n = points.shape[1]
    mismatch = ((points != np.asarray(observation)[None]) << np.arange(n)).sum(1)
    full = (1 << n) - 1
    w = weights / weights.sum()
    cols = np.stack(
        [
            np.bincount(mismatch, weights=w, minlength=1 << n),
            np.bincount(mismatch, weights=w * np.log(w), minlength=1 << n),
        ],
        1,
    )
    sums = subset_sums(cols, n)[full - np.arange(1 << n)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(sums[:, 0]) - sums[:, 1] / sums[:, 0]
