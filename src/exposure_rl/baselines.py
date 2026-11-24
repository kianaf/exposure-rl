"""Monte Carlo certainty explainers on the agent's own full-information Q (baselines for EXPOSURE).

Both estimate ``v_mask(C) = Var[Q(O, a) | o_C]`` with the agent's Q ``q_hat(o, .) = E_{z ~ e(. | o)}[Q_omega]``
evaluated at complete observations, and differ only in how hidden features are handled:

    empirical conditional   variance over replay observations that match ``o`` on ``C`` (SVERL-style;
                            coalitions with fewer than two matches fall back to 0)
    marginal imputation     variance over hybrids ``(o_C, b_{N \\ C})`` with ``b`` drawn from the replay,
                            ignoring feature dependence (off-manifold boards are possible)

The action's own cell is always kept visible, matching the oracle's action-legal conditioning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from exposure_rl import coalitions
from exposure_rl.vaeac_q import VAEACQ, belief_q, coalition_masks


def empirical_conditional_games(replay: np.ndarray, replay_q: np.ndarray, obs, action: int, hidden_value=None):
    """``(v_mask over 2^n coalitions, share of coalitions that fell back)`` from exact-match conditioning.

    ``replay``: ``(R, n)`` observations; ``replay_q``: ``(R,)`` agent's ``Q(., action)`` (NaN where illegal).
    ``hidden_value``: cell value meaning "action legal" (Minesweeper: hidden, Tic Tac Toe: empty);
    ``None`` keeps rows with finite Q.
    """

    n = replay.shape[1]
    legal = np.isfinite(replay_q) if hidden_value is None else (replay[:, action] == hidden_value)
    pts, q = replay[legal], replay_q[legal]
    weights = np.ones(len(pts))
    m = coalitions.conditional_moments(pts, weights, np.stack([np.ones_like(q), q, q**2], 1), obs)
    counts = coalitions.subset_sums(
        np.bincount(((pts != np.asarray(obs)[None]) << np.arange(n)).sum(1), minlength=1 << n).astype(float), n
    )[(1 << n) - 1 - np.arange(1 << n)]
    # The legal filter makes every kept row agree with ``obs`` on the action cell, so v(C) = v(C + a).
    v = np.maximum(m[:, 2] - m[:, 1] ** 2, 0.0)
    fallback = counts < 2
    v[fallback] = 0.0
    v[-1] = 0.0
    return v, float(fallback.mean())


def make_marginal_games(model: VAEACQ, n_impute: int, blocks: int):
    """Jitted ``(params, obs, action, imputations (B, n), eps) -> v_mask over 2^n coalitions``."""

    n = model.cfg.n_cells
    base = jnp.asarray(coalition_masks(n))

    @jax.jit
    def games(params, obs, action, imputations, eps):
        known = base.at[:, action].set(1.0)

        def block(k):  # k: (c, n) coalitions -> (c,) variance of the agent's Q over hybrids
            hybrid = jnp.where(k[:, None, :] > 0, obs[None, None, :], imputations[None, :, :])  # (c, B, n)
            q = belief_q(model, params, hybrid.reshape(-1, n), eps)[:, action].reshape(k.shape[0], n_impute)
            return q.var(axis=1)

        v = jax.lax.map(block, known.reshape(blocks, -1, n)).reshape(-1)
        return v.at[-1].set(0.0)

    return games
