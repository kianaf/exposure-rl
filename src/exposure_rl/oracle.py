"""Exact certainty games for Tic Tac Toe (milestone 1: no training).

The learner plays the expectimax-optimal policy against a fixed opponent, so
``Q(s, a)`` and the return moments ``E[G | s, a]``, ``E[G^2 | s, a]`` are exact.
Occupancy, action-legal conditioning and the 9-cell mover-relative features
come from the xrl oracle (ADR 0005).

For a target state ``x``, action ``a`` and coalition ``C`` the three games are

    v_mask(C) = Var[Q(S, a) | S_C = x_C]               state uncertainty
    v_opp(C)  = E[Var(G | S, a) | S_C = x_C]          outcome (opponent) randomness
    v_tot(C)  = Var[G | S_C = x_C, a]                 = v_opp(C) + v_mask(C)

all under the action-legal conditional occupancy. Arrays follow
``all_coalitions`` (bitmask) order: index 0 is the empty coalition, -1 the grand one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from xrl.explainability.core import shapley_operator
from xrl.explainability.tictactoe import (
    TicTacToeStateDistribution,
    build_tictactoe_state_distribution,
    state_from_observation,
)
from xrl.tictactoe.core import GameState, TicTacToeRules, X

from exposure_rl import opponents

FEATURE_COUNT = 9


class ExpectimaxOracle:
    """Exact return moments for the expectimax learner against a fixed opponent.

    Ties between optimal learner actions are broken uniformly. This leaves
    ``Q`` unchanged but adds return variance, so the tie-break must match the
    policy used to build the occupancy (``action_distribution`` below).
    """

    def __init__(self, opponent: str = "random", rules: TicTacToeRules | None = None) -> None:
        self.opponent = opponent
        self.rules = rules or TicTacToeRules()
        # Per-instance memo of the game-tree recursion (a class-level cache would pin every instance).
        self._moments = cache(self._moments_uncached)

    def _moments_uncached(self, state: GameState, learner: int) -> tuple[float, float]:
        """``(E[G], E[G^2])`` from ``state`` onward, learner's perspective."""

        if state.is_terminal:
            r0, r1 = self.rules.returns(state)
            g = r0 if learner == X else r1
            return float(g), float(g * g)
        legal = self.rules.legal_actions(state)
        children = {a: self._moments(self.rules.apply(state, a), learner) for a in legal}
        if state.current_player == learner:
            best = max(m[0] for m in children.values())
            chosen = [a for a in legal if np.isclose(children[a][0], best)]
            probs = {a: 1.0 / len(chosen) for a in chosen}
        else:
            dist = opponents.distribution(self.opponent, state, self.rules)
            probs = {a: float(dist[a]) for a in legal if dist[a] > 0}
        first = sum(p * children[a][0] for a, p in probs.items())
        second = sum(p * children[a][1] for a, p in probs.items())
        return first, second

    def action_moments(self, observation) -> tuple[np.ndarray, np.ndarray]:
        """Per-action ``E[G | s, a]`` and ``E[G^2 | s, a]``; ``nan`` for illegal actions."""

        state = state_from_observation(observation, rules=self.rules)
        learner = state.current_player
        first = np.full(FEATURE_COUNT, np.nan)
        second = np.full(FEATURE_COUNT, np.nan)
        for a in self.rules.legal_actions(state):
            first[a], second[a] = self._moments(self.rules.apply(state, a), learner)
        return first, second

    # PolicyView protocol, so the xrl occupancy builder can roll this policy out.
    def q_values(self, observation) -> np.ndarray:
        first, _ = self.action_moments(observation)
        return np.where(np.isnan(first), -np.inf, first)

    def action_distribution(self, observation, action_mask, epsilon: float = 0.0) -> np.ndarray:
        q = self.q_values(observation)
        best = np.flatnonzero(np.isclose(q, q.max()))
        probs = np.zeros(FEATURE_COUNT)
        probs[best] = 1.0 / len(best)
        return probs

    def state_value(self, observation, action_mask) -> float:
        return float(self.q_values(observation).max())

    def occupancy(self) -> TicTacToeStateDistribution:
        name, rate = opponents.parse(self.opponent)
        return build_tictactoe_state_distribution(
            self, opponent=name, perturbed_minimax_probability=rate, rules=self.rules
        )


@dataclass(frozen=True)
class CertaintyGames:
    """All coalition values of the three certainty games for one (state, action)."""

    mask: np.ndarray
    opp: np.ndarray
    tot: np.ndarray
    q_mean: np.ndarray


def certainty_games(
    oracle: ExpectimaxOracle,
    distribution: TicTacToeStateDistribution,
    observation,
    action: int,
) -> CertaintyGames:
    """Exact ``v_mask``, ``v_opp``, ``v_tot`` (and the prediction game) over all ``2^9`` coalitions."""

    view = distribution.legal_action_view(action)
    operator = view.conditional_expectation_operator(tuple(np.asarray(observation).tolist()))
    moments = [oracle.action_moments(point.observation) for point in view.points]
    q = np.array([first[action] for first, _ in moments])
    g2 = np.array([second[action] for _, second in moments])

    q_mean = operator @ q
    mask = np.maximum(operator @ q**2 - q_mean**2, 0.0)
    opp = operator @ (g2 - q**2)
    # Grand coalition: the full state is known, so state uncertainty vanishes.
    first, second = oracle.action_moments(observation)
    q_mean[-1] = first[action]
    mask[-1] = 0.0
    opp[-1] = second[action] - first[action] ** 2
    return CertaintyGames(mask=mask, opp=opp, tot=opp + mask, q_mean=q_mean)


def uncertainty_reduction_shapley(values: np.ndarray) -> np.ndarray:
    """Shapley values of ``u(C) = v(empty) - v(C)``: positive = the feature makes the agent more certain."""

    return -(shapley_operator(FEATURE_COUNT) @ values)


def leave_one_out(values: np.ndarray) -> np.ndarray:
    """Uncertainty added by hiding each single feature from the full state."""

    grand = (1 << FEATURE_COUNT) - 1
    return np.array([values[grand & ~(1 << i)] - values[grand] for i in range(FEATURE_COUNT)])


def shapley_effects(oracle: ExpectimaxOracle, distribution: TicTacToeStateDistribution, action: int) -> np.ndarray:
    """Global Shapley effects of ``Q(., a)`` under the action-legal occupancy.

    Game: ``Var(E[Q | X_C])`` (Owen 2014; Song, Nelson & Staum 2016), computed
    exactly by grouping support points on their ``C``-restricted observation.
    """

    view = distribution.legal_action_view(action)
    obs = np.array([point.observation for point in view.points])
    weights = np.array([point.weight for point in view.points])
    q = np.array([oracle.action_moments(point.observation)[0][action] for point in view.points])
    mean = float(weights @ q)
    game = np.zeros(1 << FEATURE_COUNT)
    for c in range(1 << FEATURE_COUNT):
        cols = [i for i in range(FEATURE_COUNT) if c & (1 << i)]
        if not cols:
            continue
        _, group = np.unique(obs[:, cols], axis=0, return_inverse=True)
        mass = np.bincount(group, weights=weights)
        cond = np.bincount(group, weights=weights * q) / mass
        game[c] = float(mass @ (cond - mean) ** 2)
    return shapley_operator(FEATURE_COUNT) @ game
