"""Exact return moments of a *given* agent policy (policy evaluation), for the own-policy v_state reference.

The optimal oracles (``oracle.ExpectimaxOracle``, ``minesweeper.MinesweeperOracle``) let the learner play the
optimal action. Here the learner instead plays the agent's own greedy action, ``argmax`` over legal actions of a
supplied ``q_fn(observation)`` (the agent's full-information Q, observations from the learner's view); the
opponent's moves and the mine layout stay exact. ``action_moments`` then gives ``(E_pi[G | o, a],
E_pi[G^2 | o, a])`` for the first action ``a`` followed by the agent's policy, so
``Var_pi(G | o, a) = E_pi[G^2] - E_pi[G]^2`` is the agent's own outcome variance.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from exposure_rl import opponents
from exposure_rl.minesweeper import Geometry, MinesweeperOracle
from exposure_rl.oracle import ExpectimaxOracle


class GreedyPolicy:
    """Caches the agent's greedy action per observation (``q_fn`` is evaluated once per observation)."""

    def __init__(self, q_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.q_fn = q_fn
        self._actions: dict[tuple[int, ...], int] = {}

    def action(self, obs: tuple[int, ...], legal: np.ndarray) -> int:
        a = self._actions.get(obs)
        if a is None:
            q = np.asarray(self.q_fn(np.asarray(obs, np.float32)), np.float64)
            a = int(np.flatnonzero(legal)[np.argmax(q[legal])])
            self._actions[obs] = a
        return a


class PolicyTicTacToe(ExpectimaxOracle):
    """Tic-tac-toe moments when the learner follows ``policy`` (opponent exact, as in the optimal oracle)."""

    def __init__(self, opponent: str, policy: GreedyPolicy) -> None:
        super().__init__(opponent)
        self.policy = policy

    def _moments_uncached(self, state, learner: int) -> tuple[float, float]:
        if state.is_terminal:
            r0, r1 = self.rules.returns(state)
            g = r0 if learner == 1 else r1
            return float(g), float(g * g)
        if state.current_player == learner:
            obs = tuple(int(v) for v in np.asarray(self.rules.observation(state, learner, "flat")["observation"]))
            legal = np.zeros(9, bool)
            legal[list(self.rules.legal_actions(state))] = True
            return self._moments(self.rules.apply(state, self.policy.action(obs, legal)), learner)
        dist = opponents.distribution(self.opponent, state, self.rules)
        first = second = 0.0
        for a in self.rules.legal_actions(state):
            if dist[a] > 0:
                m = self._moments(self.rules.apply(state, a), learner)
                first += float(dist[a]) * m[0]
                second += float(dist[a]) * m[1]
        return first, second


class PolicyMinesweeper(MinesweeperOracle):
    """Minesweeper moments when the learner follows ``policy`` after the first click (layouts exact)."""

    def __init__(self, geometry: Geometry, policy: GreedyPolicy) -> None:
        super().__init__(geometry)
        self.policy = policy

    def _value_uncached(self, obs: tuple[int, ...]) -> tuple[float, float]:
        return self._moments(obs, self.policy.action(obs, self.legal(obs)))
