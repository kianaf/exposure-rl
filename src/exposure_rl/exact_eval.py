"""Exact win / draw / loss probabilities of a deterministic policy against a fixed opponent.

Enumerates the game tree: the learner follows ``policy(flat_obs, legal_mask) -> action``
(mover-relative flat observation, as everywhere in xrl), the opponent follows its exact action
law. Roles are balanced 50:50 over the learner playing X and O, as in the xrl phase-one study.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
from xrl.tictactoe.core import O, TicTacToeRules, X

from exposure_rl import opponents

Policy = Callable[[np.ndarray, np.ndarray], int]


def exact_outcomes(policy: Policy, opponent: str = "random") -> dict:
    rules = TicTacToeRules()

    def outcomes_for(learner: int) -> np.ndarray:
        @cache
        def value(state) -> tuple[float, float, float]:  # (P(win), P(draw), P(loss))
            if state.is_terminal:
                r_x, r_o = rules.returns(state)
                g = r_x if learner == X else r_o
                return (float(g > 0), float(g == 0), float(g < 0))
            if state.current_player == learner:
                view = rules.observation(state, learner, "flat")
                obs = np.asarray(view["observation"], np.float32).reshape(-1)
                legal = np.asarray(view["action_mask"], bool).reshape(-1)
                action = int(policy(obs, legal))
                if not legal[action]:
                    raise ValueError(f"policy chose illegal action {action}")
                return value(rules.apply(state, action))
            probs = opponents.distribution(opponent, state, rules)
            total = np.zeros(3)
            for a, p in enumerate(probs):
                if p > 0:
                    total += p * np.asarray(value(rules.apply(state, a)))
            return tuple(total)

        return np.asarray(value(rules.initial_state()))

    by_role = {"X": outcomes_for(X), "O": outcomes_for(O)}
    mean = 0.5 * (by_role["X"] + by_role["O"])
    return {
        "win": float(mean[0]),
        "draw": float(mean[1]),
        "loss": float(mean[2]),
        "return": float(mean[0] - mean[2]),
        "win_as_X": float(by_role["X"][0]),
        "win_as_O": float(by_role["O"][0]),
    }


def greedy(q_fn: Callable[[np.ndarray], np.ndarray]) -> Policy:
    """Greedy-over-legal policy from a Q function on flat observations (first maximiser breaks ties)."""

    def policy(obs, legal):
        q = np.asarray(q_fn(obs), np.float64).reshape(-1)
        return int(np.flatnonzero(legal)[np.argmax(q[legal])])

    return policy
