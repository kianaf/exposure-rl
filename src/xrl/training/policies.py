"""Policy adapters used by training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from xrl.tictactoe.agents import make_agent
from xrl.tictactoe.core import O, X

_PERTURBED_MINIMAX_PREFIX = "perturbed_minimax"


def format_perturbed_minimax_name(discount: float) -> str:
    """Return the canonical opponent name encoding a perturbed-minimax discount factor."""

    return f"{_PERTURBED_MINIMAX_PREFIX}_d={discount:g}"


def frozen_random_move_probability(discount: float, episodes: int) -> float:
    """Return the terminal (test-time) random-move rate ``discount ** (episodes - 1)``.

    After training the annealing opponent is frozen at its final difficulty, a single
    constant rate. Exact SVERL and surrogate dataset builders consume this value.
    """

    return discount ** (max(episodes, 1) - 1)


def opponent_random_move_probability(name: str, *, discount: float, episode: int) -> float:
    """Return the opponent's random-action mixture weight at one episode."""

    normalized = name.lower()
    if normalized == "random":
        return 1.0
    if normalized in {"minimax", "perfect"}:
        return 0.0
    if normalized.startswith(_PERTURBED_MINIMAX_PREFIX):
        return discount ** max(episode - 1, 0)
    raise ValueError(f"unknown opponent {name!r}")


class LearningAgent(Protocol):
    """Small action-selection interface shared by learners and policies."""

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        agent_id: str,
        epsilon: float,
        env,
    ) -> int:
        """Return one legal action."""


@dataclass(slots=True)
class OpponentPolicy:
    """Adapter around built-in Tic Tac Toe policies."""

    name: str
    seed: int | None = None
    perturbed_minimax_discount: float = 1.0
    policy: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        policy_name = self.name
        normalized = policy_name.lower()
        self.policy = make_agent(
            policy_name,
            seed=self.seed,
            perturbed_minimax_discount=self.perturbed_minimax_discount,
        )
        if normalized == "perturbed_minimax":
            self.name = format_perturbed_minimax_name(self.perturbed_minimax_discount)

    def set_episode(self, episode: int) -> None:
        """Forward the current episode to the underlying policy when it anneals online."""

        advance = getattr(self.policy, "set_episode", None)
        if callable(advance):
            advance(episode)

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        agent_id: str,
        epsilon: float,
        env,
    ) -> int:
        mark = X if agent_id == "player_0" else O
        return self.policy(
            {"observation": observation, "action_mask": action_mask},
            action_mask,
            state=env.unwrapped.state,
            player=mark,
        )


def make_opponent(
    name: str,
    *,
    seed: int | None = None,
    perturbed_minimax_discount: float = 1.0,
):
    """Construct an opponent policy by name."""

    return OpponentPolicy(
        name=name,
        seed=seed,
        perturbed_minimax_discount=perturbed_minimax_discount,
    )
