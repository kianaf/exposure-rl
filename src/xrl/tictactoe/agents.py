"""Built-in Tic Tac Toe agents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np

from xrl.tictactoe.core import GameState, PlayerMark, TicTacToeRules, X


class Policy(Protocol):
    """Callable agent interface for UI and evaluation helpers."""

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        """Return one legal action."""


@dataclass(slots=True)
class RandomAgent:
    """Uniformly sample from the legal action mask."""

    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        legal = [index for index, value in enumerate(action_mask) if value]
        if not legal:
            raise ValueError("no legal actions available")
        return self._rng.choice(legal)


@dataclass(frozen=True, slots=True)
class MinimaxAgent:
    """Perfect-play Tic Tac Toe agent."""

    rules: TicTacToeRules = TicTacToeRules()

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        legal = self.rules.legal_actions(state)
        if not legal:
            raise ValueError("no legal actions available")

        best_action = legal[0]
        best_score = -2.0
        for action in legal:
            child = self.rules.apply(state, action)
            score = -self._negamax(child, -player)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    @cache  # noqa: B019 - intentional per-instance memo; reused via analysis._shared_minimax_agent
    def _negamax(self, state: GameState, player: PlayerMark) -> float:
        if state.is_terminal:
            player_0_return, player_1_return = self.rules.returns(state)
            return player_0_return if player == X else player_1_return

        best_score = -2.0
        for action in self.rules.legal_actions(state):
            child = self.rules.apply(state, action)
            best_score = max(best_score, -self._negamax(child, -player))
        return best_score


def annealing_discount(episodes: int, *, terminal_rate: float = 0.05) -> float:
    """Return the per-episode decay whose terminal random-move rate is ``terminal_rate``.

    :class:`PerturbedMinimaxAgent` anneals as ``rate_t = discount ** (t - 1)``,
    so the schedule is fixed by choosing where it should end. Solving
    ``discount ** (episodes - 1) = terminal_rate`` keeps the opponent's final
    strength independent of run length, which is what makes runs of different
    length comparable at all.
    """

    if episodes < 2:
        return 1.0
    if not 0.0 < terminal_rate <= 1.0:
        raise ValueError(f"terminal_rate must lie in (0, 1], got {terminal_rate}")
    return float(terminal_rate ** (1.0 / (episodes - 1)))


@dataclass(slots=True)
class PerturbedMinimaxAgent:
    """Minimax agent that anneals from fully random toward perfect play.

    The random-move rate always starts at 100% (episode 1) and decays geometrically
    by ``discount`` each episode: ``rate_t = discount ** (episode - 1)``. ``discount``
    of ``1.0`` keeps the opponent fully random forever (no annealing); ``discount < 1``
    anneals the opponent random -> minimax over training. The current episode is fed in
    online via :meth:`set_episode` from the training loop.
    """

    discount: float = 1.0
    seed: int | None = None
    rules: TicTacToeRules = TicTacToeRules()
    minimax: MinimaxAgent = field(init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)
    _probability: float = field(init=False, repr=False, default=1.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"perturbed_minimax_discount must be in [0, 1], got {self.discount}")
        self.minimax = MinimaxAgent(self.rules)
        self._rng = random.Random(self.seed)
        self._probability = 1.0

    def set_episode(self, episode: int) -> None:
        """Set the current random-move rate for ``episode`` (1-based) via the discount schedule."""

        self._probability = self.discount ** max(episode - 1, 0)

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        legal = [index for index, value in enumerate(action_mask) if int(value) == 1]
        if not legal:
            raise ValueError("no legal actions available")
        if self._rng.random() < self._probability:
            return self._rng.choice(legal)
        return self.minimax(observation, action_mask, state=state, player=player)


@dataclass(slots=True)
class QLearningPolicy:
    """Greedy policy adapter for a saved tabular Q-learning model."""

    q_model_path: str | Path
    seed: int | None = None
    q_agent: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from xrl.training.q_learning import QLearningAgent

        path = Path(self.q_model_path)
        if not path.exists():
            raise ValueError(f"qlearning model path does not exist: {path}")
        self.q_agent = QLearningAgent.load(path, seed=self.seed)

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        return self.q_agent.select_action(observation["observation"], action_mask, epsilon=0.0)


def make_agent(
    name: str,
    *,
    seed: int | None = None,
    q_model_path: str | Path | None = None,
    perturbed_minimax_discount: float = 1.0,
) -> Policy:
    """Construct a built-in policy by name."""

    normalized = name.lower()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized in {"minimax", "perfect"}:
        return MinimaxAgent()
    if normalized == "perturbed_minimax":
        return PerturbedMinimaxAgent(discount=perturbed_minimax_discount, seed=seed)
    if normalized in {"qlearning", "q-learning", "q"}:
        if q_model_path is None:
            raise ValueError("qlearning agent requires q_model_path")
        return QLearningPolicy(q_model_path=q_model_path, seed=seed)
    raise ValueError(f"unknown agent {name!r}; expected 'random', 'minimax', 'perturbed_minimax', or 'qlearning'")
