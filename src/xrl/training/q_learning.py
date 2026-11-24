"""Tabular Q-learning agent."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from xrl.training.encoders import (
    EncodedView,
    StateActionEncoder,
    StateKey,
    make_encoder,
    state_key,  # noqa: F401 - re-exported for callers importing from xrl.training.q_learning
)
from xrl.training.episode import epsilon_for_episode


def legal_actions_from_mask(action_mask: np.ndarray | Iterable[int]) -> list[int]:
    """Return legal action indices from a binary action mask."""

    return [index for index, value in enumerate(action_mask) if int(value) == 1]


def epsilon_greedy_distribution(
    values: np.ndarray | Iterable[float],
    action_mask: np.ndarray | Iterable[int],
    *,
    epsilon: float,
) -> np.ndarray:
    """Return epsilon-greedy probabilities over legal actions."""

    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    if value_array.shape != mask.shape:
        raise ValueError(f"values and action_mask must have the same shape, got {value_array.shape} and {mask.shape}")
    legal = legal_actions_from_mask(mask)
    if not legal:
        raise ValueError("no legal actions available")

    clipped_epsilon = min(max(float(epsilon), 0.0), 1.0)
    probabilities = np.zeros_like(value_array, dtype=np.float64)
    for action in legal:
        probabilities[action] += clipped_epsilon / len(legal)

    best_value = max(float(value_array[action]) for action in legal)
    best_actions = [action for action in legal if float(value_array[action]) == best_value]
    for action in best_actions:
        probabilities[action] += (1.0 - clipped_epsilon) / len(best_actions)
    return probabilities


def boltzmann_distribution(
    values: np.ndarray | Iterable[float],
    action_mask: np.ndarray | Iterable[int],
    *,
    temperature: float,
) -> np.ndarray:
    """Return softmax(values / temperature) probabilities over legal actions."""

    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
    if value_array.shape != mask.shape:
        raise ValueError(f"values and action_mask must have the same shape, got {value_array.shape} and {mask.shape}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    legal = legal_actions_from_mask(mask)
    if not legal:
        raise ValueError("no legal actions available")

    legal_values = value_array[legal] / float(temperature)
    weights = np.exp(legal_values - legal_values.max())
    probabilities = np.zeros_like(value_array, dtype=np.float64)
    probabilities[legal] = weights / weights.sum()
    return probabilities


def linear_epsilon(
    episode: int,
    *,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
) -> float:
    """Linearly decay epsilon over 0-indexed episodes."""

    return epsilon_for_episode(
        episode + 1,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
    )


@dataclass(slots=True)
class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent."""

    num_actions: int
    alpha: float = 0.3
    gamma: float = 0.95
    seed: int | None = None
    encoder: str = "identity"
    q_table: dict[StateKey, list[float]] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)
    _encoder: StateActionEncoder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")
        self._rng = random.Random(self.seed)
        self._encoder = make_encoder(self.encoder, num_actions=self.num_actions)

    def values(self, key: StateKey) -> list[float]:
        """Return the mutable Q-value row for a state key."""

        return self.q_table.setdefault(key, [0.0] * self.num_actions)

    def select_action(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        epsilon: float,
    ) -> int:
        """Select a legal action using epsilon-greedy exploration."""

        view = self.encode_view(observation, action_mask)
        table_action = self.select_table_action(view, epsilon=epsilon)
        return view.to_env_action(table_action)

    def encode_view(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
    ) -> EncodedView:
        """Return the agent's canonical view for an environment observation."""

        return self._encoder.encode(observation, action_mask)

    def select_table_action(self, view: EncodedView, *, epsilon: float) -> int:
        """Select an action in canonical Q-table coordinates."""

        legal = legal_actions_from_mask(view.action_mask)
        if not legal:
            raise ValueError("no legal actions available")
        if self._rng.random() < epsilon:
            return self._rng.choice(legal)

        values = self.values(view.state_key)
        best_value = max(values[action] for action in legal)
        best_actions = [action for action in legal if values[action] == best_value]
        return self._rng.choice(best_actions)

    def action_distribution(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return epsilon-greedy action probabilities in environment coordinates."""

        view = self.encode_view(observation, action_mask)
        row = self.q_table.get(view.state_key, [0.0 for _ in range(self.num_actions)])
        table_probabilities = epsilon_greedy_distribution(row, view.action_mask, epsilon=epsilon)
        probabilities = np.zeros(self.num_actions, dtype=np.float64)
        for table_action, probability in enumerate(table_probabilities):
            probabilities[view.to_env_action(table_action)] = probability
        return probabilities

    def q_value(self, observation: np.ndarray | Iterable[int | float], action: int) -> float:
        """Read an environment-coordinate Q-value without creating a row."""

        view = self.encode_view(observation, [1 for _ in range(self.num_actions)])
        row = self.q_table.get(view.state_key)
        if row is None:
            return 0.0
        return float(row[view.to_table_action(action)])

    def q_values_for_display(self, observation: np.ndarray | Iterable[int | float]) -> list[float]:
        """Return Q-values mapped back to environment action coordinates."""

        view = self.encode_view(observation, [1 for _ in range(self.num_actions)])
        row = self.q_table.get(view.state_key, [0.0 for _ in range(self.num_actions)])
        values = [0.0 for _ in range(self.num_actions)]
        for env_action in range(self.num_actions):
            values[env_action] = float(row[view.to_table_action(env_action)])
        return values

    def q_values(self, observation: np.ndarray | Iterable[int | float]) -> list[float]:
        """Return environment-coordinate Q-values for a flat observation."""

        return self.q_values_for_display(observation)

    def state_value(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        epsilon: float = 0.0,
    ) -> float:
        """Return the policy-weighted value estimate for an observation."""

        probabilities = self.action_distribution(observation, action_mask, epsilon=epsilon)
        values = np.asarray(self.q_values(observation), dtype=np.float64)
        return float(np.dot(probabilities, values))

    def observe_transition(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey | None,
        next_action_mask: np.ndarray | Iterable[int] | None,
        *,
        done: bool,
    ) -> float:
        """Apply one Q-learning update and return the absolute TD error."""

        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions - 1}], got {action}")
        values = self.values(state)
        if done:
            future = 0.0
        else:
            if next_state is None or next_action_mask is None:
                raise ValueError("next_state and next_action_mask are required for non-terminal updates")
            legal_next = legal_actions_from_mask(next_action_mask)
            future = max((self.values(next_state)[next_action] for next_action in legal_next), default=0.0)

        target = reward + self.gamma * future
        td_error = abs(target - values[action])
        values[action] += self.alpha * (target - values[action])
        return td_error

    def save(self, path: str | Path) -> None:
        """Save Q-values as JSON."""

        payload = {
            "num_actions": int(self.num_actions),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "encoder": self.encoder,
            "q_table": [
                {"state": list(key), "values": values}
                for key, values in sorted(self.q_table.items(), key=lambda item: repr(item[0]))
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> QLearningAgent:
        """Load Q-values from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            num_actions=int(payload["num_actions"]),
            alpha=float(payload.get("alpha", 0.3)),
            gamma=float(payload.get("gamma", 0.95)),
            seed=seed,
            encoder=str(payload.get("encoder", "identity")),
        )
        agent.q_table = {
            tuple(row["state"]): [float(value) for value in row["values"]] for row in payload.get("q_table", [])
        }
        return agent
