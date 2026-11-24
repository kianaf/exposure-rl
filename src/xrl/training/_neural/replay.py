"""Replay-buffer primitives for DQN-style training."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    """A single transition stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    # The legal-action mask of ``state`` itself. The bootstrap only ever needs
    # the successor's mask, but an objective that supervises every legal action
    # of the stored state needs this one, and it costs nothing to keep: the
    # learner step it comes from already carries it.
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple fixed-capacity replay buffer."""

    def __init__(self, capacity: int, *, seed: int | None = None):
        self.capacity = int(capacity)
        self._rng = random.Random(seed)
        self._items: list[ReplayTransition] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ReplayTransition) -> None:
        """Store a transition, overwriting oldest if at capacity."""

        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random minibatch of transitions."""

        batch = self._rng.sample(self._items, batch_size)
        return {
            "states": np.stack([transition.state for transition in batch]).astype(np.float32),
            "actions": np.asarray([transition.action for transition in batch], dtype=np.int32),
            "rewards": np.asarray([transition.reward for transition in batch], dtype=np.float32),
            "action_masks": np.stack([transition.action_mask for transition in batch]).astype(np.int8),
            "next_states": np.stack([transition.next_state for transition in batch]).astype(np.float32),
            "next_action_masks": np.stack([transition.next_action_mask for transition in batch]).astype(np.int8),
            "dones": np.asarray([transition.done for transition in batch], dtype=bool),
        }

    def snapshot(self) -> tuple[ReplayTransition, ...]:
        """Return a read-only oldest-to-newest view of the retained transitions."""

        if len(self._items) < self.capacity or self._position == 0:
            return tuple(self._items)
        return tuple((*self._items[self._position :], *self._items[: self._position]))
