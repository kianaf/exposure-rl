"""State/action encoders for tabular Q-learning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

StateKey = tuple[int | float, ...]


def state_key(observation: np.ndarray | Iterable[int | float]) -> StateKey:
    """Convert an observation vector into a hashable state key."""

    values = np.asarray(observation).reshape(-1)
    output: list[int | float] = []
    for value in values.tolist():
        if isinstance(value, float) and not value.is_integer():
            output.append(float(value))
        else:
            output.append(int(value))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class EncodedView:
    """Canonical view of an environment observation and action mask."""

    state_key: StateKey
    action_mask: np.ndarray
    env_to_table_actions: tuple[int, ...]
    table_to_env_actions: tuple[int, ...]
    transform: str = "identity"

    def to_table_action(self, env_action: int) -> int:
        return self.env_to_table_actions[int(env_action)]

    def to_env_action(self, table_action: int) -> int:
        return self.table_to_env_actions[int(table_action)]


class StateActionEncoder(Protocol):
    name: str

    def encode(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
    ) -> EncodedView:
        """Return a canonical state/action view."""


@dataclass(frozen=True, slots=True)
class IdentityEncoder:
    """Current Q-table behavior: no canonicalization."""

    num_actions: int
    name: str = "identity"

    def encode(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
    ) -> EncodedView:
        mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
        actions = tuple(range(self.num_actions))
        return EncodedView(
            state_key=state_key(observation),
            action_mask=mask,
            env_to_table_actions=actions,
            table_to_env_actions=actions,
            transform="identity",
        )


@dataclass(frozen=True, slots=True)
class BoardTransform:
    """A square-board index transform."""

    name: str
    env_to_table: tuple[int, ...]

    @property
    def table_to_env(self) -> tuple[int, ...]:
        inverse = [0 for _ in self.env_to_table]
        for env_action, table_action in enumerate(self.env_to_table):
            inverse[table_action] = env_action
        return tuple(inverse)

    def apply_vector(self, values: np.ndarray | Iterable[int | float]) -> np.ndarray:
        vector = np.asarray(values).reshape(-1)
        transformed = np.empty_like(vector)
        for env_index, table_index in enumerate(self.env_to_table):
            transformed[table_index] = vector[env_index]
        return transformed


def _idx(row: int, col: int) -> int:
    return row * 3 + col


def _transform(name: str, fn) -> BoardTransform:
    return BoardTransform(
        name=name,
        env_to_table=tuple(_idx(*fn(row, col)) for row in range(3) for col in range(3)),
    )


TICTACTOE_D4_TRANSFORMS: tuple[BoardTransform, ...] = (
    _transform("identity", lambda row, col: (row, col)),
    _transform("rot90", lambda row, col: (col, 2 - row)),
    _transform("rot180", lambda row, col: (2 - row, 2 - col)),
    _transform("rot270", lambda row, col: (2 - col, row)),
    _transform("flip_horizontal", lambda row, col: (row, 2 - col)),
    _transform("flip_vertical", lambda row, col: (2 - row, col)),
    _transform("diag", lambda row, col: (col, row)),
    _transform("anti_diag", lambda row, col: (2 - col, 2 - row)),
)


@dataclass(frozen=True, slots=True)
class TicTacToeSymmetryEncoder:
    """Canonicalize flat Tic Tac Toe observations under D4 symmetries."""

    num_actions: int = 9
    name: str = "tictactoe_symmetry"

    def encode(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
    ) -> EncodedView:
        obs = np.asarray(observation).reshape(-1)
        mask = np.asarray(action_mask, dtype=np.int8).reshape(-1)
        if self.num_actions != 9 or obs.shape != (9,) or mask.shape != (9,):
            return IdentityEncoder(self.num_actions).encode(obs, mask)

        candidates: list[tuple[StateKey, int, BoardTransform, np.ndarray]] = []
        for index, transform in enumerate(TICTACTOE_D4_TRANSFORMS):
            transformed_obs = transform.apply_vector(obs)
            transformed_mask = transform.apply_vector(mask).astype(np.int8)
            candidates.append((state_key(transformed_obs), index, transform, transformed_mask))
        key, _index, transform, transformed_mask = min(candidates, key=lambda item: (item[0], item[1]))
        return EncodedView(
            state_key=key,
            action_mask=transformed_mask,
            env_to_table_actions=transform.env_to_table,
            table_to_env_actions=transform.table_to_env,
            transform=transform.name,
        )


def make_encoder(name: str, *, num_actions: int) -> StateActionEncoder:
    normalized = name.lower()
    if normalized == "identity":
        return IdentityEncoder(num_actions)
    if normalized in {"tictactoe_symmetry", "tic_tac_toe_symmetry", "symmetry"}:
        return TicTacToeSymmetryEncoder(num_actions=num_actions)
    raise ValueError(f"unknown encoder {name!r}; expected 'identity' or 'tictactoe_symmetry'")
