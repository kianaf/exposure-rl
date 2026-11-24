"""Immutable Tic Tac Toe rules and observation encoding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

EMPTY = 0
X = 1
O = -1  # noqa: E741 - domain constant: the "O" player mark
DRAW = 0

PlayerMark = Literal[1, -1]
ObsMode = Literal["planes", "flat"]

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def row_col_to_action(row: int, col: int) -> int:
    """Convert a row/column pair to a row-major action id."""

    if not 0 <= row < 3 or not 0 <= col < 3:
        raise ValueError(f"row and col must be in [0, 2], got ({row}, {col})")
    return row * 3 + col


def action_to_row_col(action: int) -> tuple[int, int]:
    """Convert a row-major action id to a row/column pair."""

    if not 0 <= action < 9:
        raise ValueError(f"action must be in [0, 8], got {action}")
    return divmod(action, 3)


def ultimate_action(local_board: int, local_cell: int) -> int:
    """Compose an Ultimate Tic Tac Toe action id.

    Both parts use the same row-major 0..8 convention as standard Tic Tac Toe.
    """

    if not 0 <= local_board < 9:
        raise ValueError(f"local_board must be in [0, 8], got {local_board}")
    if not 0 <= local_cell < 9:
        raise ValueError(f"local_cell must be in [0, 8], got {local_cell}")
    return local_board * 9 + local_cell


def split_ultimate_action(action: int) -> tuple[int, int]:
    """Split an Ultimate Tic Tac Toe action into local board and local cell."""

    if not 0 <= action < 81:
        raise ValueError(f"ultimate action must be in [0, 80], got {action}")
    return divmod(action, 9)


@dataclass(frozen=True, slots=True)
class GameState:
    """A framework-independent immutable Tic Tac Toe position."""

    board: tuple[int, ...]
    current_player: PlayerMark = X
    move_count: int = 0
    winner: int | None = None
    is_draw: bool = False

    def __post_init__(self) -> None:
        if len(self.board) != 9:
            raise ValueError(f"board must contain 9 cells, got {len(self.board)}")
        invalid = [cell for cell in self.board if cell not in (EMPTY, X, O)]
        if invalid:
            raise ValueError(f"board cells must be -1, 0, or 1, got {invalid}")
        if self.current_player not in (X, O):
            raise ValueError(f"current_player must be 1 or -1, got {self.current_player}")
        if not 0 <= self.move_count <= 9:
            raise ValueError(f"move_count must be in [0, 9], got {self.move_count}")
        if self.winner not in (None, X, O):
            raise ValueError(f"winner must be None, 1, or -1, got {self.winner}")
        if self.winner is not None and self.is_draw:
            raise ValueError("a state cannot be both won and drawn")

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None or self.is_draw

    @property
    def board_array(self) -> np.ndarray:
        return np.asarray(self.board, dtype=np.int8).reshape(3, 3)


@dataclass(frozen=True, slots=True)
class TicTacToeRules:
    """Rules, transitions, rewards, and observations for standard Tic Tac Toe."""

    num_actions: int = 9

    def initial_state(self) -> GameState:
        return GameState(board=(EMPTY,) * 9, current_player=X, move_count=0)

    def legal_actions(self, state: GameState) -> tuple[int, ...]:
        if state.is_terminal:
            return ()
        return tuple(index for index, cell in enumerate(state.board) if cell == EMPTY)

    def action_mask(self, state: GameState) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.int8)
        for action in self.legal_actions(state):
            mask[action] = 1
        return mask

    def apply(self, state: GameState, action: int) -> GameState:
        if state.is_terminal:
            raise ValueError("cannot apply an action to a terminal state")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions - 1}], got {action}")
        if state.board[action] != EMPTY:
            raise ValueError(f"illegal action {action}: cell is already occupied")

        board = list(state.board)
        board[action] = state.current_player
        next_board = tuple(board)
        winner = self._winner(next_board)
        move_count = state.move_count + 1
        is_draw = winner is None and move_count == self.num_actions
        next_player = O if state.current_player == X else X
        return GameState(
            board=next_board,
            current_player=next_player,
            move_count=move_count,
            winner=winner,
            is_draw=is_draw,
        )

    def returns(self, state: GameState) -> tuple[float, float]:
        if state.winner == X:
            return (1.0, -1.0)
        if state.winner == O:
            return (-1.0, 1.0)
        return (0.0, 0.0)

    def observation(self, state: GameState, player: PlayerMark, obs_mode: ObsMode = "planes") -> dict[str, np.ndarray]:
        if player not in (X, O):
            raise ValueError(f"player must be 1 or -1, got {player}")
        if obs_mode == "planes":
            return {
                "observation": self._planes_observation(state, player),
                "action_mask": self.action_mask(state),
            }
        if obs_mode == "flat":
            return {
                "observation": self._flat_observation(state, player),
                "action_mask": self.action_mask(state),
            }
        raise ValueError(f"unsupported obs_mode {obs_mode!r}; expected 'planes' or 'flat'")

    def render_ansi(self, state: GameState) -> str:
        marks = {X: "X", O: "O", EMPTY: "."}
        rows = []
        for row in range(3):
            cells = [marks[state.board[row_col_to_action(row, col)]] for col in range(3)]
            rows.append(" " + " | ".join(cells) + " ")
        board = "\n---+---+---\n".join(rows)
        if state.winner == X:
            status = "X wins"
        elif state.winner == O:
            status = "O wins"
        elif state.is_draw:
            status = "Draw"
        else:
            status = f"{'X' if state.current_player == X else 'O'} to move"
        return f"{board}\n{status}"

    def _planes_observation(self, state: GameState, player: PlayerMark) -> np.ndarray:
        board = state.board_array
        current = (board == player).astype(np.int8)
        opponent = (board == -player).astype(np.int8)
        empty = (board == EMPTY).astype(np.int8)
        turn = np.full((3, 3), 1 if state.current_player == player else 0, dtype=np.int8)
        return np.stack((current, opponent, empty, turn)).astype(np.int8)

    def _flat_observation(self, state: GameState, player: PlayerMark) -> np.ndarray:
        board = np.asarray(state.board, dtype=np.int8)
        return (board * player).astype(np.int8)

    def winner(self, board: Iterable[int]) -> int | None:
        """Return the mark owning a completed line on ``board``, or ``None``."""

        return self._winner(board)

    def _winner(self, board: Iterable[int]) -> int | None:
        cells = tuple(board)
        for a, b, c in WIN_LINES:
            line_sum = cells[a] + cells[b] + cells[c]
            if line_sum == 3:
                return X
            if line_sum == -3:
                return O
        return None
