"""Minesweeper rules and flat observation encoding."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

HIDDEN = -1
EXPLODED_MINE = 9
Outcome = Literal["win", "loss"]


@dataclass(frozen=True, slots=True)
class MinesweeperConfig:
    """Board geometry for one Minesweeper run."""

    rows: int = 5
    columns: int = 5
    mines: int = 3

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError(f"rows must be positive, got {self.rows}")
        if self.columns <= 0:
            raise ValueError(f"columns must be positive, got {self.columns}")
        if not 0 <= self.mines < self.cells:
            raise ValueError(f"mines must satisfy 0 <= mines < rows * columns, got {self.mines}")

    @property
    def cells(self) -> int:
        return self.rows * self.columns

    @property
    def total_safe_cells(self) -> int:
        return self.cells - self.mines


def row_col_to_action(row: int, col: int, *, columns: int) -> int:
    """Convert a row/column pair to a row-major action id."""

    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")
    if row < 0 or col < 0 or col >= columns:
        raise ValueError(f"invalid row/column ({row}, {col}) for {columns} columns")
    return row * columns + col


def action_to_row_col(action: int, *, columns: int) -> tuple[int, int]:
    """Convert a row-major action id to a row/column pair."""

    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}")
    if action < 0:
        raise ValueError(f"action must be non-negative, got {action}")
    return divmod(action, columns)


class MinesweeperBoard:
    """Mutable single-agent Minesweeper board state.

    Mines are placed lazily on the first reveal so the first selected cell is
    guaranteed safe. The public observation hides unrevealed mines throughout
    the episode; only the exploded mine is shown on a loss.
    """

    def __init__(self, config: MinesweeperConfig):
        self.config = config
        self._fixed_mines: frozenset[int] | None = None
        self.mine_positions: frozenset[int] | None = None
        self.revealed: set[int] = set()
        self.exploded_action: int | None = None
        self.outcome: Outcome | None = None

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not None

    def reset(self, *, mine_positions: Iterable[int] | None = None) -> None:
        """Reset the board, optionally pinning a fixed mine layout."""

        self._fixed_mines = None if mine_positions is None else self._normalize_mines(mine_positions)
        self.mine_positions = self._fixed_mines
        self.revealed = set()
        self.exploded_action = None
        self.outcome = None

    def action_mask(self) -> np.ndarray:
        """Return a binary mask over revealable cells."""

        mask = np.zeros(self.config.cells, dtype=np.int8)
        if self.is_terminal:
            return mask
        for action in range(self.config.cells):
            if action not in self.revealed:
                mask[action] = 1
        return mask

    def observation(self) -> np.ndarray:
        """Return the flat hidden/clue/exploded observation vector."""

        values = np.full(self.config.cells, HIDDEN, dtype=np.int8)
        mines = self.mine_positions or frozenset()
        for action in self.revealed:
            values[action] = self._clue(action, mines)
        if self.exploded_action is not None:
            values[self.exploded_action] = EXPLODED_MINE
        return values

    def info(self) -> dict[str, int | float | str | None]:
        """Return non-secret task progress information."""

        return {
            "outcome": self.outcome,
            "safe_cells_revealed": len(self.revealed),
            "total_safe_cells": self.config.total_safe_cells,
            "completion_ratio": len(self.revealed) / self.config.total_safe_cells,
        }

    def reveal(self, action: int, *, rng: np.random.Generator) -> tuple[float, bool]:
        """Reveal one cell and return ``(reward, terminated)``."""

        action = int(action)
        self._validate_reveal(action)
        self._ensure_mines(first_action=action, rng=rng)
        mines = self.mine_positions or frozenset()

        if action in mines:
            self.exploded_action = action
            self.outcome = "loss"
            return -1.0, True

        newly_revealed = self._reveal_safe_region(action, mines)
        if len(self.revealed) == self.config.total_safe_cells:
            self.outcome = "win"
        reward = newly_revealed / self.config.total_safe_cells
        return reward, self.is_terminal

    def render_ansi(self, *, reveal_mines: bool = False) -> str:
        """Render the board as ANSI text."""

        mines = self.mine_positions or frozenset()
        rows: list[str] = []
        for row in range(self.config.rows):
            cells: list[str] = []
            for col in range(self.config.columns):
                action = row_col_to_action(row, col, columns=self.config.columns)
                cells.append(self._render_cell(action, mines, reveal_mines=reveal_mines))
            rows.append(" ".join(cells))

        if self.outcome == "win":
            status = "Win"
        elif self.outcome == "loss":
            status = "Loss"
        else:
            status = "In progress"
        return "\n".join(rows) + f"\n{status}"

    def _render_cell(self, action: int, mines: frozenset[int], *, reveal_mines: bool) -> str:
        if self.exploded_action == action:
            return "X"
        if reveal_mines and action in mines:
            return "*"
        if action not in self.revealed:
            return "."
        clue = self._clue(action, mines)
        return " " if clue == 0 else str(clue)

    def _validate_reveal(self, action: int) -> None:
        if self.is_terminal:
            raise ValueError("cannot reveal a cell after the episode is terminal")
        if not 0 <= action < self.config.cells:
            raise ValueError(f"action must be in [0, {self.config.cells - 1}], got {action}")
        if action in self.revealed:
            raise ValueError(f"illegal action {action}: cell is already revealed")

    def _ensure_mines(self, *, first_action: int, rng: np.random.Generator) -> None:
        if self.mine_positions is not None:
            if not self.revealed and first_action in self.mine_positions:
                raise ValueError("fixed mine_positions contains the first action and violates first-click safety")
            return

        candidates = [action for action in range(self.config.cells) if action != first_action]
        if self.config.mines == 0:
            self.mine_positions = frozenset()
            return
        selected = rng.choice(candidates, size=self.config.mines, replace=False)
        self.mine_positions = frozenset(int(action) for action in selected.tolist())

    def _reveal_safe_region(self, action: int, mines: frozenset[int]) -> int:
        newly_revealed = 0
        queue: deque[int] = deque([action])
        while queue:
            current = queue.popleft()
            if current in self.revealed or current in mines:
                continue
            self.revealed.add(current)
            newly_revealed += 1
            if self._clue(current, mines) != 0:
                continue
            for neighbor in self._neighbors(current):
                if neighbor not in self.revealed and neighbor not in mines:
                    queue.append(neighbor)
        return newly_revealed

    def _neighbors(self, action: int) -> tuple[int, ...]:
        row, col = action_to_row_col(action, columns=self.config.columns)
        neighbors: list[int] = []
        for next_row in range(max(0, row - 1), min(self.config.rows, row + 2)):
            for next_col in range(max(0, col - 1), min(self.config.columns, col + 2)):
                if next_row == row and next_col == col:
                    continue
                neighbors.append(row_col_to_action(next_row, next_col, columns=self.config.columns))
        return tuple(neighbors)

    def _clue(self, action: int, mines: frozenset[int]) -> int:
        return sum(1 for neighbor in self._neighbors(action) if neighbor in mines)

    def _normalize_mines(self, mine_positions: Iterable[int]) -> frozenset[int]:
        mines = frozenset(int(action) for action in mine_positions)
        if len(mines) != self.config.mines:
            raise ValueError(f"expected {self.config.mines} fixed mines, got {len(mines)}")
        invalid = [action for action in mines if not 0 <= action < self.config.cells]
        if invalid:
            raise ValueError(f"fixed mine positions out of range: {sorted(invalid)}")
        return mines
