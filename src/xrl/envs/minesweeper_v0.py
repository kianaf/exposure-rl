"""Native Gymnasium environment for single-agent Minesweeper."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

from xrl.minesweeper.core import EXPLODED_MINE, HIDDEN, MinesweeperBoard, MinesweeperConfig

ENV_ID = "xrl/Minesweeper-v0"


def env(
    *,
    rows: int = 5,
    columns: int = 5,
    mines: int = 3,
    render_mode: str | None = None,
) -> MinesweeperEnv:
    """Create a Minesweeper Gymnasium environment."""

    return MinesweeperEnv(rows=rows, columns=columns, mines=mines, render_mode=render_mode)


class MinesweeperEnv(gym.Env):
    """Single-agent reveal-only Minesweeper environment.

    The observation is a dictionary with a flat ``observation`` vector and a
    same-length ``action_mask``. Actions are row-major cell reveals.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        *,
        rows: int = 5,
        columns: int = 5,
        mines: int = 3,
        render_mode: str | None = None,
    ):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}; expected None or 'ansi'")
        self.config = MinesweeperConfig(rows=rows, columns=columns, mines=mines)
        self.render_mode = render_mode
        self.board = MinesweeperBoard(self.config)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=HIDDEN,
                    high=EXPLODED_MINE,
                    shape=(self.config.cells,),
                    dtype=np.int8,
                ),
                "action_mask": spaces.MultiBinary(self.config.cells),
            }
        )
        self.action_space = spaces.Discrete(self.config.cells)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, int | float | str | None]]:
        """Reset the board and return the initial hidden observation."""

        super().reset(seed=seed)
        options = options or {}
        self.board.reset(mine_positions=options.get("mine_positions"))
        return self._observation(), self.board.info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, int | float | str | None]]:
        """Reveal one cell."""

        reward, terminated = self.board.reveal(int(action), rng=self.np_random)
        truncated = False
        return self._observation(), float(reward), bool(terminated), truncated, self.board.info()

    def render(self) -> str | None:
        """Render ANSI text when ``render_mode='ansi'``."""

        rendered = self.board.render_ansi(reveal_mines=self.board.is_terminal)
        if self.render_mode == "ansi":
            return rendered
        return None

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "observation": self.board.observation(),
            "action_mask": self.board.action_mask(),
        }


def _register_env() -> None:
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point="xrl.envs.minesweeper_v0:env")


_register_env()


__all__ = ["ENV_ID", "MinesweeperEnv", "env"]
