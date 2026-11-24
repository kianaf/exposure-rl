"""Board-symmetry diagnostics for Minesweeper Q-values.

Symmetric boards must assign equal Q-values to symmetric actions. These
helpers measure how far a trained agent deviates from that invariant: orbit
gaps on the all-hidden initial board, and transform-consistency gaps
``|Q(s, a) - Q(g s, g a)|`` on sampled states. See ADR 0007 for how the
metric anchors the epsilon-greedy / boltzmann / soft-backup ablation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from xrl.minesweeper import HIDDEN


def board_symmetry_permutations(rows: int, columns: int) -> dict[str, np.ndarray]:
    """Return named cell-index permutations for the board's symmetry group.

    Each permutation ``perm`` transforms a flat row-major board vector via
    ``vector[perm]``: the cell at transformed position ``k`` takes the value of
    original position ``perm[k]``. Square boards get the full dihedral group
    ``D4``; rectangular boards get the flip/rotation Klein group.
    """

    if rows < 1 or columns < 1:
        raise ValueError(f"rows and columns must be positive, got {rows}x{columns}")
    grid = np.arange(rows * columns).reshape(rows, columns)
    named = {
        "identity": grid,
        "rot180": np.rot90(grid, 2),
        "flip_horizontal": np.fliplr(grid),
        "flip_vertical": np.flipud(grid),
    }
    if rows == columns:
        named.update(
            {
                "rot90": np.rot90(grid, 1),
                "rot270": np.rot90(grid, 3),
                "transpose": grid.T,
                "anti_transpose": np.rot90(grid, 2).T,
            }
        )
    return {name: transformed.reshape(-1).copy() for name, transformed in named.items()}


def initial_board_orbits(rows: int, columns: int) -> list[list[int]]:
    """Return cell-index orbits under the board's symmetry group.

    On the all-hidden initial board every symmetry fixes the state, so all
    cells within one orbit must have identical Q-values.
    """

    permutations = board_symmetry_permutations(rows, columns)
    parents = list(range(rows * columns))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for perm in permutations.values():
        for destination, source in enumerate(perm):
            parents[find(destination)] = find(int(source))

    orbits: dict[int, list[int]] = {}
    for cell in range(rows * columns):
        orbits.setdefault(find(cell), []).append(cell)
    return sorted((sorted(orbit) for orbit in orbits.values()), key=lambda orbit: orbit[0])


def initial_orbit_gaps(
    q_values: np.ndarray | Sequence[float],
    rows: int,
    columns: int,
) -> dict[str, Any]:
    """Return per-orbit Q-value spreads for the all-hidden initial board."""

    values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    if values.shape != (rows * columns,):
        raise ValueError(f"q_values must have {rows * columns} entries, got {values.shape}")
    orbits = initial_board_orbits(rows, columns)
    gaps = [float(values[orbit].max() - values[orbit].min()) for orbit in orbits]
    return {
        "orbits": orbits,
        "orbit_gaps": gaps,
        "max_orbit_gap": max(gaps),
        "mean_orbit_gap": float(np.mean(gaps)),
    }


def state_symmetry_gap(
    q_fn: Callable[[np.ndarray], np.ndarray],
    observations: Sequence[np.ndarray],
    rows: int,
    columns: int,
) -> dict[str, Any]:
    """Compare ``Q(s, a)`` with ``Q(g s, g a)`` over hidden cells for each symmetry.

    Args:
        q_fn: Maps a flat board observation to per-cell Q-values.
        observations: Flat board observations to check.
        rows: Board rows.
        columns: Board columns.

    Returns:
        Aggregate mean/max absolute gaps plus comparison counts.
    """

    permutations = {
        name: perm for name, perm in board_symmetry_permutations(rows, columns).items() if name != "identity"
    }
    gaps: list[float] = []
    for observation in observations:
        board = np.asarray(observation, dtype=np.float32).reshape(-1)
        base_q = np.asarray(q_fn(board), dtype=np.float64)
        hidden = board == HIDDEN
        for perm in permutations.values():
            legal = hidden[perm]
            if not legal.any():
                continue
            transformed_q = np.asarray(q_fn(board[perm]), dtype=np.float64)
            gaps.extend(np.abs(transformed_q[legal] - base_q[perm][legal]).tolist())
    return {
        "states": len(observations),
        "comparisons": len(gaps),
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap": float(np.max(gaps)) if gaps else 0.0,
    }


def collect_policy_observations(
    agent,
    *,
    rows: int,
    columns: int,
    mines: int,
    states: int,
    seed: int,
) -> list[np.ndarray]:
    """Roll out the agent's own policy and collect pre-action observations."""

    from xrl.envs import minesweeper_v0

    observations: list[np.ndarray] = []
    episode = 0
    while len(observations) < states:
        env = minesweeper_v0.env(rows=rows, columns=columns, mines=mines)
        observation, _ = env.reset(seed=seed + episode)
        try:
            while len(observations) < states:
                board = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
                mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
                observations.append(board)
                action = agent.select_action(board, mask, epsilon=0.0)
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
        episode += 1
    return observations


def minesweeper_q_symmetry(
    run: Annotated[Path, typer.Option("--run", help="Saved Minesweeper DQN run directory.")],
    states: int = 32,
    seed: int = 0,
) -> None:
    """Report board-symmetry gaps for a saved Minesweeper DQN run.

    Prints a JSON report with per-orbit Q-value spreads on the all-hidden
    initial board and aggregate ``|Q(s, a) - Q(g s, g a)|`` gaps over
    observations sampled from the agent's own policy. Symmetric training
    should drive both toward zero; see ADR 0007.
    """

    from xrl.training.minesweeper_dqn import MinesweeperDQNConfig, MinesweeperDQNTrainer

    run_dir = Path(run)
    config_path = run_dir / "config.json"
    model_path = run_dir / MinesweeperDQNTrainer.model_filename
    if not config_path.exists():
        raise FileNotFoundError(f"missing Minesweeper run config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing Minesweeper model: {model_path}")
    if states < 1:
        raise ValueError(f"states must be positive, got {states}")

    config = MinesweeperDQNConfig.from_json_dict(json.loads(config_path.read_text(encoding="utf-8")))
    trainer = MinesweeperDQNTrainer(config)
    agent = trainer.load_agent(run_dir)

    initial_board = np.full(config.rows * config.columns, HIDDEN, dtype=np.float32)
    initial = initial_orbit_gaps(agent.q_values(initial_board), config.rows, config.columns)
    observations = collect_policy_observations(
        agent,
        rows=config.rows,
        columns=config.columns,
        mines=config.mines,
        states=states,
        seed=seed,
    )
    sampled = state_symmetry_gap(agent.q_values, observations, config.rows, config.columns)

    print(
        json.dumps(
            {
                "run": str(run_dir),
                "rows": config.rows,
                "columns": config.columns,
                "mines": config.mines,
                "exploration": config.exploration,
                "backup": config.backup,
                "temperature": config.temperature,
                "symmetry_group": "d4" if config.rows == config.columns else "klein",
                "initial_board": initial,
                "sampled_states": sampled,
            },
            indent=2,
        )
    )
