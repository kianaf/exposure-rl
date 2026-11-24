"""Env-agnostic adapters that load trained runs into explanation workflows.

Each supported environment registers one :class:`EnvExplanationAdapter` that
knows how to rebuild the environment, load the trained policy, name the
observation features, and collect rollout backgrounds. Explanation code and
the CLI go through :func:`load_explanation_run`, so adding an environment
means registering one adapter — no explainer changes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from xrl._optional import require_neural_dependencies
from xrl.envs import minesweeper_v0, tictactoe_v0
from xrl.explainability.backgrounds import (
    BackgroundDataset,
    collect_aec_background,
    collect_gymnasium_background,
)
from xrl.explainability.core import ExplanationResult, FeatureSpec, PolicyView
from xrl.explainability.tictactoe import tictactoe_feature_spec
from xrl.minesweeper import EXPLODED_MINE, HIDDEN
from xrl.training.q_learning import QLearningAgent
from xrl.training.value_heads import SCALAR, normalize_value_head

TICTACTOE_ENV_ID = "tictactoe_v0"
MINESWEEPER_ENV_ID = "minesweeper_v0"


def run_model_type(run_dir: Path, config: Mapping[str, Any]) -> str | None:
    """Infer the model family of a saved run from its artifacts and config."""

    if (run_dir / "q_table.json").exists():
        return "tabular_q"
    if not (run_dir / "model.msgpack").exists():
        return None
    if "replay_capacity" in config or "value_head" in config or "loss" in config:
        # ``loss`` is the pre-ADR-0012 spelling; saved runs still carry it.
        head = config.get("value_head", config.get("loss", SCALAR))
        return f"dqn_{normalize_value_head(str(head))}"
    return "q_regression"


def minesweeper_feature_spec(rows: int, columns: int) -> FeatureSpec:
    """Feature spec naming each Minesweeper cell in row-major action order."""

    return FeatureSpec(
        names=tuple(f"cell_r{row}c{column}" for row in range(rows) for column in range(columns)),
        display_metadata=tuple({"row": row, "col": column} for row in range(rows) for column in range(columns)),
    )


@runtime_checkable
class EnvExplanationAdapter(Protocol):
    """Per-environment surface consumed by explanation workflows."""

    env_id: str

    def feature_spec(self, config: Mapping[str, Any]) -> FeatureSpec:
        """Return the observation feature naming for a run config."""

    def make_env(self, config: Mapping[str, Any]) -> Callable[[], Any]:
        """Return a zero-argument environment factory bound to the run config."""

    def load_policy(self, run_dir: Path, config: Mapping[str, Any]) -> PolicyView:
        """Load the trained policy saved in ``run_dir``."""

    def cohort_key(self, observation: np.ndarray) -> int:
        """Return the cohort grouping key for one observation."""

    def collect_background(
        self,
        policy: PolicyView | None,
        config: Mapping[str, Any],
        *,
        states: int,
        seed: int,
        policy_epsilon: float = 0.0,
        random_policy: bool = False,
    ) -> BackgroundDataset:
        """Collect a rollout background dataset for this environment."""

    def render_attributions(
        self,
        observation: np.ndarray,
        result: ExplanationResult,
        config: Mapping[str, Any],
    ) -> str:
        """Render the observation and its attributions as an ANSI grid."""


@dataclass(frozen=True, slots=True)
class LoadedExplanationRun:
    """A trained run resolved into everything explanation methods consume."""

    run_dir: Path
    env_id: str
    model_type: str
    config: dict[str, Any]
    policy: PolicyView
    feature_spec: FeatureSpec
    num_actions: int
    observation_size: int
    make_env: Callable[[], Any]
    adapter: EnvExplanationAdapter


_ADAPTERS: dict[str, EnvExplanationAdapter] = {}


def register_explanation_adapter(adapter: EnvExplanationAdapter) -> None:
    """Register an adapter under its ``env_id``, replacing any previous one."""

    _ADAPTERS[adapter.env_id] = adapter


def get_explanation_adapter(env_id: str) -> EnvExplanationAdapter:
    """Return the adapter registered for ``env_id``."""

    adapter = _ADAPTERS.get(env_id)
    if adapter is None:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"no explanation adapter registered for env {env_id!r}; known env ids: {known}")
    return adapter


def run_env_id(run_dir: str | Path) -> str:
    """Return a saved run's env id without loading its model."""

    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"run config must be a JSON object: {config_path}")
    return str(config.get("env_id", TICTACTOE_ENV_ID))


def load_explanation_run(run_dir: str | Path) -> LoadedExplanationRun:
    """Load a saved training run and resolve its explanation surface.

    Dispatches on ``env_id`` in ``config.json``; configs written before the
    field existed are TicTacToe runs, so absence defaults to TicTacToe.
    """

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"run config must be a JSON object: {config_path}")

    env_id = str(config.get("env_id", TICTACTOE_ENV_ID))
    adapter = get_explanation_adapter(env_id)
    model_type = run_model_type(run_dir, config)
    if model_type is None:
        raise FileNotFoundError(f"no model artifact (q_table.json or model.msgpack) found in {run_dir}")

    policy = adapter.load_policy(run_dir, config)
    features = adapter.feature_spec(config)
    return LoadedExplanationRun(
        run_dir=run_dir,
        env_id=env_id,
        model_type=model_type,
        config=config,
        policy=policy,
        feature_spec=features,
        num_actions=int(policy.num_actions),
        observation_size=features.feature_count,
        make_env=adapter.make_env(config),
        adapter=adapter,
    )


class TicTacToeExplanationAdapter:
    """Explanation adapter for the PettingZoo TicTacToe environment."""

    env_id = TICTACTOE_ENV_ID

    def feature_spec(self, config: Mapping[str, Any]) -> FeatureSpec:
        _require_flat_observations(config)
        return tictactoe_feature_spec()

    def make_env(self, config: Mapping[str, Any]) -> Callable[[], Any]:
        _require_flat_observations(config)
        return lambda: tictactoe_v0.env(obs_mode="flat")

    def load_policy(self, run_dir: Path, config: Mapping[str, Any]) -> PolicyView:
        model_type = run_model_type(run_dir, config)
        if model_type == "tabular_q":
            return QLearningAgent.load(run_dir / "q_table.json")
        if model_type == "q_regression":
            require_neural_dependencies("Loading a q_regression run")
            from xrl.training.q_regression import QRegressionConfig, QRegressionTrainer

            return QRegressionTrainer(QRegressionConfig.from_json_dict(dict(config))).load_agent(run_dir)
        if model_type is not None and model_type.startswith("dqn"):
            require_neural_dependencies("Loading a DQN run")
            from xrl.training.neural_q import NeuralQConfig, NeuralQTrainer

            return NeuralQTrainer(NeuralQConfig.from_json_dict(dict(config))).load_agent(run_dir)
        raise ValueError(f"unsupported trained run type {model_type!r} in {run_dir}")

    def cohort_key(self, observation: np.ndarray) -> int:
        # Move number: marks already placed on the board.
        return int(np.count_nonzero(np.asarray(observation).reshape(-1)))

    def collect_background(
        self,
        policy: PolicyView | None,
        config: Mapping[str, Any],
        *,
        states: int,
        seed: int,
        policy_epsilon: float = 0.0,
        random_policy: bool = False,
    ) -> BackgroundDataset:
        return collect_aec_background(
            self.make_env(config),
            policy,
            env_id=self.env_id,
            states=states,
            seed=seed,
            policy_epsilon=policy_epsilon,
            random_policy=random_policy,
        )

    def render_attributions(
        self,
        observation: np.ndarray,
        result: ExplanationResult,
        config: Mapping[str, Any],
    ) -> str:
        return render_attribution_grid(observation, result, rows=3, columns=3, formatter=_tictactoe_cell)


class MinesweeperExplanationAdapter:
    """Explanation adapter for the Gymnasium Minesweeper environment."""

    env_id = MINESWEEPER_ENV_ID

    def feature_spec(self, config: Mapping[str, Any]) -> FeatureSpec:
        rows, columns, _mines = _minesweeper_geometry(config)
        return minesweeper_feature_spec(rows, columns)

    def make_env(self, config: Mapping[str, Any]) -> Callable[[], Any]:
        rows, columns, mines = _minesweeper_geometry(config)
        return lambda: minesweeper_v0.env(rows=rows, columns=columns, mines=mines)

    def load_policy(self, run_dir: Path, config: Mapping[str, Any]) -> PolicyView:
        require_neural_dependencies("Loading a Minesweeper DQN run")
        from xrl.training.minesweeper_dqn import MinesweeperDQNConfig, MinesweeperDQNTrainer

        trainer_config = MinesweeperDQNConfig.from_json_dict(dict(config))
        return MinesweeperDQNTrainer(trainer_config).load_agent(run_dir)

    def cohort_key(self, observation: np.ndarray) -> int:
        # Progress within the game: cells already revealed.
        return int(np.count_nonzero(np.asarray(observation).reshape(-1) != float(HIDDEN)))

    def collect_background(
        self,
        policy: PolicyView | None,
        config: Mapping[str, Any],
        *,
        states: int,
        seed: int,
        policy_epsilon: float = 0.0,
        random_policy: bool = False,
    ) -> BackgroundDataset:
        return collect_gymnasium_background(
            self.make_env(config),
            policy,
            env_id=self.env_id,
            states=states,
            seed=seed,
            policy_epsilon=policy_epsilon,
            random_policy=random_policy,
        )

    def render_attributions(
        self,
        observation: np.ndarray,
        result: ExplanationResult,
        config: Mapping[str, Any],
    ) -> str:
        rows, columns, _mines = _minesweeper_geometry(config)
        return render_attribution_grid(observation, result, rows=rows, columns=columns, formatter=_minesweeper_cell)


def render_attribution_grid(
    observation: np.ndarray,
    result: ExplanationResult,
    *,
    rows: int,
    columns: int,
    formatter: Callable[[float], str],
) -> str:
    """Render a board grid next to its per-cell attribution grid."""

    board = np.asarray(observation).reshape(rows, columns)
    values = np.asarray([result.attributions[name] for name in result.feature_names]).reshape(rows, columns)
    board_lines = ["  ".join(f"{formatter(float(cell)):>2}" for cell in row) for row in board]
    value_lines = ["  ".join(f"{cell:+.3f}" for cell in row) for row in values]
    header = f"target={result.target}" + ("" if result.action is None else f" action={result.action}")
    return "\n".join(
        [
            f"board ({header}):",
            *board_lines,
            f"attributions (baseline={result.baseline_value:+.4f}, full={result.full_value:+.4f}):",
            *value_lines,
        ]
    )


def _tictactoe_cell(value: float) -> str:
    if value > 0:
        return "X"
    if value < 0:
        return "O"
    return "."


def _minesweeper_cell(value: float) -> str:
    if value == float(HIDDEN):
        return "#"
    if value == float(EXPLODED_MINE):
        return "*"
    return str(int(value))


def _require_flat_observations(config: Mapping[str, Any]) -> None:
    env_options = config.get("env_options") or {}
    obs_mode = env_options.get("obs_mode", "flat")
    if obs_mode != "flat":
        raise ValueError(f"explanations require the 'flat' observation mode, run was trained with {obs_mode!r}")


def _minesweeper_geometry(config: Mapping[str, Any]) -> tuple[int, int, int]:
    env_options = config.get("env_options") or {}
    values = []
    for key in ("rows", "columns", "mines"):
        value = config.get(key, env_options.get(key))
        if value is None:
            raise ValueError(f"minesweeper run config is missing {key!r}")
        values.append(int(value))
    return values[0], values[1], values[2]


register_explanation_adapter(TicTacToeExplanationAdapter())
register_explanation_adapter(MinesweeperExplanationAdapter())
