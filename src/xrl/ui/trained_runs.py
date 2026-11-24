"""Discovery and loading helpers for trained Tic Tac Toe runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from xrl._optional import neural_dependencies_available
from xrl.tictactoe.core import GameState, PlayerMark
from xrl.training.q_learning import QLearningAgent


class TrainedPolicy(Protocol):
    """Callable policy loaded from a saved training run."""

    num_actions: int

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        """Return one legal action."""

    def q_values(self, observation: np.ndarray) -> list[float]:
        """Return Q-values in environment action coordinates."""

    def action_distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return action probabilities in environment action coordinates."""

    def state_value(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> float:
        """Return the policy-weighted state-value estimate."""

    def q_value_support(self) -> list[float] | None:
        """Return categorical value support bins when available."""

    def q_value_distributions(self, observation: np.ndarray) -> list[list[float]] | None:
        """Return categorical value probabilities in environment action coordinates when available."""


@dataclass(frozen=True, slots=True)
class TrainedRun:
    """Metadata for one saved trained run."""

    id: str
    path: Path
    model_type: str
    loadable: bool
    status: str
    config: dict[str, Any]
    summary: dict[str, Any]
    latest_evaluations: list[dict[str, Any]]
    latest_policy: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "model_type": self.model_type,
            "label": self.label,
            "loadable": self.loadable,
            "status": self.status,
            "config": self.config,
            "summary": self.summary,
            "latest_evaluations": self.latest_evaluations,
            "latest_policy": self.latest_policy,
            "training": self.training_summary,
        }

    @property
    def label(self) -> str:
        opponent = self.config.get("opponent", "unknown")
        episodes = self.config.get("episodes", "?")
        return f"{self.model_type} vs {opponent} ({episodes} episodes)"

    @property
    def training_summary(self) -> dict[str, Any]:
        keys = [
            "episodes",
            "seed",
            "opponent",
            "encoder",
            "alpha",
            "learning_rate",
            "gamma",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay_episodes",
            "value_head",
            "replay_capacity",
            "replay_warmup",
            "batch_size",
            "target_sync_interval",
        ]
        return {key: self.config[key] for key in keys if key in self.config}


@dataclass(slots=True)
class _LoadedPolicy:
    agent: object

    @property
    def num_actions(self) -> int:
        return int(self.agent.num_actions)

    @property
    def observation_size(self) -> int:
        return int(self.agent.observation_size)

    @property
    def config(self) -> object:
        return self.agent.config

    @property
    def support(self) -> object:
        return getattr(self.agent, "support", ())

    def __call__(
        self,
        observation: dict[str, np.ndarray],
        action_mask: np.ndarray,
        *,
        state: GameState,
        player: PlayerMark,
    ) -> int:
        return self.agent.select_action(observation["observation"], action_mask, epsilon=0.0)

    def q_values(self, observation: np.ndarray) -> list[float]:
        values = self.agent.q_values(observation)
        return [float(value) for value in values]

    def masked_q_values(self, observation: np.ndarray, known_mask: np.ndarray) -> list[float]:
        method = getattr(self.agent, "masked_q_values", None)
        if not callable(method):
            raise TypeError("the loaded policy does not support masked Q queries")
        values = method(observation, known_mask)
        return [float(value) for value in values]

    def masked_q_distributions(
        self,
        observation: np.ndarray,
        known_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        method = getattr(self.agent, "masked_q_distributions", None)
        if not callable(method):
            raise TypeError("the loaded policy does not support masked Q-distribution queries")
        support, probabilities = method(observation, known_mask)
        return np.asarray(support, dtype=np.float32), np.asarray(probabilities, dtype=np.float32)

    def action_distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        return self.agent.action_distribution(observation, action_mask, epsilon=epsilon)

    def state_value(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> float:
        return float(self.agent.state_value(observation, action_mask, epsilon=epsilon))

    def q_value_support(self) -> list[float] | None:
        if not self._has_categorical_q_values():
            return None
        return [float(value) for value in np.asarray(self.agent.support, dtype=np.float32)]

    def q_value_distributions(self, observation: np.ndarray) -> list[list[float]] | None:
        if not self._has_categorical_q_values():
            return None

        result = self.agent.q_distributions(np.asarray(observation, dtype=np.float32))
        if result is None:
            return None
        _support, probabilities = result
        return [[float(value) for value in row] for row in probabilities]

    def _has_categorical_q_values(self) -> bool:
        config = getattr(self.agent, "config", None)
        return getattr(config, "value_head", None) in {"two-hot", "hl-gauss"} and hasattr(self.agent, "support")


def discover_trained_runs(runs_dir: str | Path) -> list[TrainedRun]:
    """Return loadable-looking trained runs found below ``runs_dir``."""

    root = Path(runs_dir)
    if not root.exists():
        return []

    runs: list[TrainedRun] = []
    for config_path in sorted(root.rglob("config.json")):
        run_dir = config_path.parent
        run = _run_from_dir(root, run_dir)
        if run is not None:
            runs.append(run)
    runs.sort(key=_run_sort_key, reverse=True)
    return runs


def _run_sort_key(run: TrainedRun) -> tuple[str, float, str]:
    """Sort dated run directories newest-first while keeping stable fallback ordering."""

    return (_timestamp_token(run.id), _path_mtime(run.path), run.id)


def _timestamp_token(value: str) -> str:
    match = re.search(r"(20\d{6})[-T]?(\d{6})", value)
    return "" if match is None else "".join(match.groups())


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def load_trained_policy(run: TrainedRun | str, runs: list[TrainedRun] | None = None) -> TrainedPolicy:
    """Load a greedy policy from a discovered trained run."""

    selected = run if isinstance(run, TrainedRun) else _find_run(run, runs or [])
    if not selected.loadable:
        raise RuntimeError(selected.status)

    if selected.model_type == "tabular_q":
        return _LoadedPolicy(QLearningAgent.load(selected.path / "q_table.json"))

    if selected.model_type == "q_regression":
        from xrl.training.q_regression import QRegressionConfig, QRegressionTrainer

        config = QRegressionConfig.from_json_dict(selected.config)
        agent = QRegressionTrainer(config).load_agent(selected.path)
        return _LoadedPolicy(agent)

    if selected.model_type.startswith("dqn"):
        from xrl.training.neural_q import NeuralQConfig, NeuralQTrainer

        config = NeuralQConfig.from_json_dict(selected.config)
        agent = NeuralQTrainer(config).load_agent(selected.path)
        return _LoadedPolicy(agent)

    raise ValueError(f"unsupported trained run type {selected.model_type!r}")


def _run_from_dir(root: Path, run_dir: Path) -> TrainedRun | None:
    try:
        config = _read_json(run_dir / "config.json")
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None

    model_type = _model_type(run_dir, config)
    if model_type is None:
        return None

    loadable = True
    status = "ready"
    if model_type != "tabular_q" and not neural_dependencies_available():
        loadable = False
        status = "Neural dependencies are unavailable; install xrl[neural]."

    return TrainedRun(
        id=_run_id(root, run_dir),
        path=run_dir,
        model_type=model_type,
        loadable=loadable,
        status=status,
        config=config,
        summary=_read_optional_json(run_dir / "summary.json"),
        latest_evaluations=_latest_rows_by_key(run_dir / "eval_metrics.csv", "opponent"),
        latest_policy=_latest_row(run_dir / "policy_metrics.csv"),
    )


def _model_type(run_dir: Path, config: dict[str, Any]) -> str | None:
    from xrl.explainability.adapters import run_model_type

    return run_model_type(run_dir, config)


def _find_run(run_id: str, runs: list[TrainedRun]) -> TrainedRun:
    for run in runs:
        if run.id == run_id:
            return run
    raise ValueError(f"unknown trained run {run_id!r}")


def _run_id(root: Path, run_dir: Path) -> str:
    try:
        return run_dir.relative_to(root).as_posix()
    except ValueError:
        return run_dir.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_rows_by_key(path: Path, key: str) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get(key, ""))] = row
    return [row for _, row in sorted(latest.items())]


def _latest_row(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    return rows[-1] if rows else {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    except OSError:
        return []


def _coerce_value(value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped in {"True", "False"}:
        return stripped == "True"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped
