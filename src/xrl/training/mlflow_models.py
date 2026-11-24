"""MLflow pyfunc wrappers for saved TicTacToe Q policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from xrl.training.q_learning import QLearningAgent
from xrl.training.value_heads import SCALAR, normalize_value_head

try:  # Optional monitoring dependency.
    import mlflow.pyfunc

    _PythonModelBase = mlflow.pyfunc.PythonModel
except ModuleNotFoundError:  # pragma: no cover - import without optional extra.
    mlflow = None
    _PythonModelBase = object


class TabularQPolicyPyFunc(_PythonModelBase):
    """Serve a saved tabular Q-table through MLflow's pyfunc API."""

    def load_context(self, context) -> None:
        self.agent = QLearningAgent.load(context.artifacts["q_table"])

    def predict(self, context, model_input):
        rows = _input_rows(model_input)
        outputs = []
        for row in rows:
            observation = np.asarray(row, dtype=np.float32).reshape(-1)
            mask = _legal_mask_from_observation(observation)
            q_values = self.agent.q_values_for_display(observation)
            action = self.agent.select_action(observation, mask, epsilon=0.0)
            outputs.append(
                {"action": int(action), **{f"q_{index}": float(value) for index, value in enumerate(q_values)}}
            )
        return _output_frame(outputs)


class FlaxQPolicyPyFunc(_PythonModelBase):
    """Serve a saved Flax neural Q model through MLflow's pyfunc API."""

    def load_context(self, context) -> None:
        from xrl.training.neural_q import ClassificationDQNAgent, NeuralQConfig, RegressionDQNAgent

        config = NeuralQConfig.from_json_dict(json.loads(Path(context.artifacts["config"]).read_text(encoding="utf-8")))
        self.agent = (
            RegressionDQNAgent(config, num_actions=9, observation_size=9)
            if normalize_value_head(getattr(config, "value_head", SCALAR)) == SCALAR
            else ClassificationDQNAgent(config, num_actions=9, observation_size=9)
        )
        self.agent.load_params(Path(context.artifacts["model"]))

    def predict(self, context, model_input):
        rows = _input_rows(model_input)
        outputs = []
        for row in rows:
            observation = np.asarray(row, dtype=np.float32).reshape(-1)
            mask = _legal_mask_from_observation(observation)
            q_values = self.agent.q_values(observation)
            action = self.agent.select_action(observation, mask, epsilon=0.0)
            outputs.append(
                {"action": int(action), **{f"q_{index}": float(value) for index, value in enumerate(q_values)}}
            )
        return _output_frame(outputs)


def _input_rows(model_input: Any) -> np.ndarray:
    rows = model_input.values if hasattr(model_input, "values") else model_input
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] != 9:
        raise ValueError(f"TicTacToe pyfunc policies expect 9 flat observation columns, got {rows.shape[1]}")
    return rows


def _legal_mask_from_observation(observation: np.ndarray) -> np.ndarray:
    return (np.asarray(observation).reshape(-1) == 0).astype(np.int8)


def _output_frame(rows: list[dict[str, float | int]]):
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ModuleNotFoundError:  # pragma: no cover - pandas is part of monitoring extra.
        return rows
