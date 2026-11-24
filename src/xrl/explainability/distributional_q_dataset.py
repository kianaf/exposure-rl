"""Dataset construction for distributional Q surrogate training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrl.explainability.adapters import get_explanation_adapter, load_explanation_run
from xrl.explainability.tictactoe import build_tictactoe_state_distribution, tictactoe_feature_spec
from xrl.tictactoe.core import TicTacToeRules
from xrl.training.neural_q import require_neural_dependencies


@dataclass(frozen=True, slots=True)
class SurrogateTrainingDataset:
    """Cached full-state targets used to train the masked surrogate."""

    observations: np.ndarray
    action_masks: np.ndarray
    target_distributions: np.ndarray
    weights: np.ndarray

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


def require_categorical_agent(agent: Any) -> Any:
    """Return ``agent`` after asserting it is a categorical (two_hot/hl_gauss) DQN."""

    if getattr(getattr(agent, "config", None), "value_head", None) not in {"two-hot", "hl-gauss"}:
        raise ValueError("distributional Q surrogate requires a DQN run with loss 'two-hot' or 'hl-gauss'")
    return agent


def load_categorical_dqn_agent(run_dir: Path):
    """Load a saved categorical DQN run (either environment) for surrogate training."""

    return require_categorical_agent(load_explanation_run(run_dir).policy)


def categorical_q_distributions(agent, observation: np.ndarray | Iterable[int | float]) -> np.ndarray:
    """Return a categorical DQN's action-support probabilities in environment action coordinates."""

    require_neural_dependencies()
    config = getattr(agent, "config", None)
    if getattr(config, "value_head", None) not in {"two-hot", "hl-gauss"}:
        raise ValueError("categorical_q_distributions requires a two_hot or hl_gauss DQN agent")
    result = agent.q_distributions(np.asarray(observation, dtype=np.float32).reshape(-1))
    if result is None:
        raise ValueError("categorical_q_distributions requires a categorical DQN agent")
    _support, probabilities = result
    return np.asarray(probabilities, dtype=np.float32)


def build_surrogate_training_dataset(
    base_agent,
    *,
    opponent: str,
    perturbed_minimax_probability: float | None = None,
    max_points: int | None = None,
    seed: int = 0,
    rules: TicTacToeRules | None = None,
) -> SurrogateTrainingDataset:
    """Build full-observation categorical targets from reachable learner-decision states."""

    rules = rules or TicTacToeRules()
    distribution = build_tictactoe_state_distribution(
        base_agent,
        opponent=opponent,
        perturbed_minimax_probability=perturbed_minimax_probability,
        rules=rules,
        features=tictactoe_feature_spec(),
    )
    points = list(distribution.points)
    if not points:
        raise ValueError("cannot train surrogate from an empty state distribution")
    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(seed)
        probabilities = np.asarray([point.weight for point in points], dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        indices = rng.choice(len(points), size=max_points, replace=False, p=probabilities)
        points = [points[int(index)] for index in indices]

    observations = np.stack([point.observation_array for point in points]).astype(np.float32)
    action_masks = np.stack([point.action_mask_array for point in points]).astype(np.int8)
    targets = np.stack([categorical_q_distributions(base_agent, point.observation_array) for point in points]).astype(
        np.float32
    )
    weights = np.asarray([point.weight for point in points], dtype=np.float64)
    weights = weights / weights.sum()
    return SurrogateTrainingDataset(
        observations=observations,
        action_masks=action_masks,
        target_distributions=targets,
        weights=weights,
    )


def build_rollout_surrogate_dataset(
    base_agent,
    *,
    env_id: str,
    env_config: Mapping[str, Any],
    states: int,
    seed: int = 0,
    policy_epsilon: float = 0.1,
) -> SurrogateTrainingDataset:
    """Build full-observation categorical targets from epsilon-greedy rollout states.

    Rollout visitation replaces the exact conditional-occupancy enumeration on
    environments whose reachable state space cannot be enumerated (Minesweeper).
    Repeated states already carry the visitation weighting, so per-point
    weights are uniform.
    """

    adapter = get_explanation_adapter(env_id)
    background = adapter.collect_background(
        base_agent,
        env_config,
        states=states,
        seed=seed,
        policy_epsilon=policy_epsilon,
    )
    targets = np.stack(
        [categorical_q_distributions(base_agent, observation) for observation in background.observations]
    ).astype(np.float32)
    weights = np.full(background.size, 1.0 / background.size, dtype=np.float64)
    return SurrogateTrainingDataset(
        observations=background.observations,
        action_masks=background.action_masks,
        target_distributions=targets,
        weights=weights,
    )
