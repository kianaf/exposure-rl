"""Experiment configuration for tabular RL runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_ENCODERS = ("identity", "tictactoe_symmetry")
VALID_EXPLORATIONS = ("epsilon_greedy", "boltzmann")
VALID_BACKUPS = ("hard", "soft")


def parse_hidden_sizes(value: str | Sequence[int]) -> tuple[int, ...]:
    """Normalize comma-separated or programmatic MLP widths."""

    message = "hidden_sizes must be a non-empty sequence of positive integers"
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError(message)
        try:
            parsed = tuple(int(part) for part in parts)
        except ValueError as error:
            raise ValueError(message) from error
    else:
        parsed = tuple(int(size) for size in value)
    if not parsed or any(size < 1 for size in parsed):
        raise ValueError(message)
    return parsed


def validate_exploration_config(*, exploration: str, backup: str, temperature: float) -> None:
    """Validate the exploration/backup knobs shared by the neural DQN configs.

    The temperature is only consumed by Boltzmann sampling and the soft backup,
    so it must be positive whenever either non-default option is selected.
    """

    if exploration not in VALID_EXPLORATIONS:
        raise ValueError(f"exploration must be one of {VALID_EXPLORATIONS}, got {exploration!r}")
    if backup not in VALID_BACKUPS:
        raise ValueError(f"backup must be one of {VALID_BACKUPS}, got {backup!r}")
    if (exploration == "boltzmann" or backup == "soft") and temperature <= 0.0:
        raise ValueError(f"temperature must be positive for boltzmann exploration or soft backup, got {temperature}")


def validate_common_config(
    *,
    episodes: int,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    perturbed_minimax_discount: float = 1.0,
    encoder: str = "identity",
    min_episodes: int = 1,
    metrics_window: int = 100,
) -> None:
    """Validate the hyperparameters shared by every trainer config.

    Centralizing these checks keeps the tabular, Q-regression, and DQN configs
    in agreement; previously each repeated (or omitted) them independently.

    ``min_episodes`` is 1 for trainers but 0 for the tabular config, which is
    constructed with ``episodes=0`` for evaluation-only runs (see ``eval-q``).
    """

    if episodes < min_episodes:
        raise ValueError(f"episodes must be at least {min_episodes}, got {episodes}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= epsilon_start <= 1.0:
        raise ValueError(f"epsilon_start must be in [0, 1], got {epsilon_start}")
    if not 0.0 <= epsilon_end <= 1.0:
        raise ValueError(f"epsilon_end must be in [0, 1], got {epsilon_end}")
    if epsilon_decay_episodes < 0:
        raise ValueError(f"epsilon_decay_episodes must be non-negative, got {epsilon_decay_episodes}")
    if not 0.0 <= perturbed_minimax_discount <= 1.0:
        raise ValueError(f"perturbed_minimax_discount must be in [0, 1], got {perturbed_minimax_discount}")
    if encoder not in VALID_ENCODERS:
        raise ValueError(f"encoder must be one of {VALID_ENCODERS}, got {encoder!r}")
    if metrics_window < 1:
        raise ValueError(f"metrics_window must be positive, got {metrics_window}")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for a Q-learning experiment run."""

    episodes: int = 10_000
    seed: int = 0
    alpha: float = 0.3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 8_000
    opponent: str = "random"
    eval_interval: int = 1_000
    eval_episodes: int = 200
    out_dir: Path = Path("runs/q_random")
    env_id: str = "tictactoe_v0"
    env_options: dict[str, Any] = field(default_factory=lambda: {"obs_mode": "flat"})
    perturbed_minimax_discount: float = 1.0
    encoder: str = "identity"
    metrics_window: int = 100

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        validate_common_config(
            episodes=self.episodes,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_episodes=self.epsilon_decay_episodes,
            perturbed_minimax_discount=self.perturbed_minimax_discount,
            encoder=self.encoder,
            min_episodes=0,
            metrics_window=self.metrics_window,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        normalized = dict(data)
        normalized.pop("mask_probability", None)
        normalized.pop("mask_token", None)
        normalized.pop("sample_replays", None)
        normalized.pop("perturbed_minimax_probability", None)
        normalized["out_dir"] = Path(normalized["out_dir"])
        return cls(**normalized)
