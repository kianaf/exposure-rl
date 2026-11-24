"""Composable interfaces for training and policy inspection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from xrl.training.episode import EpisodeMetrics, LearnerStats, LearnerStep, epsilon_for_episode


@runtime_checkable
class PolicyView(Protocol):
    """Policy/value surface shared by trainers and explainers."""

    num_actions: int

    def q_values(self, observation: np.ndarray | Iterable[int | float]) -> Sequence[float]:
        """Return action values in environment action coordinates."""

    def action_distribution(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return action probabilities in environment action coordinates."""

    def state_value(
        self,
        observation: np.ndarray | Iterable[int | float],
        action_mask: np.ndarray | Iterable[int],
        *,
        epsilon: float = 0.0,
    ) -> float:
        """Return the policy-weighted state-value estimate."""


class UpdateStrategy(Protocol):
    """Learner-specific transition update hook."""

    def observe_transition(
        self,
        pending: LearnerStep,
        reward: float,
        next_step: LearnerStep | None,
        done: bool,
        *,
        train: bool,
    ) -> float | None:
        """Record or apply one transition update."""

    def stats(self) -> LearnerStats:
        """Return learner counters for episode metrics."""


@runtime_checkable
class LearnerAdapter(Protocol):
    """Algorithm adapter consumed by the composable AEC runner."""

    learner: PolicyView

    def run_episode(
        self,
        *,
        opponent: object,
        episode: int,
        seed: int,
        epsilon: float,
        train: bool,
    ) -> EpisodeMetrics:
        """Run one environment episode."""


class ArtifactWriter(Protocol):
    """Model-specific artifact writer used by the shared runner."""

    out_dir: Path

    def write(
        self,
        learner: PolicyView,
        metrics: list[EpisodeMetrics],
        evaluation_metrics: list[dict[str, Any]],
        policy_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Write run artifacts and return a summary."""


@dataclass(frozen=True, slots=True)
class LinearExplorationSchedule:
    """Linear epsilon schedule as a composable object."""

    epsilon_start: float
    epsilon_end: float
    epsilon_decay_episodes: int

    def epsilon(self, episode: int) -> float:
        return epsilon_for_episode(
            episode,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_episodes=self.epsilon_decay_episodes,
        )
