"""PettingZoo AEC training and evaluation loops."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xrl.envs import tictactoe_v0
from xrl.training.analysis import minimax_action_match_summary
from xrl.training.config import ExperimentConfig
from xrl.training.episode import EpisodeMetrics
from xrl.training.metrics_io import (
    summarize_learner_metrics,
    summarize_metrics,
    write_eval_metrics_csv,
    write_metrics_csv,
    write_policy_metrics_csv,
)
from xrl.training.q_learning import QLearningAgent
from xrl.training.runner import AECExperimentRunner
from xrl.training.tracking import NoopTracker, TrainingTracker

# Re-exported for backward compatibility; the canonical definitions now live in
# ``xrl.training.metrics_io``.
__all__ = [
    "AECTrainingLoop",
    "EpisodeMetrics",
    "evaluate_policy",
    "summarize_learner_metrics",
    "summarize_metrics",
    "training_progress_summary",
    "write_eval_metrics_csv",
    "write_metrics_csv",
    "write_policy_metrics_csv",
]


class AECTrainingLoop:
    """Train and evaluate a learner through a PettingZoo AEC env."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        env_factory: Callable[..., object] | None = None,
        progress_reporter: Callable[[dict[str, object]], None] | None = None,
        progress_interval: int = 0,
        progress_window: int = 100,
        progress: Any | None = None,
        tracker: TrainingTracker | None = None,
        eval_opponents: tuple[str, ...] = ("random", "minimax"),
    ) -> None:
        self.config = config
        self.env_factory = env_factory or tictactoe_v0.env
        self.progress_reporter = progress_reporter
        self.progress_interval = max(0, progress_interval)
        self.progress_window = max(1, progress_window)
        self.progress = progress
        self.tracker = tracker or NoopTracker()
        self.eval_opponents = eval_opponents
        self.evaluation_metrics: list[dict[str, object]] = []
        self.policy_metrics: list[dict[str, object]] = []
        self._runner: AECExperimentRunner[QLearningAgent] | None = None
        self._pipeline = None

    def make_env(self):
        return self.env_factory(**self.config.env_options)

    def train(self) -> tuple[QLearningAgent, list[EpisodeMetrics]]:
        pipeline = self._make_pipeline()
        result = pipeline.fit()
        self._pipeline = pipeline
        self._runner = pipeline.runner
        self.evaluation_metrics = result.evaluation_metrics
        self.policy_metrics = result.policy_metrics
        return result.agent, result.metrics

    def evaluate(
        self,
        learner: QLearningAgent,
        *,
        episodes: int | None = None,
        opponent_name: str | None = None,
        seed: int | None = None,
    ) -> list[EpisodeMetrics]:
        pipeline = (
            self._pipeline if self._runner is not None and self._runner.learner is learner else self._make_pipeline()
        )
        return pipeline.evaluate(agent=learner, episodes=episodes, opponent_name=opponent_name, seed=seed)

    def policy_checkpoint(self, learner: QLearningAgent, *, episode: int) -> dict[str, object]:
        return {"episode": episode, **minimax_action_match_summary(learner)}

    def write_run(self, learner: QLearningAgent, metrics: list[EpisodeMetrics]) -> dict[str, object]:
        """Write experiment artifacts and return the summary."""

        pipeline = (
            self._pipeline if self._runner is not None and self._runner.learner is learner else self._make_pipeline()
        )
        return pipeline.write(agent=learner, metrics=metrics)

    def _make_pipeline(self):
        from xrl.training.pipeline import make_tabular_pipeline

        return make_tabular_pipeline(
            self.config,
            env_factory=self.env_factory,
            progress_reporter=self.progress_reporter,
            progress_interval=self.progress_interval,
            progress_window=self.progress_window,
            progress=self.progress,
            tracker=self.tracker,
            eval_opponents=self.eval_opponents,
        )


def training_progress_summary(
    metrics: list[EpisodeMetrics],
    *,
    total_episodes: int,
    window: int,
) -> dict[str, object]:
    from xrl.training.pipeline import tabular_training_progress_summary

    return tabular_training_progress_summary(metrics, total_episodes=total_episodes, window=window)


def evaluate_policy(
    learner: QLearningAgent,
    config: ExperimentConfig,
    *,
    episodes: int,
    opponent: str,
    seed: int | None = None,
) -> list[EpisodeMetrics]:
    loop = AECTrainingLoop(config)
    return loop.evaluate(learner, episodes=episodes, opponent_name=opponent, seed=seed)
