"""Shared trainer orchestration for neural AEC learners.

``BaseAECTrainer`` factors out the scaffolding that ``QRegressionTrainer`` and
``NeuralQTrainer`` previously duplicated almost verbatim: environment creation,
the train/evaluate entry points, runner caching, policy checkpoints, artifact
writing, and run loading. Subclasses supply only the parts that genuinely
differ — how to build the agent, how to build the learner adapter, and the run
name/tags.

The tabular ``AECTrainingLoop`` deliberately stays separate: it adds progress
reporting, unique-state tracking, and a custom evaluation row, and returns raw
``EpisodeMetrics`` from ``evaluate`` rather than summary dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from xrl.envs import tictactoe_v0
from xrl.training.analysis import minimax_action_match_summary
from xrl.training.episode import EpisodeMetrics, aec_probe_hook
from xrl.training.interfaces import LearnerAdapter, LinearExplorationSchedule, PolicyView
from xrl.training.metrics import PROBE_SEED_OFFSET
from xrl.training.metrics_io import StandardArtifactWriter, summarize_metrics
from xrl.training.progress import TrainingProgress
from xrl.training.runner import AECExperimentRunner, AECExperimentSpec
from xrl.training.tracking import NoopTracker, TrainingTracker

LearnerT = TypeVar("LearnerT", bound=PolicyView)


class BaseAECTrainer(Generic[LearnerT]):
    """Train and evaluate one neural learner through the composable AEC runner."""

    #: Filename used for the model checkpoint inside the run directory.
    model_filename: str = "model.msgpack"
    #: Config dataclass with ``from_json_dict`` used by :meth:`load_agent`.
    config_class: type

    def __init__(
        self,
        config: Any,
        *,
        env_factory=None,
        tracker: TrainingTracker | None = None,
        eval_opponents: tuple[str, ...] = ("random", "minimax"),
        progress_interval: int = 0,
    ):
        self.config = config
        self.env_factory = env_factory or tictactoe_v0.env
        self.tracker = tracker or NoopTracker()
        self.eval_opponents = eval_opponents
        self.progress_interval = max(0, int(progress_interval))
        self.evaluation_metrics: list[dict[str, Any]] = []
        self.policy_metrics: list[dict[str, Any]] = []
        self._runner: AECExperimentRunner[LearnerT] | None = None
        self._pipeline = None

    # --- subclass hooks -------------------------------------------------

    @property
    def run_name(self) -> str:
        raise NotImplementedError

    @property
    def tags(self) -> dict[str, str]:
        raise NotImplementedError

    def make_agent(self) -> LearnerT:
        raise NotImplementedError

    def _make_adapter(self, agent: LearnerT) -> LearnerAdapter:
        raise NotImplementedError

    def make_pipeline(self):
        """Return a composable pipeline for this trainer, if available."""
        return None

    # --- shared behavior ------------------------------------------------

    def make_env(self):
        return self.env_factory(**self.config.env_options)

    def _agent_init_shape(self) -> tuple[int, int]:
        """Return ``(num_actions, observation_size)`` from the environment."""
        env = self.make_env()
        observation = env.observation_space("player_0")["observation"]
        num_actions = int(env.action_space("player_0").n)
        observation_size = int(np.prod(observation.shape))
        env.close()
        return num_actions, observation_size

    def _spec(self) -> AECExperimentSpec:
        return AECExperimentSpec(
            episodes=self.config.episodes,
            seed=self.config.seed,
            opponent=self.config.opponent,
            eval_interval=self.config.eval_interval,
            eval_episodes=self.config.eval_episodes,
            params=self.config.to_json_dict(),
            run_name=self.run_name,
            tags=self.tags,
            exploration=LinearExplorationSchedule(
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay_episodes=self.config.epsilon_decay_episodes,
            ),
            perturbed_minimax_discount=getattr(self.config, "perturbed_minimax_discount", 1.0),
        )

    def _make_progress(self) -> TrainingProgress | None:
        """Build the shared reporter, or ``None`` when nothing would print."""

        if self.progress_interval <= 0 and int(getattr(self.config, "eval_interval", 0)) <= 0:
            return None
        return TrainingProgress(label=self.progress_label, interval=self.progress_interval)

    @property
    def progress_label(self) -> str:
        return ""

    def _make_runner(self, agent: LearnerT) -> AECExperimentRunner[LearnerT]:
        limit = int(getattr(self.config, "q_probe_states", 0))
        probe = None
        if limit > 0 and self.config.eval_interval > 0:
            probe = aec_probe_hook(self.make_env, limit=limit, seed=self.config.seed + PROBE_SEED_OFFSET)
        return AECExperimentRunner(
            self._spec(),
            self._make_adapter(agent),
            tracker=self.tracker,
            eval_opponents=self.eval_opponents,
            policy_checkpoint=lambda learner, episode: self.policy_checkpoint(learner, episode=episode),
            probe=probe,
            progress=self._make_progress(),
        )

    def train(self) -> tuple[LearnerT, list[EpisodeMetrics]]:
        pipeline = self.make_pipeline()
        if pipeline is not None:
            result = pipeline.fit()
            self._pipeline = pipeline
            self._runner = pipeline.runner
            self.evaluation_metrics = result.evaluation_metrics
            self.policy_metrics = result.policy_metrics
            return result.agent, result.metrics

        agent = self.make_agent()
        runner = self._make_runner(agent)
        self._runner = runner
        self.evaluation_metrics = runner.evaluation_metrics
        self.policy_metrics = runner.policy_metrics
        return runner.train()

    def evaluate(
        self,
        agent: LearnerT,
        *,
        episodes: int,
        seed: int,
        opponent_name: str | None = None,
    ) -> list[dict[str, Any]]:
        pipeline = (
            self._pipeline if self._runner is not None and self._runner.learner is agent else self.make_pipeline()
        )
        if pipeline is not None:
            metrics = pipeline.evaluate(agent=agent, episodes=episodes, opponent_name=opponent_name, seed=seed)
        else:
            runner = (
                self._runner if self._runner is not None and self._runner.learner is agent else self._make_runner(agent)
            )
            metrics = runner.evaluate(episodes=episodes, opponent_name=opponent_name, seed=seed)
        summary = summarize_metrics(metrics)
        return [
            {
                "episode": metrics[-1].episode if metrics else 0,
                "opponent": opponent_name or self.config.opponent,
                **summary,
            }
        ]

    def policy_checkpoint(self, agent: LearnerT, *, episode: int) -> dict[str, Any]:
        return {"episode": episode, **minimax_action_match_summary(agent)}

    def _artifact_writer(self) -> StandardArtifactWriter:
        return StandardArtifactWriter(
            self.config,
            save_model=lambda agent, out_dir: agent.save(out_dir / self.model_filename),
        )

    def write_run(self, agent: LearnerT, metrics: list[EpisodeMetrics]) -> dict[str, Any]:
        pipeline = (
            self._pipeline if self._runner is not None and self._runner.learner is agent else self.make_pipeline()
        )
        if pipeline is not None:
            return pipeline.write(agent=agent, metrics=metrics)

        writer = self._artifact_writer()
        if self._runner is not None and self._runner.learner is agent:
            return self._runner.write_run(writer, metrics)

        summary = writer.write(agent, metrics, self.evaluation_metrics, self.policy_metrics)
        self.tracker.log_artifacts(writer.out_dir)
        self.tracker.end_run()
        return summary

    def load_agent(self, run_dir: Path) -> LearnerT:
        config = self.config_class.from_json_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        trainer = type(self)(config)
        agent = trainer.make_agent()
        agent.load_params(run_dir / self.model_filename)
        return agent
