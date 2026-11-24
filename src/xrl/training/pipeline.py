"""Composable sklearn-style pipeline for XRL experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from xrl.envs import tictactoe_v0
from xrl.training.analysis import bellwether_values, is_unmasked_tictactoe_key, minimax_action_match_summary
from xrl.training.config import ExperimentConfig
from xrl.training.episode import EpisodeMetrics, LearnerStats, LearnerStep, aec_probe_hook, run_aec_episode
from xrl.training.interfaces import ArtifactWriter, LearnerAdapter, LinearExplorationSchedule, PolicyView
from xrl.training.metrics import PROBE_SEED_OFFSET
from xrl.training.metrics_io import StandardArtifactWriter, summarize_learner_metrics, summarize_metrics
from xrl.training.q_learning import QLearningAgent, StateKey
from xrl.training.runner import AECExperimentRunner, AECExperimentSpec
from xrl.training.tracking import NoopTracker, TrainingTracker
from xrl.training.value_heads import SCALAR, normalize_value_head

ProgressReporter = Callable[[dict[str, object]], None]
ProgressSummaryFn = Callable[[list[EpisodeMetrics], int, int], dict[str, object]]
EvaluationRowFn = Callable[[PolicyView, list[EpisodeMetrics], int, str], dict[str, Any]]
PolicyCheckpointFn = Callable[[PolicyView, int], dict[str, Any]]


class AgentStage(Protocol):
    """Pipeline stage that builds the fitted policy object."""

    def build_agent(self, context: PipelineContext) -> PolicyView:
        """Return a fresh learner/policy for this run."""


class EpisodeStage(Protocol):
    """Pipeline stage that adapts a policy to the episode runner."""

    def build_adapter(self, agent: PolicyView, context: PipelineContext) -> LearnerAdapter:
        """Return a runner adapter for ``agent``."""


class ArtifactStage(Protocol):
    """Pipeline stage that writes trained run artifacts."""

    def build_writer(self, context: PipelineContext) -> ArtifactWriter:
        """Return the artifact writer for the current run."""


class ExplainerStage(Protocol):
    """Pipeline stage that exposes an explainer for a fitted policy.

    The explainer object is intentionally unconstrained: exact SVERL exposes
    behaviour/outcome/prediction methods, while distributional surrogate
    explainers can expose action-value methods.
    """

    def build_explainer(
        self,
        agent: PolicyView,
        context: PipelineContext,
        result: PipelineResult | None,
    ) -> object:
        """Return an explainer bound to ``agent``."""


@dataclass(slots=True)
class PipelineContext:
    """Configuration shared by all stages in one pipeline."""

    config: Any
    spec: AECExperimentSpec
    env_factory: Callable[..., object] = tictactoe_v0.env
    tracker: TrainingTracker = field(default_factory=NoopTracker)
    eval_opponents: tuple[str, ...] = ("random", "minimax")
    progress_reporter: ProgressReporter | None = None
    progress_summary: ProgressSummaryFn | None = None
    progress_interval: int = 0
    progress_window: int = 100
    progress: Any | None = None
    evaluation_row: EvaluationRowFn | None = None
    policy_checkpoint: PolicyCheckpointFn | None = None

    def make_env(self):
        return self.env_factory(**getattr(self.config, "env_options", {}))


@dataclass(slots=True)
class PipelineResult:
    """Result cached after ``XRLPipeline.fit``."""

    agent: PolicyView
    metrics: list[EpisodeMetrics]
    evaluation_metrics: list[dict[str, Any]]
    policy_metrics: list[dict[str, Any]]
    artifacts: dict[str, Any] = field(default_factory=dict)


class XRLPipeline:
    """Composable experiment pipeline backed by ``AECExperimentRunner``."""

    def __init__(self, steps: Sequence[tuple[str, object]], *, context: PipelineContext) -> None:
        if not steps:
            raise ValueError("pipeline requires at least one stage")
        self.steps = list(steps)
        self.named_steps = {name: stage for name, stage in self.steps}
        self.context = context
        self.agent_stage = self._required_stage("build_agent", "agent")
        self.episode_stage = self._required_stage("build_adapter", "episode")
        self.artifact_stage = self._optional_stage("build_writer")
        self.explainer_stages = {name: stage for name, stage in self.steps if hasattr(stage, "build_explainer")}
        self.runner: AECExperimentRunner | None = None
        self.result: PipelineResult | None = None
        self._explainers: dict[str, object] = {}

    def with_explainer(self, name: str, stage: ExplainerStage) -> XRLPipeline:
        """Register an explainer stage and return this pipeline."""

        if not name:
            raise ValueError("explainer name must be non-empty")
        self.steps.append((name, stage))
        self.named_steps[name] = stage
        self.explainer_stages[name] = stage
        self._explainers.pop(name, None)
        return self

    def fit(self) -> PipelineResult:
        """Train the configured agent and cache the fitted pipeline result."""

        agent = self.agent_stage.build_agent(self.context)
        runner = self._make_runner(agent)
        self.runner = runner
        fitted_agent, metrics = runner.train()
        self.result = PipelineResult(
            agent=fitted_agent,
            metrics=metrics,
            evaluation_metrics=runner.evaluation_metrics,
            policy_metrics=runner.policy_metrics,
        )
        self._fit_trainable_explainers()
        return self.result

    def evaluate(
        self,
        *,
        agent: PolicyView | None = None,
        episodes: int | None = None,
        opponent_name: str | None = None,
        seed: int | None = None,
    ) -> list[EpisodeMetrics]:
        """Evaluate the fitted or supplied agent without training updates."""

        target = agent or self._fitted_agent()
        runner = self._runner_for(target)
        return runner.evaluate(episodes=episodes, opponent_name=opponent_name, seed=seed)

    def explainer(self, name: str = "default") -> object:
        """Return an explainer built for the fitted agent."""

        if name in self._explainers:
            return self._explainers[name]
        stage_name = self._resolve_explainer_name(name)
        stage = self.explainer_stages[stage_name]
        explainer = stage.build_explainer(self._fitted_agent(), self.context, self.result)
        self._explainers[name] = explainer
        if stage_name != name:
            self._explainers[stage_name] = explainer
        return explainer

    def write(
        self,
        *,
        agent: PolicyView | None = None,
        metrics: list[EpisodeMetrics] | None = None,
    ) -> dict[str, Any]:
        """Write run artifacts for the fitted or supplied agent."""

        if self.artifact_stage is None:
            raise ValueError("pipeline has no artifact stage")
        target = agent or self._fitted_agent()
        rows = metrics if metrics is not None else self._fitted_metrics()
        writer = self.artifact_stage.build_writer(self.context)
        runner = self._runner_for(target)
        if runner is self.runner:
            summary = runner.write_run(writer, rows)
        else:
            summary = writer.write(target, rows, [], [])
            self.context.tracker.log_artifacts(writer.out_dir)
            self.context.tracker.end_run()
        if self.result is not None and self.result.agent is target:
            self.result.artifacts["summary"] = summary
        return summary

    def _make_runner(self, agent: PolicyView) -> AECExperimentRunner:
        adapter = self.episode_stage.build_adapter(agent, self.context)
        return AECExperimentRunner(
            self.context.spec,
            adapter,
            tracker=self.context.tracker,
            eval_opponents=self.context.eval_opponents,
            progress_reporter=self.context.progress_reporter,
            progress_summary=self.context.progress_summary,
            progress_interval=self.context.progress_interval,
            progress_window=self.context.progress_window,
            progress=self.context.progress,
            evaluation_row=self.context.evaluation_row,
            policy_checkpoint=self.context.policy_checkpoint,
            probe=_probe_hook(self.context),
        )

    def _runner_for(self, agent: PolicyView) -> AECExperimentRunner:
        if self.runner is not None and self.runner.learner is agent:
            return self.runner
        return self._make_runner(agent)

    def _fit_trainable_explainers(self) -> None:
        if self.result is None:
            return
        for name, stage in self.explainer_stages.items():
            fit_explainer = getattr(stage, "fit_explainer", None)
            if callable(fit_explainer):
                self._explainers[name] = fit_explainer(self.result.agent, self.context, self.result)

    def _fitted_agent(self) -> PolicyView:
        if self.result is None:
            raise ValueError("pipeline must be fitted before this operation")
        return self.result.agent

    def _fitted_metrics(self) -> list[EpisodeMetrics]:
        if self.result is None:
            raise ValueError("pipeline must be fitted before this operation")
        return self.result.metrics

    def _resolve_explainer_name(self, name: str) -> str:
        if name in self.explainer_stages:
            return name
        if name == "default" and len(self.explainer_stages) == 1:
            return next(iter(self.explainer_stages))
        raise KeyError(f"unknown explainer {name!r}")

    def _required_stage(self, method: str, label: str):
        stage = self._optional_stage(method)
        if stage is None:
            raise ValueError(f"pipeline requires a {label} stage")
        return stage

    def _optional_stage(self, method: str):
        for _name, stage in self.steps:
            if hasattr(stage, method):
                return stage
        return None


@dataclass(frozen=True, slots=True)
class CallableAgentStage:
    """Agent stage backed by a factory callback."""

    factory: Callable[[PipelineContext], PolicyView]

    def build_agent(self, context: PipelineContext) -> PolicyView:
        return self.factory(context)


@dataclass(frozen=True, slots=True)
class CallableEpisodeStage:
    """Episode stage backed by an adapter factory callback."""

    factory: Callable[[PolicyView, PipelineContext], LearnerAdapter]

    def build_adapter(self, agent: PolicyView, context: PipelineContext) -> LearnerAdapter:
        return self.factory(agent, context)


@dataclass(frozen=True, slots=True)
class CallableArtifactStage:
    """Artifact stage backed by a writer factory callback."""

    factory: Callable[[PipelineContext], ArtifactWriter]

    def build_writer(self, context: PipelineContext) -> ArtifactWriter:
        return self.factory(context)


class TabularAgentStage:
    """Build a tabular Q-learning agent from an environment/config."""

    def build_agent(self, context: PipelineContext) -> QLearningAgent:
        env = context.make_env()
        try:
            num_actions = int(env.action_space("player_0").n)
        finally:
            env.close()
        return QLearningAgent(
            num_actions=num_actions,
            alpha=context.config.alpha,
            gamma=context.config.gamma,
            seed=context.config.seed,
            encoder=context.config.encoder,
        )


class TabularEpisodeStage:
    """Adapt tabular Q-learning to the shared AEC runner."""

    def __init__(self) -> None:
        self.seen_state_keys: set[StateKey] = set()

    def build_adapter(self, agent: PolicyView, context: PipelineContext) -> LearnerAdapter:
        if not isinstance(agent, QLearningAgent):
            raise TypeError("TabularEpisodeStage requires QLearningAgent")
        return _TabularLearnerAdapter(agent, make_env=context.make_env, seen_state_keys=self.seen_state_keys)


class _TabularUpdateStrategy:
    def __init__(self, learner: QLearningAgent, seen_state_keys: set[StateKey]):
        self.learner = learner
        self.seen_state_keys = seen_state_keys

    def observe_transition(
        self,
        pending: LearnerStep,
        reward: float,
        next_step: LearnerStep | None,
        done: bool,
        *,
        train: bool,
    ) -> float | None:
        if not train:
            return None
        return self.learner.observe_transition(
            pending.state,
            pending.action,
            reward,
            None if next_step is None else next_step.state,
            None if next_step is None else next_step.next_action_mask,
            done=done,
        )

    def stats(self) -> LearnerStats:
        return LearnerStats(unique_states_seen=len(self.seen_state_keys), q_table_rows=len(self.learner.q_table))


class _TabularLearnerAdapter:
    def __init__(
        self,
        learner: QLearningAgent,
        *,
        make_env: Callable[[], object],
        seen_state_keys: set[StateKey],
    ) -> None:
        self.learner = learner
        self.make_env = make_env
        self.seen_state_keys = seen_state_keys
        self.update_strategy = _TabularUpdateStrategy(learner, seen_state_keys)

    def run_episode(
        self,
        *,
        opponent: object,
        episode: int,
        seed: int,
        epsilon: float,
        train: bool,
    ) -> EpisodeMetrics:
        def select_learner_action(observation, step_epsilon: float, step_train: bool) -> LearnerStep:
            view = self.learner.encode_view(observation["observation"], observation["action_mask"])
            key = view.state_key
            if step_train and is_unmasked_tictactoe_key(key):
                self.seen_state_keys.add(key)
            table_action = self.learner.select_table_action(view, epsilon=step_epsilon)
            return LearnerStep(
                state=key,
                action=table_action,
                env_action=view.to_env_action(table_action),
                next_action_mask=view.action_mask,
            )

        return run_aec_episode(
            env=self.make_env(),
            opponent=opponent,
            episode=episode,
            seed=seed,
            epsilon=epsilon,
            train=train,
            select_learner_action=select_learner_action,
            observe_learner_transition=lambda pending, reward, next_step, done: self.update_strategy.observe_transition(
                pending,
                reward,
                next_step,
                done,
                train=train,
            ),
            learner_stats=self.update_strategy.stats,
        )


@dataclass(frozen=True, slots=True)
class TicTacToeSVERLStage:
    """Default exact Tic Tac Toe SVERL explainer stage."""

    opponent: str | None = None
    perturbed_minimax_probability: float | None = None
    rules: object | None = None
    parallel: bool | None = None
    max_workers: int | None = None

    def build_explainer(
        self,
        agent: PolicyView,
        context: PipelineContext,
        result: PipelineResult | None,
    ) -> object:
        del result
        from xrl.explainability.tictactoe import TicTacToeSVERLExplainer
        from xrl.training.policies import frozen_random_move_probability

        # SVERL explains the post-training (frozen) opponent: a stationary random-move rate
        # equal to ``discount ** (episodes - 1)``.
        probability = self.perturbed_minimax_probability
        if probability is None:
            probability = frozen_random_move_probability(
                getattr(context.config, "perturbed_minimax_discount", 1.0),
                getattr(context.config, "episodes", 1),
            )

        return TicTacToeSVERLExplainer(
            agent,
            opponent=self.opponent or context.config.opponent,
            perturbed_minimax_probability=probability,
            rules=self.rules,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )


@dataclass(frozen=True, slots=True)
class DistributionalQSurrogateStage:
    """Trainable explainer stage for existing distributional-Q surrogates."""

    config: object

    def fit_explainer(
        self,
        agent: PolicyView,
        context: PipelineContext,
        result: PipelineResult,
    ) -> object:
        del agent, context, result
        from xrl.explainability.distributional_q import DistributionalQExplainer, DistributionalQSurrogateTrainer

        surrogate, _summary = DistributionalQSurrogateTrainer(self.config).train()
        return DistributionalQExplainer(surrogate)

    def build_explainer(
        self,
        agent: PolicyView,
        context: PipelineContext,
        result: PipelineResult | None,
    ) -> object:
        del agent, context, result
        from xrl.explainability.distributional_q import DistributionalQExplainer, DistributionalQSurrogateTrainer

        surrogate = DistributionalQSurrogateTrainer(self.config).load_surrogate()
        return DistributionalQExplainer(surrogate)


def _default_explainer_stages(config: object, explainer_stage: object | None) -> list[tuple[str, object]]:
    """Return the default explainer stage entries for a trainer config.

    Exact SVERL enumerates the environment's reachable state tree, which is
    only tractable for TicTacToe — non-TicTacToe configs get no default stage
    unless the caller supplies ``explainer_stage`` explicitly.
    """

    if explainer_stage is not None:
        return [("default", explainer_stage)]
    if getattr(config, "env_id", "tictactoe_v0") != "tictactoe_v0":
        return []
    return [("default", TicTacToeSVERLStage())]


def make_tabular_pipeline(
    config: ExperimentConfig,
    *,
    env_factory: Callable[..., object] | None = None,
    progress_reporter: ProgressReporter | None = None,
    progress_interval: int = 0,
    progress_window: int = 100,
    progress: Any | None = None,
    tracker: TrainingTracker | None = None,
    eval_opponents: tuple[str, ...] = ("random", "minimax"),
    explainer_stage: object | None = None,
) -> XRLPipeline:
    """Build a pipeline for tabular Q-learning."""

    context = PipelineContext(
        config=config,
        spec=_spec_from_config(config, run_name="tabular_q", tags={"trainer": "tabular_q"}),
        env_factory=env_factory or tictactoe_v0.env,
        tracker=tracker or NoopTracker(),
        eval_opponents=eval_opponents,
        progress_reporter=progress_reporter,
        progress_summary=lambda metrics, total, window: tabular_training_progress_summary(
            metrics,
            total_episodes=total,
            window=window,
        ),
        progress_interval=max(0, progress_interval),
        progress_window=max(1, progress_window),
        progress=progress,
        evaluation_row=_tabular_evaluation_row,
        policy_checkpoint=lambda agent, episode: {"episode": episode, **minimax_action_match_summary(agent)},
    )
    return XRLPipeline(
        [
            ("agent", TabularAgentStage()),
            ("episode", TabularEpisodeStage()),
            (
                "artifacts",
                CallableArtifactStage(
                    lambda ctx: StandardArtifactWriter(
                        ctx.config,
                        save_model=lambda learner, out_dir: learner.save(out_dir / "q_table.json"),
                        model_filenames=("q_table.json",),
                    )
                ),
            ),
            *_default_explainer_stages(config, explainer_stage),
        ],
        context=context,
    )


def make_q_regression_pipeline(
    config: object,
    *,
    env_factory: Callable[..., object] | None = None,
    tracker: TrainingTracker | None = None,
    eval_opponents: tuple[str, ...] = ("random", "minimax"),
    explainer_stage: object | None = None,
) -> XRLPipeline:
    """Build a pipeline for online neural Q-regression."""

    def build_agent(context: PipelineContext) -> PolicyView:
        from xrl.training.q_regression import QRegressionAgent

        num_actions, observation_size = _agent_init_shape(context)
        return QRegressionAgent(context.config, num_actions=num_actions, observation_size=observation_size)

    def build_adapter(agent: PolicyView, context: PipelineContext) -> LearnerAdapter:
        from xrl.training.q_regression import _QRegressionLearnerAdapter

        return _QRegressionLearnerAdapter(agent, make_env=context.make_env)

    return _neural_like_pipeline(
        config,
        run_name="q_regression",
        tags={"trainer": "q_regression"},
        model_filename="model.msgpack",
        build_agent=build_agent,
        build_adapter=build_adapter,
        env_factory=env_factory,
        tracker=tracker,
        eval_opponents=eval_opponents,
        explainer_stage=explainer_stage,
    )


def make_dqn_pipeline(
    config: object,
    *,
    env_factory: Callable[..., object] | None = None,
    tracker: TrainingTracker | None = None,
    eval_opponents: tuple[str, ...] = ("random", "minimax"),
    explainer_stage: object | None = None,
    progress: Any | None = None,
) -> XRLPipeline:
    """Build a pipeline for DQN-style neural Q-learning."""

    def build_agent(context: PipelineContext) -> PolicyView:
        from xrl.training.neural_q import ClassificationDQNAgent, RegressionDQNAgent

        num_actions, observation_size = _agent_init_shape(context)
        if normalize_value_head(getattr(context.config, "value_head", SCALAR)) == SCALAR:
            return RegressionDQNAgent(context.config, num_actions=num_actions, observation_size=observation_size)
        return ClassificationDQNAgent(context.config, num_actions=num_actions, observation_size=observation_size)

    def build_adapter(agent: PolicyView, context: PipelineContext) -> LearnerAdapter:
        from xrl.training.neural_q import ReplayBuffer, _NeuralQLearnerAdapter

        replay = ReplayBuffer(context.config.replay_capacity, seed=context.config.seed)
        return _NeuralQLearnerAdapter(agent, make_env=context.make_env, replay=replay, config=context.config)

    return _neural_like_pipeline(
        config,
        run_name=f"dqn_{config.value_head}",
        tags={"trainer": "dqn", "value_head": config.value_head},
        model_filename="model.msgpack",
        build_agent=build_agent,
        build_adapter=build_adapter,
        env_factory=env_factory,
        tracker=tracker,
        eval_opponents=eval_opponents,
        explainer_stage=explainer_stage,
        progress=progress,
    )


def tabular_training_progress_summary(
    metrics: list[EpisodeMetrics],
    *,
    total_episodes: int,
    window: int,
) -> dict[str, object]:
    recent = metrics[-max(1, window) :]
    return {
        "episode": metrics[-1].episode if metrics else 0,
        "episodes": total_episodes,
        "epsilon": metrics[-1].epsilon if metrics else 0.0,
        "td_error_mean": metrics[-1].td_error_mean if metrics else 0.0,
        "td_error_max": metrics[-1].td_error_max if metrics else 0.0,
        "unique_states_seen": metrics[-1].unique_states_seen if metrics else 0,
        "q_table_rows": metrics[-1].q_table_rows if metrics else 0,
        "window": len(recent),
        "window_summary": summarize_learner_metrics(recent),
        "overall_summary": summarize_learner_metrics(metrics),
    }


def _neural_like_pipeline(
    config: object,
    *,
    run_name: str,
    tags: dict[str, str],
    model_filename: str,
    build_agent: Callable[[PipelineContext], PolicyView],
    build_adapter: Callable[[PolicyView, PipelineContext], LearnerAdapter],
    env_factory: Callable[..., object] | None,
    tracker: TrainingTracker | None,
    eval_opponents: tuple[str, ...],
    explainer_stage: object | None = None,
    progress: Any | None = None,
) -> XRLPipeline:
    context = PipelineContext(
        config=config,
        spec=_spec_from_config(config, run_name=run_name, tags=tags),
        env_factory=env_factory or tictactoe_v0.env,
        tracker=tracker or NoopTracker(),
        eval_opponents=eval_opponents,
        policy_checkpoint=lambda agent, episode: {"episode": episode, **minimax_action_match_summary(agent)},
        progress=progress,
    )
    return XRLPipeline(
        [
            ("agent", CallableAgentStage(build_agent)),
            ("episode", CallableEpisodeStage(build_adapter)),
            (
                "artifacts",
                CallableArtifactStage(
                    lambda ctx: StandardArtifactWriter(
                        ctx.config,
                        save_model=lambda learner, out_dir: learner.save(out_dir / model_filename),
                        model_filenames=(model_filename,),
                    )
                ),
            ),
            *_default_explainer_stages(config, explainer_stage),
        ],
        context=context,
    )


def _probe_hook(context: PipelineContext) -> Callable[[PolicyView], dict[str, float]] | None:
    """Build the runner's Q-probe hook, or ``None`` when probing is disabled."""

    limit = int(getattr(context.config, "q_probe_states", 0))
    if limit <= 0 or context.spec.eval_interval <= 0:
        return None
    return aec_probe_hook(context.make_env, limit=limit, seed=context.spec.seed + PROBE_SEED_OFFSET)


def _spec_from_config(config: object, *, run_name: str, tags: dict[str, str]) -> AECExperimentSpec:
    return AECExperimentSpec(
        episodes=config.episodes,
        seed=config.seed,
        opponent=config.opponent,
        eval_interval=config.eval_interval,
        eval_episodes=config.eval_episodes,
        params=config.to_json_dict(),
        run_name=run_name,
        tags=tags,
        exploration=LinearExplorationSchedule(
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_episodes=config.epsilon_decay_episodes,
        ),
        perturbed_minimax_discount=getattr(config, "perturbed_minimax_discount", 1.0),
        metrics_window=getattr(config, "metrics_window", 100),
    )


def _agent_init_shape(context: PipelineContext) -> tuple[int, int]:
    env = context.make_env()
    try:
        observation = env.observation_space("player_0")["observation"]
        num_actions = int(env.action_space("player_0").n)
        observation_size = int(np.prod(observation.shape))
    finally:
        env.close()
    return num_actions, observation_size


def _tabular_evaluation_row(
    learner: PolicyView,
    eval_metrics: list[EpisodeMetrics],
    episode: int,
    opponent_name: str,
) -> dict[str, Any]:
    summary = summarize_metrics(eval_metrics)
    return {
        "episode": episode,
        "opponent": opponent_name,
        "mean_return": summary["mean_learner_return"],
        "bellwether": bellwether_values(learner),
        **summary,
    }
