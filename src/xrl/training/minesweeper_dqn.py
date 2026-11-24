"""DQN training vertical for native Gymnasium Minesweeper."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from xrl._masking import validate_mask_mixture
from xrl.envs import minesweeper_v0
from xrl.minesweeper import MinesweeperConfig
from xrl.training._neural.replay import ReplayBuffer
from xrl.training._neural.value_projection import (
    DEFAULT_HL_GAUSS_SIGMA_BINS,
    DEFAULT_NUM_BINS,
    DEFAULT_V_MAX,
    DEFAULT_V_MIN,
)
from xrl.training.config import parse_hidden_sizes, validate_common_config, validate_exploration_config
from xrl.training.episode import LearnerStats, LearnerStep
from xrl.training.flops import FLOPS_ROW_FIELDS, learner_flops, learner_training_accounting
from xrl.training.interfaces import LinearExplorationSchedule
from xrl.training.metrics import (
    PROBE_SEED_OFFSET,
    PROBE_STAT_FIELDS,
    ProbeSetBuilder,
    mlflow_metrics_projection,
    probe_mlflow_metrics,
    probe_q_stats,
    rolling_means,
)
from xrl.training.metrics_io import DEFAULT_ENV_ID, guard_run_directory, prune_stale_artifacts
from xrl.training.neural_q import (
    DEFAULT_REPLAY_WARMUP,
    MASKED_TARGETS,
    TD_ACTION_TARGET,
    ClassificationDQNAgent,
    NeuralQAgent,
    _ReplayUpdateStrategy,
)
from xrl.training.progress import ProgressReporter, TrainingProgress
from xrl.training.run_metadata import RunIdentity, execute_training_run
from xrl.training.runner import CHECKPOINT_EVAL_SEED_OFFSET
from xrl.training.seed_sweep import SeedSweepSpec, parse_seeds, run_seed_sweep, trainer_run_seed
from xrl.training.tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    NoopTracker,
    TrackerConfig,
    TrainingTracker,
    tracker_from_cli_args,
)
from xrl.training.value_heads import SCALAR, normalize_value_head


@dataclass(frozen=True, slots=True)
class MinesweeperDQNConfig:
    """Configuration for the Minesweeper DQN vertical."""

    episodes: int = 1_000
    seed: int = 0
    value_head: str = SCALAR
    rows: int = 5
    columns: int = 5
    mines: int = 3
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    exploration: str = "epsilon_greedy"
    backup: str = "hard"
    temperature: float = 0.05
    replay_capacity: int = 10_000
    replay_warmup: int = DEFAULT_REPLAY_WARMUP
    batch_size: int = 32
    target_sync_interval: int = 100
    hidden_sizes: tuple[int, ...] = (64, 64)
    v_min: float = DEFAULT_V_MIN
    v_max: float = DEFAULT_V_MAX
    num_bins: int = DEFAULT_NUM_BINS
    hl_gauss_sigma_bins: float = DEFAULT_HL_GAUSS_SIGMA_BINS
    encoder: str = "identity"
    feature_mask_input: bool = False
    masked_batch_ratio: float = 0.0
    masked_target: str = TD_ACTION_TARGET
    mask_empty_example_interval: int = 256
    mask_shapley_probability: float = 0.7
    mask_uniform_probability: float = 0.3
    eval_interval: int = 50
    eval_episodes: int = 20
    metrics_window: int = 100
    q_probe_states: int = 64
    out_dir: Path = Path("runs/minesweeper-dqn")

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_head", normalize_value_head(self.value_head))
        object.__setattr__(self, "hidden_sizes", parse_hidden_sizes(self.hidden_sizes))
        if self.encoder != "identity":
            raise ValueError("Minesweeper v1 supports only the 'identity' encoder")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.replay_capacity < 1:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be positive")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain one or more positive layer widths")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        if self.q_probe_states < 0:
            raise ValueError("q_probe_states must be non-negative")
        if not np.isfinite(self.masked_batch_ratio) or not 0 <= self.masked_batch_ratio <= 1:
            raise ValueError("masked_batch_ratio must be finite and in [0, 1]")
        if self.masked_batch_ratio > 0 and not self.feature_mask_input:
            raise ValueError("masked_batch_ratio requires feature_mask_input=true")
        if self.masked_batch_ratio > 0 and round(self.masked_batch_ratio * self.batch_size) < 1:
            raise ValueError("masked_batch_ratio is too small to add an example at this batch_size")
        if self.masked_target not in MASKED_TARGETS:
            raise ValueError(f"masked_target must be one of {MASKED_TARGETS}, got {self.masked_target!r}")
        validate_mask_mixture(
            empty_mask_example_interval=self.mask_empty_example_interval,
            shapley_probability=self.mask_shapley_probability,
            uniform_probability=self.mask_uniform_probability,
        )
        # Checkpoint selection promotes the best evaluation to model.msgpack. An
        # enabled interval with no episodes scores every checkpoint 0.0, which
        # would silently ship the earliest unevaluated model as the "best" one.
        if self.eval_interval > 0 and self.eval_episodes < 1:
            raise ValueError("eval_episodes must be positive when eval_interval is enabled")
        MinesweeperConfig(rows=self.rows, columns=self.columns, mines=self.mines)
        validate_exploration_config(
            exploration=self.exploration,
            backup=self.backup,
            temperature=self.temperature,
        )
        validate_common_config(
            episodes=self.episodes,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_episodes=self.epsilon_decay_episodes,
            encoder=self.encoder,
            metrics_window=self.metrics_window,
        )

    @property
    def env_options(self) -> dict[str, int]:
        return {"rows": self.rows, "columns": self.columns, "mines": self.mines}

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        data["hidden_sizes"] = list(self.hidden_sizes)
        data["env_id"] = "minesweeper_v0"
        data["env_options"] = self.env_options
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> MinesweeperDQNConfig:
        normalized = dict(data)
        legacy = {"explainable_q", "masked_loss_weight", "mask_empty_interval"} & normalized.keys()
        if legacy:
            raise ValueError(
                "legacy mask configuration is unsupported; migrate to feature_mask_input, "
                "masked_batch_ratio, and mask_empty_example_interval"
            )
        normalized["value_head"] = normalize_value_head(normalized.pop("loss", normalized.get("value_head", SCALAR)))
        env_options = dict(normalized.pop("env_options", {}) or {})
        normalized.pop("env_id", None)
        for key in ("rows", "columns", "mines"):
            if key not in normalized and key in env_options:
                normalized[key] = env_options[key]
        normalized["out_dir"] = Path(normalized["out_dir"])
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (64, 64)))
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class MinesweeperEpisodeMetrics:
    """Metrics captured for one single-agent Minesweeper episode."""

    episode: int
    seed: int
    outcome: str
    learner_return: float
    episode_length: int
    epsilon: float
    elapsed_seconds: float
    rows: int
    columns: int
    mines: int
    safe_cells_revealed: int
    total_safe_cells: int
    completion_ratio: float
    loss_mean: float = 0.0
    loss_max: float = 0.0
    gradient_updates: int = 0
    replay_size: int = 0
    win: int = 0
    steps_per_second: float = 0.0
    rolling_win_rate: float = 0.0
    rolling_completion_ratio: float = 0.0
    rolling_learner_return: float = 0.0
    rolling_episode_length: float = 0.0


# Per-episode facts whose trailing-window means are the actual learning curves.
_ROLLING_COLUMNS: dict[str, str] = {
    "win": "rolling_win_rate",
    "completion_ratio": "rolling_completion_ratio",
    "learner_return": "rolling_learner_return",
    "episode_length": "rolling_episode_length",
}

# Constants and identifiers stay MLflow params; only genuine series become metrics.
_MLFLOW_EXCLUDED_METRIC_FIELDS: frozenset[str] = frozenset(
    {"episode", "seed", "outcome", "rows", "columns", "mines", "total_safe_cells"}
)
_MLFLOW_EXCLUDED_EVAL_FIELDS: frozenset[str] = frozenset(
    {"episode", "seed", "rows", "columns", "mines", "episodes", *PROBE_STAT_FIELDS}
)

# Columns aggregated and plotted by multi-seed sweeps of this vertical.
MINESWEEPER_SWEEP_METRIC_COLUMNS: tuple[str, ...] = (
    "rolling_win_rate",
    "rolling_completion_ratio",
    "rolling_learner_return",
    "rolling_episode_length",
    "loss_mean",
    "steps_per_second",
)
MINESWEEPER_SWEEP_EVAL_COLUMNS: tuple[str, ...] = (
    "win_rate",
    "mean_completion_ratio",
    "mean_return",
    "mean_episode_length",
    "training_flops",
    *PROBE_STAT_FIELDS,
)


def minesweeper_mlflow_metrics(metric: MinesweeperEpisodeMetrics) -> dict[str, float | int]:
    """Project one episode's metrics onto grouped ``train/...`` MLflow keys."""

    return mlflow_metrics_projection(asdict(metric), exclude=_MLFLOW_EXCLUDED_METRIC_FIELDS, prefix="train/")


MINESWEEPER_EVAL_FIELDS: tuple[str, ...] = (
    "episode",
    "episodes",
    "seed",
    "rows",
    "columns",
    "mines",
    "wins",
    "losses",
    "win_rate",
    "loss_rate",
    "mean_learner_return",
    "mean_return",
    "mean_episode_length",
    "mean_completion_ratio",
    *PROBE_STAT_FIELDS,
    *FLOPS_ROW_FIELDS,
)


class MinesweeperDQNTrainer:
    """Train and evaluate DQN agents on native Gymnasium Minesweeper."""

    model_filename = "model.msgpack"
    final_model_filename = "final_model.msgpack"
    config_class = MinesweeperDQNConfig

    def __init__(
        self,
        config: MinesweeperDQNConfig,
        *,
        env_factory=None,
        tracker: TrainingTracker | None = None,
        progress: ProgressReporter | None = None,
    ):
        self.config = config
        self.env_factory = env_factory or minesweeper_v0.env
        self.tracker = tracker or NoopTracker()
        self.progress = progress
        self.evaluation_metrics: list[dict[str, Any]] = []
        self._probe_set: tuple[np.ndarray, np.ndarray] | None = None
        self.best_evaluation: dict[str, Any] | None = None
        self._best_model_bytes: bytes | None = None

    @property
    def run_name(self) -> str:
        return f"minesweeper_dqn_{self.config.value_head}"

    @property
    def tags(self) -> dict[str, str]:
        return {"trainer": "dqn", "value_head": self.config.value_head, "environment": "minesweeper_v0"}

    def make_env(self):
        return self.env_factory(**self.config.env_options)

    def make_agent(self) -> NeuralQAgent:
        num_actions, observation_size = self._agent_init_shape()
        if self.config.value_head == SCALAR:
            return NeuralQAgent(self.config, num_actions=num_actions, observation_size=observation_size)
        return ClassificationDQNAgent(self.config, num_actions=num_actions, observation_size=observation_size)

    def train(self) -> tuple[NeuralQAgent, list[MinesweeperEpisodeMetrics]]:
        agent = self.make_agent()
        self.evaluation_metrics = []
        self._probe_set = None
        self.best_evaluation = None
        self._best_model_bytes = None
        if self.config.q_probe_states > 0 and self.config.eval_interval > 0:
            self._probe_set = collect_minesweeper_probe_states(
                self.make_env,
                agent,
                limit=self.config.q_probe_states,
                seed=self.config.seed + PROBE_SEED_OFFSET,
            )
        replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        update_strategy = _ReplayUpdateStrategy(agent, replay, self.config)
        exploration = LinearExplorationSchedule(
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay_episodes=self.config.epsilon_decay_episodes,
        )
        metrics: list[MinesweeperEpisodeMetrics] = []

        self.tracker.start_run(run_name=self.run_name, tags=self.tags)
        params = self.config.to_json_dict()
        params["total_safe_cells"] = self.config.rows * self.config.columns - self.config.mines
        self.tracker.log_params(params)
        try:
            for episode in range(1, self.config.episodes + 1):
                epsilon = exploration.epsilon(episode)
                metric = run_minesweeper_dqn_episode(
                    env=self.make_env(),
                    agent=agent,
                    update_strategy=update_strategy,
                    config=self.config,
                    episode=episode,
                    seed=self.config.seed + episode,
                    epsilon=epsilon,
                    train=True,
                )
                metric = self._with_rolling_metrics(metrics, metric)
                metrics.append(metric)
                self.tracker.log_metrics(minesweeper_mlflow_metrics(metric), step=episode)
                if self.progress is not None:
                    self.progress.episode(metric, episode=episode, total=self.config.episodes)
                self._checkpoint_if_due(agent, episode)
            # Force a final checkpoint so every run ends with an eval row, even
            # when the episode count is not a multiple of the eval interval.
            if self.config.eval_interval > 0 and self.config.episodes % self.config.eval_interval != 0:
                self._evaluation_checkpoint(agent, self.config.episodes)
        except Exception:
            self.tracker.end_run(status="FAILED")
            raise

        return agent, metrics

    def _with_rolling_metrics(
        self,
        history: list[MinesweeperEpisodeMetrics],
        metric: MinesweeperEpisodeMetrics,
    ) -> MinesweeperEpisodeMetrics:
        window = self.config.metrics_window
        recent = [*history[max(0, len(history) - window + 1) :], metric]
        return replace(metric, **rolling_means(recent, columns=_ROLLING_COLUMNS, window=window))

    def evaluate_metrics(
        self,
        agent: NeuralQAgent,
        *,
        episodes: int,
        seed: int,
    ) -> list[MinesweeperEpisodeMetrics]:
        update_strategy = _ReplayUpdateStrategy(agent, replay=None, config=self.config)
        return [
            run_minesweeper_dqn_episode(
                env=self.make_env(),
                agent=agent,
                update_strategy=update_strategy,
                config=self.config,
                episode=episode,
                seed=seed + episode,
                epsilon=0.0,
                train=False,
            )
            for episode in range(1, episodes + 1)
        ]

    def evaluate(
        self,
        agent: NeuralQAgent,
        *,
        episodes: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        metrics = self.evaluate_metrics(agent, episodes=episodes, seed=seed)
        return [minesweeper_evaluation_row(metrics, episode=metrics[-1].episode if metrics else 0, seed=seed)]

    def write_run(self, agent: NeuralQAgent, metrics: list[MinesweeperEpisodeMetrics]) -> dict[str, Any]:
        out_dir = self.config.out_dir
        config_json = self.config.to_json_dict()
        out_dir.mkdir(parents=True, exist_ok=True)
        guard_run_directory(out_dir, str(config_json.get("env_id", DEFAULT_ENV_ID)))
        (out_dir / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")
        agent.save(out_dir / self.final_model_filename)
        if self._best_model_bytes is None:
            agent.save(out_dir / self.model_filename)
        else:
            (out_dir / self.model_filename).write_bytes(self._best_model_bytes)
        write_minesweeper_metrics_csv(out_dir / "metrics.csv", metrics)
        write_minesweeper_eval_metrics_csv(out_dir / "eval_metrics.csv", self.evaluation_metrics)
        summary = summarize_minesweeper_metrics(metrics)
        summary.update(learner_training_accounting(agent))
        if self.evaluation_metrics:
            summary["evaluations"] = self.evaluation_metrics
        if self.best_evaluation is None:
            summary["model_checkpoint"] = "final"
        else:
            summary.update(
                {
                    "model_checkpoint": "best_evaluation",
                    "best_eval_episode": int(self.best_evaluation["episode"]),
                    "best_eval_win_rate": float(self.best_evaluation["win_rate"]),
                }
            )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        # Minesweeper writes no policy_metrics.csv; without this an earlier Tic
        # Tac Toe run in the same directory would leave one behind to be read as
        # this run's own.
        prune_stale_artifacts(
            out_dir,
            (
                "config.json",
                "eval_metrics.csv",
                "metrics.csv",
                "run_metadata.json",
                "summary.json",
                self.model_filename,
                self.final_model_filename,
            ),
        )
        self.tracker.log_artifacts(out_dir)
        self.tracker.end_run()
        return summary

    def load_agent(self, run_dir: Path) -> NeuralQAgent:
        config = self.config_class.from_json_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        trainer = type(self)(config)
        agent = trainer.make_agent()
        agent.load_params(run_dir / self.model_filename)
        return agent

    def _agent_init_shape(self) -> tuple[int, int]:
        env = self.make_env()
        try:
            observation = env.observation_space["observation"]
            num_actions = int(env.action_space.n)
            observation_size = int(np.prod(observation.shape))
        finally:
            env.close()
        return num_actions, observation_size

    def _checkpoint_if_due(self, agent: NeuralQAgent, episode: int) -> None:
        if self.config.eval_interval <= 0 or episode % self.config.eval_interval != 0:
            return
        self._evaluation_checkpoint(agent, episode)

    def _evaluation_checkpoint(self, agent: NeuralQAgent, episode: int) -> None:
        # A fixed eval seed replays the identical board set at every checkpoint,
        # so eval curves are comparable across checkpoints and across runs.
        seed = self.config.seed + CHECKPOINT_EVAL_SEED_OFFSET
        metrics = self.evaluate_metrics(agent, episodes=self.config.eval_episodes, seed=seed)
        row = minesweeper_evaluation_row(metrics, episode=episode, seed=seed)
        probe_stats = self._probe_stats(agent)
        row.update(probe_stats)
        row.update(learner_flops(agent))
        self.evaluation_metrics.append(row)
        if self.progress is not None:
            self.progress.evaluation(row, episode=episode, total=self.config.episodes)
        if self.best_evaluation is None or float(row["win_rate"]) > float(self.best_evaluation["win_rate"]):
            self.best_evaluation = dict(row)
            self._best_model_bytes = agent.serialized_params()
        self.tracker.log_metrics(
            mlflow_metrics_projection(row, exclude=_MLFLOW_EXCLUDED_EVAL_FIELDS, prefix="eval/"),
            step=episode,
        )
        if probe_stats:
            self.tracker.log_metrics(probe_mlflow_metrics(probe_stats), step=episode)

    def _probe_stats(self, agent: NeuralQAgent) -> dict[str, float]:
        if self._probe_set is None:
            return {}
        states, masks = self._probe_set
        return probe_q_stats(agent.q_values_batch(states), masks)


def run_minesweeper_dqn_episode(
    *,
    env,
    agent: NeuralQAgent,
    update_strategy: _ReplayUpdateStrategy,
    config: MinesweeperDQNConfig,
    episode: int,
    seed: int,
    epsilon: float,
    train: bool,
) -> MinesweeperEpisodeMetrics:
    """Run one Gymnasium Minesweeper episode."""

    update_strategy.begin_episode()
    observation, info = env.reset(seed=seed)
    learner_return = 0.0
    episode_length = 0
    losses: list[float] = []
    started = time.perf_counter()

    try:
        while True:
            learner_step = _select_action_step(agent, observation, epsilon=epsilon if train else 0.0)
            next_observation, reward, terminated, truncated, info = env.step(learner_step.env_action)
            done = bool(terminated or truncated)
            learner_return += float(reward)
            episode_length += 1
            next_step = None if done else _encoded_observation_step(agent, next_observation)
            loss = update_strategy.observe_transition(
                learner_step,
                float(reward),
                next_step,
                done,
                train=train,
            )
            if loss is not None:
                losses.append(loss)
            observation = next_observation
            if done:
                break

        stats = update_strategy.stats()
        outcome = info.get("outcome")
        if outcome not in {"win", "loss"}:
            outcome = "win" if learner_return > 0 else "loss"
        return _episode_metrics(
            episode=episode,
            seed=seed,
            epsilon=epsilon,
            learner_return=learner_return,
            episode_length=episode_length,
            elapsed_seconds=time.perf_counter() - started,
            outcome=str(outcome),
            info=info,
            losses=losses,
            stats=stats,
            config=config,
        )
    finally:
        env.close()


def _progress_for(
    config: MinesweeperDQNConfig,
    interval: int,
    *,
    label: str = "",
) -> TrainingProgress | None:
    """Build a reporter, or ``None`` when nothing would ever be printed."""

    if interval <= 0 and config.eval_interval <= 0:
        return None
    return TrainingProgress(label=label, interval=max(0, interval))


def run_minesweeper_seed_sweep(
    base_config: MinesweeperDQNConfig,
    *,
    seeds: tuple[int, ...],
    tracker_config: TrackerConfig | None = None,
    plot: bool = True,
    progress_interval: int = 0,
) -> dict[str, Any]:
    """Train one Minesweeper DQN run per seed under ``base_config.out_dir``.

    ``base_config.seed`` is ignored; each run uses one of ``seeds`` and writes
    to ``<out_dir>/seed_<k>/``. Aggregates land at the sweep root (see
    ``xrl.training.seed_sweep``).
    """

    spec = SeedSweepSpec(
        seeds=seeds,
        root=base_config.out_dir,
        run_name=f"minesweeper_dqn_{base_config.value_head}_seeds{len(seeds)}",
        tags={
            "workflow": "seed_sweep",
            "trainer": "dqn",
            "value_head": base_config.value_head,
            "environment": "minesweeper_v0",
        },
        base_params=base_config.to_json_dict(),
        metric_columns=MINESWEEPER_SWEEP_METRIC_COLUMNS,
        eval_columns=MINESWEEPER_SWEEP_EVAL_COLUMNS,
    )

    def make_trainer(config: MinesweeperDQNConfig, tracker) -> MinesweeperDQNTrainer:
        return MinesweeperDQNTrainer(
            config,
            tracker=tracker,
            progress=_progress_for(config, progress_interval, label=f"seed {config.seed}"),
        )

    run_seed = trainer_run_seed(
        base_config,
        make_trainer,
        identity=RunIdentity(
            environment="minesweeper",
            algorithm="dqn",
            value_head=base_config.value_head,
        ),
    )
    return run_seed_sweep(spec, run_seed, tracker_config=tracker_config, plot=plot)


def collect_minesweeper_probe_states(
    make_env,
    agent: NeuralQAgent,
    *,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect distinct encoded probe states from random-legal-policy rollouts.

    Episode seeds start at ``seed`` (pass a base offset far from the training
    and eval streams, see ``PROBE_SEED_OFFSET``); the action RNG is independent
    of the agent's RNG so training reproducibility is untouched. The episode
    budget is capped at ``10 * limit`` so tiny boards yield smaller probe sets
    instead of looping forever.
    """

    if limit <= 0:
        return None
    builder = ProbeSetBuilder(limit)
    rng = np.random.default_rng(seed)
    for index in range(10 * limit):
        if builder.full:
            break
        env = make_env()
        try:
            observation, _info = env.reset(seed=seed + index)
            while True:
                obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
                mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
                view = agent.encode_view(obs_vector, mask)
                builder.add(np.asarray(view.state_key, dtype=np.float32), np.asarray(view.action_mask))
                legal = np.flatnonzero(mask)
                if legal.size == 0:
                    break
                observation, _reward, terminated, truncated, _info = env.step(int(rng.choice(legal)))
                if terminated or truncated:
                    break
        finally:
            env.close()
    return builder.build()


def write_minesweeper_metrics_csv(path: Path, metrics: list[MinesweeperEpisodeMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(MinesweeperEpisodeMetrics)])
        writer.writeheader()
        for row in metrics:
            writer.writerow(asdict(row))


def write_minesweeper_eval_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MINESWEEPER_EVAL_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def summarize_minesweeper_metrics(metrics: list[MinesweeperEpisodeMetrics]) -> dict[str, Any]:
    total = len(metrics)
    wins = sum(1 for metric in metrics if metric.outcome == "win")
    losses = sum(1 for metric in metrics if metric.outcome == "loss")
    mean_return = sum(metric.learner_return for metric in metrics) / total if total else 0.0
    mean_episode_length = sum(metric.episode_length for metric in metrics) / total if total else 0.0
    mean_completion = sum(metric.completion_ratio for metric in metrics) / total if total else 0.0
    return {
        "episodes": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "mean_learner_return": mean_return,
        "mean_return": mean_return,
        "mean_episode_length": mean_episode_length,
        "mean_completion_ratio": mean_completion,
    }


def minesweeper_evaluation_row(
    metrics: list[MinesweeperEpisodeMetrics],
    *,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    summary = summarize_minesweeper_metrics(metrics)
    geometry = _geometry_from_metrics(metrics)
    return {"episode": episode, "seed": seed, **geometry, **summary}


def train_minesweeper_dqn(
    episodes: int = 1_000,
    value_head: str = SCALAR,
    rows: int = 5,
    columns: int = 5,
    mines: int = 3,
    out: Path = Path("runs/minesweeper-dqn"),
    seed: int = 0,
    learning_rate: float = 1e-3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 800,
    exploration: str = "epsilon_greedy",
    backup: str = "hard",
    temperature: float = 0.05,
    batch_size: int = 32,
    replay_capacity: int = 10_000,
    replay_warmup: int = 32,
    target_sync_interval: int = 100,
    hidden_sizes: Annotated[
        str | tuple[int, ...],
        typer.Option(help="Comma-separated positive MLP hidden-layer widths, for example 128,128,128."),
    ] = "64,64",
    num_bins: int = 51,
    v_min: float = -1.0,
    v_max: float = 1.0,
    hl_gauss_sigma_bins: float = 0.75,
    encoder: str = "identity",
    feature_mask_input: bool = False,
    masked_batch_ratio: float = 0.0,
    mask_empty_example_interval: int = 256,
    mask_shapley_probability: float = 0.7,
    mask_uniform_probability: float = 0.3,
    eval_interval: int = 50,
    eval_episodes: int = 20,
    metrics_window: int = 100,
    q_probe_states: int = 64,
    progress_interval: int = 0,
    seeds: Annotated[
        str | None,
        typer.Option(help="Comma-separated seed list for a multi-seed sweep (overrides --seed)."),
    ] = None,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip sweep plot generation.")] = False,
    mlflow: bool = False,
    mlflow_experiment: str = "xrl",
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, Any]:
    """Train DQN with regression, Two-Hot, or HL-Gauss loss on Minesweeper."""

    tracker_config = TrackerConfig(
        mlflow=mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    config = MinesweeperDQNConfig(
        episodes=episodes,
        seed=seed,
        value_head=value_head,
        rows=rows,
        columns=columns,
        mines=mines,
        gamma=gamma,
        learning_rate=learning_rate,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        exploration=exploration,
        backup=backup,
        temperature=temperature,
        replay_capacity=replay_capacity,
        replay_warmup=replay_warmup,
        batch_size=batch_size,
        target_sync_interval=target_sync_interval,
        hidden_sizes=parse_hidden_sizes(hidden_sizes),
        v_min=v_min,
        v_max=v_max,
        num_bins=num_bins,
        hl_gauss_sigma_bins=hl_gauss_sigma_bins,
        encoder=encoder,
        feature_mask_input=feature_mask_input,
        masked_batch_ratio=masked_batch_ratio,
        mask_empty_example_interval=mask_empty_example_interval,
        mask_shapley_probability=mask_shapley_probability,
        mask_uniform_probability=mask_uniform_probability,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        metrics_window=metrics_window,
        q_probe_states=q_probe_states,
        out_dir=out,
    )
    if seeds is not None:
        result = run_minesweeper_seed_sweep(
            config,
            seeds=parse_seeds(seeds),
            tracker_config=tracker_config,
            plot=not no_plots,
            progress_interval=progress_interval,
        )
        return result
    trainer = MinesweeperDQNTrainer(
        config,
        tracker=tracker_from_cli_args(tracker_config),
        progress=_progress_for(config, progress_interval),
    )
    summary = execute_training_run(
        config,
        trainer,
        RunIdentity(environment="minesweeper", algorithm="dqn", value_head=config.value_head),
    )
    return {"out_dir": str(config.out_dir), "summary": summary}


def visualize_minesweeper(
    run: Annotated[Path, typer.Option("--run", help="Saved Minesweeper DQN run directory.")],
    seed: int = 0,
    out: Path = Path("replay.txt"),
) -> None:
    """Write an ANSI replay for a saved Minesweeper DQN run.

    The replay is generated greedily (epsilon=0) from ``model.msgpack`` and
    ``config.json`` in ``run``. Frames are written as plain text so they can be
    viewed in a terminal, attached to experiment artifacts, or converted by a
    future UI layer.
    """

    run_dir = Path(run)
    config_path = run_dir / "config.json"
    model_path = run_dir / MinesweeperDQNTrainer.model_filename
    if not config_path.exists():
        raise FileNotFoundError(f"missing Minesweeper run config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"missing Minesweeper model: {model_path}")

    config = MinesweeperDQNConfig.from_json_dict(json.loads(config_path.read_text(encoding="utf-8")))
    trainer = MinesweeperDQNTrainer(config)
    agent = trainer.load_agent(run_dir)
    env = minesweeper_v0.env(rows=config.rows, columns=config.columns, mines=config.mines, render_mode="ansi")

    frames: list[str] = []
    observation, info = env.reset(seed=seed)
    frames.append(_minesweeper_replay_frame(step=0, action=None, reward=0.0, info=info, board=env.render() or ""))
    total_reward = 0.0
    step = 0
    try:
        while True:
            action = _select_action_step(agent, observation, epsilon=0.0).env_action
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
            total_reward += float(reward)
            frames.append(
                _minesweeper_replay_frame(
                    step=step,
                    action=action,
                    reward=float(reward),
                    info=info,
                    board=env.render() or "",
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(frames) + f"\n\nTotal reward: {total_reward:.6g}\n"
    out.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "run": str(run_dir),
                "seed": seed,
                "out": str(out),
                "steps": step,
                "total_reward": total_reward,
                "outcome": info.get("outcome"),
            },
            indent=2,
        )
    )


def _minesweeper_replay_frame(
    *,
    step: int,
    action: int | None,
    reward: float,
    info: dict[str, Any],
    board: str,
) -> str:
    action_label = "initial" if action is None else str(action)
    return "\n".join(
        [
            f"Step {step} | action={action_label} | reward={reward:.6g} | outcome={info.get('outcome')}",
            f"safe={info.get('safe_cells_revealed')}/{info.get('total_safe_cells')} completion={float(info.get('completion_ratio', 0.0)):.3f}",
            board,
        ]
    )


def _select_action_step(agent: NeuralQAgent, observation: dict[str, np.ndarray], *, epsilon: float) -> LearnerStep:
    obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
    view = agent.encode_view(obs_vector, mask)
    env_action = agent.select_action(obs_vector, mask, epsilon=epsilon)
    return LearnerStep(
        state=np.asarray(view.state_key, dtype=np.float32),
        action=view.to_table_action(env_action),
        env_action=env_action,
        next_action_mask=np.asarray(view.action_mask, dtype=np.int8),
    )


def _encoded_observation_step(agent: NeuralQAgent, observation: dict[str, np.ndarray]) -> LearnerStep:
    obs_vector = np.asarray(observation["observation"], dtype=np.float32).reshape(-1)
    mask = np.asarray(observation["action_mask"], dtype=np.int8).reshape(-1)
    view = agent.encode_view(obs_vector, mask)
    return LearnerStep(
        state=np.asarray(view.state_key, dtype=np.float32),
        action=0,
        env_action=0,
        next_action_mask=np.asarray(view.action_mask, dtype=np.int8),
    )


def _episode_metrics(
    *,
    episode: int,
    seed: int,
    epsilon: float,
    learner_return: float,
    episode_length: int,
    elapsed_seconds: float,
    outcome: str,
    info: dict[str, Any],
    losses: list[float],
    stats: LearnerStats,
    config: MinesweeperDQNConfig,
) -> MinesweeperEpisodeMetrics:
    return MinesweeperEpisodeMetrics(
        episode=episode,
        seed=seed,
        outcome=outcome,
        learner_return=learner_return,
        episode_length=episode_length,
        epsilon=epsilon,
        elapsed_seconds=elapsed_seconds,
        rows=config.rows,
        columns=config.columns,
        mines=config.mines,
        safe_cells_revealed=int(info.get("safe_cells_revealed", 0)),
        total_safe_cells=int(info.get("total_safe_cells", config.rows * config.columns - config.mines)),
        completion_ratio=float(info.get("completion_ratio", 0.0)),
        loss_mean=sum(losses) / len(losses) if losses else 0.0,
        loss_max=max(losses, default=0.0),
        gradient_updates=stats.gradient_updates,
        replay_size=stats.replay_size,
        win=int(outcome == "win"),
        steps_per_second=episode_length / elapsed_seconds if elapsed_seconds > 0 else 0.0,
    )


def _geometry_from_metrics(metrics: list[MinesweeperEpisodeMetrics]) -> dict[str, int]:
    if not metrics:
        return {"rows": 0, "columns": 0, "mines": 0}
    first = metrics[0]
    return {"rows": first.rows, "columns": first.columns, "mines": first.mines}
