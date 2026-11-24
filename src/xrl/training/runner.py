"""Composable AEC experiment runner."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any, Generic, TypeVar

from xrl.training.episode import EpisodeMetrics
from xrl.training.flops import learner_flops
from xrl.training.interfaces import ArtifactWriter, LearnerAdapter, LinearExplorationSchedule, PolicyView
from xrl.training.metrics import (
    PROBE_STAT_FIELDS,
    mlflow_metrics_projection,
    probe_mlflow_metrics,
    rolling_means,
)
from xrl.training.metrics_io import summarize_learner_metrics
from xrl.training.policies import make_opponent, opponent_random_move_probability
from xrl.training.progress import ProgressReporter as TrainingProgressReporter
from xrl.training.tracking import NoopTracker, TrainingTracker

LearnerT = TypeVar("LearnerT", bound=PolicyView)

# Seed spacing keeps evaluation episode streams disjoint: per-opponent blocks within a
# checkpoint, and checkpoint evaluations away from the training-episode seed range.
EVAL_OPPONENT_SEED_STRIDE = 10_000
CHECKPOINT_EVAL_SEED_OFFSET = 100_000

# Per-episode facts whose trailing-window means are the actual learning curves.
_ROLLING_COLUMNS: dict[str, str] = {
    "win": "rolling_win_rate",
    "draw": "rolling_draw_rate",
    "learner_return": "rolling_learner_return",
    "episode_length": "rolling_episode_length",
}

# Identifiers stay MLflow params/tags; strings and booleans are dropped anyway,
# but excluding them here keeps the intent explicit.
_MLFLOW_EXCLUDED_METRIC_FIELDS: frozenset[str] = frozenset({"episode", "seed", "opponent", "winner", "draw"})

ProgressReporter = Callable[[dict[str, object]], None]
"""``report : Summary → ()`` — sink for one progress summary (e.g. print, log)."""

ProgressSummaryFn = Callable[[list[EpisodeMetrics], int, int], dict[str, object]]
"""``summarize : [EpisodeMetrics] × ℕ_total × ℕ_window → Summary`` over the trailing window."""

EvaluationRowFn = Callable[[LearnerT, list[EpisodeMetrics], int, str], dict[str, Any]]
"""``row : L × [EpisodeMetrics] × ℕ_episode × OpponentName → Row`` — one evaluation table row."""

PolicyCheckpointFn = Callable[[LearnerT, int], dict[str, Any]]
"""``checkpoint : L × ℕ_episode → Row`` — snapshot of policy-level statistics."""

ProbeFn = Callable[[LearnerT], dict[str, float]]
"""``probe : L → Stats`` — Q-value health statistics over a fixed probe set."""


@dataclass(frozen=True, slots=True)
class AECExperimentSpec:
    """Shared training-loop configuration for an AEC experiment.

    The spec is the product object of all static (per-run constant) hyperparameters::

        Spec ≅ ℕ_episodes × ℕ_seed × OpponentName × ℕ_eval_interval × ℕ_eval_episodes
               × Params × RunName × Tags × Schedule × [0, 1]_discount

    where ``Schedule = LinearExplorationSchedule``, an object with one arrow
    ``epsilon : ℕ_episode → [0, 1]``. Everything that varies per episode
    (epsilon, seeds, opponent difficulty) is *derived* from the spec inside the
    runner; the spec itself is frozen and hashable-by-parts, so one spec value
    fully determines a reproducible run.

    Usage::

        spec = AECExperimentSpec(
            episodes=5_000, seed=7, opponent="random",
            eval_interval=500, eval_episodes=200,
            params={"alpha": 0.1}, run_name="q-baseline", tags={"algo": "q"},
            exploration=LinearExplorationSchedule(1.0, 0.05, 4_000),
        )
    """

    episodes: int
    seed: int
    opponent: str
    eval_interval: int
    eval_episodes: int
    params: dict[str, Any]
    run_name: str
    tags: dict[str, str]
    exploration: LinearExplorationSchedule
    perturbed_minimax_discount: float = 1.0
    metrics_window: int = 100


class AECExperimentRunner(Generic[LearnerT]):
    """Train and evaluate one learner through a composable AEC adapter.

    The runner is generic over the category of learners: an object ``L`` is any
    type satisfying ``PolicyView`` (it has ``q_values``, ``action_distribution``,
    ``state_value``), and a ``LearnerAdapter`` packages one such learner together
    with the single morphism the runner needs::

        run_episode : Opponent × ℕ_episode × ℕ_seed × [0, 1]_ε × 𝔹_train → EpisodeMetrics

    Given a frozen ``AECExperimentSpec`` ``S`` and an adapter ``A``, the runner
    composes that morphism into the run-level operations::

        train    : () → L × [EpisodeMetrics]      # ε annealed by S.exploration
        evaluate : () → [EpisodeMetrics]          # ε = 0, opponent frozen
        write_run : ArtifactWriter × [EpisodeMetrics] → Summary

    All optional hooks (progress, evaluation rows, policy checkpoints) are plain
    functions injected at construction, so specialising the runner for a new
    algorithm means supplying morphisms, not subclassing.

    Usage::

        runner = AECExperimentRunner(spec, adapter, tracker=tracker)
        learner, metrics = runner.train()          # also fires periodic checkpoints
        summary = runner.write_run(writer, metrics)
    """

    def __init__(
        self,
        spec: AECExperimentSpec,
        adapter: LearnerAdapter,
        *,
        tracker: TrainingTracker | None = None,
        eval_opponents: tuple[str, ...] = ("random", "minimax"),
        progress_reporter: ProgressReporter | None = None,
        progress_summary: ProgressSummaryFn | None = None,
        progress_interval: int = 0,
        progress_window: int = 100,
        progress: TrainingProgressReporter | None = None,
        evaluation_row: Callable[[LearnerT, list[EpisodeMetrics], int, str], dict[str, Any]] | None = None,
        policy_checkpoint: Callable[[LearnerT, int], dict[str, Any]] | None = None,
        probe: ProbeFn | None = None,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.tracker = tracker or NoopTracker()
        self.eval_opponents = eval_opponents
        self.progress_reporter = progress_reporter
        self.progress_summary = progress_summary
        self.progress = progress
        self.progress_interval = max(0, progress_interval)
        self.progress_window = max(1, progress_window)
        self.evaluation_row = evaluation_row or default_evaluation_row
        self.policy_checkpoint = policy_checkpoint
        self.probe = probe
        self.evaluation_metrics: list[dict[str, Any]] = []
        self.policy_metrics: list[dict[str, Any]] = []

    @property
    def learner(self) -> LearnerT:
        """The adapter's learner, narrowed from ``PolicyView`` to the concrete ``LearnerT``."""
        return self.adapter.learner  # type: ignore[return-value]

    def train(self) -> tuple[LearnerT, list[EpisodeMetrics]]:
        """Run the full training loop and return the learner with per-episode metrics.

        Functional form, with spec ``S``, adapter ``A``, and ``ε = S.exploration.epsilon``::

            train : () → L × [EpisodeMetrics]
            train() = (A.learner,
                       [A.run_episode(opp_e, e, S.seed + e, ε(e), train=True)
                        | e ← 1 .. S.episodes])

        i.e. an unfold over the episode index in which each step is the adapter
        morphism applied at that episode's epsilon and derived seed. Side
        channels per step: metrics are logged to the tracker, progress is
        reported every ``progress_interval`` episodes, and an evaluation
        checkpoint fires every ``S.eval_interval`` episodes (appending to
        ``evaluation_metrics`` / ``policy_metrics``). The opponent ``opp_e`` is
        one stateful object advanced via ``set_episode(e)``, which lets
        annealing opponents change difficulty across the run.

        Usage::

            learner, metrics = runner.train()
            final_return = metrics[-1].learner_return
        """
        opponent = make_opponent(
            self.spec.opponent,
            seed=self.spec.seed + 1,
            perturbed_minimax_discount=self.spec.perturbed_minimax_discount,
        )
        metrics: list[EpisodeMetrics] = []
        self.tracker.start_run(run_name=self.spec.run_name, tags=self.spec.tags)
        self.tracker.log_params(self.spec.params)

        try:
            for episode in range(1, self.spec.episodes + 1):
                epsilon = self.spec.exploration.epsilon(episode)
                opponent.set_episode(episode)
                metric = self.adapter.run_episode(
                    opponent=opponent,
                    episode=episode,
                    seed=self.spec.seed + episode,
                    epsilon=epsilon,
                    train=True,
                )
                metric = replace(
                    metric,
                    opponent_random_move_probability=opponent_random_move_probability(
                        self.spec.opponent,
                        discount=self.spec.perturbed_minimax_discount,
                        episode=episode,
                    ),
                )
                metric = self._with_rolling_metrics(metrics, metric)
                metrics.append(metric)
                self.tracker.log_metrics(
                    mlflow_metrics_projection(asdict(metric), exclude=_MLFLOW_EXCLUDED_METRIC_FIELDS, prefix="train/"),
                    step=episode,
                )
                self._report_progress(metrics)
                self._checkpoint_if_due(episode)
            # Force a final checkpoint so every run ends with an eval row, even
            # when the episode count is not a multiple of the eval interval.
            if self.spec.eval_interval > 0 and self.spec.episodes % self.spec.eval_interval != 0:
                self._evaluation_checkpoint(self.spec.episodes)
        except Exception:
            self.tracker.end_run(status="FAILED")
            raise

        return self.learner, metrics

    def evaluate(
        self,
        *,
        episodes: int | None = None,
        opponent_name: str | None = None,
        seed: int | None = None,
    ) -> list[EpisodeMetrics]:
        """Run greedy evaluation episodes against one frozen opponent.

        Functional form, with defaults drawn from the spec ``S`` when an
        argument is ``None``::

            evaluate : ℕ_episodes? × OpponentName? × ℕ_seed? → [EpisodeMetrics]
            evaluate(n, o, s) = [A.run_episode(freeze(o), e, s + e, ε=0, train=False)
                                 | e ← 1 .. n]

        The input object differs from training in two ways that make this a
        measurement rather than a process: epsilon is pinned to ``0`` (pure
        exploitation) and ``freeze(o)`` fixes the opponent at its terminal
        episode-``S.episodes`` difficulty, so every checkpoint is scored against
        the same stationary yardstick. Learner updates are disabled
        (``train=False``); the learner is read, never written.

        Usage::

            rows = runner.evaluate(episodes=200, opponent_name="minimax")
            win_rate = sum(r.learner_return > 0 for r in rows) / len(rows)
        """
        total = self.spec.eval_episodes if episodes is None else episodes
        base_seed = self.spec.seed if seed is None else seed
        name = opponent_name or self.spec.opponent
        opponent = make_opponent(
            name,
            seed=base_seed + 1,
            perturbed_minimax_discount=self.spec.perturbed_minimax_discount,
        )
        # Freeze an annealing opponent at its terminal (hardest) difficulty so evaluation
        # is a stationary yardstick across checkpoints rather than drifting with training.
        opponent.set_episode(self.spec.episodes)
        return [
            self.adapter.run_episode(
                opponent=opponent,
                episode=episode,
                seed=base_seed + episode,
                epsilon=0.0,
                train=False,
            )
            for episode in range(1, total + 1)
        ]

    def evaluation_checkpoint(
        self,
        *,
        episode: int,
        seed: int,
        opponents: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate against every configured opponent and summarize each into one row.

        Functional form — a map of ``evaluation_row ∘ evaluate`` over the
        opponent tuple, with per-opponent seed offsets to keep streams disjoint::

            evaluation_checkpoint : ℕ_episode × ℕ_seed × OpponentName* → [Row]
            checkpoint(e, s, os) = [evaluation_row(L, evaluate(S.eval_episodes, o_i, s + 10_000·i), e, o_i)
                                    | (i, o_i) ← enumerate(os)]

        ``Row`` is whatever the injected ``EvaluationRowFn`` returns; the
        default is ``{episode, opponent} ∪ summarize(metrics)``. This is the
        pure core reused both by the periodic in-training checkpoint and by
        callers wanting an ad-hoc multi-opponent snapshot.

        Usage::

            rows = runner.evaluation_checkpoint(episode=1_000, seed=spec.seed)
            by_opponent = {row["opponent"]: row for row in rows}
        """
        rows: list[dict[str, Any]] = []
        for index, opponent_name in enumerate(opponents or self.eval_opponents):
            eval_metrics = self.evaluate(
                episodes=self.spec.eval_episodes,
                opponent_name=opponent_name,
                seed=seed + index * EVAL_OPPONENT_SEED_STRIDE,
            )
            rows.append(self.evaluation_row(self.learner, eval_metrics, episode, opponent_name))
        return rows

    def write_run(
        self,
        writer: ArtifactWriter,
        metrics: list[EpisodeMetrics],
    ) -> dict[str, Any]:
        """Persist run artifacts through the writer and close the tracking run.

        Functional form::

            write_run : ArtifactWriter × [EpisodeMetrics] → Summary
            write_run(W, ms) = W.write(L, ms, evaluation_metrics, policy_metrics)

        ``ArtifactWriter`` is the model-specific serialization object: any ``W``
        with an ``out_dir`` and a ``write`` arrow taking the learner plus the
        three metric streams the runner accumulated. As a side effect the
        tracker logs ``W.out_dir`` as artifacts and the run is ended, so this is
        the terminal operation of a run — call it once, after ``train``.

        Usage::

            learner, metrics = runner.train()
            summary = runner.write_run(writer, metrics)
        """
        summary = writer.write(self.learner, metrics, self.evaluation_metrics, self.policy_metrics)
        self.tracker.log_artifacts(writer.out_dir)
        self.tracker.end_run()
        return summary

    def _with_rolling_metrics(self, history: list[EpisodeMetrics], metric: EpisodeMetrics) -> EpisodeMetrics:
        window = self.spec.metrics_window
        recent = [*history[max(0, len(history) - window + 1) :], metric]
        return replace(metric, **rolling_means(recent, columns=_ROLLING_COLUMNS, window=window))

    def _report_progress(self, metrics: list[EpisodeMetrics]) -> None:
        if self.progress is not None:
            # The reporter owns its own cadence, so no interval check here.
            self.progress.episode(metrics[-1], episode=metrics[-1].episode, total=self.spec.episodes)
            return
        if self.progress_reporter is None or self.progress_summary is None or self.progress_interval <= 0:
            return
        episode = metrics[-1].episode
        if episode != self.spec.episodes and episode % self.progress_interval != 0:
            return
        self.progress_reporter(self.progress_summary(metrics, self.spec.episodes, self.progress_window))

    def _checkpoint_if_due(self, episode: int) -> None:
        if self.spec.eval_interval <= 0 or episode % self.spec.eval_interval != 0:
            return
        self._evaluation_checkpoint(episode)

    def _evaluation_checkpoint(self, episode: int) -> None:
        # A fixed eval seed replays the identical episode stream at every
        # checkpoint, so eval curves are comparable across checkpoints and runs.
        eval_rows = self.evaluation_checkpoint(
            episode=episode,
            seed=self.spec.seed + CHECKPOINT_EVAL_SEED_OFFSET,
            opponents=self.eval_opponents,
        )
        eval_rows = [{**row, **learner_flops(self.learner)} for row in eval_rows]
        self.evaluation_metrics.extend(eval_rows)
        if self.progress is not None:
            for row in eval_rows:
                self.progress.evaluation(
                    row,
                    episode=episode,
                    total=self.spec.episodes,
                    context=str(row.get("opponent", "")),
                )
        for row in eval_rows:
            opponent = _sanitize_metric_key(str(row["opponent"]))
            self.tracker.log_metrics(
                mlflow_metrics_projection(row, exclude=frozenset({"episode", "opponent"}), prefix=f"eval/{opponent}/"),
                step=episode,
            )
        probe_stats = self.probe(self.learner) if self.probe is not None else {}
        if probe_stats:
            self.tracker.log_metrics(probe_mlflow_metrics(probe_stats), step=episode)
        if self.policy_checkpoint is not None:
            # Probe stats ride along in the per-checkpoint policy row so they
            # land in policy_metrics.csv; MLflow gets them under probe/ above.
            policy_row = {**self.policy_checkpoint(self.learner, episode), **probe_stats}
            self.policy_metrics.append(policy_row)
            self.tracker.log_metrics(
                mlflow_metrics_projection(
                    policy_row,
                    exclude=frozenset({"episode", *PROBE_STAT_FIELDS}),
                    prefix="policy/",
                ),
                step=episode,
            )


def default_evaluation_row(
    learner: LearnerT,
    metrics: list[EpisodeMetrics],
    episode: int,
    opponent_name: str,
) -> dict[str, Any]:
    """Build the default evaluation row: identifying keys plus the shared metric summary.

    Functional form (an ``EvaluationRowFn`` that ignores the learner)::

        default_evaluation_row : L × [EpisodeMetrics] × ℕ_episode × OpponentName → Row
        default_evaluation_row(_, ms, e, o) = {episode: e, opponent: o} ∪ summarize(ms)

    Custom ``EvaluationRowFn`` implementations use the learner argument to add
    policy-dependent columns (e.g. Q-table size); this default only aggregates
    the episode metrics.

    Usage::

        row = default_evaluation_row(learner, eval_metrics, episode=500, opponent_name="random")
        # {"episode": 500, "opponent": "random", "episodes": 200, "win_rate": ..., ...}
    """
    del learner
    return {"episode": episode, "opponent": opponent_name, **_summarize_metrics(metrics)}


def _summarize_metrics(metrics: list[EpisodeMetrics]) -> dict[str, Any]:
    # Evaluation rows expose an extra ``episodes`` alias on top of the shared summary.
    return {"episodes": len(metrics), **summarize_learner_metrics(metrics)}


# MLflow metric keys allow only alphanumerics and ``_-./ :``; opponent names such as
# ``perturbed_minimax_d=0.99`` contain ``=``, which raises on log. Map illegal chars to ``_``.
_METRIC_KEY_ILLEGAL = re.compile(r"[^A-Za-z0-9_\-./: ]")


def _sanitize_metric_key(name: str) -> str:
    """Return ``name`` with characters MLflow rejects in metric keys replaced by ``_``."""

    return _METRIC_KEY_ILLEGAL.sub("_", name)
