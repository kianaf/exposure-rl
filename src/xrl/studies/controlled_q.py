"""Thin reproducible driver for the controlled mask-ratio DQN study."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from xrl.studies.analysis import write_table
from xrl.studies.cell_worker import CONTROLLED_Q_CELL_HANDLER, SubprocessCellWork
from xrl.studies.core import (
    StudyCellContext,
    StudyCellWork,
    StudyCondition,
    StudyManifestMismatchError,
    StudyProfile,
    StudyResult,
    StudyRunner,
    StudySpec,
)
from xrl.studies.fidelity import ControlledQPosthocResult, run_controlled_q_posthoc
from xrl.studies.report import PRIMARY_POLICY_METRICS, ControlledQReportResult, build_controlled_q_report
from xrl.studies.tictactoe import ExactTicTacToeEvaluator
from xrl.tictactoe.agents import annealing_discount
from xrl.training.flops import learner_training_accounting
from xrl.training.interfaces import PolicyView
from xrl.training.neural_q import (
    MASKED_TARGETS,
    SELF_DISTILL_TARGET,
    TD_ACTION_TARGET,
    NeuralQAgent,
    NeuralQConfig,
    NeuralQTrainer,
)
from xrl.training.policies import opponent_random_move_probability
from xrl.training.value_heads import HL_GAUSS, SCALAR, VALUE_HEADS, normalize_value_head

CONTROLLED_STUDY_SCHEMA_VERSION = 2
CONTROLLED_MANIFEST_FILENAME = "controlled_study_manifest.json"
CONTROLLED_RESULT_FILENAME = "controlled_study_result.json"

MASK_RATIOS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class ControlledQProfile:
    """Immutable engineering or research budget for the complete study."""

    name: str
    episodes: int
    epsilon_decay_episodes: int
    checkpoint_interval: int
    primary_seeds: tuple[int, ...]
    diagnostic_seeds: tuple[int, ...]
    explanation_seeds: tuple[int, ...]
    surrogate_updates: int
    shared_explanation_states: int
    policy_explanation_states: int
    bootstrap_resamples: int
    training_opponent: str = "random"
    terminal_random_move_rate: float = 0.05
    masked_target: str = TD_ACTION_TARGET
    mask_ratios: tuple[float, ...] = MASK_RATIOS
    primary_value_heads: tuple[str, ...] = VALUE_HEADS
    raw_input_value_heads: tuple[str, ...] = VALUE_HEADS
    posthoc_protocol: str = "legacy"
    policy_buffer_sizes: tuple[int, ...] = (128,)
    exact_policy_full_support: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "episodes",
            "epsilon_decay_episodes",
            "checkpoint_interval",
            "surrogate_updates",
            "shared_explanation_states",
            "policy_explanation_states",
            "bootstrap_resamples",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.episodes % self.checkpoint_interval != 0:
            raise ValueError("episodes must be divisible by checkpoint_interval")
        if not self.primary_seeds or not self.diagnostic_seeds or not self.explanation_seeds:
            raise ValueError("all seed sets must be non-empty")
        if not set(self.diagnostic_seeds) <= set(self.primary_seeds):
            raise ValueError("diagnostic_seeds must be a subset of primary_seeds")
        if not set(self.explanation_seeds) <= set(self.primary_seeds):
            raise ValueError("explanation_seeds must be a subset of primary_seeds")
        if self.training_opponent not in {"random", "perturbed_minimax"}:
            raise ValueError(f"training_opponent must be random or perturbed_minimax, got {self.training_opponent!r}")
        if not 0.0 < self.terminal_random_move_rate <= 1.0:
            raise ValueError("terminal_random_move_rate must lie in (0, 1]")
        if self.masked_target not in MASKED_TARGETS:
            raise ValueError(f"masked_target must be one of {MASKED_TARGETS}, got {self.masked_target!r}")
        if not self.mask_ratios:
            raise ValueError("mask_ratios must be non-empty")
        if len(set(self.mask_ratios)) != len(self.mask_ratios):
            raise ValueError(f"mask_ratios must be distinct, got {self.mask_ratios}")
        if any(not 0.0 <= ratio <= 1.0 for ratio in self.mask_ratios):
            raise ValueError(f"every mask ratio must lie in [0, 1], got {self.mask_ratios}")
        for field_name in ("primary_value_heads", "raw_input_value_heads"):
            heads = tuple(normalize_value_head(head) for head in getattr(self, field_name))
            if not heads or len(set(heads)) != len(heads):
                raise ValueError(f"{field_name} must contain distinct value heads")
        if self.posthoc_protocol not in {"legacy", "phase-one"}:
            raise ValueError("posthoc_protocol must be 'legacy' or 'phase-one'")
        if not self.policy_buffer_sizes or any(size < 1 for size in self.policy_buffer_sizes):
            raise ValueError("policy_buffer_sizes must contain positive sizes")
        if tuple(sorted(set(self.policy_buffer_sizes))) != self.policy_buffer_sizes:
            raise ValueError("policy_buffer_sizes must be strictly increasing")
        if self.shared_explanation_states % 2 or self.policy_explanation_states % 2:
            raise ValueError("shared and policy explanation state counts must be even")
        if self.posthoc_protocol == "phase-one" and len(self.policy_buffer_sizes) != 4:
            raise ValueError("Phase 1 requires exactly four policy buffer sizes")

    def study_profile(self) -> StudyProfile:
        """Resolve every training and post-hoc knob into the hashed manifest."""

        return StudyProfile(
            self.name,
            {
                "episodes": self.episodes,
                "epsilon_start": 1.0,
                "epsilon_end": 0.05,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
                "checkpoint_interval": self.checkpoint_interval,
                "training_opponent": self.training_opponent,
                "terminal_random_move_rate": self.terminal_random_move_rate,
                "perturbed_minimax_discount": (
                    annealing_discount(self.episodes, terminal_rate=self.terminal_random_move_rate)
                    if self.training_opponent == "perturbed_minimax"
                    else 1.0
                ),
                "gamma": 0.95,
                "learning_rate": 1e-3,
                "hidden_sizes": [64, 64],
                "batch_size": 32,
                "replay_warmup": 32,
                "replay_capacity": 10_000,
                "target_sync_interval": 100,
                "num_bins": 51,
                "v_min": -1.0,
                "v_max": 1.0,
                "hl_gauss_sigma_bins": 0.75,
                "masked_target": self.masked_target,
                "mask_empty_example_interval": 256,
                "mask_shapley_probability": 0.7,
                "mask_uniform_probability": 0.3,
                "surrogate_updates": self.surrogate_updates,
                "surrogate_batch_size": 64,
                "surrogate_hidden_sizes": [64, 64],
                "surrogate_full_mask_example_interval": 512,
                "surrogate_empty_mask_example_interval": 512,
                "shared_explanation_states": self.shared_explanation_states,
                "policy_explanation_states": self.policy_explanation_states,
                "explanation_seeds": list(self.explanation_seeds),
                "bootstrap_resamples": self.bootstrap_resamples,
                "posthoc_protocol": self.posthoc_protocol,
                "policy_buffer_sizes": list(self.policy_buffer_sizes),
                "exact_policy_full_support": self.exact_policy_full_support,
            },
        )


FAST_PROFILE = ControlledQProfile(
    name="fast",
    episodes=1_000,
    epsilon_decay_episodes=800,
    checkpoint_interval=250,
    primary_seeds=(0, 1, 2),
    diagnostic_seeds=(0, 1, 2),
    explanation_seeds=(0,),
    surrogate_updates=64,
    shared_explanation_states=32,
    policy_explanation_states=128,
    bootstrap_resamples=500,
)

RESEARCH_PROFILE = ControlledQProfile(
    name="research",
    episodes=10_000,
    epsilon_decay_episodes=8_000,
    checkpoint_interval=1_000,
    primary_seeds=tuple(range(10)),
    diagnostic_seeds=(0, 1, 2),
    explanation_seeds=tuple(range(5)),
    surrogate_updates=4_000,
    shared_explanation_states=32,
    policy_explanation_states=128,
    bootstrap_resamples=10_000,
)

# The predeclared matrix trains against a stationary random opponent, so its
# agents are near-optimal against random play and essentially fully exploitable
# by a best response. These profiles replicate the same matrix against an
# opponent annealing from fully random toward minimax, holding every other knob
# fixed so the opponent is the only treatment.
FAST_MINIMAX_PROFILE = replace(
    FAST_PROFILE,
    name="fast-minimax",
    training_opponent="perturbed_minimax",
)

RESEARCH_MINIMAX_PROFILE = replace(
    RESEARCH_PROFILE,
    name="research-minimax",
    training_opponent="perturbed_minimax",
)

# The direct arm's masked rows regress a bootstrapped single-action TD target, so
# no mask ratio reaches the coalition game (ADR 0044). These profiles replicate
# the matrix with the masked rows regressing every legal action against the
# network's own detached full-observation values, holding every other knob fixed
# so the objective is the only treatment. Two ratios rather than four: the direct
# arm's dose-response was measured over a 4x budget change, which r = 0.25 -> 1.0
# reproduces, and r = 0 trains no masked rows at all.
DISTILLED_MASK_RATIOS = (0.25, 1.0)

FAST_DISTILLED_PROFILE = replace(
    FAST_PROFILE,
    name="fast-distilled",
    masked_target=SELF_DISTILL_TARGET,
    mask_ratios=DISTILLED_MASK_RATIOS,
)

RESEARCH_DISTILLED_PROFILE = replace(
    RESEARCH_PROFILE,
    name="research-distilled",
    masked_target=SELF_DISTILL_TARGET,
    mask_ratios=DISTILLED_MASK_RATIOS,
)

# Phase 1 is deliberately smaller in factors and larger in evidence. The pilot
# runs the complete research workload for two seeds; changing only the seed sets
# yields the confirmatory ten-seed profile and makes its runtime projection
# meaningful.
PHASE_ONE_PILOT_PROFILE = ControlledQProfile(
    name="phase-one-pilot",
    episodes=10_000,
    epsilon_decay_episodes=8_000,
    checkpoint_interval=1_000,
    primary_seeds=(0, 1),
    diagnostic_seeds=(0, 1),
    explanation_seeds=(0, 1),
    surrogate_updates=16_000,
    shared_explanation_states=32,
    policy_explanation_states=128,
    bootstrap_resamples=2_000,
    training_opponent="perturbed_minimax",
    masked_target=SELF_DISTILL_TARGET,
    mask_ratios=MASK_RATIOS,
    primary_value_heads=(HL_GAUSS,),
    raw_input_value_heads=(HL_GAUSS,),
    posthoc_protocol="phase-one",
    policy_buffer_sizes=(128, 512, 2_048, 8_192),
    exact_policy_full_support=True,
)

PHASE_ONE_PROFILE = replace(
    PHASE_ONE_PILOT_PROFILE,
    name="phase-one",
    primary_seeds=tuple(range(10)),
    diagnostic_seeds=tuple(range(10)),
    explanation_seeds=tuple(range(10)),
    bootstrap_resamples=10_000,
)

CONTROLLED_Q_PROFILES: dict[str, ControlledQProfile] = {
    FAST_PROFILE.name: FAST_PROFILE,
    RESEARCH_PROFILE.name: RESEARCH_PROFILE,
    FAST_MINIMAX_PROFILE.name: FAST_MINIMAX_PROFILE,
    RESEARCH_MINIMAX_PROFILE.name: RESEARCH_MINIMAX_PROFILE,
    FAST_DISTILLED_PROFILE.name: FAST_DISTILLED_PROFILE,
    RESEARCH_DISTILLED_PROFILE.name: RESEARCH_DISTILLED_PROFILE,
    PHASE_ONE_PILOT_PROFILE.name: PHASE_ONE_PILOT_PROFILE,
    PHASE_ONE_PROFILE.name: PHASE_ONE_PROFILE,
}


def _profile_payload(profile: ControlledQProfile) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(json.dumps(asdict(profile), allow_nan=False))
    return payload


@dataclass(frozen=True, slots=True)
class ControlledQStudyPlan:
    """Primary studies plus an independently manifested scalar supplement."""

    root: Path
    profile: ControlledQProfile
    primary: StudySpec
    raw_input: StudySpec
    scalar_learning: StudySpec | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def manifest_payload(self) -> dict[str, Any]:
        # The scalar supplement is intentionally absent: changing or versioning
        # a diagnostic must not invalidate either categorical study manifest.
        # Its own StudySpec fingerprint is recorded in the completed result.
        return {
            "schema_version": CONTROLLED_STUDY_SCHEMA_VERSION,
            "name": "controlled-mask-ratio-q",
            "profile": _profile_payload(self.profile),
            "primary": self.primary.manifest_document(),
            "raw_input": self.raw_input.manifest_document(),
        }

    def manifest_document(self) -> dict[str, Any]:
        return {**self.manifest_payload(), "fingerprint": self.fingerprint}

    def prepare_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / CONTROLLED_MANIFEST_FILENAME
        expected = self.manifest_document()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != expected:
                raise StudyManifestMismatchError(
                    f"{self.root} belongs to a different controlled-study manifest; "
                    f"expected {self.fingerprint}, found {existing.get('fingerprint', 'missing')}"
                )
            return
        if any(self.root.iterdir()):
            raise StudyManifestMismatchError(
                f"{self.root} is non-empty but has no {CONTROLLED_MANIFEST_FILENAME}; choose a fresh root"
            )
        _write_json_atomic(path, expected)


@dataclass(frozen=True, slots=True)
class ControlledQPolicyStudyResult:
    root: Path
    manifest_fingerprint: str
    primary: StudyResult
    raw_input: StudyResult
    scalar_learning: StudyResult | None = None
    report: ControlledQReportResult | None = None

    def to_json_dict(self) -> dict[str, Any]:
        result = {
            "root": str(self.root),
            "manifest_fingerprint": self.manifest_fingerprint,
            "primary_cells": len(self.primary.cells),
            "raw_input_cells": len(self.raw_input.cells),
            "scalar_learning_cells": (len(self.scalar_learning.cells) if self.scalar_learning is not None else 0),
            "scalar_learning_manifest_fingerprint": (
                self.scalar_learning.manifest_fingerprint if self.scalar_learning is not None else None
            ),
            "reused_cells": (
                self.primary.reused_cells
                + self.raw_input.reused_cells
                + (self.scalar_learning.reused_cells if self.scalar_learning is not None else 0)
            ),
        }
        if self.report is not None:
            result["report"] = self.report.to_json_dict(root=self.root)
        return result


def controlled_q_profile(name: str) -> ControlledQProfile:
    try:
        return CONTROLLED_Q_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"profile must be one of {tuple(CONTROLLED_Q_PROFILES)}, got {name!r}") from exc


def primary_conditions(profile: ControlledQProfile) -> tuple[StudyCondition, ...]:
    """The head x ratio grid the profile declares.

    The masked target is a profile setting rather than a condition parameter:
    every cell in one root shares it, and crossing it with the ratio inside one
    matrix would pool two objectives behind one manifest.
    """

    return tuple(
        StudyCondition(
            f"primary-{head}-r{_ratio_key(ratio)}",
            {
                "arm": "primary",
                "value_head": head,
                "feature_mask_input": True,
                "masked_batch_ratio": ratio,
            },
        )
        for head in profile.primary_value_heads
        for ratio in profile.mask_ratios
    )


def raw_input_conditions(value_heads: tuple[str, ...] = VALUE_HEADS) -> tuple[StudyCondition, ...]:
    return tuple(
        StudyCondition(
            f"raw-input-{head}",
            {
                "arm": "raw-input-diagnostic",
                "value_head": head,
                "feature_mask_input": False,
                "masked_batch_ratio": 0.0,
            },
        )
        for head in value_heads
    )


def scalar_learning_conditions() -> tuple[StudyCondition, ...]:
    """Return the two scalar-Q adapter controls kept outside the main manifest."""

    return (
        StudyCondition(
            "scalar-mask-input-r0",
            {
                "arm": "scalar-learning-mask-input",
                "value_head": SCALAR,
                "feature_mask_input": True,
                "masked_batch_ratio": 0.0,
            },
        ),
        StudyCondition(
            "scalar-raw-input",
            {
                "arm": "scalar-learning-raw-input",
                "value_head": SCALAR,
                "feature_mask_input": False,
                "masked_batch_ratio": 0.0,
            },
        ),
    )


def build_controlled_q_study_plan(
    root: Path,
    *,
    profile: str | ControlledQProfile = "fast",
) -> ControlledQStudyPlan:
    resolved = controlled_q_profile(profile) if isinstance(profile, str) else profile
    study_profile = resolved.study_profile()
    root = Path(root)
    scalar_learning = None
    if resolved.posthoc_protocol == "phase-one":
        scalar_learning = StudySpec(
            name="controlled-scalar-learning-diagnostic-v1",
            root=root / "scalar_learning_v1",
            profile=study_profile,
            conditions=scalar_learning_conditions(),
            seeds=resolved.diagnostic_seeds,
        )
    return ControlledQStudyPlan(
        root=root,
        profile=resolved,
        primary=StudySpec(
            name="controlled-mask-ratio-primary",
            root=root / "primary",
            profile=study_profile,
            conditions=primary_conditions(resolved),
            seeds=resolved.primary_seeds,
        ),
        raw_input=StudySpec(
            name="controlled-mask-ratio-raw-input",
            root=root / "raw_input",
            profile=study_profile,
            conditions=raw_input_conditions(resolved.raw_input_value_heads),
            seeds=resolved.diagnostic_seeds,
        ),
        scalar_learning=scalar_learning,
    )


def neural_q_config_for_cell(context: StudyCellContext) -> NeuralQConfig:
    """Resolve one manifest condition into the ordinary DQN config."""

    settings = context.profile.settings
    parameters = context.condition.parameters
    return NeuralQConfig(
        episodes=int(settings["episodes"]),
        seed=context.seed,
        value_head=str(parameters["value_head"]),
        opponent=str(settings["training_opponent"]),
        perturbed_minimax_discount=float(settings["perturbed_minimax_discount"]),
        gamma=float(settings["gamma"]),
        learning_rate=float(settings["learning_rate"]),
        epsilon_start=float(settings["epsilon_start"]),
        epsilon_end=float(settings["epsilon_end"]),
        epsilon_decay_episodes=int(settings["epsilon_decay_episodes"]),
        replay_capacity=int(settings["replay_capacity"]),
        replay_warmup=int(settings["replay_warmup"]),
        batch_size=int(settings["batch_size"]),
        target_sync_interval=int(settings["target_sync_interval"]),
        hidden_sizes=tuple(int(value) for value in settings["hidden_sizes"]),
        v_min=float(settings["v_min"]),
        v_max=float(settings["v_max"]),
        num_bins=int(settings["num_bins"]),
        hl_gauss_sigma_bins=float(settings["hl_gauss_sigma_bins"]),
        feature_mask_input=bool(parameters["feature_mask_input"]),
        masked_batch_ratio=float(parameters["masked_batch_ratio"]),
        masked_target=str(settings["masked_target"]),
        mask_empty_example_interval=int(settings["mask_empty_example_interval"]),
        mask_shapley_probability=float(settings["mask_shapley_probability"]),
        mask_uniform_probability=float(settings["mask_uniform_probability"]),
        eval_interval=int(settings["checkpoint_interval"]),
        eval_episodes=1,
        q_probe_states=0,
        out_dir=context.out_dir,
        env_options={"obs_mode": "flat"},
    )


class StudyNeuralQTrainer(NeuralQTrainer):
    """DQN trainer whose checkpoints are exact, serialized study artifacts."""

    def __init__(
        self,
        config: NeuralQConfig,
        *,
        condition: StudyCondition,
        evaluator: ExactTicTacToeEvaluator | None = None,
    ) -> None:
        super().__init__(config, eval_opponents=())
        self.condition = condition
        self.evaluation_opponent_random_move_probability = opponent_random_move_probability(
            config.opponent,
            discount=config.perturbed_minimax_discount,
            episode=config.episodes,
        )
        self.evaluator = evaluator or ExactTicTacToeEvaluator(
            opponent_random_move_probability=self.evaluation_opponent_random_move_probability
        )
        self.exact_policy_metrics: list[dict[str, Any]] = []
        self.exact_role_metrics: list[dict[str, Any]] = []
        self.exact_opening_metrics: list[dict[str, Any]] = []
        self.checkpoint_index: list[dict[str, Any]] = []
        self.exact_checkpoint_seconds = 0.0
        self.checkpoint_publication_seconds = 0.0

    def make_pipeline(self):
        # The ordinary pipeline hardcodes the legacy minimax policy row. This
        # trainer injects exact role/opening rows and serialized checkpoints.
        return None

    def train(self):
        agent = self.make_agent()
        self._record_checkpoint(agent, episode=0)
        runner = self._make_runner(agent)
        self._runner = runner
        self.evaluation_metrics = runner.evaluation_metrics
        self.policy_metrics = runner.policy_metrics
        return runner.train()

    def policy_checkpoint(self, agent: NeuralQAgent, *, episode: int) -> dict[str, Any]:
        return self._record_checkpoint(agent, episode=episode)

    def _record_checkpoint(self, agent: NeuralQAgent, *, episode: int) -> dict[str, Any]:
        if any(int(row["episode"]) == episode for row in self.checkpoint_index):
            raise ValueError(f"duplicate study checkpoint at episode {episode}")
        checkpoint_dir = self.config.out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        digits = max(5, len(str(self.config.episodes)))
        relative_path = Path("checkpoints") / f"episode_{episode:0{digits}d}.msgpack"
        checkpoint_path = self.config.out_dir / relative_path
        publication_started = time.perf_counter()
        agent.save(checkpoint_path)
        self.checkpoint_publication_seconds += time.perf_counter() - publication_started
        exact_started = time.perf_counter()
        evaluation = self.evaluator.evaluate(cast(PolicyView, agent))
        self.exact_checkpoint_seconds += time.perf_counter() - exact_started
        common = {
            **self._condition_fields(),
            "seed": self.config.seed,
            "episode": episode,
            "evaluation_opponent": self.config.opponent,
            "evaluation_opponent_random_move_probability": self.evaluation_opponent_random_move_probability,
            **_accounting_row(agent),
        }
        policy_row = {**common, **evaluation.role_balanced_row}
        self.exact_policy_metrics.append(policy_row)
        self.exact_role_metrics.extend({**common, **row} for row in evaluation.role_rows())
        self.exact_opening_metrics.extend({**common, **row} for row in evaluation.opening_rows())
        self.checkpoint_index.append(
            {
                **common,
                "checkpoint": relative_path.as_posix(),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        )
        return policy_row

    def _condition_fields(self) -> dict[str, Any]:
        parameters = self.condition.parameters
        return {
            "condition_key": self.condition.key,
            "arm": parameters["arm"],
            "value_head": parameters["value_head"],
            "feature_mask_input": parameters["feature_mask_input"],
            "masked_batch_ratio": parameters["masked_batch_ratio"],
        }

    def write_study_artifacts(self) -> tuple[Path, ...]:
        if not self.checkpoint_index or int(self.checkpoint_index[-1]["episode"]) != self.config.episodes:
            raise ValueError("training did not produce the declared final checkpoint")
        return (
            write_table(self.config.out_dir / "checkpoint_index.parquet", self.checkpoint_index),
            write_table(self.config.out_dir / "exact_policy_metrics.parquet", self.exact_policy_metrics),
            write_table(self.config.out_dir / "exact_role_metrics.parquet", self.exact_role_metrics),
            write_table(self.config.out_dir / "exact_opening_metrics.parquet", self.exact_opening_metrics),
        )

    def replay_snapshot(self) -> tuple[Any, ...]:
        """Return the final retained replay transitions without exposing mutable storage."""

        if self._runner is None:
            return ()
        update_strategy = getattr(self._runner.adapter, "update_strategy", None)
        replay = getattr(update_strategy, "replay", None)
        snapshot = getattr(replay, "snapshot", None)
        return tuple(snapshot()) if callable(snapshot) else ()


TrainerFactory = Callable[[NeuralQConfig, StudyCellContext], StudyNeuralQTrainer]
PosthocRunner = Callable[[NeuralQAgent, StudyCellContext], ControlledQPosthocResult]


def train_controlled_q_policy_cell(
    context: StudyCellContext,
    *,
    trainer_factory: TrainerFactory | None = None,
    posthoc_runner: PosthocRunner = run_controlled_q_posthoc,
) -> Mapping[str, Any]:
    """Train, exactly evaluate, and persist one atomic condition/seed cell."""

    config = neural_q_config_for_cell(context)
    trainer = (
        trainer_factory(config, context)
        if trainer_factory is not None
        else StudyNeuralQTrainer(config, condition=context.condition)
    )
    cell_started = time.perf_counter()
    training_started = time.perf_counter()
    agent, metrics = trainer.train()
    training_total_seconds = time.perf_counter() - training_started
    publication_started = time.perf_counter()
    summary = trainer.write_run(agent, metrics)
    _canonicalize_saved_config(context.out_dir / "config.json")
    trainer.write_study_artifacts()
    policy_publication_seconds = time.perf_counter() - publication_started + trainer.checkpoint_publication_seconds
    posthoc_started = time.perf_counter()
    posthoc = (
        run_controlled_q_posthoc(agent, context, replay_transitions=trainer.replay_snapshot())
        if posthoc_runner is run_controlled_q_posthoc
        else posthoc_runner(agent, context)
    )
    posthoc_total_seconds = time.perf_counter() - posthoc_started
    final_policy = trainer.exact_policy_metrics[-1]
    learner_training_seconds = max(
        0.0,
        training_total_seconds - trainer.exact_checkpoint_seconds - trainer.checkpoint_publication_seconds,
    )
    return {
        "episodes": config.episodes,
        "checkpoints": len(trainer.checkpoint_index),
        "final_expected_return": final_policy["expected_return"],
        "final_exploitability": final_policy["exploitability"],
        "posthoc": posthoc.to_json_dict(),
        "training_summary": summary,
        "timing": {
            "learner_training_seconds": learner_training_seconds,
            "exact_checkpoint_seconds": trainer.exact_checkpoint_seconds,
            "policy_publication_seconds": policy_publication_seconds,
            "law_seconds": posthoc.law_seconds,
            "surrogate_seconds": posthoc.surrogate_seconds,
            "inference_seconds": posthoc.inference_seconds,
            "posthoc_overhead_seconds": max(
                0.0,
                posthoc_total_seconds - posthoc.law_seconds - posthoc.surrogate_seconds - posthoc.inference_seconds,
            ),
            "cell_total_seconds": time.perf_counter() - cell_started,
        },
    }


def run_controlled_q_study(
    root: Path,
    *,
    profile: str | ControlledQProfile = "fast",
    reuse: bool = True,
    trainer_factory: TrainerFactory | None = None,
    posthoc_runner: PosthocRunner = run_controlled_q_posthoc,
    primary_metric: str = "expected_return",
    workers: int = 1,
) -> ControlledQPolicyStudyResult:
    """Run or validate-resume the complete controlled study."""

    # Guard before the plan is built so the rejection is testable without the
    # neural stack and leaves no directory behind.
    if workers < 1:
        raise ValueError("workers must be positive")
    injected = trainer_factory is not None or posthoc_runner is not run_controlled_q_posthoc
    if workers > 1 and injected:
        raise ValueError(
            "workers > 1 requires the default trainer factory and post-hoc runner; "
            "injected callables cannot cross a process boundary"
        )

    plan = build_controlled_q_study_plan(root, profile=profile)
    plan.prepare_root()

    def work(context: StudyCellContext) -> Mapping[str, Any]:
        return train_controlled_q_policy_cell(
            context,
            trainer_factory=trainer_factory,
            posthoc_runner=posthoc_runner,
        )

    # Every non-injected run goes through a worker process, one worker or many,
    # so the pinned reduction order is identical at every ``workers`` value and
    # numerics cannot depend on the schedule.
    primary_work: StudyCellWork = work
    raw_input_work: StudyCellWork = work
    scalar_learning_work: StudyCellWork = work
    if not injected:
        primary_work = SubprocessCellWork(root=plan.primary.root, handler=CONTROLLED_Q_CELL_HANDLER)
        raw_input_work = SubprocessCellWork(root=plan.raw_input.root, handler=CONTROLLED_Q_CELL_HANDLER)
        if plan.scalar_learning is not None:
            scalar_learning_work = SubprocessCellWork(
                root=plan.scalar_learning.root,
                handler=CONTROLLED_Q_CELL_HANDLER,
            )

    primary = StudyRunner(plan.primary).run(primary_work, reuse=reuse, max_workers=workers)
    raw_input = StudyRunner(plan.raw_input).run(raw_input_work, reuse=reuse, max_workers=workers)
    scalar_learning = (
        StudyRunner(plan.scalar_learning).run(
            scalar_learning_work,
            reuse=reuse,
            max_workers=workers,
        )
        if plan.scalar_learning is not None
        else None
    )
    result = ControlledQPolicyStudyResult(
        root=plan.root,
        manifest_fingerprint=plan.fingerprint,
        primary=primary,
        raw_input=raw_input,
        scalar_learning=scalar_learning,
    )
    report = build_controlled_q_report(plan, result, primary_metric=primary_metric)
    result = replace(result, report=report)
    _write_json_atomic(plan.root / CONTROLLED_RESULT_FILENAME, result.to_json_dict())
    return result


# Compatibility name retained from the policy-only implementation increment.
run_controlled_q_policy_study = run_controlled_q_study


def controlled_q_study(
    out: Annotated[
        Path,
        typer.Option("--out", help="Fresh or manifest-matched study output root."),
    ] = Path("runs/controlled_mask_ratio_q"),
    profile: Annotated[
        str,
        typer.Option("--profile", help=f"Budget profile: {', '.join(CONTROLLED_Q_PROFILES)}."),
    ] = "fast",
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            help=(
                "Cells trained concurrently, each in its own thread-pinned process. "
                "Numerics are independent of this value."
            ),
        ),
    ] = 1,
    no_reuse: Annotated[
        bool,
        typer.Option("--no-reuse", help="Reject existing completed cells instead of validating and reusing them."),
    ] = False,
    primary_metric: Annotated[
        str,
        typer.Option(
            "--primary-metric",
            help=f"Report endpoint; one of {', '.join(PRIMARY_POLICY_METRICS)}.",
        ),
    ] = "expected_return",
) -> None:
    """Run the controlled mask-ratio DQN matrix."""

    result = run_controlled_q_study(
        out,
        profile=profile,
        reuse=not no_reuse,
        primary_metric=primary_metric,
        workers=workers,
    )
    print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))


def _ratio_key(ratio: float) -> str:
    return f"{ratio:g}"


def _accounting_row(agent: NeuralQAgent) -> dict[str, Any]:
    accounting = learner_training_accounting(agent)
    return {
        "replay_transition_count": int(accounting.get("replay_transition_count", 0)),
        "gradient_update_count": int(accounting.get("gradient_update_count", 0)),
        "full_training_example_count": int(accounting.get("full_training_example_count", 0)),
        "masked_training_example_count": int(accounting.get("masked_training_example_count", 0)),
        "masked_legal_target_count": int(accounting.get("masked_legal_target_count", 0)),
        "predictor_training_example_count": int(accounting.get("predictor_training_example_count", 0)),
        "training_flops": float(accounting.get("training_flops", 0.0)),
        "flops_source": accounting.get("flops_source") or "unmeasured",
    }


def _canonicalize_saved_config(path: Path) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["out_dir"] = "."
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


__all__ = [
    "CONTROLLED_MANIFEST_FILENAME",
    "CONTROLLED_Q_PROFILES",
    "CONTROLLED_RESULT_FILENAME",
    "FAST_MINIMAX_PROFILE",
    "DISTILLED_MASK_RATIOS",
    "RESEARCH_DISTILLED_PROFILE",
    "FAST_DISTILLED_PROFILE",
    "FAST_PROFILE",
    "MASK_RATIOS",
    "PHASE_ONE_PILOT_PROFILE",
    "PHASE_ONE_PROFILE",
    "RESEARCH_MINIMAX_PROFILE",
    "RESEARCH_PROFILE",
    "ControlledQPolicyStudyResult",
    "ControlledQProfile",
    "ControlledQStudyPlan",
    "StudyNeuralQTrainer",
    "build_controlled_q_study_plan",
    "controlled_q_profile",
    "controlled_q_study",
    "neural_q_config_for_cell",
    "primary_conditions",
    "raw_input_conditions",
    "scalar_learning_conditions",
    "run_controlled_q_policy_study",
    "run_controlled_q_study",
    "train_controlled_q_policy_cell",
]
