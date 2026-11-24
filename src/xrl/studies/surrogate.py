"""Head-aware masked-Q surrogates for controlled studies."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from xrl._masking import CoalitionMaskSampler, CoalitionMaskSamplerConfig, masked_inputs
from xrl._optional import jax, jnp, nn, optax, require_neural_dependencies, serialization
from xrl.explainability.tictactoe import TicTacToeStateDistribution
from xrl.studies.analysis import write_table
from xrl.training.flops import StepCost, learner_flops, measure_step_cost, surrogate_update_flops
from xrl.training.value_heads import SCALAR, normalize_value_head

ReferenceName = Literal["shared", "policy", "uniform", "replay"]
TargetKind = Literal["scalar", "categorical"]
TargetRepresentation = Literal["native", "scalar"]

SURROGATE_SCHEMA_VERSION = 2

if nn is not None:

    class MaskedQSurrogateNetwork(nn.Module):
        """One MLP shape with a scalar or categorical action-value output."""

        num_actions: int
        hidden_sizes: tuple[int, ...]
        num_bins: int | None = None

        @nn.compact
        def __call__(self, x):
            for hidden_size in self.hidden_sizes:
                x = nn.relu(nn.Dense(hidden_size)(x))
            output_size = self.num_actions if self.num_bins is None else self.num_actions * self.num_bins
            x = nn.Dense(output_size)(x)
            if self.num_bins is None:
                return x
            return x.reshape((x.shape[0], self.num_actions, self.num_bins))

else:
    MaskedQSurrogateNetwork = None  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class MaskedQSurrogateConfig:
    """Matched training contract for either study reference law and Q head."""

    reference: ReferenceName
    seed: int = 0
    updates: int = 4_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    full_mask_example_interval: int | None = 512
    empty_mask_example_interval: int | None = 512
    shapley_probability: float = 0.7
    uniform_probability: float = 0.3
    log_interval: int = 50
    target_representation: TargetRepresentation = "native"

    def __post_init__(self) -> None:
        if self.reference not in {"shared", "policy", "uniform", "replay"}:
            raise ValueError("reference must be 'shared', 'policy', 'uniform', or 'replay'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
        if self.target_representation not in {"native", "scalar"}:
            raise ValueError("target_representation must be 'native' or 'scalar'")
        # Validate the mixture independently of a game-specific feature count.
        self.mask_sampler_config(feature_count=2)

    def mask_sampler_config(self, *, feature_count: int) -> CoalitionMaskSamplerConfig:
        """Return the shared unguided coalition law for a resolved input size."""

        return CoalitionMaskSamplerConfig(
            feature_count=feature_count,
            full_mask_example_interval=self.full_mask_example_interval,
            empty_mask_example_interval=self.empty_mask_example_interval,
            shapley_probability=self.shapley_probability,
            uniform_probability=self.uniform_probability,
            guided_probability=0.0,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> MaskedQSurrogateConfig:
        normalized = dict(data)
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (64, 64)))
        return cls(**normalized)


def _readonly_float_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).copy()
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class MaskedQSurrogateDataset:
    """Frozen base-Q targets sampled under one declared state law."""

    reference: ReferenceName
    target_kind: TargetKind
    value_head: str
    observations: np.ndarray
    action_masks: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    support: np.ndarray | None = None

    def __post_init__(self) -> None:
        value_head = normalize_value_head(self.value_head)
        if self.reference not in {"shared", "policy", "uniform", "replay"}:
            raise ValueError("reference must be 'shared', 'policy', 'uniform', or 'replay'")
        expected_kind = "scalar" if value_head == SCALAR else "categorical"
        if self.target_kind != expected_kind:
            raise ValueError(f"{value_head!r} requires target_kind={expected_kind!r}")

        observations = _readonly_float_array(self.observations)
        action_masks = _readonly_float_array(self.action_masks)
        targets = _readonly_float_array(self.targets)
        weights = _readonly_float_array(self.weights).reshape(-1)
        if observations.ndim != 2 or observations.shape[0] < 1:
            raise ValueError("observations must have shape (points, features) with at least one point")
        if action_masks.ndim != 2 or action_masks.shape[0] != observations.shape[0]:
            raise ValueError("action_masks must have one row per observation")
        if not np.all(np.isin(action_masks, (0.0, 1.0))) or np.any(action_masks.sum(axis=1) < 1):
            raise ValueError("each action mask must be binary and contain a legal action")
        if weights.shape != (observations.shape[0],) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite, non-negative, and have one entry per observation")
        total_weight = float(weights.sum())
        if total_weight <= 0:
            raise ValueError("weights must have positive mass")
        weights = _readonly_float_array(weights / total_weight).reshape(-1)

        support: np.ndarray | None
        if self.target_kind == "scalar":
            if targets.shape != action_masks.shape:
                raise ValueError("scalar targets must have shape (points, actions)")
            support = None
        else:
            support = _readonly_float_array(self.support).reshape(-1)
            if support.shape[0] < 2:
                raise ValueError("categorical targets require at least two support bins")
            if targets.shape != (*action_masks.shape, support.shape[0]):
                raise ValueError("categorical targets must have shape (points, actions, bins)")
            if np.any(targets < -1e-7) or not np.allclose(targets.sum(axis=-1), 1.0, atol=1e-5):
                raise ValueError("categorical targets must be normalized probability vectors")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite")

        object.__setattr__(self, "value_head", value_head)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "action_masks", action_masks)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "support", support)

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.observations.shape[1])

    @property
    def num_actions(self) -> int:
        return int(self.action_masks.shape[1])


def build_masked_q_surrogate_dataset(
    base_agent: Any,
    distribution: TicTacToeStateDistribution,
    *,
    reference: ReferenceName,
    target_representation: TargetRepresentation = "native",
) -> MaskedQSurrogateDataset:
    """Freeze scalar values or categorical probabilities under ``distribution``."""

    if not distribution.points:
        raise ValueError("reference distribution must contain at least one point")
    observations = np.stack([point.observation_array for point in distribution.points]).astype(np.float32)
    action_masks = np.stack([point.action_mask_array for point in distribution.points]).astype(np.float32)
    weights = np.asarray([point.weight for point in distribution.points], dtype=np.float32)
    return build_masked_q_surrogate_dataset_from_arrays(
        base_agent,
        observations,
        action_masks,
        weights,
        reference=reference,
        target_representation=target_representation,
    )


def build_masked_q_surrogate_dataset_from_arrays(
    base_agent: Any,
    observations: np.ndarray,
    action_masks: np.ndarray,
    weights: np.ndarray,
    *,
    reference: ReferenceName,
    target_representation: TargetRepresentation = "native",
) -> MaskedQSurrogateDataset:
    """Freeze teacher targets for a weighted array-backed state law."""

    source_value_head = normalize_value_head(str(getattr(base_agent.config, "value_head", SCALAR)))
    if target_representation not in {"native", "scalar"}:
        raise ValueError("target_representation must be 'native' or 'scalar'")
    value_head = SCALAR if target_representation == "scalar" else source_value_head
    observations = np.asarray(observations, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    if value_head == SCALAR:
        batched = getattr(base_agent, "q_values_observations", None)
        targets = (
            np.asarray(batched(observations), dtype=np.float32)
            if callable(batched)
            else np.stack([base_agent.q_values(observation) for observation in observations]).astype(np.float32)
        )
        return MaskedQSurrogateDataset(
            reference=reference,
            target_kind="scalar",
            value_head=value_head,
            observations=observations,
            action_masks=action_masks,
            targets=targets,
            weights=weights,
        )

    batched_distributions = getattr(base_agent, "q_distributions_observations", None)
    batch_result = batched_distributions(observations) if callable(batched_distributions) else None
    if batch_result is not None:
        support, probabilities_array = batch_result
        support = np.asarray(support, dtype=np.float32)
        probabilities_array = np.asarray(probabilities_array, dtype=np.float32)
    else:
        supports: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for observation in observations:
            result = base_agent.q_distributions(observation)
            if result is None:
                raise TypeError(f"categorical head {value_head!r} did not expose q_distributions")
            point_support, point_probabilities = result
            supports.append(np.asarray(point_support, dtype=np.float32))
            probabilities.append(np.asarray(point_probabilities, dtype=np.float32))
        support = supports[0]
        if any(candidate.shape != support.shape or not np.allclose(candidate, support) for candidate in supports[1:]):
            raise ValueError("categorical support changed across reference states")
        probabilities_array = np.stack(probabilities)
    return MaskedQSurrogateDataset(
        reference=reference,
        target_kind="categorical",
        value_head=value_head,
        observations=observations,
        action_masks=action_masks,
        targets=probabilities_array,
        weights=weights,
        support=support,
    )


class MaskedQSurrogate:
    """Masked Q model with a common expected-value surface for either head."""

    def __init__(
        self,
        config: MaskedQSurrogateConfig,
        *,
        target_kind: TargetKind,
        value_head: str,
        num_actions: int,
        observation_size: int,
        support: Sequence[float] | None = None,
    ) -> None:
        require_neural_dependencies()
        value_head = normalize_value_head(value_head)
        expected_kind = "scalar" if value_head == SCALAR else "categorical"
        if target_kind != expected_kind:
            raise ValueError(f"{value_head!r} requires target_kind={expected_kind!r}")
        if num_actions < 1 or observation_size < 2:
            raise ValueError("num_actions must be positive and observation_size must be at least two")
        support_array = None if support is None else np.asarray(support, dtype=np.float32).reshape(-1)
        if target_kind == "categorical" and (support_array is None or support_array.shape[0] < 2):
            raise ValueError("categorical surrogates require at least two support bins")
        if target_kind == "scalar" and support_array is not None:
            raise ValueError("scalar surrogates do not have categorical support")

        self.config = config
        self.target_kind: TargetKind = target_kind
        self.value_head = value_head
        self.num_actions = int(num_actions)
        self.observation_size = int(observation_size)
        self.support = support_array if support_array is not None else np.asarray([], dtype=np.float32)
        self.model = MaskedQSurrogateNetwork(
            num_actions=self.num_actions,
            hidden_sizes=config.hidden_sizes,
            num_bins=None if target_kind == "scalar" else len(self.support),
        )
        self._key = jax.random.PRNGKey(config.seed)
        dummy = jnp.zeros((1, self.observation_size * 2), dtype=jnp.float32)
        self.params = self.model.init(self._key, dummy)
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self._train_step = self._make_train_step()
        self._step_cost: StepCost | None = None
        self.update_count = 0

    def _make_train_step(self):
        model = self.model
        optimizer = self.optimizer
        target_kind = self.target_kind

        def train_step(params, opt_state, inputs, targets, action_masks):
            def loss_fn(current_params):
                outputs = cast(Any, model.apply(current_params, inputs))
                if target_kind == "scalar":
                    per_action = jnp.square(outputs - targets)
                else:
                    per_action = -jnp.sum(targets * jax.nn.log_softmax(outputs, axis=-1), axis=-1)
                mask = action_masks.astype(jnp.float32)
                return jnp.sum(per_action * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss

        return jax.jit(train_step)

    def update(
        self,
        states: np.ndarray,
        known_masks: np.ndarray,
        targets: np.ndarray,
        action_masks: np.ndarray,
    ) -> float:
        """Apply one legal-action scalar-MSE or categorical-CE update."""

        inputs = jnp.asarray(masked_inputs(states, known_masks), dtype=jnp.float32)
        target_array = jnp.asarray(targets, dtype=jnp.float32)
        action_mask_array = jnp.asarray(action_masks, dtype=jnp.float32)
        expected_shape: tuple[int, ...] = (inputs.shape[0], self.num_actions)
        if self.target_kind == "categorical":
            expected_shape = (*expected_shape, len(self.support))
        if tuple(target_array.shape) != expected_shape:
            raise ValueError(f"expected targets with shape {expected_shape}, got {tuple(target_array.shape)}")
        if tuple(action_mask_array.shape) != tuple(expected_shape[:2]):
            raise ValueError(
                f"expected action_masks with shape {tuple(expected_shape[:2])}, got {tuple(action_mask_array.shape)}"
            )
        step_args = (self.params, self.opt_state, inputs, target_array, action_mask_array)
        if self._step_cost is None:
            output_size = self.num_actions if self.target_kind == "scalar" else self.num_actions * len(self.support)
            self._step_cost = measure_step_cost(
                self._train_step,
                step_args,
                fallback=lambda: surrogate_update_flops(
                    input_width=self.observation_size * 2,
                    hidden_sizes=self.config.hidden_sizes,
                    output_size=output_size,
                    batch_size=int(inputs.shape[0]),
                ),
            )
        self.params, self.opt_state, loss = self._train_step(*step_args)
        self.update_count += 1
        return float(loss)

    @property
    def step_cost(self) -> StepCost | None:
        return self._step_cost

    @property
    def training_flops(self) -> float:
        return 0.0 if self._step_cost is None else self._step_cost.total(self.update_count)

    @property
    def flops_source(self) -> str | None:
        return None if self._step_cost is None else self._step_cost.source

    def _outputs_batch(self, observations: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        inputs = jnp.asarray(masked_inputs(observations, known_masks), dtype=jnp.float32)
        return np.asarray(self.model.apply(self.params, inputs), dtype=np.float32)

    def q_values_batch(self, observations: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        """Return expected Q for a batch of explicit coalitions."""

        outputs = self._outputs_batch(observations, known_masks)
        if self.target_kind == "scalar":
            return outputs
        probabilities = np.asarray(jax.nn.softmax(jnp.asarray(outputs), axis=-1), dtype=np.float32)
        return np.sum(probabilities * self.support[None, None, :], axis=-1)

    def q_values(
        self,
        observation: np.ndarray | Iterable[int | float],
        known_mask: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        """Return expected Q for one explicit coalition."""

        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        mask_array = np.asarray(known_mask, dtype=np.float32).reshape(1, -1)
        return self.q_values_batch(observation_array, mask_array)[0]

    def q_distributions_batch(self, observations: np.ndarray, known_masks: np.ndarray) -> np.ndarray:
        """Return categorical probabilities, rejecting scalar surrogates."""

        if self.target_kind != "categorical":
            raise TypeError("q_distributions requires a categorical surrogate")
        outputs = self._outputs_batch(observations, known_masks)
        return np.asarray(jax.nn.softmax(jnp.asarray(outputs), axis=-1), dtype=np.float32)

    def q_distributions(
        self,
        observation: np.ndarray | Iterable[int | float],
        known_mask: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        """Return categorical probabilities for one explicit coalition."""

        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        mask_array = np.asarray(known_mask, dtype=np.float32).reshape(1, -1)
        return self.q_distributions_batch(observation_array, mask_array)[0]

    def save(self, path: Path) -> None:
        path.write_bytes(serialization.to_bytes(self.params))

    def load_params(self, path: Path) -> None:
        self.params = serialization.from_bytes(self.params, path.read_bytes())


@dataclass(frozen=True, slots=True)
class SurrogateTrainingMetric:
    update: int
    loss: float

    def to_row(self) -> dict[str, float | int]:
        return {"update": self.update, "loss": self.loss}


@dataclass(frozen=True, slots=True)
class SurrogateTrainingSummary:
    reference: ReferenceName
    target_kind: TargetKind
    value_head: str
    source_value_head: str
    target_representation: TargetRepresentation
    updates: int
    training_points: int
    final_loss: float
    full_mask_examples: int
    empty_mask_examples: int
    training_flops: float
    flops_source: str | None
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaskedQSurrogateTrainingResult:
    surrogate: MaskedQSurrogate
    dataset: MaskedQSurrogateDataset
    metrics: tuple[SurrogateTrainingMetric, ...]
    summary: SurrogateTrainingSummary


class MaskedQSurrogateTrainer:
    """Train the same masked adapter against scalar or categorical base Q."""

    def __init__(self, config: MaskedQSurrogateConfig) -> None:
        self.config = config

    def train(
        self,
        base_agent: Any,
        distribution: TicTacToeStateDistribution,
    ) -> MaskedQSurrogateTrainingResult:
        started = time.perf_counter()
        source_value_head = normalize_value_head(str(getattr(base_agent.config, "value_head", SCALAR)))
        dataset = build_masked_q_surrogate_dataset(
            base_agent,
            distribution,
            reference=self.config.reference,
            target_representation=self.config.target_representation,
        )
        return self.train_dataset(
            dataset,
            source_value_head=source_value_head,
            started=started,
        )

    def train_dataset(
        self,
        dataset: MaskedQSurrogateDataset,
        *,
        source_value_head: str,
        started: float | None = None,
    ) -> MaskedQSurrogateTrainingResult:
        """Train from a frozen weighted dataset, including replay-backed laws."""

        started = time.perf_counter() if started is None else started
        source_value_head = normalize_value_head(source_value_head)
        if dataset.reference != self.config.reference:
            raise ValueError(f"dataset reference {dataset.reference!r} != trainer reference {self.config.reference!r}")
        expected_head = SCALAR if self.config.target_representation == "scalar" else source_value_head
        if dataset.value_head != expected_head:
            raise ValueError(f"dataset head {dataset.value_head!r} != expected {expected_head!r}")
        surrogate = MaskedQSurrogate(
            self.config,
            target_kind=dataset.target_kind,
            value_head=dataset.value_head,
            num_actions=dataset.num_actions,
            observation_size=dataset.feature_count,
            support=None if dataset.support is None else tuple(float(value) for value in dataset.support),
        )
        rng = np.random.default_rng(self.config.seed)
        sampler = CoalitionMaskSampler(
            self.config.mask_sampler_config(feature_count=dataset.feature_count),
            seed=self.config.seed,
        )
        metrics: list[SurrogateTrainingMetric] = []
        full_masks = 0
        empty_masks = 0
        for update in range(self.config.updates):
            indices = rng.choice(dataset.size, size=self.config.batch_size, replace=True, p=dataset.weights)
            known_masks = sampler.sample_batch(
                self.config.batch_size,
                example_offset=update * self.config.batch_size,
            )
            full_masks += int(np.sum(np.all(known_masks == 1, axis=1)))
            empty_masks += int(np.sum(np.all(known_masks == 0, axis=1)))
            loss = surrogate.update(
                dataset.observations[indices],
                known_masks,
                dataset.targets[indices],
                dataset.action_masks[indices],
            )
            if update % self.config.log_interval == 0 or update == self.config.updates - 1:
                metrics.append(SurrogateTrainingMetric(update=update + 1, loss=loss))

        accounting = learner_flops(surrogate)
        summary = SurrogateTrainingSummary(
            reference=self.config.reference,
            target_kind=dataset.target_kind,
            value_head=dataset.value_head,
            source_value_head=source_value_head,
            target_representation=self.config.target_representation,
            updates=self.config.updates,
            training_points=dataset.size,
            final_loss=float(metrics[-1].loss),
            full_mask_examples=full_masks,
            empty_mask_examples=empty_masks,
            training_flops=float(accounting.get("training_flops", 0.0)),
            flops_source=accounting.get("flops_source"),
            duration_seconds=max(0.0, time.perf_counter() - started),
        )
        return MaskedQSurrogateTrainingResult(
            surrogate=surrogate,
            dataset=dataset,
            metrics=tuple(metrics),
            summary=summary,
        )

    def write_artifacts(self, result: MaskedQSurrogateTrainingResult, out_dir: Path) -> tuple[Path, ...]:
        """Persist a self-describing surrogate inside an atomic study cell."""

        out_dir.mkdir(parents=True, exist_ok=True)
        model_record = {
            "target_kind": result.surrogate.target_kind,
            "value_head": result.surrogate.value_head,
            "num_actions": result.surrogate.num_actions,
            "observation_size": result.surrogate.observation_size,
            "support": result.surrogate.support.tolist(),
        }
        config_path = out_dir / "surrogate_config.json"
        params_path = out_dir / "surrogate.msgpack"
        metrics_path = out_dir / "surrogate_metrics.parquet"
        summary_path = out_dir / "surrogate_summary.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": SURROGATE_SCHEMA_VERSION,
                    "config": self.config.to_json_dict(),
                    "model": model_record,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        result.surrogate.save(params_path)
        write_table(metrics_path, (metric.to_row() for metric in result.metrics))
        summary_path.write_text(
            json.dumps(result.summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return config_path, params_path, metrics_path, summary_path


def load_masked_q_surrogate(run_dir: Path) -> MaskedQSurrogate:
    """Load a self-describing head-aware study surrogate."""

    record = json.loads((run_dir / "surrogate_config.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != SURROGATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported surrogate schema version {record.get('schema_version')!r}")
    config = MaskedQSurrogateConfig.from_json_dict(record["config"])
    model_record = record["model"]
    support = model_record.get("support") or None
    surrogate = MaskedQSurrogate(
        config,
        target_kind=model_record["target_kind"],
        value_head=model_record["value_head"],
        num_actions=int(model_record["num_actions"]),
        observation_size=int(model_record["observation_size"]),
        support=support,
    )
    surrogate.load_params(run_dir / "surrogate.msgpack")
    return surrogate
