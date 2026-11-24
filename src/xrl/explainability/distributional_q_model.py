"""Masked distributional Q surrogate model and config."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xrl.explainability.distributional_q_masks import CoalitionMaskSamplerConfig, masked_surrogate_inputs
from xrl.training.flops import StepCost, measure_step_cost, surrogate_update_flops
from xrl.training.neural_q import jax, jnp, nn, optax, require_neural_dependencies, serialization

_PROBABILITY_FLOOR = 1e-8

if nn is not None:

    class MaskedDistributionalQNetwork(nn.Module):
        """MLP surrogate that predicts categorical Q distributions from masked states."""

        num_actions: int
        num_bins: int
        hidden_sizes: tuple[int, ...] = (64, 64)

        @nn.compact
        def __call__(self, x):
            for hidden_size in self.hidden_sizes:
                x = nn.relu(nn.Dense(hidden_size)(x))
            x = nn.Dense(self.num_actions * self.num_bins)(x)
            return x.reshape((x.shape[0], self.num_actions, self.num_bins))

else:
    MaskedDistributionalQNetwork = None


@dataclass(frozen=True, slots=True)
class DistributionalQSurrogateConfig:
    """Configuration for a masked distributional Q surrogate."""

    base_run_dir: Path
    out_dir: Path = Path("runs/q_surrogate")
    seed: int = 0
    updates: int = 1_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)
    loss: str = "cross_entropy"
    feature_count: int = 9
    dataset: str = "auto"
    rollout_states: int = 2_048
    rollout_epsilon: float = 0.1
    opponent: str = "random"
    perturbed_minimax_probability: float | None = None
    max_distribution_points: int | None = None
    full_mask_example_interval: int = 512
    empty_mask_example_interval: int = 512
    shapley_probability: float = 0.7
    uniform_probability: float = 0.3
    guided_warmup_updates: int = 200
    guided_refresh_interval: int = 100
    guided_probability: float = 0.25
    guided_epsilon: float = 1e-3
    log_interval: int = 50

    def __post_init__(self) -> None:
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss not in {"cross_entropy", "js"}:
            raise ValueError("loss must be 'cross_entropy' or 'js'")
        if self.dataset not in {"auto", "exact", "rollout"}:
            raise ValueError("dataset must be 'auto', 'exact', or 'rollout'")
        if self.rollout_states < 1:
            raise ValueError("rollout_states must be positive")
        if not 0.0 <= self.rollout_epsilon <= 1.0:
            raise ValueError("rollout_epsilon must be in [0, 1]")
        if self.max_distribution_points is not None and self.max_distribution_points < 1:
            raise ValueError("max_distribution_points must be positive when set")
        if self.guided_warmup_updates < 0:
            raise ValueError("guided_warmup_updates must be non-negative")
        if self.guided_refresh_interval < 1:
            raise ValueError("guided_refresh_interval must be positive")
        if self.log_interval < 1:
            raise ValueError("log_interval must be positive")
        CoalitionMaskSamplerConfig(
            feature_count=self.feature_count,
            full_mask_example_interval=self.full_mask_example_interval,
            empty_mask_example_interval=self.empty_mask_example_interval,
            shapley_probability=self.shapley_probability,
            uniform_probability=self.uniform_probability,
            guided_probability=self.guided_probability,
            guided_epsilon=self.guided_epsilon,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_run_dir"] = str(self.base_run_dir)
        data["out_dir"] = str(self.out_dir)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> DistributionalQSurrogateConfig:
        normalized = dict(data)
        # Provenance is a property of the written artifact, not of the requested
        # configuration, so it is recorded beside the config and dropped here.
        normalized.pop("base_run_started_at", None)
        normalized["base_run_dir"] = Path(normalized["base_run_dir"])
        normalized["out_dir"] = Path(normalized["out_dir"])
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (64, 64)))
        return cls(**normalized)

    @property
    def mask_sampler_config(self) -> CoalitionMaskSamplerConfig:
        return CoalitionMaskSamplerConfig(
            feature_count=self.feature_count,
            full_mask_example_interval=self.full_mask_example_interval,
            empty_mask_example_interval=self.empty_mask_example_interval,
            shapley_probability=self.shapley_probability,
            uniform_probability=self.uniform_probability,
            guided_probability=self.guided_probability,
            guided_epsilon=self.guided_epsilon,
        )


class MaskedDistributionalQSurrogate:
    """Trainable masked surrogate for a categorical DQN policy."""

    def __init__(
        self,
        config: DistributionalQSurrogateConfig,
        *,
        num_actions: int,
        observation_size: int,
        support: Sequence[float],
    ) -> None:
        require_neural_dependencies()
        if int(observation_size) != config.feature_count:
            raise ValueError(
                f"observation_size ({int(observation_size)}) must match config.feature_count ({config.feature_count})"
            )
        self.config = config
        self.num_actions = int(num_actions)
        self.observation_size = int(observation_size)
        self.support = np.asarray(support, dtype=np.float32).reshape(-1)
        self.model = MaskedDistributionalQNetwork(
            num_actions=self.num_actions,
            num_bins=len(self.support),
            hidden_sizes=config.hidden_sizes,
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
        loss_name = self.config.loss

        def train_step(params, opt_state, inputs, target_probabilities, action_masks):
            def loss_fn(current_params):
                logits = model.apply(current_params, inputs)
                mask = action_masks.astype(jnp.float32)
                if loss_name == "cross_entropy":
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    per_action = -jnp.sum(target_probabilities * log_probs, axis=-1)
                else:
                    probs = jax.nn.softmax(logits, axis=-1)
                    midpoint = 0.5 * (probs + target_probabilities)
                    target_kl = jnp.sum(
                        target_probabilities
                        * (
                            jnp.log(jnp.maximum(target_probabilities, _PROBABILITY_FLOOR))
                            - jnp.log(jnp.maximum(midpoint, _PROBABILITY_FLOOR))
                        ),
                        axis=-1,
                    )
                    pred_kl = jnp.sum(
                        probs
                        * (
                            jnp.log(jnp.maximum(probs, _PROBABILITY_FLOOR))
                            - jnp.log(jnp.maximum(midpoint, _PROBABILITY_FLOOR))
                        ),
                        axis=-1,
                    )
                    per_action = 0.5 * (target_kl + pred_kl)
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
        target_probabilities: np.ndarray,
        action_masks: np.ndarray,
    ) -> float:
        """Apply one surrogate gradient update."""

        inputs = jnp.asarray(masked_surrogate_inputs(states, known_masks), dtype=jnp.float32)
        targets = jnp.asarray(target_probabilities, dtype=jnp.float32)
        masks = jnp.asarray(action_masks, dtype=jnp.float32)
        step_args = (self.params, self.opt_state, inputs, targets, masks)
        if self._step_cost is None:
            self._step_cost = measure_step_cost(
                self._train_step,
                step_args,
                fallback=lambda: surrogate_update_flops(
                    input_width=self.observation_size * 2,
                    hidden_sizes=self.config.hidden_sizes,
                    output_size=self.num_actions * len(self.support),
                    batch_size=int(inputs.shape[0]),
                ),
            )
        self.params, self.opt_state, loss = self._train_step(*step_args)
        self.update_count += 1
        return float(loss)

    @property
    def step_cost(self) -> StepCost | None:
        """Cost of one surrogate update, or ``None`` before the first update."""

        return self._step_cost

    @property
    def training_flops(self) -> float:
        """Cumulative FLOPs spent distilling this surrogate.

        Charged to the surrogate-backed arm on top of its base DQN: the
        distillation stage is compute that the direct mask-aware arm never
        spends.
        """

        return 0.0 if self._step_cost is None else self._step_cost.total(self.update_count)

    @property
    def flops_source(self) -> str | None:
        """Which method produced :attr:`training_flops`, or ``None`` if unmeasured."""

        return None if self._step_cost is None else self._step_cost.source

    def q_distributions(
        self,
        observation: np.ndarray | Iterable[int | float],
        known_mask: np.ndarray | Iterable[int],
    ) -> np.ndarray:
        """Return categorical Q distributions for one masked observation."""

        state = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        mask = np.asarray(known_mask, dtype=np.float32).reshape(1, -1)
        inputs = jnp.asarray(masked_surrogate_inputs(state, mask), dtype=jnp.float32)
        logits = self.model.apply(self.params, inputs)
        return np.asarray(jax.nn.softmax(logits, axis=-1)[0], dtype=np.float32)

    def q_values(
        self, observation: np.ndarray | Iterable[int | float], known_mask: np.ndarray | Iterable[int]
    ) -> np.ndarray:
        """Return expected Q-values decoded from categorical distributions."""

        probabilities = self.q_distributions(observation, known_mask)
        return np.sum(probabilities * self.support[None, :], axis=-1)

    def save(self, path: Path) -> None:
        """Write surrogate parameters to disk."""

        require_neural_dependencies()
        path.write_bytes(serialization.to_bytes(self.params))

    def load_params(self, path: Path) -> None:
        """Load surrogate parameters from disk."""

        require_neural_dependencies()
        self.params = serialization.from_bytes(self.params, path.read_bytes())
