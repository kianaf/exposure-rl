"""DQN-style Q-learning agents backed by JAX/Flax function approximation."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xrl._masking import CoalitionMaskSampler, CoalitionMaskSamplerConfig, masked_inputs, validate_mask_mixture
from xrl._optional import jax, jnp, nn, optax, require_neural_dependencies, serialization
from xrl.training._neural.adapters import neural_select_learner_action
from xrl.training._neural.replay import ReplayBuffer, ReplayTransition
from xrl.training._neural.trainer import BaseAECTrainer
from xrl.training._neural.value_projection import (
    DEFAULT_HL_GAUSS_SIGMA_BINS,
    DEFAULT_NUM_BINS,
    DEFAULT_V_MAX,
    DEFAULT_V_MIN,
    _project_values_jax,
    hl_gauss_projection,  # noqa: F401 - re-exported for callers importing from xrl.training.neural_q
    support_values,
    two_hot_projection,  # noqa: F401 - re-exported for callers importing from xrl.training.neural_q
)
from xrl.training.config import parse_hidden_sizes, validate_common_config, validate_exploration_config
from xrl.training.encoders import EncodedView, StateActionEncoder, make_encoder
from xrl.training.episode import EpisodeMetrics, LearnerStats, LearnerStep, run_aec_episode
from xrl.training.flops import StepCost, dqn_update_flops, measure_step_cost
from xrl.training.q_learning import (
    boltzmann_distribution,
    epsilon_greedy_distribution,
    legal_actions_from_mask,
)
from xrl.training.run_metadata import RunIdentity, execute_training_run
from xrl.training.tracking import DEFAULT_MLFLOW_TRACKING_URI, TrackerConfig, tracker_from_cli_args
from xrl.training.value_heads import CATEGORICAL_VALUE_HEADS, SCALAR, normalize_value_head

DEFAULT_REPLAY_WARMUP = 32

# What the appended masked rows regress against. The historical objective gives
# them the same bootstrapped, single-action TD target as the full rows, so its
# population limit is the replay law's conditional TD value rather than the
# coalition game. Self-distillation replaces the target and the action coverage
# and leaves the state law alone. See ADR 0044.
TD_ACTION_TARGET = "td-action"
SELF_DISTILL_TARGET = "self-distill"
MASKED_TARGETS = (TD_ACTION_TARGET, SELF_DISTILL_TARGET)


def _masked_self_distillation_loss(
    masked_outputs,
    full_outputs,
    masked_source_indices,
    action_masks,
    *,
    value_head: str,
):
    """Return one global mean over the masked batch's legal action targets.

    Categorical heads retain the teacher's full probability vector. Decoding to
    expected Q and projecting back to bins would replace the characteristic
    target with a different distribution. The teacher is detached before either
    the scalar MSE or categorical cross-entropy is formed.
    """

    teacher_outputs = jax.lax.stop_gradient(full_outputs)[masked_source_indices]
    legal = action_masks[masked_source_indices].astype(jnp.float32)
    if value_head == SCALAR:
        per_action = jnp.square(masked_outputs - teacher_outputs)
    else:
        teacher_probabilities = jax.nn.softmax(teacher_outputs, axis=-1)
        student_log_probabilities = jax.nn.log_softmax(masked_outputs, axis=-1)
        per_action = -jnp.sum(teacher_probabilities * student_log_probabilities, axis=-1)
    legal_target_count = jnp.sum(legal)
    return jnp.sum(per_action * legal) / jnp.maximum(legal_target_count, 1.0)


if nn is not None:

    class MLPQNetwork(nn.Module):
        """Small MLP Q-network.

        Args:
            num_actions: Number of discrete actions.
            hidden_sizes: Hidden layer widths. Defaults to (64, 64).
            num_bins: Number of categorical bins, or None for scalar regression output.
        """

        num_actions: int
        hidden_sizes: tuple[int, ...] = (64, 64)
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
    MLPQNetwork = None


@dataclass(frozen=True, slots=True)
class NeuralQConfig:
    """Configuration for DQN regression and scalar-target classification losses.

    Attributes:
        episodes: Number of training episodes. Defaults to 1_000.
        seed: Random seed. Defaults to 0.
        value_head: Value representation: 'scalar', 'two-hot', or 'hl-gauss'. Defaults to 'scalar'.
        opponent: Self-play opponent type. Defaults to 'random'.
        gamma: Discount factor. Defaults to 0.95.
        learning_rate: Adam learning rate. Defaults to 1e-3.
        epsilon_start: Initial exploration rate. Defaults to 1.0.
        epsilon_end: Final exploration rate. Defaults to 0.05.
        epsilon_decay_episodes: Episodes over which epsilon linearly decays. Defaults to 800.
        exploration: Action-selection scheme: 'epsilon_greedy' or 'boltzmann'.
            Boltzmann agents sample softmax(Q / temperature) over legal actions
            and ignore the epsilon schedule. Defaults to 'epsilon_greedy'.
        backup: TD target operator: 'hard' (max) or 'soft'
            (temperature * logsumexp(Q / temperature)). Defaults to 'hard'.
        temperature: Softmax temperature shared by boltzmann exploration and the
            soft backup. Defaults to 0.05.
        replay_capacity: Maximum replay buffer size. Defaults to 10_000.
        replay_warmup: Transitions to collect before training begins. Defaults to 32.
        batch_size: Minibatch size. Defaults to 32.
        target_sync_interval: Gradient steps between target network syncs. Defaults to 100.
        hidden_sizes: MLP hidden layer widths. Defaults to (64, 64).
        v_min: Minimum categorical support value.
        v_max: Maximum categorical support value.
        num_bins: Number of categorical support bins. Defaults to 51.
        hl_gauss_sigma_bins: HL-Gauss smoothing in bin units. Defaults to 0.75.
        encoder: State encoder name. Defaults to 'identity'.
        feature_mask_input: Append the feature mask to model inputs. Ordinary
            policy and target calls use an all-ones mask.
        masked_batch_ratio: Appended masked current-state examples per full
            replay example. Must be in [0, 1].
        masked_target: What the appended masked examples regress against.
            'td-action' supervises the single replayed action with the same
            bootstrapped TD target as the full rows; 'self-distill' supervises
            every legal action of the stored state against the network's own
            detached full-observation expected Q (scalar head) or probability
            vector (categorical head).
        mask_empty_example_interval: Inject one empty mask after this many
            masked examples, independent of optimizer update and batch size.
        eval_interval: Episodes between evaluations (0 = disabled).
        eval_episodes: Number of evaluation episodes.
        out_dir: Output directory.
        env_options: Additional keyword arguments passed to the environment factory.
        perturbed_minimax_discount: Per-episode decay of the perturbed_minimax random-move
            rate (always starts at 100% random); 1.0 disables annealing.
    """

    episodes: int = 1_000
    seed: int = 0
    value_head: str = SCALAR
    opponent: str = "random"
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
    eval_interval: int = 0
    eval_episodes: int = 20
    metrics_window: int = 100
    q_probe_states: int = 64
    out_dir: Path = Path("runs/dqn")
    env_options: dict[str, Any] = field(default_factory=lambda: {"obs_mode": "flat"})
    perturbed_minimax_discount: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_head", normalize_value_head(self.value_head))
        object.__setattr__(self, "hidden_sizes", parse_hidden_sizes(self.hidden_sizes))
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.replay_capacity < 1:
            raise ValueError("replay_capacity must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be positive")
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if self.v_max <= self.v_min:
            raise ValueError("v_max must be greater than v_min")
        if self.q_probe_states < 0:
            raise ValueError("q_probe_states must be non-negative")
        if not math.isfinite(self.masked_batch_ratio) or not 0 <= self.masked_batch_ratio <= 1:
            raise ValueError("masked_batch_ratio must be finite and in [0, 1]")
        if self.masked_batch_ratio > 0 and not self.feature_mask_input:
            raise ValueError("masked_batch_ratio requires feature_mask_input=true")
        if self.masked_target not in MASKED_TARGETS:
            raise ValueError(f"masked_target must be one of {MASKED_TARGETS}, got {self.masked_target!r}")
        if self.masked_batch_ratio > 0 and round(self.masked_batch_ratio * self.batch_size) < 1:
            raise ValueError("masked_batch_ratio is too small to add an example at this batch_size")
        if self.feature_mask_input and self.encoder != "identity":
            raise ValueError("feature_mask_input currently requires the 'identity' encoder")
        validate_mask_mixture(
            empty_mask_example_interval=self.mask_empty_example_interval,
            shapley_probability=self.mask_shapley_probability,
            uniform_probability=self.mask_uniform_probability,
        )
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
            perturbed_minimax_discount=self.perturbed_minimax_discount,
            encoder=self.encoder,
            metrics_window=self.metrics_window,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> NeuralQConfig:
        normalized = dict(data)
        legacy = {"explainable_q", "masked_loss_weight", "mask_empty_interval"} & normalized.keys()
        if legacy:
            raise ValueError(
                "legacy mask configuration is unsupported; migrate to feature_mask_input, "
                "masked_batch_ratio, and mask_empty_example_interval"
            )
        normalized["value_head"] = normalize_value_head(normalized.pop("loss", normalized.get("value_head", SCALAR)))
        normalized.pop("perturbed_minimax_probability", None)
        normalized["out_dir"] = Path(normalized["out_dir"])
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (64, 64)))
        return cls(**normalized)


class NeuralQAgent:
    """Shared DQN agent wrapper for regression and scalar-target classification losses.

    Args:
        config: DQN configuration.
        num_actions: Number of discrete environment actions.
        observation_size: Dimension of the flattened observation vector.
    """

    def __init__(self, config: NeuralQConfig, *, num_actions: int, observation_size: int):
        require_neural_dependencies()
        self.config = config
        self.num_actions = int(num_actions)
        self.observation_size = int(observation_size)
        self.encoder: StateActionEncoder = make_encoder(config.encoder, num_actions=self.num_actions)
        self._rng = random.Random(config.seed)
        self._key = jax.random.PRNGKey(config.seed)
        self.support = jnp.asarray(support_values(config.v_min, config.v_max, config.num_bins))
        self.model = MLPQNetwork(
            num_actions=self.num_actions,
            hidden_sizes=config.hidden_sizes,
            num_bins=None if config.value_head == SCALAR else config.num_bins,
        )
        input_size = self.observation_size * 2 if config.feature_mask_input else self.observation_size
        dummy = jnp.zeros((1, input_size), dtype=jnp.float32)
        self.params = self.model.init(self._key, dummy)
        self.target_params = self.params
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self._train_step = self._make_train_step()
        self._q_values_fn = self._make_q_values_fn()
        self._q_distributions_fn = self._make_q_distributions_fn()
        self._mask_sampler = (
            CoalitionMaskSampler(
                CoalitionMaskSamplerConfig(
                    feature_count=self.observation_size,
                    # Direct training already sees the full observation in its
                    # TD term every update, so the periodic full mask is off.
                    # The interior coalition measure is unchanged, which is what
                    # keeps this arm comparable to the surrogate. See ADR 0026.
                    full_mask_example_interval=None,
                    empty_mask_example_interval=config.mask_empty_example_interval,
                    shapley_probability=config.mask_shapley_probability,
                    uniform_probability=config.mask_uniform_probability,
                    guided_probability=0.0,
                ),
                seed=config.seed,
            )
            if config.masked_batch_ratio > 0
            else None
        )
        self._mask_row_rng = random.Random(config.seed + 1_000_003)
        self.update_count = 0
        self.transition_count = 0
        self.full_example_count = 0
        self.masked_example_count = 0
        self.masked_legal_target_count = 0
        self._input_size = input_size
        self._step_cost: StepCost | None = None

    @property
    def name(self) -> str:
        return f"dqn_{self.config.value_head}"

    def encode_view(self, observation: np.ndarray, action_mask: np.ndarray) -> EncodedView:
        """Encode an observation and mask for the internal encoder.

        Args:
            observation: Raw environment observation.
            action_mask: Binary legal-action mask.

        Returns:
            EncodedView with the encoded state key and action-mapping tables.
        """
        return self.encoder.encode(observation, action_mask)

    def _make_q_values_fn(self):
        """Build and JIT-compile the batched Q-value query.

        Built once per agent, in the same style as :meth:`_make_train_step`:
        the model, head, and support are captured in a closure so the returned
        function is a pure function of ``(params, states)``. Compiling it once
        is the whole point -- an uncompiled ``model.apply`` re-traces the
        network on every call, which costs milliseconds per query and dominates
        exact evaluation, where the query is issued thousands of times per
        checkpoint. Rebuilding this per call would reproduce that cost.
        """

        model = self.model
        value_head = self.config.value_head
        support = self.support

        def q_values_fn(params, states):
            outputs = model.apply(params, states)
            if value_head == SCALAR:
                return outputs
            probs = jax.nn.softmax(outputs, axis=-1)
            return jnp.sum(probs * support[None, None, :], axis=-1)

        return jax.jit(q_values_fn)

    def _make_q_distributions_fn(self):
        if self.config.value_head == SCALAR:
            return None
        model = self.model

        def q_distributions_fn(params, states):
            return jax.nn.softmax(model.apply(params, states), axis=-1)

        return jax.jit(q_distributions_fn)

    def _q_values_from_params(self, params, states):
        """Decode scalar Q-values from network outputs.

        For the regression loss the outputs are already scalar per action; for
        categorical losses they are logits over ``support`` bins decoded to their
        expected value. ``states`` is a ``(batch, observation_size)`` array.

        Args:
            params: Flax model parameters.
            states: Batched observations, shape (batch, observation_size).

        Returns:
            Q-values, shape (batch, num_actions).
        """
        return self._q_values_fn(params, states)

    def _model_inputs(
        self,
        states: np.ndarray,
        known_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return network inputs, supplying a full known-mask when omitted."""

        states_array = np.asarray(states, dtype=np.float32)
        if states_array.ndim == 1:
            states_array = states_array[None, :]
        if states_array.ndim != 2 or states_array.shape[1] != self.observation_size:
            raise ValueError(f"expected states with shape (batch, {self.observation_size}), got {states_array.shape}")
        if not self.config.feature_mask_input:
            if known_masks is not None:
                raise ValueError("known_mask queries require feature_mask_input=true")
            return states_array
        masks = np.ones_like(states_array, dtype=np.float32) if known_masks is None else known_masks
        return masked_inputs(states_array, np.asarray(masks, dtype=np.float32))

    def _make_train_step(self):
        """Build and JIT-compile the per-batch gradient step.

        Captures the model, optimizer, discount, and loss configuration in a
        closure so the returned ``train_step`` is a pure function of its array
        arguments (params, target params, optimizer state, and the transition
        batch), which JAX can compile and reuse across updates.
        """
        model = self.model
        optimizer = self.optimizer
        gamma = self.config.gamma
        value_head = self.config.value_head
        support = self.support
        sigma_bins = self.config.hl_gauss_sigma_bins
        backup = self.config.backup
        temperature = self.config.temperature
        has_masked_training = self.config.masked_batch_ratio > 0
        self_distill = self.config.masked_target == SELF_DISTILL_TARGET

        def q_values_from_outputs(outputs):
            if value_head == SCALAR:
                return outputs
            probs = jax.nn.softmax(outputs, axis=-1)
            return jnp.sum(probs * support[None, None, :], axis=-1)

        def train_step(
            params,
            target_params,
            opt_state,
            states,
            masked_states,
            masked_source_indices,
            actions,
            action_masks,
            rewards,
            next_states,
            next_masks,
            dones,
        ):
            target_outputs = model.apply(target_params, next_states)
            target_q = q_values_from_outputs(target_outputs)
            masked_target_q = jnp.where(next_masks, target_q, -jnp.inf)
            if backup == "soft":
                future = temperature * jax.nn.logsumexp(masked_target_q / temperature, axis=1)
            else:
                future = jnp.max(masked_target_q, axis=1)
            future = jnp.where(jnp.isfinite(future), future, 0.0)
            scalar_targets = rewards + gamma * jnp.where(dones, 0.0, future)

            def loss_fn(current_params):
                full_outputs = model.apply(current_params, states)

                def losses_from_outputs(outputs, chosen_actions, chosen_targets):
                    chosen = outputs[jnp.arange(chosen_actions.shape[0]), chosen_actions]
                    if value_head == SCALAR:
                        return jnp.square(chosen - chosen_targets)

                    target_probs = _project_values_jax(
                        chosen_targets,
                        support,
                        value_head,
                        sigma_bins,
                    )
                    log_probs = jax.nn.log_softmax(chosen, axis=-1)
                    return -jnp.sum(target_probs * log_probs, axis=-1)

                full_losses = losses_from_outputs(full_outputs, actions, scalar_targets)
                if not has_masked_training:
                    return jnp.mean(full_losses)
                masked_outputs = model.apply(current_params, masked_states)
                if self_distill:
                    partial_loss = _masked_self_distillation_loss(
                        masked_outputs,
                        full_outputs,
                        masked_source_indices,
                        action_masks,
                        value_head=value_head,
                    )
                    masked_weight = masked_outputs.shape[0]
                    return (jnp.sum(full_losses) + masked_weight * partial_loss) / (
                        full_losses.shape[0] + masked_weight
                    )
                else:
                    partial_losses = losses_from_outputs(
                        masked_outputs,
                        actions[masked_source_indices],
                        scalar_targets[masked_source_indices],
                    )
                return (jnp.sum(full_losses) + jnp.sum(partial_losses)) / (
                    full_losses.shape[0] + partial_losses.shape[0]
                )

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss

        return jax.jit(train_step)

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return scalar Q-values for every action in environment coordinates.

        Args:
            observation: Raw environment observation.

        Returns:
            Float32 array of length num_actions.
        """
        view = self.encode_view(observation, np.ones(self.num_actions, dtype=np.int8))
        state = jnp.asarray(
            self._model_inputs(np.asarray(view.state_key, dtype=np.float32)),
            dtype=jnp.float32,
        )
        table_values = np.asarray(self._q_values_from_params(self.params, state)[0])
        values: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        for env_action in range(self.num_actions):
            values[env_action] = table_values[view.to_table_action(env_action)]
        return values

    def _require_mask_query(self, *, allow_untrained: bool) -> None:
        if not self.config.feature_mask_input:
            raise ValueError("masked Q queries require feature_mask_input=true")
        if self.config.masked_batch_ratio <= 0 and not allow_untrained:
            raise ValueError(
                "partial masks were not trained at masked_batch_ratio=0; "
                "pass allow_untrained=True only for an OOD negative control"
            )

    def masked_q_values(
        self,
        observation: np.ndarray,
        known_mask: np.ndarray,
        *,
        allow_untrained: bool = False,
    ) -> np.ndarray:
        """Return Q-values for an explicit feature coalition.

        This surface is intentionally separate from :meth:`q_values`, whose
        one-argument contract remains the full-observation ``PolicyView`` API.
        """

        self._require_mask_query(allow_untrained=allow_untrained)
        view = self.encode_view(observation, np.ones(self.num_actions, dtype=np.int8))
        state = jnp.asarray(
            self._model_inputs(np.asarray(view.state_key, dtype=np.float32), known_mask),
            dtype=jnp.float32,
        )
        table_values = np.asarray(self._q_values_from_params(self.params, state)[0])
        values: np.ndarray = np.zeros(self.num_actions, dtype=np.float32)
        for env_action in range(self.num_actions):
            values[env_action] = table_values[view.to_table_action(env_action)]
        return values

    def q_values_observations(self, observations: np.ndarray) -> np.ndarray:
        """Return env-coordinate Q-values for many raw observations at once.

        The batched sibling of :meth:`q_values`. Encoding and the table->env
        action mapping stay per-observation, because a non-identity encoder may
        canonicalise each state differently; only the network query is batched.
        Callers that already hold encoder state keys want
        :meth:`q_values_batch` instead.
        """

        raw = np.asarray(observations, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"expected observations with shape (batch, features), got {raw.shape}")
        full_mask = np.ones(self.num_actions, dtype=np.int8)
        views = [self.encode_view(row, full_mask) for row in raw]
        keys = np.stack([np.asarray(view.state_key, dtype=np.float32) for view in views])
        table_values = self.q_values_batch(keys)
        values = np.zeros((len(views), self.num_actions), dtype=np.float32)
        for index, view in enumerate(views):
            for env_action in range(self.num_actions):
                values[index, env_action] = table_values[index, view.to_table_action(env_action)]
        return values

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Return scalar Q-values for a batch of encoded states.

        ``states`` is a ``(batch, observation_size)`` array of encoder state
        keys (see ``encode_view``); the result has shape
        ``(batch, num_actions)`` in the encoder's table coordinates, which
        equal environment actions for the identity encoder.
        """
        batch = jnp.asarray(self._model_inputs(np.asarray(states, dtype=np.float32)), dtype=jnp.float32)
        return np.asarray(self._q_values_from_params(self.params, batch))

    def masked_q_values_batch(
        self,
        states: np.ndarray,
        known_masks: np.ndarray,
        *,
        allow_untrained: bool = False,
    ) -> np.ndarray:
        """Return Q-values for a batch of encoded states under explicit coalitions.

        The batched sibling of :meth:`masked_q_values`, sharing
        :meth:`q_values_batch`'s contract: ``states`` and ``known_masks`` are
        ``(batch, observation_size)`` arrays of encoder state keys, and the
        result is ``(batch, num_actions)`` in the encoder's table coordinates,
        which equal environment actions for the identity encoder.

        Amortized explainers evaluate coalitions in the hundreds of thousands,
        where the per-row path dominates; this issues one forward pass instead.
        """

        self._require_mask_query(allow_untrained=allow_untrained)
        batch = jnp.asarray(
            self._model_inputs(
                np.asarray(states, dtype=np.float32),
                np.asarray(known_masks, dtype=np.float32),
            ),
            dtype=jnp.float32,
        )
        return np.asarray(self._q_values_from_params(self.params, batch))

    def q_distributions(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return categorical Q support and probabilities in environment coordinates.

        Regression agents have scalar outputs and return ``None``. For Two-Hot and
        HL-Gauss agents, the probability matrix has shape
        ``(num_actions, num_bins)`` and each row sums to one.
        """

        if self.config.value_head == SCALAR:
            return None

        view = self.encode_view(observation, np.ones(self.num_actions, dtype=np.int8))
        state = jnp.asarray(
            self._model_inputs(np.asarray(view.state_key, dtype=np.float32)),
            dtype=jnp.float32,
        )
        table_logits = self.model.apply(self.params, state)[0]
        table_probabilities = np.asarray(jax.nn.softmax(table_logits, axis=-1), dtype=np.float32)
        probabilities = np.zeros_like(table_probabilities)
        for env_action in range(self.num_actions):
            probabilities[env_action] = table_probabilities[view.to_table_action(env_action)]
        return np.asarray(self.support, dtype=np.float32), probabilities

    def q_distributions_observations(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Return categorical support and probabilities for raw observations in one batch."""

        if self.config.value_head == SCALAR:
            return None
        raw = np.asarray(observations, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"expected observations with shape (batch, features), got {raw.shape}")
        full_mask = np.ones(self.num_actions, dtype=np.int8)
        views = [self.encode_view(row, full_mask) for row in raw]
        keys = np.stack([np.asarray(view.state_key, dtype=np.float32) for view in views])
        inputs = jnp.asarray(self._model_inputs(keys), dtype=jnp.float32)
        table_probabilities = np.asarray(self._q_distributions_fn(self.params, inputs), dtype=np.float32)
        probabilities = np.zeros_like(table_probabilities)
        for index, view in enumerate(views):
            for env_action in range(self.num_actions):
                probabilities[index, env_action] = table_probabilities[index, view.to_table_action(env_action)]
        return np.asarray(self.support, dtype=np.float32), probabilities

    def masked_q_distributions(
        self,
        observation: np.ndarray,
        known_mask: np.ndarray,
        *,
        allow_untrained: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return categorical Q distributions for an explicit coalition."""

        self._require_mask_query(allow_untrained=allow_untrained)
        if self.config.value_head == SCALAR:
            raise TypeError("masked_q_distributions requires a categorical value head")
        view = self.encode_view(observation, np.ones(self.num_actions, dtype=np.int8))
        state = jnp.asarray(
            self._model_inputs(np.asarray(view.state_key, dtype=np.float32), known_mask),
            dtype=jnp.float32,
        )
        table_logits = self.model.apply(self.params, state)[0]
        table_probabilities = np.asarray(jax.nn.softmax(table_logits, axis=-1), dtype=np.float32)
        probabilities = np.zeros_like(table_probabilities)
        for env_action in range(self.num_actions):
            probabilities[env_action] = table_probabilities[view.to_table_action(env_action)]
        return np.asarray(self.support, dtype=np.float32), probabilities

    def action_distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return the agent's action probabilities in environment coordinates.

        Epsilon-greedy agents return the epsilon-greedy distribution. Boltzmann
        agents return ``softmax(Q / temperature)`` over legal actions — the
        policy the Q-values parametrize — and ignore ``epsilon``.

        Args:
            observation: Raw environment observation.
            action_mask: Binary legal-action mask.
            epsilon: Exploration rate (0 = pure greedy, 1 = uniform random).
                Ignored for boltzmann agents.

        Returns:
            Probability distribution over actions.
        """

        if self.config.exploration == "boltzmann":
            return boltzmann_distribution(
                self.q_values(observation),
                action_mask,
                temperature=self.config.temperature,
            )
        return epsilon_greedy_distribution(self.q_values(observation), action_mask, epsilon=epsilon)

    def state_value(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> float:
        """Return the policy-weighted value estimate for an observation.

        Args:
            observation: Raw environment observation.
            action_mask: Binary legal-action mask.
            epsilon: Exploration rate.

        Returns:
            Expected Q-value under the agent's policy.
        """

        q_values = self.q_values(observation)
        probabilities = self.action_distribution(observation, action_mask, epsilon=epsilon)
        return float(np.dot(probabilities, np.asarray(q_values, dtype=np.float64)))

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, *, epsilon: float) -> int:
        """Pick a legal action under the agent's policy, in environment coordinates.

        Epsilon-greedy agents explore with rate ``epsilon``; boltzmann agents
        sample ``softmax(Q / temperature)`` and ignore ``epsilon``.

        Args:
            observation: Raw environment observation.
            action_mask: Binary legal-action mask.
            epsilon: Exploration rate. Ignored for boltzmann agents.

        Returns:
            The chosen action index.

        Raises:
            ValueError: If no legal actions are available.
        """
        view = self.encode_view(observation, action_mask)
        legal = legal_actions_from_mask(view.action_mask)
        if not legal:
            raise ValueError("no legal actions available")

        if self.config.exploration == "boltzmann":
            state = jnp.asarray(self._model_inputs(np.asarray(view.state_key, dtype=np.float32)), dtype=jnp.float32)
            values = np.asarray(self._q_values_from_params(self.params, state)[0])
            probabilities = boltzmann_distribution(
                values,
                view.action_mask,
                temperature=self.config.temperature,
            )
            table_action = self._rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            return view.to_env_action(table_action)

        if self._rng.random() < epsilon:
            return view.to_env_action(self._rng.choice(legal))

        state = jnp.asarray(self._model_inputs(np.asarray(view.state_key, dtype=np.float32)), dtype=jnp.float32)
        values = np.asarray(self._q_values_from_params(self.params, state)[0])
        best_value = max(values[action] for action in legal)
        best_actions = [action for action in legal if values[action] == best_value]
        return view.to_env_action(self._rng.choice(best_actions))

    def update(self, batch: dict[str, np.ndarray]) -> float:
        """Apply one gradient step to a replay batch and return the scalar loss.

        Syncs the target network every ``target_sync_interval`` updates.

        Args:
            batch: Dict with keys states, actions, action_masks, rewards,
                next_states, next_action_masks, dones.

        Returns:
            The scalar training loss for this batch.
        """
        raw_states = np.asarray(batch["states"], dtype=np.float32)
        states = jnp.asarray(self._model_inputs(raw_states), dtype=jnp.float32)
        masked_count = round(self.config.masked_batch_ratio * raw_states.shape[0])
        if self._mask_sampler is None:
            masked_source_indices = np.empty(0, dtype=np.int32)
            masked_states = np.empty((0, self._input_size), dtype=np.float32)
        else:
            row_order = list(range(raw_states.shape[0]))
            self._mask_row_rng.shuffle(row_order)
            masked_source_indices = np.asarray(row_order[:masked_count], dtype=np.int32)
            known_masks = self._mask_sampler.sample_batch(
                masked_count,
                example_offset=self.masked_example_count,
            )
            masked_states = self._model_inputs(raw_states[masked_source_indices], known_masks)
        actions = jnp.asarray(batch["actions"], dtype=jnp.int32)
        action_masks = jnp.asarray(batch["action_masks"], dtype=bool)
        rewards = jnp.asarray(batch["rewards"], dtype=jnp.float32)

        next_states = jnp.asarray(
            self._model_inputs(np.asarray(batch["next_states"], dtype=np.float32)), dtype=jnp.float32
        )
        next_masks = jnp.asarray(batch["next_action_masks"], dtype=bool)
        dones = jnp.asarray(batch["dones"], dtype=bool)

        step_args = (
            self.params,
            self.target_params,
            self.opt_state,
            states,
            jnp.asarray(masked_states, dtype=jnp.float32),
            jnp.asarray(masked_source_indices, dtype=jnp.int32),
            actions,
            action_masks,
            rewards,
            next_states,
            next_masks,
            dones,
        )
        if self._step_cost is None:
            # Costed from the first real batch so the measured shapes are the
            # ones training actually uses; the step is already compiled by then.
            self._step_cost = self._measure_step_cost(step_args)
        self.params, self.opt_state, loss = self._train_step(*step_args)
        self.update_count += 1
        self.full_example_count += raw_states.shape[0]
        self.masked_example_count += masked_count
        if self.config.masked_target == SELF_DISTILL_TARGET:
            legal_masks = np.asarray(batch["action_masks"], dtype=np.int8)[masked_source_indices]
            self.masked_legal_target_count += int(np.sum(legal_masks))
        else:
            self.masked_legal_target_count += masked_count
        if self.update_count % self.config.target_sync_interval == 0:
            self.sync_target()
        return float(loss)

    def _measure_step_cost(self, step_args: tuple[Any, ...]) -> StepCost:
        output_size = self.num_actions if self.config.value_head == SCALAR else self.num_actions * self.config.num_bins
        full_batch_size = int(step_args[3].shape[0])
        masked_batch_size = int(step_args[4].shape[0])
        return measure_step_cost(
            self._train_step,
            step_args,
            fallback=lambda: dqn_update_flops(
                input_width=self._input_size,
                hidden_sizes=self.config.hidden_sizes,
                output_size=output_size,
                full_batch_size=full_batch_size,
                masked_batch_size=masked_batch_size,
            ),
        )

    @property
    def step_cost(self) -> StepCost | None:
        """Cost of one replay update, or ``None`` before the first update."""

        return self._step_cost

    @property
    def training_flops(self) -> float:
        """Cumulative FLOPs spent on gradient updates so far."""

        return 0.0 if self._step_cost is None else self._step_cost.total(self.update_count)

    @property
    def predictor_example_count(self) -> int:
        """Full plus appended masked current-state examples seen so far."""

        return self.full_example_count + self.masked_example_count

    @property
    def flops_source(self) -> str | None:
        """Which method produced :attr:`training_flops`, or ``None`` if unmeasured."""

        return None if self._step_cost is None else self._step_cost.source

    def sync_target(self) -> None:
        """Copy online parameters to the target network."""
        self.target_params = self.params

    def serialized_params(self) -> bytes:
        """Return the current model parameters as Flax msgpack bytes."""

        require_neural_dependencies()
        return serialization.to_bytes(self.params)

    def save(self, path: Path) -> None:
        """Serialize model parameters to disk.

        Args:
            path: File path to write the msgpack-serialized parameters.
        """
        path.write_bytes(self.serialized_params())

    def load_params(self, path: Path) -> None:
        """Load model parameters from disk and sync the target network.

        Args:
            path: File path to read the msgpack-serialized parameters from.

        Raises:
            ValueError: If the checkpoint's shapes contradict this agent's
                configuration.
        """
        require_neural_dependencies()
        restored = serialization.from_bytes(self.params, path.read_bytes())
        require_matching_param_shapes(self.params, restored, path=path, hint=self._input_width_hint())
        self.params = restored
        self.sync_target()

    def _input_width_hint(self) -> str:
        """Name the flag that decides this model's input width."""

        factor = "observation_size * 2" if self.config.feature_mask_input else "observation_size"
        return f"feature_mask_input={self.config.feature_mask_input} sets the input width to {factor}"


def require_matching_param_shapes(expected: Any, restored: Any, *, path: Path, hint: str) -> None:
    """Reject a checkpoint whose parameter shapes contradict the built model.

    Msgpack records values, not shapes, and ``serialization.from_bytes``
    restores by position in the parameter tree. A checkpoint from a differently
    shaped model therefore loads without complaint and fails much later, inside
    a matrix product, far from the configuration that caused it. See ADR 0027.
    """

    expected_leaves = jax.tree_util.tree_leaves(expected)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if len(expected_leaves) != len(restored_leaves):
        raise ValueError(
            f"checkpoint {path} has {len(restored_leaves)} parameter arrays, expected {len(expected_leaves)}; {hint}"
        )
    mismatches = [
        (index, tuple(np.shape(want)), tuple(np.shape(got)))
        for index, (want, got) in enumerate(zip(expected_leaves, restored_leaves, strict=True))
        if tuple(np.shape(want)) != tuple(np.shape(got))
    ]
    if not mismatches:
        return
    index, want, got = mismatches[0]
    raise ValueError(
        f"checkpoint {path} has parameter {index} of shape {got}, expected {want} "
        f"({len(mismatches)} of {len(expected_leaves)} arrays mismatched); {hint}"
    )


class RegressionDQNAgent(NeuralQAgent):
    """Scalar TD-regression DQN baseline."""

    def __init__(self, config: NeuralQConfig, *, num_actions: int, observation_size: int):
        super().__init__(
            NeuralQConfig.from_json_dict({**config.to_json_dict(), "value_head": SCALAR}),
            num_actions=num_actions,
            observation_size=observation_size,
        )


class ClassificationDQNAgent(NeuralQAgent):
    """Cross-entropy scalar-target value classification DQN."""

    def __init__(self, config: NeuralQConfig, *, num_actions: int, observation_size: int):
        if config.value_head not in {"two-hot", "hl-gauss"}:
            raise ValueError(f"classification DQN requires a categorical value head: {CATEGORICAL_VALUE_HEADS}")
        super().__init__(config, num_actions=num_actions, observation_size=observation_size)


class _ReplayUpdateStrategy:
    def __init__(self, agent: NeuralQAgent, replay: ReplayBuffer | None, config: NeuralQConfig):
        self.agent = agent
        self.replay = replay
        self.config = config
        self.losses: list[float] = []

    def begin_episode(self) -> None:
        self.losses = []

    def observe_transition(
        self,
        pending: LearnerStep,
        reward: float,
        next_step: LearnerStep | None,
        done: bool,
        *,
        train: bool,
    ) -> float | None:
        if not train or self.replay is None:
            return None
        self.agent.transition_count += 1
        self.replay.add(
            ReplayTransition(
                state=pending.state,
                action=pending.action,
                reward=reward,
                # A learner step's ``next_action_mask`` is the mask of its own
                # state; it earns that name only from the successor's point of
                # view. So the pending step already carries the current state's
                # legal actions.
                action_mask=np.asarray(pending.next_action_mask, dtype=np.int8),
                next_state=pending.state if next_step is None else next_step.state,
                next_action_mask=(
                    np.zeros(self.agent.num_actions, dtype=np.int8)
                    if next_step is None
                    else np.asarray(next_step.next_action_mask, dtype=np.int8)
                ),
                done=done,
            )
        )
        before = len(self.losses)
        self._maybe_update()
        return self.losses[-1] if len(self.losses) > before else None

    def stats(self) -> LearnerStats:
        replay_size = len(self.replay) if self.replay is not None else 0
        return LearnerStats(
            gradient_updates=self.agent.update_count,
            replay_size=replay_size,
            unique_states_seen=replay_size,
        )

    def _maybe_update(self) -> None:
        if self.replay is None:
            return
        if len(self.replay) < self.config.replay_warmup or len(self.replay) < self.config.batch_size:
            return
        self.losses.append(self.agent.update(self.replay.sample(self.config.batch_size)))


class _NeuralQLearnerAdapter:
    def __init__(
        self,
        agent: NeuralQAgent,
        *,
        make_env,
        replay: ReplayBuffer | None,
        config: NeuralQConfig,
    ) -> None:
        self.learner = agent
        self.make_env = make_env
        self.update_strategy = _ReplayUpdateStrategy(agent, replay, config)

    def run_episode(
        self,
        *,
        opponent: object,
        episode: int,
        seed: int,
        epsilon: float,
        train: bool,
    ) -> EpisodeMetrics:
        self.update_strategy.begin_episode()
        return run_aec_episode(
            env=self.make_env(),
            opponent=opponent,
            episode=episode,
            seed=seed,
            epsilon=epsilon,
            train=train,
            select_learner_action=neural_select_learner_action(self.learner),
            observe_learner_transition=lambda pending, reward, next_step, done: self.update_strategy.observe_transition(
                pending,
                reward,
                next_step,
                done,
                train=train,
            ),
            learner_stats=self.update_strategy.stats,
        )


class NeuralQTrainer(BaseAECTrainer[NeuralQAgent]):
    """Train DQN-style agents through the composable AEC runner."""

    config_class = NeuralQConfig

    @property
    def run_name(self) -> str:
        return f"dqn_{self.config.value_head}"

    @property
    def tags(self) -> dict[str, str]:
        return {"trainer": "dqn", "value_head": self.config.value_head}

    def make_agent(self) -> NeuralQAgent:
        num_actions, observation_size = self._agent_init_shape()
        if self.config.value_head == SCALAR:
            return RegressionDQNAgent(self.config, num_actions=num_actions, observation_size=observation_size)
        return ClassificationDQNAgent(self.config, num_actions=num_actions, observation_size=observation_size)

    def _make_adapter(self, agent: NeuralQAgent) -> _NeuralQLearnerAdapter:
        # The replay buffer is only consulted while training; evaluation runs
        # with train=False and never touches it.
        replay = ReplayBuffer(self.config.replay_capacity, seed=self.config.seed)
        return _NeuralQLearnerAdapter(agent, make_env=self.make_env, replay=replay, config=self.config)

    def make_pipeline(self):
        from xrl.training.pipeline import make_dqn_pipeline

        return make_dqn_pipeline(
            self.config,
            env_factory=self.env_factory,
            tracker=self.tracker,
            eval_opponents=self.eval_opponents,
            progress=self._make_progress(),
        )


def train_dqn(
    episodes: int = 1_000,
    value_head: str = SCALAR,
    opponent: str = "random",
    out: Path = Path("runs/dqn"),
    seed: int = 0,
    learning_rate: float = 1e-3,
    hidden_sizes: str | tuple[int, ...] = (64, 64),
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
    eval_interval: int = 0,
    eval_episodes: int = 20,
    metrics_window: int = 100,
    q_probe_states: int = 64,
    progress_interval: int = 0,
    perturbed_minimax_discount: float = 1.0,
    seeds: str | None = None,
    no_plots: bool = False,
    mlflow: bool = False,
    mlflow_experiment: str = "xrl",
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, Any]:
    """Train DQN with regression, Two-Hot, or HL-Gauss scalar-target value loss.

    Pass ``--seeds 0,1,2`` to run a multi-seed sweep (one ``seed_<k>/`` run per
    seed plus aggregate artifacts; ``--seed`` is ignored in that case).
    """
    tracker_config = TrackerConfig(
        mlflow=mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    config = NeuralQConfig(
        episodes=episodes,
        seed=seed,
        value_head=value_head,
        opponent=opponent,
        gamma=gamma,
        learning_rate=learning_rate,
        hidden_sizes=parse_hidden_sizes(hidden_sizes),
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
        perturbed_minimax_discount=perturbed_minimax_discount,
    )
    if seeds is not None:
        from xrl.training.metrics_io import (
            AEC_DQN_SWEEP_EVAL_COLUMNS,
            AEC_SWEEP_METRIC_COLUMNS,
            AEC_SWEEP_POLICY_COLUMNS,
        )
        from xrl.training.seed_sweep import SeedSweepSpec, parse_seeds, run_seed_sweep, trainer_run_seed

        seed_tuple = parse_seeds(seeds)
        spec = SeedSweepSpec(
            seeds=seed_tuple,
            root=config.out_dir,
            run_name=f"dqn_{config.value_head}_seeds{len(seed_tuple)}",
            tags={"workflow": "seed_sweep", "trainer": "dqn", "value_head": config.value_head},
            base_params=config.to_json_dict(),
            metric_columns=AEC_SWEEP_METRIC_COLUMNS,
            eval_columns=AEC_DQN_SWEEP_EVAL_COLUMNS,
            eval_group_by="opponent",
            policy_columns=AEC_SWEEP_POLICY_COLUMNS,
        )
        run_seed = trainer_run_seed(
            config,
            lambda cfg, tracker: NeuralQTrainer(cfg, tracker=tracker, progress_interval=progress_interval),
            identity=RunIdentity(
                environment="tictactoe",
                algorithm="dqn",
                value_head=config.value_head,
                eval_opponents=("random", "minimax"),
            ),
        )
        return run_seed_sweep(spec, run_seed, tracker_config=tracker_config, plot=not no_plots)
    trainer = NeuralQTrainer(
        config,
        tracker=tracker_from_cli_args(tracker_config),
        progress_interval=progress_interval,
    )
    summary = execute_training_run(
        config,
        trainer,
        RunIdentity(
            environment="tictactoe",
            algorithm="dqn",
            value_head=config.value_head,
            eval_opponents=("random", "minimax"),
        ),
    )
    return {"out_dir": str(config.out_dir), "summary": summary}
