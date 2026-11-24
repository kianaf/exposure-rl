"""Online neural Q-regression baseline."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xrl.training._neural.adapters import neural_select_learner_action
from xrl.training._neural.trainer import BaseAECTrainer
from xrl.training.config import parse_hidden_sizes, validate_common_config
from xrl.training.encoders import EncodedView, make_encoder
from xrl.training.episode import EpisodeMetrics, LearnerStats, LearnerStep, run_aec_episode
from xrl.training.neural_q import MLPQNetwork, jax, jnp, optax, require_neural_dependencies, serialization
from xrl.training.q_learning import epsilon_greedy_distribution, legal_actions_from_mask
from xrl.training.run_metadata import RunIdentity, execute_training_run
from xrl.training.tracking import DEFAULT_MLFLOW_TRACKING_URI, TrackerConfig, tracker_from_cli_args


@dataclass(frozen=True, slots=True)
class QRegressionConfig:
    """Configuration for online neural Q-regression."""

    episodes: int = 1_000
    seed: int = 0
    opponent: str = "random"
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    hidden_sizes: tuple[int, ...] = (64, 64)
    encoder: str = "identity"
    eval_interval: int = 0
    eval_episodes: int = 20
    metrics_window: int = 100
    q_probe_states: int = 64
    out_dir: Path = Path("runs/q_regression")
    env_options: dict[str, Any] = field(default_factory=lambda: {"obs_mode": "flat"})
    perturbed_minimax_discount: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", parse_hidden_sizes(self.hidden_sizes))
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.q_probe_states < 0:
            raise ValueError("q_probe_states must be non-negative")
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
    def from_json_dict(cls, data: dict[str, Any]) -> QRegressionConfig:
        normalized = dict(data)
        normalized.pop("perturbed_minimax_probability", None)
        normalized["out_dir"] = Path(normalized["out_dir"])
        normalized["hidden_sizes"] = tuple(normalized.get("hidden_sizes", (64, 64)))
        return cls(**normalized)


class QRegressionAgent:
    """Online MLP Q-regression agent without replay or a target network."""

    name = "q_regression"

    def __init__(self, config: QRegressionConfig, *, num_actions: int, observation_size: int):
        require_neural_dependencies()
        self.config = config
        self.num_actions = int(num_actions)
        self.observation_size = int(observation_size)
        self.encoder = make_encoder(config.encoder, num_actions=self.num_actions)
        self._rng = random.Random(config.seed)
        self._key = jax.random.PRNGKey(config.seed)
        self.model = MLPQNetwork(num_actions=self.num_actions, hidden_sizes=config.hidden_sizes, num_bins=None)
        dummy = jnp.zeros((1, self.observation_size), dtype=jnp.float32)
        self.params = self.model.init(self._key, dummy)
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        self._train_step = self._make_train_step()
        self.update_count = 0

    def _make_train_step(self):
        """Build and JIT-compile the single-transition gradient step.

        Unlike DQN, this is online regression: each call updates on one
        transition with no replay buffer and no target network. The TD target
        uses the live network's masked max next-Q (bootstrapping), stopped from
        gradient flow.
        """
        model = self.model
        optimizer = self.optimizer
        gamma = self.config.gamma

        def train_step(params, opt_state, state, action, reward, next_state, next_mask, done):
            next_q = model.apply(params, next_state[None, :])[0]
            masked_next_q = jnp.where(next_mask, next_q, -jnp.inf)
            future = jnp.max(masked_next_q)
            future = jnp.where(jnp.isfinite(future), future, 0.0)
            target = jax.lax.stop_gradient(reward + gamma * jnp.where(done, 0.0, future))

            def loss_fn(current_params):
                q_values = model.apply(current_params, state[None, :])[0]
                prediction = q_values[action]
                return jnp.square(prediction - target)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, next_opt_state = optimizer.update(grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss

        return jax.jit(train_step)

    def encode_view(self, observation: np.ndarray, action_mask: np.ndarray) -> EncodedView:
        """Encode a raw observation/mask into the encoder's canonical view."""
        return self.encoder.encode(observation, action_mask)

    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return scalar Q-values for every action in environment coordinates."""
        view = self.encode_view(np.asarray(observation, dtype=np.float32), np.ones(self.num_actions, dtype=np.int8))
        state = jnp.asarray(np.asarray(view.state_key, dtype=np.float32)[None, :])
        table_values = np.asarray(self.model.apply(self.params, state)[0])
        values = np.zeros(self.num_actions, dtype=np.float32)
        for env_action in range(self.num_actions):
            values[env_action] = table_values[view.to_table_action(env_action)]
        return values

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Return scalar Q-values for a batch of encoded states.

        ``states`` is a ``(batch, observation_size)`` array of encoder state
        keys (see ``encode_view``); the result has shape
        ``(batch, num_actions)`` in the encoder's table coordinates.
        """
        batch = jnp.asarray(np.asarray(states, dtype=np.float32))
        return np.asarray(self.model.apply(self.params, batch))

    def action_distribution(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """Return epsilon-greedy action probabilities in environment coordinates."""

        return epsilon_greedy_distribution(self.q_values(observation), action_mask, epsilon=epsilon)

    def state_value(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float = 0.0,
    ) -> float:
        """Return the policy-weighted value estimate for an observation."""

        q_values = self.q_values(observation)
        probabilities = epsilon_greedy_distribution(q_values, action_mask, epsilon=epsilon)
        return float(np.dot(probabilities, np.asarray(q_values, dtype=np.float64)))

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray, *, epsilon: float) -> int:
        """Pick an epsilon-greedy legal action, returned in environment coordinates."""
        view = self.encode_view(np.asarray(observation, dtype=np.float32), action_mask)
        legal = legal_actions_from_mask(view.action_mask)
        if not legal:
            raise ValueError("no legal actions available")
        if self._rng.random() < epsilon:
            return view.to_env_action(self._rng.choice(legal))

        state = jnp.asarray(np.asarray(view.state_key, dtype=np.float32)[None, :])
        values = np.asarray(self.model.apply(self.params, state)[0])
        best_value = max(values[action] for action in legal)
        best_actions = [action for action in legal if values[action] == best_value]
        return view.to_env_action(self._rng.choice(best_actions))

    def update_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        next_action_mask: np.ndarray | None,
        *,
        done: bool,
    ) -> float:
        """Apply one online gradient update for a transition and return the loss."""
        if next_state is None:
            next_state = state
        if next_action_mask is None:
            next_action_mask = np.zeros(self.num_actions, dtype=np.int8)
        self.params, self.opt_state, loss = self._train_step(
            self.params,
            self.opt_state,
            jnp.asarray(state, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.int32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_state, dtype=jnp.float32),
            jnp.asarray(next_action_mask, dtype=bool),
            jnp.asarray(done, dtype=bool),
        )
        self.update_count += 1
        return float(loss)

    def save(self, path: Path) -> None:
        require_neural_dependencies()
        path.write_bytes(serialization.to_bytes(self.params))

    def load_params(self, path: Path) -> None:
        require_neural_dependencies()
        self.params = serialization.from_bytes(self.params, path.read_bytes())


class _QRegressionUpdateStrategy:
    def __init__(self, agent: QRegressionAgent):
        self.agent = agent

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
        return self.agent.update_transition(
            pending.state,
            pending.action,
            reward,
            None if next_step is None else next_step.state,
            None if next_step is None else next_step.next_action_mask,
            done=done,
        )

    def stats(self) -> LearnerStats:
        return LearnerStats(gradient_updates=self.agent.update_count)


class _QRegressionLearnerAdapter:
    def __init__(self, agent: QRegressionAgent, *, make_env):
        self.learner = agent
        self.make_env = make_env
        self.update_strategy = _QRegressionUpdateStrategy(agent)

    def run_episode(
        self,
        *,
        opponent: object,
        episode: int,
        seed: int,
        epsilon: float,
        train: bool,
    ) -> EpisodeMetrics:
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


class QRegressionTrainer(BaseAECTrainer[QRegressionAgent]):
    """Train online Q-regression through the shared AEC episode driver."""

    config_class = QRegressionConfig

    @property
    def run_name(self) -> str:
        return "q_regression"

    @property
    def tags(self) -> dict[str, str]:
        return {"trainer": "q_regression"}

    def make_agent(self) -> QRegressionAgent:
        num_actions, observation_size = self._agent_init_shape()
        return QRegressionAgent(self.config, num_actions=num_actions, observation_size=observation_size)

    def _make_adapter(self, agent: QRegressionAgent) -> _QRegressionLearnerAdapter:
        return _QRegressionLearnerAdapter(agent, make_env=self.make_env)

    def make_pipeline(self):
        from xrl.training.pipeline import make_q_regression_pipeline

        return make_q_regression_pipeline(
            self.config,
            env_factory=self.env_factory,
            tracker=self.tracker,
            eval_opponents=self.eval_opponents,
        )


def train_q_regression(
    episodes: int = 1_000,
    opponent: str = "random",
    out: Path = Path("runs/q_regression"),
    seed: int = 0,
    learning_rate: float = 1e-3,
    hidden_sizes: str | tuple[int, ...] = (64, 64),
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 800,
    encoder: str = "identity",
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
    """Train online Q-regression: Q-learning with an MLP, no replay buffer, no target network.

    Pass ``--seeds 0,1,2`` to run a multi-seed sweep (one ``seed_<k>/`` run per
    seed plus aggregate artifacts; ``--seed`` is ignored in that case).
    """
    tracker_config = TrackerConfig(
        mlflow=mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    config = QRegressionConfig(
        episodes=episodes,
        seed=seed,
        opponent=opponent,
        gamma=gamma,
        learning_rate=learning_rate,
        hidden_sizes=parse_hidden_sizes(hidden_sizes),
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        encoder=encoder,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        metrics_window=metrics_window,
        q_probe_states=q_probe_states,
        out_dir=out,
        perturbed_minimax_discount=perturbed_minimax_discount,
    )
    if seeds is not None:
        from xrl.training.metrics_io import (
            AEC_SWEEP_EVAL_COLUMNS,
            AEC_SWEEP_METRIC_COLUMNS,
            AEC_SWEEP_POLICY_COLUMNS,
        )
        from xrl.training.seed_sweep import SeedSweepSpec, parse_seeds, run_seed_sweep, trainer_run_seed

        seed_tuple = parse_seeds(seeds)
        spec = SeedSweepSpec(
            seeds=seed_tuple,
            root=config.out_dir,
            run_name=f"q_regression_seeds{len(seed_tuple)}",
            tags={"workflow": "seed_sweep", "trainer": "q_regression"},
            base_params=config.to_json_dict(),
            metric_columns=AEC_SWEEP_METRIC_COLUMNS,
            eval_columns=AEC_SWEEP_EVAL_COLUMNS,
            eval_group_by="opponent",
            policy_columns=AEC_SWEEP_POLICY_COLUMNS,
        )
        run_seed = trainer_run_seed(
            config,
            lambda cfg, tracker: QRegressionTrainer(cfg, tracker=tracker, progress_interval=progress_interval),
            identity=RunIdentity(
                environment="tictactoe",
                algorithm="online-q",
                eval_opponents=("random", "minimax"),
            ),
        )
        return run_seed_sweep(spec, run_seed, tracker_config=tracker_config, plot=not no_plots)
    trainer = QRegressionTrainer(
        config,
        tracker=tracker_from_cli_args(tracker_config),
        progress_interval=progress_interval,
    )
    summary = execute_training_run(
        config,
        trainer,
        RunIdentity(
            environment="tictactoe",
            algorithm="online-q",
            eval_opponents=("random", "minimax"),
        ),
    )
    return {"out_dir": str(config.out_dir), "summary": summary}
