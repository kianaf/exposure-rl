"""Scalar masked surrogate for a scalar Direct-Q: Philipp's xrl surrogate with the categorical part removed.

xrl's ``train-q-surrogate`` distils only categorical (two-hot / HL-Gauss) DQNs: it regresses masked inputs on
the base agent's return distribution with a cross-entropy. For the all-scalar comparison the base Direct-Q has
a scalar head, so this module keeps every other part of the xrl recipe and reuses xrl's own code for it:

    training states   tic-tac-toe: xrl ``build_tictactoe_state_distribution`` of the base policy (weights =
                      occupancy; greedy, or epsilon-greedy with ``occupancy_epsilon``); Minesweeper: xrl ``collect_background`` (epsilon-greedy rollouts, eps 0.1)
    inputs / masks    xrl ``masked_surrogate_inputs``, xrl ``CoalitionMaskSampler`` (Shapley 0.7 / uniform
                      0.3, full and empty anchors every 512 examples, guided masks after 200 updates from
                      exact / sampled Shapley magnitudes via xrl ``DistributionalQExplainer``)
    network / optim   MLP 64 x 64 ReLU, Adam 1e-3, batch 64 (xrl surrogate defaults)

The only change: the target is the base agent's scalar Q and the loss is the squared error averaged over legal
actions (xrl averages its per-action cross-entropy over legal actions in the same way).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from xrl._masking import CoalitionMaskSamplerConfig
from xrl.explainability.adapters import TICTACTOE_ENV_ID, get_explanation_adapter, load_explanation_run
from xrl.explainability.core import ExactShapleyEstimator, SampledShapleyEstimator
from xrl.explainability.distributional_q_explainer import EXACT_SHAPLEY_MAX_FEATURES, DistributionalQExplainer
from xrl.explainability.distributional_q_masks import CoalitionMaskSampler, masked_surrogate_inputs
from xrl.explainability.tictactoe import build_tictactoe_state_distribution, tictactoe_feature_spec

from exposure_rl import opponents


@dataclass(frozen=True)
class ScalarSurrogateConfig:
    base_run_dir: str
    opponent: str = "random"  # tic-tac-toe opponent spec for the training states (exposure_rl.opponents)
    seed: int = 0
    updates: int = 16_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden: tuple[int, ...] = (64, 64)
    rollout_states: int = 16_384
    rollout_epsilon: float = 0.1
    occupancy_epsilon: float = 0.0  # tic-tac-toe: exact occupancy of the epsilon-greedy base policy (0: greedy)
    full_mask_example_interval: int = 512
    empty_mask_example_interval: int = 512
    shapley_probability: float = 0.7
    uniform_probability: float = 0.3
    guided_probability: float = 0.25
    guided_epsilon: float = 1e-3
    guided_warmup_updates: int = 200
    guided_refresh_interval: int = 100


class _Net(nn.Module):
    hidden: tuple[int, ...]
    num_actions: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.num_actions)(x)


class ScalarMaskedSurrogate:
    """``q_values(observation, known_mask)``: the masked surrogate's Q for every action."""

    support: tuple = ()

    def __init__(self, cfg: ScalarSurrogateConfig, num_actions: int, observation_size: int, params=None) -> None:
        self.cfg, self.num_actions, self.observation_size = cfg, num_actions, observation_size
        self.model = _Net(tuple(cfg.hidden), num_actions)
        dummy = jnp.zeros((1, 2 * observation_size), jnp.float32)
        self.params = params if params is not None else self.model.init(jax.random.PRNGKey(cfg.seed), dummy)

    def q_batch(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        inputs = jnp.asarray(masked_surrogate_inputs(np.asarray(states, np.float32), np.asarray(masks, np.float32)))
        return np.asarray(self.model.apply(self.params, inputs), np.float64)

    def q_values(self, observation, known_mask) -> np.ndarray:
        return self.q_batch(np.asarray(observation)[None], np.asarray(known_mask)[None])[0]


class _EpsilonGreedy:
    """The base policy with a fixed epsilon (xrl's occupancy builder asks for the greedy distribution)."""

    def __init__(self, policy, epsilon: float) -> None:
        self.policy, self.epsilon = policy, epsilon

    def action_distribution(self, observation, action_mask, *, epsilon: float = 0.0) -> np.ndarray:
        return self.policy.action_distribution(observation, action_mask, epsilon=self.epsilon)


def _dataset(cfg: ScalarSurrogateConfig, run):
    base = run.policy
    if run.env_id == TICTACTOE_ENV_ID:
        name, rate = opponents.parse(cfg.opponent)
        behaviour = _EpsilonGreedy(base, cfg.occupancy_epsilon) if cfg.occupancy_epsilon > 0 else base
        dist = build_tictactoe_state_distribution(
            behaviour, opponent=name, perturbed_minimax_probability=rate, features=tictactoe_feature_spec()
        )
        obs = np.stack([p.observation_array for p in dist.points]).astype(np.float32)
        legal = np.stack([p.action_mask_array for p in dist.points]).astype(np.float32)
        w = np.asarray([p.weight for p in dist.points], np.float64)
    else:
        background = get_explanation_adapter(run.env_id).collect_background(
            base, run.config, states=cfg.rollout_states, seed=cfg.seed, policy_epsilon=cfg.rollout_epsilon
        )
        obs = np.asarray(background.observations, np.float32)
        legal = (obs == -1).astype(np.float32)  # Minesweeper: hidden cells are the legal clicks
        w = np.ones(len(obs))
    targets = np.stack([np.asarray(base.q_values(o), np.float32) for o in obs])
    return obs, legal, np.nan_to_num(targets), w / w.sum()


def train(cfg: ScalarSurrogateConfig, out_dir: Path) -> ScalarMaskedSurrogate:
    run = load_explanation_run(Path(cfg.base_run_dir))
    obs, legal, targets, w = _dataset(cfg, run)
    n = obs.shape[1]
    surrogate = ScalarMaskedSurrogate(cfg, int(run.policy.num_actions), n)
    tx = optax.adam(cfg.learning_rate)
    opt_state = tx.init(surrogate.params)

    @jax.jit
    def step(params, opt_state, inputs, target, mask):
        def loss_fn(p):
            sq = (surrogate.model.apply(p, inputs) - target) ** 2
            return jnp.sum(sq * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    sampler = CoalitionMaskSampler(
        CoalitionMaskSamplerConfig(
            feature_count=n,
            full_mask_example_interval=cfg.full_mask_example_interval,
            empty_mask_example_interval=cfg.empty_mask_example_interval,
            shapley_probability=cfg.shapley_probability,
            uniform_probability=cfg.uniform_probability,
            guided_probability=cfg.guided_probability,
            guided_epsilon=cfg.guided_epsilon,
        ),
        seed=cfg.seed,
    )
    estimator = (
        ExactShapleyEstimator()
        if n <= EXACT_SHAPLEY_MAX_FEATURES
        else SampledShapleyEstimator(num_permutations=32, seed=cfg.seed)
    )
    rng = np.random.default_rng(cfg.seed)
    magnitudes, history = None, []
    for update in range(cfg.updates):
        if update >= cfg.guided_warmup_updates and update % cfg.guided_refresh_interval == 0:
            i = int(rng.choice(len(obs), p=w))  # as xrl: Shapley magnitudes of the current surrogate at one state
            q = surrogate.q_values(obs[i], np.ones(n, np.int8))
            action = max(np.flatnonzero(legal[i] > 0), key=lambda a: float(q[a]))
            explanation = DistributionalQExplainer(surrogate, estimator=estimator).explain_action(
                obs[i], legal[i].astype(np.int8), action=int(action)
            )
            magnitudes = np.asarray([abs(v) for v in explanation.attributions.values()], np.float64)
        idx = rng.choice(len(obs), size=cfg.batch_size, replace=True, p=w)
        masks = sampler.sample_batch(
            cfg.batch_size, example_offset=update * cfg.batch_size, attribution_magnitudes=magnitudes
        )
        inputs = jnp.asarray(masked_surrogate_inputs(obs[idx], np.asarray(masks, np.float32)))
        surrogate.params, opt_state, loss = step(
            surrogate.params, opt_state, inputs, jnp.asarray(targets[idx]), jnp.asarray(legal[idx])
        )
        if update % 500 == 0 or update == cfg.updates - 1:
            history.append({"update": update + 1, "loss": float(loss)})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scalar_surrogate.msgpack").write_bytes(serialization.to_bytes(surrogate.params))
    (out_dir / "scalar_surrogate_config.json").write_text(
        json.dumps(
            {**asdict(cfg), "num_actions": surrogate.num_actions, "observation_size": n, "points": len(obs)}, indent=2
        )
    )
    (out_dir / "scalar_surrogate_history.json").write_text(json.dumps(history))
    return surrogate


def load(out_dir: Path) -> ScalarMaskedSurrogate:
    meta = json.loads((out_dir / "scalar_surrogate_config.json").read_text())
    num_actions, n = meta.pop("num_actions"), meta.pop("observation_size")
    meta.pop("points", None)
    meta["hidden"] = tuple(meta["hidden"])
    cfg = ScalarSurrogateConfig(**meta)
    surrogate = ScalarMaskedSurrogate(cfg, num_actions, n)
    surrogate.params = serialization.from_bytes(surrogate.params, (out_dir / "scalar_surrogate.msgpack").read_bytes())
    return surrogate
