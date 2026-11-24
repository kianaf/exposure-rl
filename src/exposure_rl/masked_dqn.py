"""Baselines with certainty heads: mask-aware DQN (MA-DQN) and DQN + surrogate, each optionally with a variance head
``s`` and a second-moment head ``m2`` (2026-11-07, reviewer objection: "the baselines could have s too").

Re-implementation in our code of the xrl Direct-Q / DQN recipe of Table 1 (raw input ``[o * m, m]``, MLP 2 x 256
ReLU, Adam 1e-3 constant, batch 512, replay 1e5, warmup 5000, target sync 1000, gamma = 1, epsilon 1 -> 0.05),
so that extra heads can be added without touching the xrl package. Heads (one output per action each):

  q   regressed on the TD target y = r + max_a' Q_target(o', a') (full next observation)
  s   regressed on the TD return-variance target y_s = clip((y - Q_target(o, a))^2 + S_target(o', a*), 0, 1)
      (the same recursion as EXPOSURE's variance head, Tamar et al.)
  m2  regressed on Q_target(o, a)^2

MA-DQN (``mask_aware=True``): every update trains the heads twice, on the full input and on a masked input
``[o * m, m]`` with the same full-information targets. A masked regression on full-information targets converges to
their conditional mean given the visible cells, so

  q(o_C) = E[Q(O, a) | o_C],   s(o_C) = E[Var(G | O, a) | o_C] = v_inh(C),   m2(o_C) - q(o_C)^2 = v_obs(C),

the last as a difference of two regressions (clipped at 0). Standard DQN (``mask_aware=False``): full input only;
the surrogate (``fit_surrogate``) then regresses the DQN's q, s and q^2 on masked inputs over the DQN's
epsilon-greedy occupancy (epsilon = 0.1), as the xrl surrogate does for q.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from exposure_rl import opponents
from exposure_rl.td import MinesweeperGame, Replay, TDConfig, TicTacToeVsOpponent

HEADS = ("q", "s", "m2")


@dataclass(frozen=True)
class DQNConfig:
    n_cells: int = 9
    hidden: int = 256
    depth: int = 2
    lr: float = 1e-3
    batch_size: int = 512
    mask_aware: bool = True
    var_head: bool = True
    moment_head: bool = True
    kernel_share: float = 0.7  # mask sizes: Shapley kernel 70 %, uniform 30 %, empty 1/256 (xrl-matched)
    empty_prob: float = 1 / 256
    seed: int = 0


class QNet(nn.Module):
    cfg: DQNConfig

    @nn.compact
    def __call__(self, obs, mask):  # (B, n), (B, n) -> (B, 3, n): q, s, m2 per action
        h = jnp.concatenate([obs * mask, mask], axis=-1)
        for _ in range(self.cfg.depth):
            h = nn.relu(nn.Dense(self.cfg.hidden)(h))
        return nn.Dense(len(HEADS) * self.cfg.n_cells)(h).reshape(*obs.shape[:-1], len(HEADS), self.cfg.n_cells)


def sample_masks(rng: np.random.Generator, b: int, cfg: DQNConfig) -> np.ndarray:
    n = cfg.n_cells
    sizes = np.arange(1, n)
    kernel = (n - 1) / (sizes * (n - sizes))
    kernel = kernel / kernel.sum()
    k = np.where(rng.random(b) < cfg.kernel_share, rng.choice(sizes, size=b, p=kernel), rng.choice(sizes, size=b))
    k = np.where(rng.random(b) < cfg.empty_prob, 0, k)
    return (rng.random((b, n)).argsort(axis=1) < k[:, None]).astype(np.float32)


def _taken(values, actions):  # (B, n) at per-row action -> (B,)
    return jnp.take_along_axis(values, actions[:, None], axis=-1)[:, 0]


def make_loss(net: QNet, cfg: DQNConfig, td: TDConfig):
    def loss_fn(params, target_params, obs, mask, act, rew, next_obs, next_legal, done):
        full = jnp.ones_like(obs)
        tq_next = net.apply(target_params, next_obs, full)
        q_next = jnp.where(next_legal, tq_next[:, 0], -jnp.inf)
        a_star = jnp.argmax(q_next, axis=-1)
        future = jnp.max(q_next, axis=-1)
        alive = jnp.isfinite(future) & (done < 0.5)
        y = rew + td.gamma * jnp.where(alive, future, 0.0)
        tq_now = net.apply(target_params, obs, full)
        q_bar = _taken(tq_now[:, 0], act)
        s_next = _taken(tq_next[:, 1], a_star)
        y_s = jnp.maximum((y - q_bar) ** 2 + td.gamma**2 * jnp.where(alive, s_next, 0.0), 0.0)  # no upper clip, as in td.py
        targets = jax.lax.stop_gradient(jnp.stack([y, y_s, q_bar**2], axis=1))  # (B, 3)
        use = jnp.array([1.0, float(cfg.var_head), float(cfg.moment_head)])

        def err(m):
            out = net.apply(params, obs, m)
            pred = jnp.take_along_axis(out, jnp.broadcast_to(act[:, None, None], (len(act), len(HEADS), 1)), -1)[..., 0]
            return ((pred - targets) ** 2 * use).sum(-1)

        total = err(full)
        if cfg.mask_aware:
            total = total + err(mask)
        td_mse = ((_taken(net.apply(params, obs, full)[:, 0], act) - y) ** 2).mean()
        return total.mean(), {"loss": total.mean(), "td_mse": td_mse}

    return loss_fn


def init(cfg: DQNConfig):
    net = QNet(cfg)
    z = jnp.zeros((1, cfg.n_cells))
    return net, net.init(jax.random.PRNGKey(cfg.seed), z, z)


def make_env(td: TDConfig, rng):
    if td.domain == "minesweeper":
        return MinesweeperGame(td.rows, td.columns, td.mines, rng)
    return TicTacToeVsOpponent(td.opponent, rng)


def train(cfg: DQNConfig, td: TDConfig, log_every: int = 20_000):
    """TD training as exposure_rl.td.train_td (one env step, one update), the agent acting on its full-input Q."""

    rng = np.random.default_rng(cfg.seed)
    env = make_env(td, rng)
    replay = Replay(td.replay_size, cfg.n_cells)
    net, params = init(cfg)
    target_params = params
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)
    loss_fn = make_loss(net, cfg, td)

    @jax.jit
    def q_full(params, obs):
        return net.apply(params, obs[None], jnp.ones((1, cfg.n_cells)))[0, 0]

    @jax.jit
    def step(params, target_params, opt_state, batch):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, target_params, *batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    def epsilon(t):
        return td.eps_start + min(1.0, t / max(1, td.eps_decay_share * td.updates)) * (td.eps_end - td.eps_start)

    def act(obs, legal, eps_now):
        idx = np.flatnonzero(legal)
        if rng.random() < eps_now:
            return int(rng.choice(idx))
        q = np.asarray(q_full(params, jnp.asarray(obs, jnp.float32)))
        return int(idx[np.argmax(q[idx])])

    obs, legal = env.reset()
    for _ in range(td.warmup):
        a = int(rng.choice(np.flatnonzero(legal)))
        nxt, nl, r, done = env.step(a)
        replay.add(obs, a, r, nxt, nl, float(done))
        obs, legal = env.reset() if done else (nxt, nl)
    _, target_rate = opponents.parse(td.opponent) if td.domain == "tictactoe" else (None, None)
    if td.anneal_opponent and target_rate is not None:
        env.rate = 1.0
    returns, ret, history = [], 0.0, []
    for t in range(1, td.updates + 1):
        if td.anneal_opponent and target_rate is not None:
            env.rate = target_rate ** (t / td.updates)
        a = act(obs, legal, epsilon(t))
        nxt, nl, r, done = env.step(a)
        replay.add(obs, a, r, nxt, nl, float(done))
        ret += r
        if done:
            returns.append(ret)
            ret = 0.0
            obs, legal = env.reset()
        else:
            obs, legal = nxt, nl
        o, ac, rw, no, nlg, dn = replay.sample(rng, cfg.batch_size)
        batch = tuple(jnp.asarray(v) for v in (o, sample_masks(rng, cfg.batch_size, cfg), ac, rw, no, nlg, dn))
        params, opt_state, metrics = step(params, target_params, opt_state, batch)
        if t % td.target_every == 0:
            target_params = params
        if log_every and t % log_every == 0:
            entry = {"update": t, "epsilon": epsilon(t), "mean_return_last_500": float(np.mean(returns[-500:]))}
            entry.update({k: float(v) for k, v in metrics.items()})
            history.append(entry)
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()), flush=True)
    return net, params, history


def occupancy_states(net, params, cfg: DQNConfig, td: TDConfig, n_states: int, eps: float = 0.1, seed: int = 0):
    """Decision observations visited by the epsilon-greedy DQN (the surrogate's training data)."""

    rng = np.random.default_rng(10_000 + seed)
    env = make_env(td, rng)
    q_full = jax.jit(lambda o: net.apply(params, o[None], jnp.ones((1, cfg.n_cells)))[0, 0])
    states, (obs, legal) = [], env.reset()
    while len(states) < n_states:
        states.append(np.asarray(obs, np.float32))
        idx = np.flatnonzero(legal)
        a = int(rng.choice(idx)) if rng.random() < eps else int(idx[np.argmax(np.asarray(q_full(jnp.asarray(obs, jnp.float32)))[idx])])
        nxt, nl, _, done = env.step(a)
        obs, legal = env.reset() if done else (nxt, nl)
    return np.stack(states)


def fit_surrogate(net, params, cfg: DQNConfig, states: np.ndarray, steps: int = 50_000, seed: int = 0, moment: bool = True):
    """Masked surrogate of a standard DQN: regresses its q, s and q^2 (all actions) on masked inputs."""

    scfg = DQNConfig(**{**cfg.__dict__, "mask_aware": True, "seed": seed + 777})
    snet, sparams = init(scfg)
    base =np.concatenate([np.asarray(net.apply(params, jnp.asarray(c), jnp.ones_like(jnp.asarray(c)))) for c in np.array_split(states, 64)])
    targets = np.stack([base[:, 0], base[:, 1], base[:, 0] ** 2], axis=1)  # (N, 3, n)
    use = jnp.array([1.0, float(cfg.var_head), float(moment)])[None, :, None]  # q^2 from the DQN's own q
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(sparams)

    @jax.jit
    def step(p, opt_state, o, m, y):
        def loss(p):
            return (((snet.apply(p, o, m) - y) ** 2) * use).sum((1, 2)).mean()

        l, g = jax.value_and_grad(loss)(p)
        u, opt_state = tx.update(g, opt_state, p)
        return optax.apply_updates(p, u), opt_state, l

    rng = np.random.default_rng(seed + 778)
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(states), cfg.batch_size)
        m = sample_masks(rng, cfg.batch_size, cfg)
        m[: cfg.batch_size // 8] = 1.0  # some full inputs, so the surrogate also reproduces the DQN itself
        sparams, opt_state, l = step(sparams, opt_state, jnp.asarray(states[idx]), jnp.asarray(m), jnp.asarray(targets[idx]))
        if t % 10_000 == 0:
            print(f"  surrogate step {t}  loss={float(l):.5f}", flush=True)
    return snet, sparams
