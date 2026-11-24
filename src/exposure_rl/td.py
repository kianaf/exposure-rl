"""Milestone 2b: VAEAC-Q trained by TD while the agent plays through its own latent belief.

Same model and losses as milestone 2a, except the Q term: instead of the exact Q, the Q head
is regressed on the one-step target

    y = r + gamma * (1 - done) * max_{a' legal} Q_target(x', a'),   Q(x, a) = E_{z ~ full belief}[Q_omega(z, a)]

with ``x'`` the next learner-decision board (after the opponent's reply), as in the xrl DQN.
The Q head still only sees ``z`` drawn from the full belief; masked Q still comes only from
averaging over the belief. The agent acts epsilon-greedily on the belief-averaged Q, the same
object that is explained.

``gamma = 1`` (the xrl default is 0.95): rewards are terminal-only and episodes are at most nine
plies, so the undiscounted target keeps the exact expectimax oracle a valid reference.

With a variance head (``cfg.var_weight > 0``) its target is also a one-step bootstrap, from the
return-variance recursion Var(G | x, a) = E[(r + gamma Q(x', a*) - Q(x, a))^2] + gamma^2 E[Var(G | x', a*)]:

    y_s = (r + gamma (1 - done) Q_target(x', a*) - Q_target(x, a))^2 + gamma^2 (1 - done) S_target(x', a*)

with ``a*`` the greedy next action and ``S = E_z[s_omega]`` the belief-averaged variance head, so no
oracle is used in training. ``domain="minesweeper"`` plays xrl's Minesweeper rules (exposure_rl.minesweeper).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from xrl.tictactoe.core import O, TicTacToeRules, X

from exposure_rl import opponents
from exposure_rl.vaeac_q import (
    VAEACQ,
    Config,
    belief_kl,
    belief_q,
    init_params,
    learning_rate,
    mixture_q_moments,
    prior_regulariser,
    reconstruction_nll,
    sample_masks,
)


@dataclass(frozen=True)
class TDConfig:
    updates: int = 150_000
    gamma: float = 1.0
    replay_size: int = 100_000
    warmup: int = 5_000
    target_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_share: float = 0.5  # share of updates over which epsilon decays linearly
    opponent: str = "random"  # or "pm05": xrl perturbed minimax (see exposure_rl.opponents)
    # Anneal a perturbed-minimax opponent's random-move rate from 1 to its target over training (xrl CLI schedule).
    anneal_opponent: bool = False
    domain: str = "tictactoe"  # or "minesweeper" (Geometry below)
    rows: int = 4
    columns: int = 4
    mines: int = 3


class TicTacToeVsOpponent:
    """Learner vs a fixed opponent; the learner plays X or O with probability 1/2 (the oracle's profile)."""

    def __init__(self, opponent: str, rng: np.random.Generator) -> None:
        self.rules = TicTacToeRules()
        self.opponent = opponent
        self.rng = rng
        self.rate: float | None = None  # current random-move rate when annealing a perturbed minimax

    def _observe(self):
        view = self.rules.observation(self.state, self.learner, "flat")
        return np.asarray(view["observation"], np.float32).reshape(-1), np.asarray(view["action_mask"], bool)

    def _learner_return(self) -> float:
        r_x, r_o = self.rules.returns(self.state)
        return float(r_x if self.learner == X else r_o)

    def _opponent_moves(self) -> None:
        probs = opponents.distribution(self.opponent, self.state, self.rules, rate=self.rate)
        self.state = self.rules.apply(self.state, int(self.rng.choice(9, p=probs)))

    def reset(self):
        self.learner = X if self.rng.random() < 0.5 else O
        self.state = self.rules.initial_state()
        if self.learner == O:
            self._opponent_moves()
        return self._observe()

    def step(self, action: int):
        """Learner move, then the opponent's reply. Returns ``(next_obs, next_legal, reward, done)``."""

        self.state = self.rules.apply(self.state, action)
        if not self.state.is_terminal:
            self._opponent_moves()
        if self.state.is_terminal:
            return np.zeros(9, np.float32), np.zeros(9, bool), self._learner_return(), True
        obs, legal = self._observe()
        return obs, legal, 0.0, False


class MinesweeperGame:
    """xrl Minesweeper rules via the exact oracle's ``reveal``: mines are placed after the first click (so it is
    always safe), a safe reveal pays newly revealed / total safe, a mine pays -1 and ends the episode."""

    def __init__(self, rows: int, columns: int, mines: int, rng: np.random.Generator) -> None:
        from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle

        self.oracle = MinesweeperOracle(Geometry(rows, columns, mines))
        self.hidden, self.n, self.rng = HIDDEN, rows * columns, rng

    def _observe(self):
        obs = np.asarray(self.obs, np.float32)
        return obs, obs == self.hidden

    def reset(self):
        self.obs, self.layout = (self.hidden,) * self.n, None
        return self._observe()

    def step(self, action: int):
        if self.layout is None:  # first click: draw the mines among the other cells
            choices = [lay for lay in self.oracle.layouts if action not in lay]
            self.layout = choices[int(self.rng.integers(len(choices)))]
        if action in self.layout:  # a mine: -1 and the episode ends (``reveal`` is the oracle's safe-reveal step)
            return np.zeros(self.n, np.float32), np.zeros(self.n, bool), -1.0, True
        self.obs, reward, done = self.oracle.reveal(self.obs, action, self.layout)
        if done:
            return np.zeros(self.n, np.float32), np.zeros(self.n, bool), float(reward), True
        obs, legal = self._observe()
        return obs, legal, float(reward), False


class Replay:
    def __init__(self, size: int, n_cells: int = 9) -> None:
        self.size, self.n, self.i = size, 0, 0
        self.obs = np.zeros((size, n_cells), np.float32)
        self.act = np.zeros(size, np.int32)
        self.rew = np.zeros(size, np.float32)
        self.next_obs = np.zeros((size, n_cells), np.float32)
        self.next_legal = np.zeros((size, n_cells), bool)
        self.done = np.zeros(size, np.float32)

    def add(self, obs, act, rew, next_obs, next_legal, done) -> None:
        j = self.i
        self.obs[j], self.act[j], self.rew[j] = obs, act, rew
        self.next_obs[j], self.next_legal[j], self.done[j] = next_obs, next_legal, done
        self.i, self.n = (j + 1) % self.size, min(self.n + 1, self.size)

    def sample(self, rng: np.random.Generator, batch: int):
        idx = rng.integers(0, self.n, size=batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.next_legal[idx], self.done[idx]


def make_td_loss(model: VAEACQ, cfg: Config, td: TDConfig):
    def loss_fn(params, target_params, obs, known, act, rew, next_obs, next_legal, done, eps, eps_target):
        out = model.apply({"params": params}, obs, known, eps)
        recon = reconstruction_nll(out, obs, known)
        kl = belief_kl(out)

        q_next = belief_q(model, target_params, next_obs, eps_target)
        masked_next = jnp.where(next_legal, q_next, -jnp.inf)
        a_star = jnp.argmax(masked_next, axis=-1)
        future = jnp.max(masked_next, axis=-1)
        future = jnp.where(jnp.isfinite(future), future, 0.0)
        y = jax.lax.stop_gradient(rew + td.gamma * (1.0 - done) * future)

        def taken(values, actions):  # (B, M, n) at per-row action -> (B, M)
            idx = jnp.broadcast_to(actions[:, None, None], (*values.shape[:2], 1))
            return jnp.take_along_axis(values, idx, axis=-1)[..., 0]

        pi_f = jnp.exp(out["log_pi_f"])
        td_err = jnp.sum(pi_f * (taken(out["q"], act) - y[:, None]) ** 2, axis=-1)
        total = recon + cfg.beta * kl + cfg.q_weight * td_err + prior_regulariser(out, cfg)
        metrics = {"loss": total.mean(), "recon": recon.mean(), "kl": kl.mean(), "td_mse": td_err.mean()}

        if cfg.var_weight > 0:  # TD target for the variance head (return-variance recursion), no oracle
            log_pi, mu, ls = model.apply({"params": target_params}, next_obs, method=VAEACQ.full_belief)
            s_next = mixture_q_moments(model, target_params, log_pi, mu, ls, eps_target, head=VAEACQ.variance)[0]
            s_future = jnp.take_along_axis(s_next, a_star[:, None], axis=-1)[:, 0]
            q_now = jnp.take_along_axis(belief_q(model, target_params, obs, eps_target), act[:, None], axis=-1)[:, 0]
            y_s = jax.lax.stop_gradient(
                jnp.maximum(  # no upper clip: a single target (y - Q)^2 can exceed Var(G) <= 1, clipping biases it down
                    (y - q_now) ** 2 + td.gamma**2 * (1.0 - done) * jnp.where(jnp.isfinite(future), s_future, 0.0),
                    0.0,
                )
            )
            var_err = jnp.sum(pi_f * (taken(out["s"], act) - y_s[:, None]) ** 2, axis=-1)
            total = total + cfg.var_weight * var_err
            metrics.update(loss=total.mean(), var_mse=var_err.mean())
        return total.mean(), metrics

    return loss_fn


def train_td(cfg: Config, td: TDConfig, log_every: int = 10_000, on_log=None):
    """Interleave one environment step and one gradient update; returns ``(model, params, history)``."""

    rng = np.random.default_rng(cfg.seed)
    if td.domain == "minesweeper":
        env = MinesweeperGame(td.rows, td.columns, td.mines, rng)
    else:
        env = TicTacToeVsOpponent(td.opponent, rng)
    replay = Replay(td.replay_size, cfg.n_cells)

    model = VAEACQ(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    params = init_params(model, cfg.seed)
    # Small fixed sample set for the belief-averaged Q used to act and to form TD targets.
    eps_act = jax.random.normal(jax.random.PRNGKey(cfg.seed + 99), (16, cfg.latent_dim))
    target_params = params
    tx = optax.adam(learning_rate(cfg, td.updates))
    opt_state = tx.init(params)
    loss_fn = make_td_loss(model, cfg, td)

    @jax.jit
    def greedy_q(params, obs):
        return belief_q(model, params, obs[None], eps_act)[0]

    @jax.jit
    def step(params, target_params, opt_state, batch, eps):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, target_params, *batch, eps, eps_act)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    def epsilon(t: int) -> float:
        frac = min(1.0, t / max(1, td.eps_decay_share * td.updates))
        return td.eps_start + frac * (td.eps_end - td.eps_start)

    def act(obs, legal, eps_now) -> int:
        legal_idx = np.flatnonzero(legal)
        if rng.random() < eps_now:
            return int(rng.choice(legal_idx))
        q = np.asarray(greedy_q(params, jnp.asarray(obs)))
        return int(legal_idx[np.argmax(q[legal_idx])])

    obs, legal = env.reset()
    history, episode_returns, episode_return = [], [], 0.0
    for _ in range(td.warmup):
        a = int(rng.choice(np.flatnonzero(legal)))
        next_obs, next_legal, r, done = env.step(a)
        replay.add(obs, a, r, next_obs, next_legal, float(done))
        obs, legal = env.reset() if done else (next_obs, next_legal)

    _, target_rate = opponents.parse(td.opponent) if td.domain == "tictactoe" else (None, None)
    if td.anneal_opponent and target_rate is not None:
        env.rate = 1.0  # warmup against a fully random opponent, then decay to the target rate
    for t in range(1, td.updates + 1):
        if td.anneal_opponent and target_rate is not None:
            env.rate = target_rate ** (t / td.updates)
        a = act(obs, legal, epsilon(t))
        next_obs, next_legal, r, done = env.step(a)
        replay.add(obs, a, r, next_obs, next_legal, float(done))
        episode_return += r
        if done:  # the episode's return (sum of rewards; in tic-tac-toe the single terminal reward)
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs, legal = env.reset()
        else:
            obs, legal = next_obs, next_legal

        o, ac, rw, no, nl, dn = replay.sample(rng, cfg.batch_size)
        batch = tuple(jnp.asarray(v) for v in (o, sample_masks(rng, cfg.batch_size, cfg), ac, rw, no, nl, dn))
        key, sub = jax.random.split(key)
        params, opt_state, metrics = step(
            params, target_params, opt_state, batch, jax.random.normal(sub, (cfg.batch_size, cfg.latent_dim))
        )
        if t % td.target_every == 0:
            target_params = params
        if log_every and t % log_every == 0:
            recent = float(np.mean(episode_returns[-500:])) if episode_returns else float("nan")
            entry = {"update": t, "epsilon": epsilon(t), "mean_return_last_500": recent}
            entry.update({k: float(v) for k, v in metrics.items()})
            if on_log is not None:
                entry.update(on_log(model, params))
            history.append(entry)
            print(
                "  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()),
                flush=True,
            )
    return model, params, history
