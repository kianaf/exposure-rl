"""VAEAC-Q: a Q-network whose masked inputs are answered by averaging over a latent belief.

One mask-aware belief network serves every coalition (``tied_encoder=True``, the default,
decided 2026-11-03):

    belief network  e_psi(z | x_S, S)          mixture of M diagonal Gaussians; S = N is the
                                               agent's full-information belief
    decoder         p_theta(x_Sbar | z, x_S)   reconstructs the hidden cells (training only)
    Q head          Q_omega(z, a)              trained only on z drawn from the full belief

Training objective per (board, mask): the VAEAC bound with the posterior being the same network
at the full mask, KL(e_psi(. | x_N) || e_psi(. | x_S)), plus reconstruction of the hidden cells
and the Q term, all as exact expectations over the full belief's mixture components.

Masked Q is never trained directly: Q(x_S, a) = E_{z ~ e_psi(. | x_S)}[Q_omega(z, a)], and the
certainty game is v_mask(S) = Var_{z ~ e_psi(. | x_S)}[Q_omega(z, a)]. The agent acts on the
same belief-averaged Q at S = N, so the acted-on Q and the explained Q are one object.

``tied_encoder=False`` is the earlier two-network ablation (separate Gaussian full encoder
q_phi(z | x) and masked prior p_psi(z | x_S, S)), kept for comparison.

Milestone 2a supervises Q_omega with the exact expectimax Q (no RL); milestone 2b uses TD.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

N_CELLS = 9
N_VALUES = 3  # Tic Tac Toe: mover-relative cell value in {-1, 0, 1}; Minesweeper uses 10 (hidden, clues 0-8)


@dataclass(frozen=True)
class Config:
    latent_dim: int = 16
    hidden: int = 256
    depth: int = 3
    # Hidden activation; "relu" with hidden=64, depth=2 matches the xrl Direct-Q MLP (Philipp's default).
    activation: str = "gelu"
    # Input encoding of a masked board: "onehot" (default) or "raw" (xrl: [value * known, known]).
    input_encoding: str = "onehot"
    # "cosine" decay to 5% (default) or "constant" (xrl's Adam).
    lr_schedule: str = "cosine"
    # Belief e_psi(z | x_S) as a mixture of this many diagonal Gaussians (1 = single Gaussian).
    n_components: int = 10
    # One belief network for every coalition (True) or separate full encoder + masked prior (False).
    tied_encoder: bool = True
    # Number of distinct cell values; a value v is one-hot encoded at index v + 1 (so -1 is index 0).
    n_values: int = N_VALUES
    # Number of cells / features (9 for 3x3 boards, 16 for Minesweeper 4x4).
    n_cells: int = N_CELLS
    beta: float = 1.0  # KL weight; back to 1 on 2026-11-07 (0.5 was tried, results in results/beta05_backup)
    q_weight: float = 50.0
    # Variance head s_omega(z, a) >= 0 regressed on Var(G | o, a); 0 disables it (no extra parameters).
    var_weight: float = 0.0
    # Q and variance heads share their hidden layers (one Q-head on z with two output layers; the paper's model).
    # False: two separate MLPs (the earlier model; checkpoint suffix "_separate").
    shared_heads: bool = True
    prior_mu_l2: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 512
    steps: int = 60_000
    # ADR 0015 mask mixture: Shapley-kernel vs uniform partial sizes; empty mask per row with this probability.
    kernel_share: float = 0.7
    empty_prob: float = 1 / 8
    seed: int = 0


def one_hot(obs: jnp.ndarray, n_values: int = N_VALUES) -> jnp.ndarray:
    """``(..., n)`` cell values in {-1, ..., n_values - 2} → ``(..., n * n_values)``."""

    return jax.nn.one_hot(obs.astype(jnp.int32) + 1, n_values).reshape(*obs.shape[:-1], obs.shape[-1] * n_values)


def masked_input(
    obs: jnp.ndarray, known: jnp.ndarray, n_values: int = N_VALUES, encoding: str = "onehot"
) -> jnp.ndarray:
    """``[onehot(x) * s, s]``: unknown cells are all-zero one-hots, distinguished by the mask channel."""

    if encoding == "raw":  # xrl's encoding: [cell value * known, known], unknown values zero-filled
        return jnp.concatenate([obs.astype(jnp.float32) * known, known], axis=-1)
    hot = one_hot(obs, n_values).reshape(*obs.shape, n_values) * known[..., None]
    return jnp.concatenate([hot.reshape(*obs.shape[:-1], -1), known], axis=-1)


ACTIVATIONS = {"gelu": nn.gelu, "relu": nn.relu}


class MLP(nn.Module):
    hidden: int
    depth: int
    out: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        act = ACTIVATIONS[self.activation]
        for _ in range(self.depth):
            x = act(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out)(x)


class MixtureGaussian(nn.Module):
    """``(log_pi, mu, log_sigma)`` with shapes ``(..., M)``, ``(..., M, d)``, ``(..., M, d)``."""

    hidden: int
    depth: int
    latent_dim: int
    n_components: int
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        m, d = self.n_components, self.latent_dim
        h = MLP(self.hidden, self.depth, m * (2 * d + 1), self.activation)(x)
        logits, rest = h[..., :m], h[..., m:].reshape(*h.shape[:-1], m, 2 * d)
        mu, log_sigma = jnp.split(rest, 2, axis=-1)
        return jax.nn.log_softmax(logits, axis=-1), mu, jnp.clip(log_sigma, -7.0, 3.0)


class VAEACQ(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        act = c.activation
        self.belief_net = MixtureGaussian(c.hidden, c.depth, c.latent_dim, c.n_components, act)
        if not c.tied_encoder:
            self.full_net = MixtureGaussian(c.hidden, c.depth, c.latent_dim, 1, act)
        self.decoder = MLP(c.hidden, c.depth, c.n_cells * c.n_values, act)
        if c.shared_heads and c.var_weight > 0:
            self.head_trunk = MLP(c.hidden, c.depth - 1, c.hidden, act)  # + activation: depth hidden layers
            self.q_out = nn.Dense(c.n_cells)
            self.var_out = nn.Dense(c.n_cells)
        else:
            self.q_head = MLP(c.hidden, c.depth, c.n_cells, act)
            if c.var_weight > 0:
                self.var_head = MLP(c.hidden, c.depth, c.n_cells, act)

    def belief(self, obs, known):
        """Belief given coalition ``known``; with the tied encoder ``known = 1`` is the full-information belief."""

        return self.belief_net(masked_input(obs, known, self.cfg.n_values, self.cfg.input_encoding))

    def full_belief(self, obs):
        """The agent's own belief at full information (what it acts on and what the grand coalition uses)."""

        if self.cfg.tied_encoder:
            return self.belief(obs, jnp.ones_like(obs))
        return self.full_net(one_hot(obs, self.cfg.n_values))

    def decode(self, z, obs, known):
        logits = self.decoder(
            jnp.concatenate([z, masked_input(obs, known, self.cfg.n_values, self.cfg.input_encoding)], axis=-1)
        )
        return logits.reshape(*logits.shape[:-1], self.cfg.n_cells, self.cfg.n_values)

    def _trunk(self, z):
        return ACTIVATIONS[self.cfg.activation](self.head_trunk(z))

    def q(self, z):
        if self.cfg.shared_heads and self.cfg.var_weight > 0:
            return self.q_out(self._trunk(z))
        return self.q_head(z)

    def variance(self, z):
        """``s_omega(z, a)``: the return variance the agent expects at latent ``z`` (state uncertainty)."""

        if self.cfg.shared_heads:
            return jax.nn.softplus(self.var_out(self._trunk(z)))
        return jax.nn.softplus(self.var_head(z))

    def __call__(self, obs, known, eps):
        """Training pass: one reparameterised sample per full-belief component (``eps``: ``(B, d)``)."""

        log_pi_f, mu_f, ls_f = self.full_belief(obs)
        log_pi_p, mu_p, ls_p = self.belief(obs, known)
        z = mu_f + jnp.exp(ls_f) * eps[:, None, :]  # (B, M_full, d)
        m = z.shape[1]
        obs_m = jnp.broadcast_to(obs[:, None], (obs.shape[0], m, obs.shape[-1]))
        known_m = jnp.broadcast_to(known[:, None], obs_m.shape)
        return {
            "z": z,
            "log_pi_f": log_pi_f,
            "mu_f": mu_f,
            "ls_f": ls_f,
            "log_pi_p": log_pi_p,
            "mu_p": mu_p,
            "ls_p": ls_p,
            "logits": self.decode(z, obs_m, known_m),  # (B, M_full, 9, 3)
            "q": self.q(z),  # (B, M_full, n)
            **({"s": self.variance(z)} if self.cfg.var_weight > 0 else {}),
        }


def gaussian_log_density(z, mu, ls):
    return jnp.sum(-0.5 * ((z - mu) / jnp.exp(ls)) ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def gaussian_kl(mu_q, ls_q, mu_p, ls_p):
    """``KL(N(mu_q, e^{2 ls_q}) || N(mu_p, e^{2 ls_p}))`` summed over latent dims."""

    return jnp.sum(ls_p - ls_q + (jnp.exp(2 * ls_q) + (mu_q - mu_p) ** 2) / (2 * jnp.exp(2 * ls_p)) - 0.5, axis=-1)


def mixture_log_density(z, log_pi, mu, ls):
    """``log sum_m pi_m N(z; mu_m, sigma_m)`` for ``z`` of shape ``(..., K, d)`` against a ``(..., M, d)`` mixture."""

    comp = gaussian_log_density(z[..., :, None, :], mu[..., None, :, :], ls[..., None, :, :])  # (..., K, M)
    return jax.nn.logsumexp(log_pi[..., None, :] + comp, axis=-1)


def belief_kl(out) -> jnp.ndarray:
    """``KL(full belief || coalition belief)``: closed form for two single Gaussians, else a stratified estimate.

    The estimate is ``sum_m pi_m [log e(z_m | x_N) - log e(z_m | x_S)]`` with one reparameterised
    sample ``z_m`` per full-belief component, differentiable in the weights ``pi``.
    """

    if out["mu_f"].shape[-2] == 1 and out["mu_p"].shape[-2] == 1:
        return gaussian_kl(out["mu_f"][:, 0], out["ls_f"][:, 0], out["mu_p"][:, 0], out["ls_p"][:, 0])
    z = out["z"]
    log_q = mixture_log_density(z, out["log_pi_f"], out["mu_f"], out["ls_f"])
    log_p = mixture_log_density(z, out["log_pi_p"], out["mu_p"], out["ls_p"])
    return jnp.sum(jnp.exp(out["log_pi_f"]) * (log_q - log_p), axis=-1)


def reconstruction_nll(out, obs, known) -> jnp.ndarray:
    """Expected NLL of the hidden cells under the full belief (exact over its components)."""

    target = (obs + 1).astype(jnp.int32)[:, None, :, None]
    nll = -jnp.take_along_axis(
        jax.nn.log_softmax(out["logits"]), jnp.broadcast_to(target, (*out["logits"].shape[:3], 1)), axis=-1
    )[..., 0]
    per_component = jnp.sum(nll * (1.0 - known)[:, None, :], axis=-1)
    return jnp.sum(jnp.exp(out["log_pi_f"]) * per_component, axis=-1)


def prior_regulariser(out, cfg: Config) -> jnp.ndarray:
    return cfg.prior_mu_l2 * jnp.sum(out["mu_p"] ** 2, axis=(-2, -1))


def sample_masks(rng: np.random.Generator, n: int, cfg: Config) -> np.ndarray:
    """ADR 0015 mixture over partial coalition sizes 1..8, plus the empty mask; the full mask is never sampled.

    With the tied encoder the full mask is still trained in every row, as the posterior side of the bound.
    """

    n_cells = cfg.n_cells
    sizes = np.arange(1, n_cells)
    kernel = (n_cells - 1) / (sizes * (n_cells - sizes))
    kernel = kernel / kernel.sum()
    k = np.where(
        rng.random(n) < cfg.kernel_share,
        rng.choice(sizes, size=n, p=kernel),
        rng.choice(sizes, size=n),
    )
    k = np.where(rng.random(n) < cfg.empty_prob, 0, k)
    order = rng.random((n, n_cells)).argsort(axis=1)
    return (order < k[:, None]).astype(np.float32)


def make_loss(model: VAEACQ, cfg: Config):
    def loss_fn(params, obs, known, q_target, legal, eps, var_target=None):
        out = model.apply({"params": params}, obs, known, eps)
        recon = reconstruction_nll(out, obs, known)
        kl = belief_kl(out)
        sq = (
            jnp.sum(legal[:, None, :] * (out["q"] - q_target[:, None, :]) ** 2, axis=-1)
            / jnp.maximum(legal.sum(-1), 1.0)[:, None]
        )
        q_err = jnp.sum(jnp.exp(out["log_pi_f"]) * sq, axis=-1)
        total = recon + cfg.beta * kl + cfg.q_weight * q_err + prior_regulariser(out, cfg)
        metrics = {"loss": total.mean(), "recon": recon.mean(), "kl": kl.mean(), "q_mse": q_err.mean()}
        if var_target is not None:
            sq_v = (
                jnp.sum(legal[:, None, :] * (out["s"] - var_target[:, None, :]) ** 2, axis=-1)
                / jnp.maximum(legal.sum(-1), 1.0)[:, None]
            )
            v_err = jnp.sum(jnp.exp(out["log_pi_f"]) * sq_v, axis=-1)
            total = total + cfg.var_weight * v_err
            metrics.update(loss=total.mean(), var_mse=v_err.mean())
        return total.mean(), metrics

    return loss_fn


def init_params(model: VAEACQ, seed: int):
    c = model.cfg
    return model.init(
        jax.random.PRNGKey(seed), jnp.zeros((1, c.n_cells)), jnp.ones((1, c.n_cells)), jnp.zeros((1, c.latent_dim))
    )["params"]


def train_supervised(
    obs: np.ndarray,
    weights: np.ndarray,
    q_exact: np.ndarray,
    cfg: Config,
    log_every: int = 2000,
    var_exact: np.ndarray | None = None,
) -> tuple[VAEACQ, dict]:  # (model, params)
    """Milestone 2a: states drawn from the occupancy, Q head regressed on exact Q for legal actions.

    With ``var_exact`` (exact ``Var(G | o, a)``, NaN where illegal) and ``cfg.var_weight > 0`` the
    variance head is regressed on it as well.
    """

    rng = np.random.default_rng(cfg.seed)
    legal = (~np.isnan(q_exact)).astype(np.float32)
    q_target = np.nan_to_num(q_exact).astype(np.float32)
    var_target = None
    if var_exact is not None and cfg.var_weight > 0:
        var_target = np.maximum(np.nan_to_num(var_exact), 0.0).astype(np.float32)
    probs = weights / weights.sum()

    model = VAEACQ(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    params = init_params(model, cfg.seed)
    tx = optax.adam(learning_rate(cfg, cfg.steps))
    opt_state = tx.init(params)
    loss_fn = make_loss(model, cfg)

    @jax.jit
    def step(params, opt_state, batch, eps, var_t):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *batch, eps, var_t)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    for t in range(1, cfg.steps + 1):
        idx = rng.choice(len(obs), size=cfg.batch_size, p=probs)
        batch = (
            jnp.asarray(obs[idx], jnp.float32),
            jnp.asarray(sample_masks(rng, cfg.batch_size, cfg)),
            jnp.asarray(q_target[idx]),
            jnp.asarray(legal[idx]),
        )
        key, sub = jax.random.split(key)
        var_t = None if var_target is None else jnp.asarray(var_target[idx])
        params, opt_state, metrics = step(
            params, opt_state, batch, jax.random.normal(sub, (cfg.batch_size, cfg.latent_dim)), var_t
        )
        if log_every and t % log_every == 0:
            print(f"  step {t:6d}  " + "  ".join(f"{k}={float(v):.4f}" for k, v in metrics.items()), flush=True)
    return model, params


def mixture_q_moments(model: VAEACQ, params, log_pi, mu, ls, eps, head=VAEACQ.q):
    """``E[Q_omega]`` and ``E[Q_omega^2]`` per action under a mixture belief, stratified over components.

    ``mu``/``ls``: ``(..., M, d)``; ``eps``: ``(K, d)`` common random numbers. Returns ``(..., 9)`` twice.
    Samples are processed in chunks of ``chunk`` (sequential ``lax.map``) to bound GPU memory.
    """

    k = eps.shape[0]
    chunk = min(32, k)
    if k % chunk:
        raise ValueError(f"number of latent samples ({k}) must be a multiple of {chunk}")

    def partial(e):  # e: (chunk, d) -> sums of Q and Q^2 over this chunk, (..., M, 9) each
        z = mu[..., :, None, :] + jnp.exp(ls)[..., :, None, :] * e  # (..., M, chunk, d)
        q = model.apply({"params": params}, z, method=head)
        return q.sum(axis=-2), (q**2).sum(axis=-2)

    sums, sq_sums = jax.lax.map(partial, eps.reshape(k // chunk, chunk, -1))
    pi = jnp.exp(log_pi)[..., None]
    return jnp.sum(pi * sums.sum(0) / k, axis=-2), jnp.sum(pi * sq_sums.sum(0) / k, axis=-2)


def belief_q(model: VAEACQ, params, obs, eps):
    """The agent's Q: ``E_{z ~ full belief}[Q_omega(z, .)]`` for a batch of boards ``(B, 9) -> (B, 9)``."""

    log_pi, mu, ls = model.apply({"params": params}, obs, method=VAEACQ.full_belief)
    return mixture_q_moments(model, params, log_pi, mu, ls, eps)[0]


def coalition_masks(n_cells: int) -> np.ndarray:
    """All ``2^n`` coalitions as 0/1 rows, bitmask order (bit ``i`` = cell ``i``), empty first."""

    bits = np.arange(1 << n_cells)
    return ((bits[:, None] >> np.arange(n_cells)) & 1).astype(np.float32)


ALL_KNOWN = coalition_masks(N_CELLS)


def make_coalition_games(model: VAEACQ, legal_action_known: bool = True, blocks: int = 1, with_state: bool = False):
    """Jitted ``(params, obs, action, eps) -> (Q(x_C, a), v_mask(C))`` over all ``2^n`` coalitions.

    Every row uses the belief for its coalition; the grand coalition is the agent's own full belief
    (with the tied encoder this is the same network at the all-ones mask). ``eps`` (``K x d``) is
    shared by every coalition and component (common random numbers). Mixture moments are combined
    with the exact weights (stratified): ``Var = sum_m pi_m E_m[Q^2] - (sum_m pi_m E_m[Q])^2``.

    With ``legal_action_known`` the action's own cell is shown as known-empty in every coalition,
    matching the oracle's conditioning on "a legal" (ADR 0005); its player then gets zero credit.

    ``blocks > 1`` evaluates the coalitions in that many sequential blocks (GPU memory; 16 cells).
    ``with_state`` (tied encoder with a variance head) also returns
    ``v_state(C) = E_{z ~ e(. | x_C)}[s_omega(z, a)]``.
    """

    base = jnp.asarray(coalition_masks(model.cfg.n_cells))

    def block_moments(params, obs, action, known, eps):
        x = jnp.broadcast_to(obs.astype(jnp.float32), known.shape)
        log_pi, mu, ls = model.apply({"params": params}, x, known, method=VAEACQ.belief)
        moments = mixture_q_moments(model, params, log_pi, mu, ls, eps)
        if with_state:
            moments = (*moments, mixture_q_moments(model, params, log_pi, mu, ls, eps, head=VAEACQ.variance)[0])
        return tuple(jnp.take(t, action, axis=-1) for t in moments)

    @jax.jit
    def games(params, obs, action, eps):
        known = base.at[:, action].set(1.0) if legal_action_known else base
        blocked = known.reshape(blocks, -1, known.shape[-1])
        first, second, *state = (
            t.reshape(-1) for t in jax.lax.map(lambda k: block_moments(params, obs, action, k, eps), blocked)
        )
        x = jnp.broadcast_to(obs.astype(jnp.float32), (1, known.shape[-1]))
        if not model.cfg.tied_encoder:
            lp, m, s = model.apply({"params": params}, x, method=VAEACQ.full_belief)
            f_full, s_full = (jnp.take(t, action, axis=-1) for t in mixture_q_moments(model, params, lp, m, s, eps))
            first, second = first.at[-1].set(f_full[0]), second.at[-1].set(s_full[0])
        v_mask = jnp.maximum(second - first**2, 0.0)
        return (first, v_mask, state[0]) if with_state else (first, v_mask)

    return games


def make_entropy_game(model: VAEACQ, blocks: int = 1, legal_action_known: bool = True):
    """Jitted ``(params, obs, action, eps) -> H(C)``: entropy of the belief ``e_psi(. | o_C)`` for all coalitions.

    The representation-sharpness game. Stratified Monte Carlo over mixture components with shared ``eps``
    (``K x d``): ``H = -sum_m pi_m mean_k log e(mu_m + sigma_m eps_k)``. Only coordinate-free quantities
    enter, and an invertible linear map of ``z`` shifts ``H`` by a constant, which leaves Shapley values
    unchanged. The action's cell is known in every coalition, as in ``make_coalition_games``.
    """

    base = jnp.asarray(coalition_masks(model.cfg.n_cells))

    def block_entropy(params, obs, known, eps):
        x = jnp.broadcast_to(obs.astype(jnp.float32), known.shape)
        log_pi, mu, ls = model.apply({"params": params}, x, known, method=VAEACQ.belief)
        z = mu[:, :, None, :] + jnp.exp(ls)[:, :, None, :] * eps  # (c, M, K, d)
        log_e = jax.vmap(lambda zz: mixture_log_density(zz, log_pi, mu, ls), in_axes=1, out_axes=1)(z)  # (c, M, K)
        return -jnp.sum(jnp.exp(log_pi) * log_e.mean(-1), axis=-1)

    @jax.jit
    def game(params, obs, action, eps):
        known = base.at[:, action].set(1.0) if legal_action_known else base
        blocked = known.reshape(blocks, -1, known.shape[-1])
        return jax.lax.map(lambda k: block_entropy(params, obs, k, eps), blocked).reshape(-1)

    return game


def make_predictive_entropy_game(model: VAEACQ, k: int, r: int, blocks: int = 1, legal_action_known: bool = True):
    """Jitted ``(params, obs, action, eps, key) -> H(C)``: Shannon entropy (nats) of the belief over the hidden cells.

    The certainty game of the belief in the oracle's units: the belief over boards is the VAEAC imputation
    ``p(x_hidden | o_C) = E_{z ~ e(. | o_C)} prod_{i not in C} p(x_i | z, o_C)``, and
    ``H(C) = -E_x log p(x_hidden | o_C)``. Nested Monte Carlo: ``Z = M * k`` latents (``eps``: ``k x d``, stratified
    over the mixture components), ``r`` boards decoded per latent, each board's likelihood averaged over the same
    ``Z`` latents. The action's cell is known in every coalition, as in ``make_coalition_games``.
    """

    n = model.cfg.n_cells
    base = jnp.asarray(coalition_masks(n))

    def block(params, obs, known, eps, key):
        c = known.shape[0]
        x = jnp.broadcast_to(obs.astype(jnp.float32), known.shape)
        log_pi, mu, ls = model.apply({"params": params}, x, known, method=VAEACQ.belief)  # (c, M, d)
        m = log_pi.shape[1]
        z = (mu[:, :, None, :] + jnp.exp(ls)[:, :, None, :] * eps).reshape(c, m * k, -1)  # (c, Z, d)
        log_wz = jnp.broadcast_to(log_pi[:, :, None], (c, m, k)).reshape(c, m * k) - jnp.log(k)  # weight of each z
        xz = jnp.broadcast_to(x[:, None], (c, m * k, n))
        kz = jnp.broadcast_to(known[:, None], (c, m * k, n))
        logp = jax.nn.log_softmax(model.apply({"params": params}, z, xz, kz, method=VAEACQ.decode))  # (c, Z, n, V)
        samples = jax.random.categorical(key, logp, axis=-1, shape=(r, *logp.shape[:-1]))  # (r, c, Z, n)
        onehot = jax.nn.one_hot(samples, logp.shape[-1]).transpose(1, 2, 0, 3, 4).reshape(c, m * k * r, n, -1)
        ll = jnp.einsum("csnv,cznv,cn->csz", onehot, logp, 1.0 - known)  # log p(x_s | z') over the hidden cells
        log_px = jax.nn.logsumexp(ll + log_wz[:, None, :], axis=-1)  # (c, S)
        w_s = jnp.exp(jnp.repeat(log_wz, r, axis=1)) / r  # sample s was decoded from latent s // r
        return -jnp.sum(w_s * log_px, axis=-1)

    @jax.jit
    def game(params, obs, action, eps, key):
        known = base.at[:, action].set(1.0) if legal_action_known else base
        parts = (known.reshape(blocks, -1, n), jax.random.split(key, blocks))
        return jax.lax.map(lambda t: block(params, obs, t[0], eps, t[1]), parts).reshape(-1)

    return game


XRL_MATCHED = {  # everything matched to the xrl Direct-Q (Philipp's) configuration
    "hidden": 64,
    "depth": 2,
    "activation": "relu",
    "input_encoding": "raw",
    "lr_schedule": "constant",
    "empty_prob": 1 / 256,
}


def learning_rate(cfg: Config, steps: int):
    if cfg.lr_schedule == "constant":
        return cfg.lr
    return optax.cosine_decay_schedule(cfg.lr, steps, alpha=0.05)


def size_tag(cfg: Config) -> str:
    """Checkpoint suffix: "" for the default, "_xrl" for the xrl-matched preset, else the network size."""

    default = Config()
    if all(getattr(cfg, k) == v for k, v in XRL_MATCHED.items()):
        return "_xrl"
    if all(getattr(cfg, k) == v for k, v in XRL_MATCHED.items() if k != "hidden"):
        return f"_xrl{cfg.hidden}"  # xrl-matched at another width (network-size sweep)
    if (cfg.hidden, cfg.depth, cfg.activation) == (default.hidden, default.depth, default.activation):
        return ""
    return f"_h{cfg.hidden}x{cfg.depth}{cfg.activation}"
