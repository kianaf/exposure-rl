"""Observational uncertainty through the decoder's imputations, for Table 1 (2026-11-08, review point 1).

The latent route reads v_obs(C) = Var_{z ~ e(.|o_C)}[Q_omega(z, a)]; it mixes the spread over completions with latent
noise for the same completion and is > 0 at C = N. The imputation route reads the definition directly:

    v_obs(C) = Var[Q_agent(O', a)],   z ~ e(. | o_C),  O' ~ p_theta(. | z, o_C)  (visible cells kept),

with Q_agent the agent's own full-information Q (exposure_rl.evaluate.full_information_q, same random numbers), so
v_obs(N) = 0 exactly. Latents: k per mixture component, weighted by pi_m / k; r boards decoded per latent.
Q is evaluated once per distinct decoded board (boards are keyed as integers, cell value v -> digit v + 1).

Scored like Table 1 (scripts/replay_matching_eval.py, score_rank_columns.py): same observations and actions, exact
v_obs game of the agent's own Q on the optimal policy's occupancy, Shapley values of the reduction, per-board Spearman
rho summarized by the mean (and median) over boards.

Usage: uv run python scripts/imputation_vobs.py [--domains random pm05 ms] [--k 8] [--r 4]
Writes results/imputation_vobs.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, spearmanr

from exposure_rl import coalitions
from exposure_rl.vaeac_q import VAEACQ, belief_q, coalition_masks

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag, abl = _module("vobs_diagnostic"), _module("replay_matching_eval")
EVAL_EPS_KEY, EVAL_SAMPLES = 12345, 64  # as exposure_rl.evaluate.full_information_q


class QCache:
    """The agent's full-information Q for decoded boards, computed once per distinct board."""

    def __init__(self, model, params, n, n_values):
        self.model, self.params, self.n, self.v = model, params, n, n_values
        self.pow = (n_values ** np.arange(n)).astype(np.int64)
        eps = jax.random.normal(jax.random.PRNGKey(EVAL_EPS_KEY), (EVAL_SAMPLES, model.cfg.latent_dim))
        self.q = jax.jit(lambda o: belief_q(model, params, o, eps))
        self.keys = np.zeros(0, np.int64)
        self.vals = np.zeros((0, n), np.float64)

    def evaluate(self, keys: np.ndarray) -> np.ndarray:
        """Q for the boards with these (distinct) keys, no caching."""

        digits = (keys[:, None] // self.pow) % self.v
        boards = digits.astype(np.float32) - 1.0
        size = 4096  # fixed batch shape (padded), so the jitted Q compiles once
        pad = (-len(boards)) % size
        boards = np.concatenate([boards, np.repeat(boards[:1], pad, axis=0)])
        out = np.concatenate([np.asarray(self.q(jnp.asarray(c)), np.float64) for c in boards.reshape(-1, size, self.n)])
        return out[: len(keys)]

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        new = np.setdiff1d(np.unique(keys), self.keys, assume_unique=False)
        if len(new):
            digits = (new[:, None] // self.pow) % self.v  # value index = v + 1
            boards = digits.astype(np.float32) - 1.0
            q = np.concatenate(
                [np.asarray(self.q(jnp.asarray(c)), np.float64) for c in np.array_split(boards, max(1, len(boards) // 1024))]
            )
            order = np.argsort(np.concatenate([self.keys, new]))
            self.keys = np.concatenate([self.keys, new])[order]
            self.vals = np.concatenate([self.vals, q])[order]
        return self.vals[np.searchsorted(self.keys, keys)]


def make_decoder(model, k: int, r: int, n: int):
    @jax.jit
    def decode(params, obs, known, eps, key):
        c = known.shape[0]
        x = jnp.broadcast_to(obs.astype(jnp.float32), known.shape)
        log_pi, mu, ls = model.apply({"params": params}, x, known, method=VAEACQ.belief)
        m = log_pi.shape[1]
        z = (mu[:, :, None, :] + jnp.exp(ls)[:, :, None, :] * eps).reshape(c, m * k, -1)
        w = jnp.exp(jnp.broadcast_to(log_pi[:, :, None], (c, m, k)).reshape(c, m * k)) / k
        xz = jnp.broadcast_to(x[:, None], (c, m * k, n))
        kz = jnp.broadcast_to(known[:, None], (c, m * k, n))
        logp = jax.nn.log_softmax(model.apply({"params": params}, z, xz, kz, method=VAEACQ.decode))
        draws = jax.random.categorical(key, logp, axis=-1, shape=(r, *logp.shape[:-1]))
        idx = jnp.where(kz[None] > 0, (xz[None] + 1).astype(jnp.int32), draws)  # (r, c, mk, n), value index
        return idx.transpose(1, 2, 0, 3).reshape(c, m * k * r, n), jnp.repeat(w, r, axis=1) / r

    return decode


def vobs_games(model, params, obs, actions, n, k, r, seed, block=4096):
    cache = QCache(model, params, n, model.cfg.n_values)
    decode = make_decoder(model, k, r, n)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (k, model.cfg.latent_dim))
    pw = cache.pow
    rows = []
    for j, (o, a) in enumerate(zip(obs, actions)):
        masks = coalition_masks(n)
        masks[:, int(a)] = 1.0
        keys, weights = [], []
        for b, m in enumerate(np.array_split(masks, max(1, len(masks) // block))):
            idx, w = decode(params, jnp.asarray(o), jnp.asarray(m), eps, jax.random.PRNGKey(10_000 * seed + 100 * j + b))
            keys.append((np.asarray(idx).astype(np.int64) * pw).sum(-1))
            weights.append(np.asarray(w, np.float64))
        keys, w = np.concatenate(keys), np.concatenate(weights)  # (2^n, samples)
        # Q once per distinct decoded board of this observation (a global cache is too slow for 2^16 coalitions)
        uniq, inv = np.unique(keys.ravel(), return_inverse=True)
        q = cache.evaluate(uniq)[:, int(a)][inv].reshape(keys.shape)
        mean = (w * q).sum(1)
        rows.append(np.maximum((w * q**2).sum(1) - mean**2, 0.0))
        if j % 50 == 0:
            print(f"    board {j}/{len(obs)}, {len(uniq)} distinct decoded boards for this one", flush=True)
    return np.stack(rows)


def score(v_m, v_e, n):
    interior = coalitions.coalition_sizes(n) < n
    phi_m, phi_e = coalitions.uncertainty_reduction_shapley(v_m, n), coalitions.uncertainty_reduction_shapley(v_e, n)
    rhos = [spearmanr(x, y).statistic for x, y in zip(phi_m, phi_e) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return {
        "phi_obs_spearman_mean": float(np.mean(rhos)),
        "phi_obs_spearman_median": float(np.median(rhos)),
        "v_obs_pearson_interior": float(pearsonr(v_m[:, interior].ravel(), v_e[:, interior].ravel()).statistic),
        "v_obs_full_coalition": float(v_m[:, -1].mean()),
        "v_obs_empty_model_vs_exact": [float(v_m[:, 0].mean()), float(v_e[:, 0].mean())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["random", "pm05", "ms"])
    parser.add_argument("--k", type=int, default=8, help="latents per mixture component")
    parser.add_argument("--r", type=int, default=4, help="boards decoded per latent")
    args = parser.parse_args()
    path = ROOT / "results" / "imputation_vobs.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in args.domains:
        points, weights, obs, actions, legal_of, n, load = diag.setting(name)
        res = {}
        for seed in range(3):
            model, params = load(seed)
            q_agent = np.asarray(
                np.concatenate([np.asarray(QCache(model, params, n, model.cfg.n_values).q(jnp.asarray(c))) for c in np.array_split(points.astype(np.float32), max(1, len(points) // 1024))]),
                np.float64,
            )
            _, v_e = abl.own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
            print(f"{name} seed {seed}: exact games done, decoding {len(obs)} boards x 2^{n} coalitions", flush=True)
            v_m = vobs_games(model, params, obs, actions, n, args.k, args.r, seed)
            res[seed] = score(v_m, v_e, n) | {"k": args.k, "r": args.r}
            print(name, seed, json.dumps(res[seed]), flush=True)
        out[name] = res
        path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
