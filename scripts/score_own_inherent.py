"""phi^inh of every Table 1 method against the exact return variance of its *own* greedy policy (2026-11-15).

Table 1 scored v_inh against the optimal policy's variance, E[Var_opt(G | O, a) | o_C]. Here the reference is
E[Var_pi(G | O, a) | o_C] with pi the method's own greedy policy on its full-information Q (exposure_rl.policy_eval;
opponent moves and mine layouts stay exact). Occupancy, scored observations, actions, coalitions and each method's
model games are those of Table 1 (score_rank_columns.py for EXPOSURE, certainty_baselines.py for the baselines),
so only the reference changes. ``--reference optimal`` rebuilds the old reference as a check of the pipeline.
``--reference agent`` (2026-11-16, consistent with phi^Q and phi^obs): E[s(O, a) | o_C], the exact conditional mean over
the occupancy of the agent's own full-observation variance head (EXPOSURE: E_z[s_omega] under the full belief, as its
full-observation Q). The model games are cached in results/own_inherent/vm/, so a new reference needs no recomputation.

The DQN + surrogate agent acts with its DQN, so its policy is the DQN's greedy one and its games are the surrogate's.

Usage: uv run python scripts/score_own_inherent.py --domain random|pm05|ms --method exposure|ma|surr --seed 0
       [--reference own|optimal|agent] [--width 512] -> results/own_inherent/<domain>_<method>[_w512]_s<seed>[_optimal].json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from exposure_rl import coalitions, masked_dqn
from exposure_rl.evaluate import OracleReference, full_information_q, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import exact_games, load_oracle, scored_indices
from exposure_rl.oracle import certainty_games
from exposure_rl.policy_eval import GreedyPolicy, PolicyMinesweeper, PolicyTicTacToe
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, make_coalition_games, mixture_q_moments, size_tag

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "own_inherent"
_spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts" / "certainty_baselines.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)
TAG = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": 256}))


def exposure(domain: str, seed: int, obs: np.ndarray, actions: np.ndarray, n: int):
    """(full-information Q and s on a batch of observations, v_inh games of the scored observations), as Table 1."""

    if domain == "ms":
        cfg = dataclasses.replace(Config(), seed=seed, n_values=10, n_cells=n, var_weight=50.0, **{**XRL_MATCHED, "hidden": 256})
        model = VAEACQ(cfg)
        path = ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms4x4_3_var{size_tag(cfg)}_td_m10_s{seed}.msgpack"
        params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
        games = make_coalition_games(model, blocks=(1 << n) // 2048, with_state=True)
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (128, model.cfg.latent_dim))
    else:
        _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, TAG, domain, td_var=True))
        games = make_coalition_games(model, with_state=True)
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (256, model.cfg.latent_dim))
    eps_full = jax.random.normal(jax.random.PRNGKey(12345), (64, model.cfg.latent_dim))  # as full_information_q
    s_full = jax.jit(lambda x: mixture_q_moments(model, params, *model.apply({"params": params}, x, method=VAEACQ.full_belief), eps_full, head=VAEACQ.variance)[0])
    s_fn = lambda x: np.concatenate([np.asarray(s_full(jnp.asarray(c, jnp.float32)), np.float64) for c in np.array_split(np.asarray(x), max(1, len(x) // 256))])
    v_m = lambda: np.stack([np.asarray(games(params, jnp.asarray(o, jnp.float32), int(a), eps)[2]) for o, a in zip(obs, actions)])
    return lambda x: full_information_q(model, params, np.asarray(x, np.float32)), s_fn, v_m


def baseline(domain: str, method: str, seed: int, obs: np.ndarray, actions: np.ndarray, n: int, width: int = 256):
    """The same for MA-DQN (``ma``) and DQN + surrogate (``surr``), from the certainty_baselines.py checkpoints."""

    cfg = masked_dqn.DQNConfig(n_cells=n, hidden=width, mask_aware=method == "ma", var_head=True, moment_head=method == "ma", seed=seed)
    net, params = masked_dqn.init(cfg)
    stem = cb.MODELS / f"{domain}_{method}{'2' * (method == 'surr')}{f'_w{width}' * (width != 256)}_noclip_s{seed}"  # the unclipped Table 1 runs
    params = serialization.from_bytes(params, Path(f"{stem}.msgpack").read_bytes())
    explainer, eparams = net, params
    if method == "surr":
        explainer, eparams = masked_dqn.init(dataclasses.replace(cfg, mask_aware=True, seed=seed + 777))
        eparams = serialization.from_bytes(eparams, Path(f"{stem}_surrogate.msgpack").read_bytes())
    q = lambda x: np.asarray(net.apply(params, jnp.asarray(x, jnp.float32), jnp.ones_like(jnp.asarray(x, jnp.float32)))[:, 0], np.float64)
    s = lambda x: np.asarray(net.apply(params, jnp.asarray(x, jnp.float32), jnp.ones_like(jnp.asarray(x, jnp.float32)))[:, 1], np.float64)
    s_fn = lambda x: np.concatenate([s(c) for c in np.array_split(np.asarray(x), max(1, len(x) // 4096))])
    v_m = lambda: np.stack([cb.heads_all_coalitions(explainer, eparams, o, int(a), n)[:, 1] for o, a in zip(obs, actions)])
    return q, s_fn, v_m


def reference_tictactoe(domain: str, q_fn, own: bool) -> np.ndarray:
    ref = OracleReference(domain)
    oracle = PolicyTicTacToe(domain, GreedyPolicy(lambda o: q_fn(np.asarray(o)[None])[0])) if own else ref.oracle
    return np.stack([certainty_games(oracle, ref.dist, p.observation, int(a)).opp for p, a in zip(ref.dist.points, ref.actions)])


def reference_minesweeper(q_fn, own: bool) -> np.ndarray:
    g = Geometry(4, 4, 3)
    data = load_oracle(g, ROOT / "results" / "cache")
    points, weights = data["points"], data["weights"]
    actions = np.nanargmax(data["q"], axis=1)
    scored, _ = scored_indices(weights)
    if not own:
        return np.stack([exact_games(data, int(i))[2] for i in scored])
    oracle = PolicyMinesweeper(g, GreedyPolicy(lambda o: q_fn(np.asarray(o)[None])[0]))
    keys = [tuple(int(v) for v in p) for p in points]
    v_e, t = [], time.time()
    for k, i in enumerate(scored):  # as score_own_state.py
        a = int(actions[i])
        legal = np.flatnonzero(points[:, a] == HIDDEN)
        moments = np.array([oracle._moments(keys[j], a) for j in legal])
        q_pi, var_pi = moments[:, 0], np.maximum(moments[:, 1] - moments[:, 0] ** 2, 0.0)
        own_q, own_g2 = oracle._moments(keys[i], a)
        v_e.append(coalitions.certainty_games(points[legal], weights[legal], q_pi, var_pi, points[i], own_q, max(own_g2 - own_q**2, 0.0))[2])
        if k % 50 == 0:
            print(f"  {k}/{len(scored)} observations, {time.time() - t:.0f}s", flush=True)
    return np.stack(v_e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("random", "pm05", "ms"), required=True)
    parser.add_argument("--method", choices=("exposure", "ma", "surr"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", choices=("own", "optimal", "agent"), default="own")
    parser.add_argument("--width", type=int, default=256, help="MA-DQN capacity check: 512")
    args = parser.parse_args()
    points, weights, obs, actions, legal_of, n, _, _, _ = cb.setting(args.domain)
    load = exposure if args.method == "exposure" else (lambda *a: baseline(a[0], args.method, *a[1:], width=args.width))
    q_fn, s_fn, games = load(args.domain, args.seed, obs, actions, n)
    stem = f"{args.domain}_{args.method}{f'_w{args.width}' * (args.width != 256)}_s{args.seed}"
    cache = OUT / "vm" / f"{stem}.npy"
    if cache.exists():
        v_m = np.load(cache)
    else:
        v_m = games()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, v_m)
    own = args.reference == "own"
    if args.reference == "agent":  # the agent's own full-observation s, conditioned exactly like Q for phi^Q / phi^obs
        s_all = s_fn(points)
        v_e = np.stack([
            coalitions.conditional_moments(points[legal_of(int(a))], weights[legal_of(int(a))], s_all[legal_of(int(a)), int(a)][:, None], o)[:, 0]
            for o, a in zip(obs, actions)
        ])
    else:
        v_e = reference_minesweeper(q_fn, own) if args.domain == "ms" else reference_tictactoe(args.domain, q_fn, own)
    result = {
        "domain": args.domain,
        "method": args.method,
        "seed": args.seed,
        "reference": args.reference,
        "v_inh": cb.score(v_m, v_e, n, reduction=True),
        "v_inh_full_coalition_exact": float(v_e[:, -1].mean()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{stem}{ {'own': '', 'optimal': '_optimal', 'agent': '_agent'}[args.reference] }.json"
    (OUT / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
