"""Baselines with certainty heads, scored like Table 1 (2026-11-07).

Trains (exposure_rl.masked_dqn) and scores one baseline on one domain and seed:
  --method ma     MA-DQN with q, s and m2 heads (masked inputs, full-information targets)
  --method surr   standard DQN with q and s heads + masked surrogate of its q, s and q^2 (epsilon-greedy occupancy)
Games per scored observation, every coalition, the action's cell always visible:
  predicted outcome  q(o_C)                 vs exact E[Q_own(O, a) | o_C]
  v_obs              max(m2(o_C) - q^2, 0)  vs exact Var[Q_own(O, a) | o_C]
  v_inh              s(o_C)                 vs exact E[Var_opt(G | O, a) | o_C]  (optimal policy, as Table 1)
Q_own is the agent's own full-information Q (MA-DQN: itself; surrogate: its DQN). Scores: median Spearman rho of the
Shapley values (reductions for the uncertainty games) and Pearson r of interior coalition values, plus wins.
Budgets, scored observations and explained actions are those of Table 1 (scripts/reproduce_paper.sh,
replay_matching_eval.py).

Usage: uv run python scripts/certainty_baselines.py --domain random|pm05|ms --method ma|surr --seed 0 [--reuse] [--q-only] [--width 512]
Writes results/certainty_baselines/<domain>_<method>_s<seed>.json (+ checkpoints in results/models/certainty_baselines/).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import pearsonr

from exposure_rl import coalitions, masked_dqn
from exposure_rl.evaluate import OracleReference
from exposure_rl.exact_eval import exact_outcomes, greedy
from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle
from exposure_rl.minesweeper_exact import load_oracle, scored_indices
from exposure_rl.td import TDConfig

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "replay_matching_eval.py")
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)
OUT = ROOT / "results" / "certainty_baselines"
MODELS = ROOT / "results" / "models" / "certainty_baselines"
UPDATES = {  # matched to Table 1 (reproduce_paper.sh)
    "random": (165290, 165847, 165839),
    "pm05": (186075, 186720, 186324),
    "ms": (157853, 158532, 157568),
}


def setting(domain: str):
    """(points, weights, scored obs, actions, legal_of, n, optimal-policy variance, play, TDConfig kwargs)."""

    if domain in ("random", "pm05"):
        ref = OracleReference(domain)
        var_opt = np.stack([m[1] - m[0] ** 2 for m in (ref.oracle.action_moments(p.observation) for p in ref.dist.points)])
        legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])

        def play(q_fn):
            return {"agent_" + k: float(v) for k, v in exact_outcomes(greedy(q_fn), domain).items() if isinstance(v, (int, float))}

        td = {"opponent": domain, "anneal_opponent": domain == "pm05"}
        return ref.obs.astype(np.float32), ref.w, ref.obs.astype(np.float32), ref.actions, legal_of, 9, var_opt, play, td
    g = Geometry(4, 4, 3)
    oracle = load_oracle(g, ROOT / "results" / "cache")
    points = oracle["points"].astype(np.float32)
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(oracle["weights"])
    rules = MinesweeperOracle(g)
    legal_of = lambda a: oracle["points"][:, a] == HIDDEN

    def play(q_fn):
        def policy(obs, legal):
            q = np.asarray(q_fn(obs))
            return int(np.flatnonzero(legal)[np.argmax(q[legal])])

        return {"agent_" + k: v for k, v in abl.ms_script.exact_policy_outcome(rules, policy).items()}

    var_opt = oracle["g2"] - oracle["q"] ** 2
    td = {"domain": "minesweeper", "rows": 4, "columns": 4, "mines": 3}
    return points, oracle["weights"], points[scored], actions[scored], legal_of, 16, var_opt, play, td


def heads_all_coalitions(net, params, o, a, n, chunk=16384):
    masks = coalitions_masks(n)
    masks[:, a] = 1.0
    outs = []
    for m in np.array_split(masks, max(1, len(masks) // chunk)):
        out = np.asarray(net.apply(params, jnp.broadcast_to(jnp.asarray(o), m.shape), jnp.asarray(m)), np.float64)
        outs.append(out[:, :, a])
    return np.concatenate(outs)  # (2^n, 3): q, s, m2


def coalitions_masks(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float32)


def score(v_m, v_e, n, reduction):
    interior = coalitions.coalition_sizes(n) < n
    shap = coalitions.uncertainty_reduction_shapley if reduction else coalitions.shapley
    ok = np.isfinite(v_e).all(1)
    return {
        "rho": abl.median_spearman(shap(v_m[ok], n), shap(v_e[ok], n)),
        "rho_mean": abl.mean_spearman(shap(v_m[ok], n), shap(v_e[ok], n)),
        "r": float(pearsonr(v_m[ok][:, interior].ravel(), v_e[ok][:, interior].ravel()).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("random", "pm05", "ms"), required=True)
    parser.add_argument("--method", choices=("ma", "surr"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=None, help="override the Table 1 budget (smoke tests)")
    parser.add_argument("--surrogate-states", type=int, default=200_000)
    parser.add_argument("--surrogate-steps", type=int, default=50_000)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--width", type=int, default=256, help="hidden width (capacity check: 512)")
    parser.add_argument("--q-only", action="store_true", help="no s / m2 heads: reproduces the Table 1 baseline (check)")
    args = parser.parse_args()
    points, weights, obs, actions, legal_of, n, var_opt, play, td_kw = setting(args.domain)
    cfg = masked_dqn.DQNConfig(
        n_cells=n,
        hidden=args.width,
        mask_aware=args.method == "ma",
        var_head=not args.q_only,
        moment_head=not args.q_only and args.method == "ma",  # a standard DQN needs no m2 head (diverged in
        # Minesweeper, 2026-11-08): its surrogate regresses q^2 of the DQN's own q
        seed=args.seed,
    )
    td = dataclasses.replace(TDConfig(), updates=args.updates or UPDATES[args.domain][args.seed], **td_kw)
    tag = f"{args.domain}_{args.method}{'2' * (args.method == 'surr')}{'_qonly' * args.q_only}{f'_w{args.width}' * (args.width != 256)}_s{args.seed}" + (f"_u{args.updates}" if args.updates else "")
    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    net, params = masked_dqn.init(cfg)
    ckpt = MODELS / f"{tag}.msgpack"
    history = []
    if args.reuse and ckpt.exists():
        params = serialization.from_bytes(params, ckpt.read_bytes())
    else:
        print(f"training {args.method} on {args.domain}, seed {args.seed}, {td.updates} updates", flush=True)
        net, params, history = masked_dqn.train(cfg, td)
        ckpt.write_bytes(serialization.to_bytes(params))

    full_q = lambda x: np.asarray(net.apply(params, jnp.asarray(x), jnp.ones_like(jnp.asarray(x)))[:, 0], np.float64)
    explainer, eparams = net, params
    if args.method == "surr":
        sckpt = MODELS / f"{tag}_surrogate.msgpack"
        snet, sparams = masked_dqn.init(dataclasses.replace(cfg, mask_aware=True, seed=args.seed + 777))
        if args.reuse and sckpt.exists():
            sparams = serialization.from_bytes(sparams, sckpt.read_bytes())
        else:
            states = masked_dqn.occupancy_states(net, params, cfg, td, args.surrogate_states, seed=args.seed)
            snet, sparams = masked_dqn.fit_surrogate(
                net, params, cfg, states, args.surrogate_steps, seed=args.seed, moment=not args.q_only
            )
            sckpt.write_bytes(serialization.to_bytes(sparams))
        explainer, eparams = snet, sparams

    q_agent = np.concatenate([full_q(c) for c in np.array_split(points, 32)])
    q_e, vobs_e = abl.own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
    vinh_e = np.stack(
        [
            coalitions.conditional_moments(points[legal_of(int(a))], weights[legal_of(int(a))], var_opt[legal_of(int(a)), int(a)][:, None], o)[:, 0]
            for o, a in zip(obs, actions)
        ]
    )
    g = np.stack([heads_all_coalitions(explainer, eparams, o, int(a), n) for o, a in zip(obs, actions)])  # (B, 2^n, 3)
    q_m, s_m, m2_m = g[..., 0], g[..., 1], g[..., 2]
    vobs_m = np.maximum(m2_m - q_m**2, 0.0)
    summary = {
        "domain": args.domain,
        "method": args.method,
        "seed": args.seed,
        "updates": td.updates,
        "predicted_outcome": score(q_m, q_e, n, reduction=False),
        "v_obs": score(vobs_m, vobs_e, n, reduction=True),
        "v_inh": score(s_m, vinh_e, n, reduction=True),
        "v_obs_negative_share": float((m2_m - q_m**2 < 0).mean()),  # before clipping: difference of two regressions
        "v_obs_full_coalition": float(vobs_m[:, -1].mean()),
        **play(lambda o: full_q(np.asarray(o, np.float32)[None])[0]),
        "history": history,
    }
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
