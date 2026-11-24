"""v_state of our TD agents against the exact v_state of their *own* policy (policy evaluation), seed by seed.

Table 1 so far scores v_state against the optimal policy's return variance. Here the reference is the agent's own
outcome variance, Var_pi(G | o, a) with pi the agent's greedy policy (exposure_rl.policy_eval), conditioned with
the same occupancy, observations, actions and coalitions as before:

  v_state^pi(C) = E[Var_pi(G | O, a) | o_C],   grand coalition: Var_pi(G | o, a).

Scores: r = Pearson over all coalitions, rho = median Spearman of the uncertainty-reduction Shapley values (as the
v_state columns of Table 1).

Usage: uv run python scripts/score_own_state.py --domain {tictactoe,pm05,minesweeper} [--width 256] [--seed S]
       -> results/own_state<tag>_<domain>_s<S>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, spearmanr

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, full_information_q, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_oracle, scored_indices
from exposure_rl.oracle import certainty_games
from exposure_rl.policy_eval import GreedyPolicy, PolicyMinesweeper, PolicyTicTacToe
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, make_coalition_games, size_tag

ROOT = Path(__file__).resolve().parents[1]


def scores(vs_m: np.ndarray, vs_e: np.ndarray, n: int) -> dict:
    phi_m = coalitions.uncertainty_reduction_shapley(vs_m, n)
    phi_e = coalitions.uncertainty_reduction_shapley(vs_e, n)
    rhos = [spearmanr(a, b).statistic for a, b in zip(phi_m, phi_e) if np.std(a) > 1e-9 and np.std(b) > 1e-9]
    return {
        "own_policy_v_state_pearson": float(pearsonr(vs_m.ravel(), vs_e.ravel()).statistic),
        "own_policy_phi_state_spearman_median": float(np.median(rhos)),
        "own_policy_phi_state_spearman_mean": float(np.mean(rhos)),
    }


def q_function(model, params):
    return lambda obs: full_information_q(model, params, np.asarray(obs, np.float32)[None])[0]


def tictactoe(opponent: str, width: int, seed: int) -> dict:
    ref = OracleReference(opponent)
    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": width}))
    _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, tag, opponent, td_var=True))
    oracle = PolicyTicTacToe(opponent, GreedyPolicy(q_function(model, params)))
    vs_e = np.stack(
        [certainty_games(oracle, ref.dist, p.observation, int(a)).opp for p, a in zip(ref.dist.points, ref.actions)]
    )
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (256, model.cfg.latent_dim))
    games_s = make_coalition_games(model, with_state=True)
    vs_m = np.stack([np.asarray(games_s(params, jnp.asarray(o), int(a), eps)[2]) for o, a in zip(ref.obs, ref.actions)])
    return scores(vs_m, vs_e, 9)


def minesweeper(width: int, seed: int) -> dict:
    g = Geometry(4, 4, 3)
    data = load_oracle(g, ROOT / "results" / "cache")
    points, weights = data["points"], data["weights"]
    actions = np.nanargmax(data["q"], axis=1)
    scored, _ = scored_indices(weights)
    cfg = dataclasses.replace(
        Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **{**XRL_MATCHED, "hidden": width}
    )
    model = VAEACQ(cfg)
    path = ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms4x4_3_var{size_tag(cfg)}_td_m10_s{seed}.msgpack"
    params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
    oracle = PolicyMinesweeper(g, GreedyPolicy(q_function(model, params)))
    keys = [tuple(int(v) for v in p) for p in points]
    vs_e, t = [], time.time()
    for k, i in enumerate(scored):
        a = int(actions[i])
        legal = np.flatnonzero(points[:, a] == HIDDEN)
        moments = np.array([oracle._moments(keys[j], a) for j in legal])  # (E_pi[G], E_pi[G^2]) per support point
        q_pi, var_pi = moments[:, 0], np.maximum(moments[:, 1] - moments[:, 0] ** 2, 0.0)
        own_q, own_g2 = oracle._moments(keys[i], a)
        _, _, v_state = coalitions.certainty_games(
            points[legal], weights[legal], q_pi, var_pi, points[i], own_q, max(own_g2 - own_q**2, 0.0)
        )
        vs_e.append(v_state)
        if k % 25 == 0:
            print(f"  {k}/{len(scored)} observations, {time.time() - t:.0f}s", flush=True)
    vs_e = np.stack(vs_e)
    games_s = make_coalition_games(model, blocks=(1 << g.cells) // 2048, with_state=True)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (128, cfg.latent_dim))
    vs_m = np.stack(
        [np.asarray(games_s(params, jnp.asarray(points[i], jnp.float32), int(actions[i]), eps)[2]) for i in scored]
    )
    return scores(vs_m, vs_e, g.cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "pm05", "minesweeper"), required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": args.width}))
    if args.domain == "minesweeper":
        result = minesweeper(args.width, args.seed)
    else:
        result = tictactoe("random" if args.domain == "tictactoe" else "pm05", args.width, args.seed)
    out = ROOT / "results" / f"own_state{tag}_{args.domain}_s{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(args.domain, args.seed, result, flush=True)


if __name__ == "__main__":
    main()
