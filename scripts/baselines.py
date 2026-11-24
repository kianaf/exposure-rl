"""Monte Carlo certainty explainers vs EXPOSURE's belief games, all scored against the exact v_mask.

Baselines use the agent's own full-information Q (the same trained model), so only the treatment of
hidden features differs (see ``exposure_rl.baselines``):

    empirical conditional, replay of R observations drawn from the occupancy (R = 1000, 10000)
    marginal imputation with 16 replay boards per coalition

Usage:
    uv run python scripts/baselines.py --domain tictactoe --seed 0      # 2a and TD agents, M = 10
    uv run python scripts/baselines.py --domain minesweeper --seed 0    # 4x4 / 3, variance-head model
Writes results/baselines_<domain>_s<seed>.json.
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
from scipy.stats import spearmanr

from exposure_rl import coalitions
from exposure_rl.baselines import empirical_conditional_games, make_marginal_games
from exposure_rl.evaluate import (
    OracleReference,
    compare,
    full_information_q,
    load_checkpoint,
    model_games,
    tictactoe_checkpoint,
)
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_games, load_oracle, scored_indices, tag
from exposure_rl.oracle import certainty_games
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, make_coalition_games, size_tag

ROOT = Path(__file__).resolve().parents[1]
SIZE: dict = {}  # network size from the command line (default: 3 x 256 GELU)
OPPONENT = ["random"]
ALL_TD = [False]  # --all-td: score only the TD agents (variance head on its TD target)  # tic-tac-toe opponent spec
REPLAYS = (1000, 10000)
N_IMPUTE = 16


def score(w, v, v_e, phi_e) -> dict:
    scores, _ = compare(w, np.zeros_like(v), v, np.zeros_like(v), v_e, phi_e)
    scores.pop("q_coalition_mae")
    scores.pop("calibration_by_coalition_size_model_vs_exact")
    return scores


def run_baselines(model, params, all_obs, all_w, q_full, actions, scored, w_scored, v_e, phi_e, hidden, seed):
    """Scores of both baselines on the scored observations; ``q_full``: agent's Q for every occupancy point."""

    rng = np.random.default_rng(seed)
    n = all_obs.shape[1]
    out = {}
    for r in REPLAYS:
        idx = rng.choice(len(all_obs), size=r, p=all_w / all_w.sum())
        t = time.time()
        rows = [
            empirical_conditional_games(all_obs[idx], q_full[idx, int(a)], all_obs[i], int(a), hidden)
            for i, a in zip(scored, actions[scored])
        ]
        v = np.stack([x[0] for x in rows])
        out[f"empirical_conditional_R{r}"] = {
            **score(w_scored, v, v_e, phi_e),
            "fallback_share": float(np.mean([x[1] for x in rows])),
        }
        print(f"empirical R={r}: {time.time() - t:.0f}s", flush=True)

    idx = rng.choice(len(all_obs), size=REPLAYS[-1], p=all_w / all_w.sum())
    games = make_marginal_games(model, N_IMPUTE, blocks=max(1, (1 << n) // 1024))
    eps = jax.random.normal(jax.random.PRNGKey(seed + 7), (16, model.cfg.latent_dim))
    t = time.time()
    v = np.stack(
        [
            np.asarray(
                games(
                    params,
                    jnp.asarray(all_obs[i], jnp.float32),
                    int(a),
                    jnp.asarray(all_obs[rng.choice(idx, size=N_IMPUTE)], jnp.float32),
                    eps,
                )
            )
            for i, a in zip(scored, actions[scored])
        ]
    )
    out["marginal_imputation"] = score(w_scored, v, v_e, phi_e)
    print(f"marginal: {time.time() - t:.0f}s", flush=True)
    return out


def tictactoe(seed: int) -> dict:
    ref = OracleReference(OPPONENT[0])
    result = {}
    for kind in ("td",) if ALL_TD[0] else ("vaeac_q", "td"):
        suffix = size_tag(dataclasses.replace(Config(), **SIZE))
        _, model, params = load_checkpoint(tictactoe_checkpoint(kind, seed, suffix, OPPONENT[0], td_var=ALL_TD[0]))
        q_full = full_information_q(model, params, ref.obs)
        q_full = np.where(np.isnan(ref.q_exact), np.nan, q_full)
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (256, model.cfg.latent_dim))
        _, v_m = model_games(make_coalition_games(model), params, ref.obs, ref.actions, eps)
        scored = np.arange(len(ref.obs))
        state = {}
        if model.cfg.var_weight > 0:  # variance head: score v_state against the oracle's exact E[Var(G | O, a) | o_C]
            exact = [
                certainty_games(ref.oracle, ref.dist, p.observation, int(a)).opp
                for p, a in zip(ref.dist.points, ref.actions)
            ]
            vs_e = np.stack(exact)
            games_s = make_coalition_games(model, with_state=True)
            vs_m = np.stack(
                [np.asarray(games_s(params, jnp.asarray(o), int(a), eps)[2]) for o, a in zip(ref.obs, ref.actions)]
            )
            phi_s_m = coalitions.uncertainty_reduction_shapley(vs_m, 9)
            phi_s_e = coalitions.uncertainty_reduction_shapley(vs_e, 9)
            big = np.abs(phi_s_e) > 1e-3
            state = {
                "v_state_pearson": float(np.corrcoef(vs_m.ravel(), vs_e.ravel())[0, 1]),
                "phi_state_sign_agreement": float((np.sign(phi_s_m) == np.sign(phi_s_e))[big].mean()),
                "phi_state_spearman_median": float(
                    np.median(
                        [
                            spearmanr(x, y).statistic
                            for x, y in zip(phi_s_m, phi_s_e)
                            if np.std(x) > 1e-9 and np.std(y) > 1e-9
                        ]
                    )
                ),
                "phi_state_exact_positive_share": float((phi_s_e[big] > 0).mean()),
            }
        result[kind] = {
            "exposure_rl": score(ref.w, v_m, ref.v_e, ref.phi_e) | state,
            **run_baselines(
                model, params, ref.obs, ref.w, q_full, ref.actions, scored, ref.w, ref.v_e, ref.phi_e, None, seed
            ),
        }
    return result


def minesweeper(seed: int) -> dict:
    g = Geometry(4, 4, 3)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    obs = oracle["points"].astype(np.float32)
    w = oracle["weights"] / oracle["weights"].sum()
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, w_scored = scored_indices(oracle["weights"])
    games = load_games(g, oracle, np.unique(scored), cache)
    keep_row = {int(i): k for k, i in enumerate(games["keep"])}
    v_e = games["v_mask_keep"][[keep_row[int(i)] for i in scored]]
    phi_e = games["phi_mask"][scored]

    cfg = dataclasses.replace(Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **SIZE)
    model = VAEACQ(cfg)
    path = (
        ROOT
        / "results"
        / "models"
        / "minesweeper"
        / f"vaeac_q_ms{tag(g)}_var{size_tag(cfg)}{'_td' * ALL_TD[0]}_m10_s{seed}.msgpack"
    )
    params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
    q_full = full_information_q(model, params, obs)
    q_full = np.where(np.isnan(oracle["q"]), np.nan, q_full)
    model_run = json.loads((ROOT / "results" / f"minesweeper_{tag(g)}_var{size_tag(cfg)}_s{seed}.json").read_text())
    return {
        "exposure_rl": {k: v for k, v in model_run["model_v_mask_vs_oracle"].items() if not isinstance(v, dict)},
        **run_baselines(model, params, obs, w, q_full, actions, scored, w_scored, v_e, phi_e, HIDDEN, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "minesweeper"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random", help="tic-tac-toe opponent: random or pm05")
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--activation", default=Config.activation, choices=("gelu", "relu"))
    parser.add_argument("--xrl-matched", action="store_true", help="xrl Direct-Q configuration (vaeac_q.XRL_MATCHED)")
    parser.add_argument("--width", type=int, default=0, help="with --xrl-matched: hidden width (0: the xrl 64)")
    parser.add_argument("--all-td", action="store_true", help="only the TD agents with a TD-trained variance head")
    args = parser.parse_args()
    SIZE.update(
        {**XRL_MATCHED, **({"hidden": args.width} if args.width else {})}
        if args.xrl_matched
        else {"hidden": args.hidden, "depth": args.depth, "activation": args.activation}
    )
    OPPONENT[0] = args.opponent
    ALL_TD[0] = args.all_td
    size_suffix_opp = "" if args.opponent == "random" else f"_{args.opponent}"
    size_suffix = size_tag(dataclasses.replace(Config(), **SIZE))
    result = tictactoe(args.seed) if args.domain == "tictactoe" else minesweeper(args.seed)
    (ROOT / "results" / f"baselines_{args.domain}{size_suffix}{size_suffix_opp}_s{args.seed}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
