"""Shannon version of the certainty game of the belief: entropy of the model's predicted completions vs the exact one.

The paper's model side is the differential entropy of the latent belief, in other units than the exact
completion entropy H[O | o_C]. Here the model side is the Shannon entropy of the boards it predicts (VAEAC
imputation: z from the coalition belief, hidden cells from the decoder), in nats like the exact game:

    p(x_hidden | o_C) = E_{z ~ e(. | o_C)} prod_{i not in C} p(x_i | z, o_C),   H(C) = -E_x log p(x_hidden | o_C)

Nested Monte Carlo: Z = M * K latents (K stratified samples per mixture component), R boards decoded per latent,
each board's likelihood averaged over the same Z latents. The action's cell is known in every coalition, as in
the other games. Scores as in Table 1 plus Pearson on the raw coalition values (same units now).

Usage: uv run python scripts/predictive_entropy.py --domain {tictactoe,pm05,minesweeper} [--k 8] [--r 2]\n       (paper: tic-tac-toe --k 32 --r 4; Minesweeper --k 16 --r 4, 2^16 coalitions per board)
       -> results/predictive_entropy_xrl256_<domain>.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, spearmanr

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_games, load_oracle, scored_indices
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    init_params,
    make_entropy_game,
    make_predictive_entropy_game,
    size_tag,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts" / "representation_entropy.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)
WIDTH = 256


def scores(h_m, h_e, h_l, n) -> dict:
    shap = lambda v: coalitions.uncertainty_reduction_shapley(v, n)
    phi_e, phi_m, phi_l = (np.stack([shap(h) for h in hs]) for hs in (h_e, h_m, h_l))
    interior = coalitions.coalition_sizes(n) < n
    big = np.abs(phi_e) > 1e-3
    flat = lambda h: h[:, interior].ravel()
    return {
        "predictive_v_pearson_interior": float(pearsonr(flat(h_m), flat(h_e)).statistic),
        "predictive_v_spearman_interior": float(spearmanr(flat(h_m), flat(h_e)).statistic),
        "predictive_phi_spearman_median": rep.median_spearman(phi_m, phi_e),
        "predictive_phi_spearman_mean": rep.mean_spearman(phi_m, phi_e),
        "predictive_phi_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
        "latent_phi_spearman_median": rep.median_spearman(phi_l, phi_e),
        "latent_v_spearman_interior": float(spearmanr(flat(h_l), flat(h_e)).statistic),
        "mean_h_exact_empty": float(np.nanmean(h_e[:, 0])),
        "mean_h_model_empty": float(h_m[:, 0].mean()),
    }


def model_games(model, params, obs, actions, seed, k, r, blocks):
    d = model.cfg.latent_dim
    game = make_predictive_entropy_game(model, k, r, blocks=blocks)
    latent = make_entropy_game(model, blocks=max(1, blocks // 4))
    eps = jax.random.normal(jax.random.PRNGKey(seed + 11), (k, d))
    eps_l = jax.random.normal(jax.random.PRNGKey(seed + 3), (rep.SAMPLES, d))
    keys = jax.random.split(jax.random.PRNGKey(seed + 13), len(obs))
    f = lambda o: jnp.asarray(o, jnp.float32)
    h_m = np.stack([np.asarray(game(params, f(o), int(a), eps, kk)) for o, a, kk in zip(obs, actions, keys)])
    h_l = np.stack([np.asarray(latent(params, f(o), int(a), eps_l)) for o, a in zip(obs, actions)])
    return h_m, h_l


def tictactoe(opponent, k, r) -> dict:
    ref = OracleReference(opponent)
    legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
    h_e = np.stack(
        [
            coalitions.completion_entropy(ref.obs[legal_of(int(a))], ref.w[legal_of(int(a))], o)
            for o, a in zip(ref.obs, ref.actions)
        ]
    )
    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": WIDTH}))
    out = {}
    for seed in range(3):
        _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, tag, opponent, td_var=True))
        h_m, h_l = model_games(model, params, ref.obs, ref.actions, seed, k, r, 1)
        out[seed] = scores(h_m, h_e, h_l, 9)
        print(opponent, seed, json.dumps(out[seed]), flush=True)
    return out


def minesweeper(k, r) -> dict:
    g = Geometry(4, 4, 3)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    points, weights = oracle["points"], oracle["weights"]
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(weights)
    load_games(g, oracle, np.unique(scored), cache)
    legal_of = lambda a: points[:, a] == HIDDEN
    obs, acts = points[scored], actions[scored]
    h_e = np.stack(
        [
            coalitions.completion_entropy(points[legal_of(int(a))], weights[legal_of(int(a))], o)
            for o, a in zip(obs, acts)
        ]
    )
    out = {}
    for seed in range(3):
        cfg = dataclasses.replace(
            Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **{**XRL_MATCHED, "hidden": WIDTH}
        )
        model = VAEACQ(cfg)
        path = (
            ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms4x4_3_var{size_tag(cfg)}_td_m10_s{seed}.msgpack"
        )
        params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
        h_m, h_l = model_games(model, params, obs, acts, seed, k, r, 256)
        out[seed] = scores(h_m, h_e, h_l, g.cells)
        print("minesweeper", seed, json.dumps(out[seed]), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "pm05", "minesweeper"), required=True)
    parser.add_argument("--k", type=int, default=8, help="latent samples per mixture component")
    parser.add_argument("--r", type=int, default=2, help="boards decoded per latent")
    args = parser.parse_args()
    if args.domain == "minesweeper":
        out = minesweeper(args.k, args.r)
    else:
        out = tictactoe("random" if args.domain == "tictactoe" else "pm05", args.k, args.r)
    (ROOT / "results" / f"predictive_entropy_xrl{WIDTH}_{args.domain}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
