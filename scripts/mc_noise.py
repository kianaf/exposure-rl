"""Monte Carlo noise of the model games (2026-11-08, review point 4): Minesweeper, one checkpoint, fixed boards.

For checkpoint seed 0 and the 300 scored observations of Table 1, every model game is recomputed with three
sampling repeats; repeat 0 uses the paper's random numbers (reproduces Table 1 for this checkpoint), repeats 1 and 2
change only the sampling. Exact games stay fixed, so the spread of rho over repeats is the Monte Carlo noise, to be
compared with the seed SD in Table 1.

  phi^Q, phi^inh  latent route, 128 latent samples per coalition (as scripts/minesweeper.py)
  phi^obs         imputation route, 8 latents per component x 4 boards (scripts/imputation_vobs.py)
  phi^H           nested Monte Carlo, k = 16, r = 4 (scripts/predictive_entropy.py)

Usage: uv run python scripts/mc_noise.py [--repeats 3] -> results/mc_noise_minesweeper.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from exposure_rl import coalitions
from exposure_rl.minesweeper import Geometry
from exposure_rl.minesweeper_exact import exact_games, load_oracle, scored_indices
from exposure_rl.vaeac_q import make_coalition_games

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


imp, pe = _module("imputation_vobs"), _module("predictive_entropy")


def mean_rho(v_m, v_e, n, reduction):
    shap = coalitions.uncertainty_reduction_shapley if reduction else coalitions.shapley
    a, b = shap(v_m, n), shap(v_e, n)
    return float(np.mean([spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    seed = 0
    points, weights, obs, actions, legal_of, n, load = imp.diag.setting("ms")
    model, params = load(seed)
    oracle = load_oracle(Geometry(4, 4, 3), ROOT / "results" / "cache")
    scored, _ = scored_indices(oracle["weights"])

    cache = imp.QCache(model, params, n, model.cfg.n_values)
    q_agent = np.concatenate([cache.evaluate(((p + 1).astype(np.int64) * cache.pow).sum(-1)) for p in np.array_split(points, 16)])
    q_e, vobs_e = imp.abl.own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
    vinh_e = np.stack([exact_games(oracle, int(i))[2] for i in scored])
    h_e = np.stack([coalitions.completion_entropy(points[legal_of(int(a))], weights[legal_of(int(a))], o) for o, a in zip(obs, actions)])
    games_fn = make_coalition_games(model, blocks=(1 << n) // 2048, with_state=True)

    out = {}
    for rep in range(args.repeats):
        off = 1000 * rep  # repeat 0 = the paper's random numbers
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1 + off), (128, model.cfg.latent_dim))
        rows = [[np.asarray(t) for t in games_fn(params, jnp.asarray(o, jnp.float32), int(a), eps)] for o, a in zip(obs, actions)]
        q_m, vinh_m = np.stack([r[0] for r in rows]), np.stack([r[2] for r in rows])
        vobs_m = imp.vobs_games(model, params, obs, actions, n, 8, 4, seed + off)
        h_m, _ = pe.model_games(model, params, obs, actions, seed + off, 16, 4, 256)
        out[rep] = {
            "phi_Q": mean_rho(q_m, q_e, n, False),
            "phi_inh": mean_rho(vinh_m, vinh_e, n, True),
            "phi_obs": mean_rho(vobs_m, vobs_e, n, True),
            "phi_H": mean_rho(h_m, h_e, n, True),
        }
        print("repeat", rep, json.dumps(out[rep]), flush=True)
        (ROOT / "results" / "mc_noise_minesweeper.json").write_text(json.dumps(out, indent=2))
    spread = {g: float(np.std([r[g] for r in out.values()], ddof=1)) for g in out[0]}
    out["sd_over_repeats"] = spread
    (ROOT / "results" / "mc_noise_minesweeper.json").write_text(json.dumps(out, indent=2))
    print("SD over sampling repeats:", json.dumps(spread), flush=True)


if __name__ == "__main__":
    main()
