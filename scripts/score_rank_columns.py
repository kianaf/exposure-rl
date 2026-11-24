"""The two Table 1 rank scores the main evals do not produce, for our TD agents (seed by seed).

  v_mask own rho   median Spearman of phi^mask of the agent's belief game vs the exact v_mask of its own Q
                   (the reference of the "own v_mask r" column, replay_matching_eval.py --replay)
  v_state own rho  tic-tac-toe only (Minesweeper has it already): median Spearman of phi^state vs the oracle's
                   exact E[Var(G | O, a) | o_C] (the reference of the "own v_state r" column, baselines.py)

Same observations, actions, latent samples and Shapley convention as those scripts; the functions are imported
from them so nothing is redefined.

Usage: uv run python scripts/score_rank_columns.py [--width 256] -> results/rank_columns<tag>.json
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

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, full_information_q, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import exact_games, load_oracle, scored_indices
from exposure_rl.oracle import certainty_games
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, make_coalition_games, size_tag

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ablations", ROOT / "scripts" / "replay_matching_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def vmask_rho(model, params, points, weights, obs, actions, legal_of, n):
    q_agent = full_information_q(model, params, points.astype(np.float32))
    _, v_e = ev.own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
    v_m = ev.exposure_vmask_games(model, params, obs, actions, n)
    a, b = coalitions.shapley(v_m, n), coalitions.shapley(v_e, n)
    return ev.median_spearman(a, b), ev.mean_spearman(a, b)


def both(prefix, pair):
    return {f"{prefix}_spearman_median": pair[0], f"{prefix}_spearman_mean": pair[1]}


def state_rho(vs_m, vs_e, n):
    a, b = coalitions.uncertainty_reduction_shapley(vs_m, n), coalitions.uncertainty_reduction_shapley(vs_e, n)
    return ev.median_spearman(a, b), ev.mean_spearman(a, b)


def tictactoe(width: int, opponent: str) -> dict:
    ref = OracleReference(opponent)
    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": width}))
    legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
    exact_state = np.stack(
        [certainty_games(ref.oracle, ref.dist, p.observation, int(a)).opp for p, a in zip(ref.dist.points, ref.actions)]
    )
    out = {}
    for seed in range(3):
        _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, tag, opponent, td_var=True))
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (256, model.cfg.latent_dim))
        games_s = make_coalition_games(model, with_state=True)
        vs_m = np.stack(
            [np.asarray(games_s(params, jnp.asarray(o), int(a), eps)[2]) for o, a in zip(ref.obs, ref.actions)]
        )
        out[seed] = both("own_phi_mask", vmask_rho(model, params, ref.obs, ref.w, ref.obs, ref.actions, legal_of, 9)) | both(
            "phi_state", state_rho(vs_m, exact_state, 9)
        )
        print("tictactoe", opponent, seed, out[seed], flush=True)
    return out


def minesweeper(width: int) -> dict:
    g = Geometry(4, 4, 3)
    oracle = load_oracle(g, ROOT / "results" / "cache")
    points, weights = oracle["points"], oracle["weights"]
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(weights)
    legal_of = lambda a: points[:, a] == HIDDEN
    out = {}
    for seed in range(3):
        cfg = dataclasses.replace(
            Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **{**XRL_MATCHED, "hidden": width}
        )
        model = VAEACQ(cfg)
        path = (
            ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms4x4_3_var{size_tag(cfg)}_td_m10_s{seed}.msgpack"
        )

        params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
        obs = points[scored].astype(np.float32)
        # v_inh (2026-11-08; Table 1 took it from scripts/minesweeper.py): exact games of the optimal policy
        vs_e = np.stack([exact_games(oracle, int(i))[2] for i in scored])
        games_s = make_coalition_games(model, blocks=(1 << 16) // 2048, with_state=True)  # as scripts/minesweeper.py
        eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (128, model.cfg.latent_dim))
        vs_m = np.stack([np.asarray(games_s(params, jnp.asarray(o), int(a), eps)[2]) for o, a in zip(obs, actions[scored])])
        out[seed] = both(
            "own_phi_mask", vmask_rho(model, params, points.astype(np.float32), weights, obs, actions[scored], legal_of, 16)
        ) | both("phi_state", state_rho(vs_m, vs_e, 16))
        print("minesweeper", seed, out[seed], flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": args.width}))
    result = {
        "ttt_random": tictactoe(args.width, "random"),
        "ttt_pm05": tictactoe(args.width, "pm05"),
        "ms": minesweeper(args.width),
    }
    (ROOT / "results" / f"rank_columns{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
