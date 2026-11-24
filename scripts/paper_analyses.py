"""The paper's analyses that no other script stores: exact references and checks quoted in the text.

  chain_rule   phi^H vs phi^Q (tic-tac-toe, random opponent). Exact: H[O | o_C] = H[V | o_C] + H[O | V, o_C] with
               V = Q*(O, a); median per-board Spearman of |phi^Q| with phi^H, its value part and its within-value
               part. Model (3 TD seeds): the agent's own Shannon phi^H vs its own |phi^Q|.
  parity       oracle-free check of the Shannon belief entropy: with one masked cell (action cell visible),
               piece-count parity bounds H by log 2; share of coalitions below log 2 + 0.05, mean model vs exact H.
  probes       linear probes on the full-information belief mean (3 TD seeds, all reachable boards): outcome class
               (balanced accuracy), V and the outcome entropy H[G | x] (ridge R^2), each vs the raw board.
  optimal      exact win rates of the optimal policy (Table 1 headers): tic-tac-toe vs random and perturbed minimax
               (enumerated game tree), Minesweeper from the exact oracle.

Usage: uv run python scripts/paper_analyses.py [chain_rule parity probes optimal]  (default: all)
       -> results/paper_analyses.json (sections merged into the existing file)
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
from scipy.stats import spearmanr

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, load_checkpoint, tictactoe_checkpoint
from exposure_rl.exact_eval import exact_outcomes, greedy
from exposure_rl.oracle import ExpectimaxOracle, certainty_games
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    make_coalition_games,
    make_predictive_entropy_game,
    size_tag,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "paper_analyses.json"
TAG = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": 256}))
SEEDS = (0, 1, 2)


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def median_spearman(a, b) -> float:
    rhos = [spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return float(np.mean(rhos))  # mean over boards since 2026-11-08 (was the median), as Table 1


def model_games(seed: int, opponent: str = "random"):
    _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, TAG, opponent, td_var=True))
    d = model.cfg.latent_dim
    entropy = make_predictive_entropy_game(model, 32, 4)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 11), (32, d))  # as scripts/predictive_entropy.py
    return model, params, entropy, eps


def chain_rule() -> dict:
    er = _module("entropy_references")
    ref = OracleReference("random")
    shap = lambda v: coalitions.uncertainty_reduction_shapley(v, 9)
    phi = {"H": [], "V": [], "Q": []}
    for p, o, a in zip(ref.dist.points, ref.obs, ref.actions):
        a = int(a)
        legal = ~np.isnan(ref.q_exact[:, a])
        pts, w = ref.obs[legal], ref.w[legal]
        _, idx = np.unique(np.round(ref.q_exact[legal, a], 9), return_inverse=True)
        phi["H"].append(shap(coalitions.completion_entropy(pts, w, o)))
        phi["V"].append(shap(er.class_entropy(pts, w, o, np.eye(idx.max() + 1)[idx])))  # exact H[V | o_C]
        phi["Q"].append(np.abs(coalitions.shapley(certainty_games(ref.oracle, ref.dist, p.observation, a).q_mean, 9)))
    phi = {k: np.stack(v) for k, v in phi.items()}
    within = phi["H"] - phi["V"]
    out = {
        "exact_value_share_of_phi_H": float(phi["V"].sum() / phi["H"].sum()),
        "exact_rho_phi_H_vs_absQ": median_spearman(phi["H"], phi["Q"]),
        "exact_rho_value_part_vs_absQ": median_spearman(phi["V"], phi["Q"]),
        "exact_rho_within_part_vs_absQ": median_spearman(within, phi["Q"]),
        "model_rho_phi_H_vs_absQ": {},
    }
    for seed in SEEDS:
        model, params, entropy, eps = model_games(seed)
        games = make_coalition_games(model)
        eps_q = jax.random.normal(jax.random.PRNGKey(seed + 1), (64, model.cfg.latent_dim))
        keys = jax.random.split(jax.random.PRNGKey(seed + 13), len(ref.obs))
        ph, pq = [], []
        for o, a, k in zip(ref.obs, ref.actions, keys):
            pq.append(np.abs(coalitions.shapley(np.asarray(games(params, jnp.asarray(o), int(a), eps_q)[0]), 9)))
            ph.append(shap(np.asarray(entropy(params, jnp.asarray(o), int(a), eps, k))))
        out["model_rho_phi_H_vs_absQ"][seed] = median_spearman(np.stack(ph), np.stack(pq))
        print("chain_rule", seed, out["model_rho_phi_H_vs_absQ"][seed], flush=True)
    return out


def parity() -> dict:
    out = {}
    sizes, masks = coalitions.coalition_sizes(9), np.arange(512)
    for opponent in ("random", "pm05"):
        ref = OracleReference(opponent)
        one = lambda a: (sizes == 8) & (((masks >> a) & 1) == 1)  # one masked cell, action cell visible
        h_e = np.concatenate(
            [
                coalitions.completion_entropy(
                    ref.obs[~np.isnan(ref.q_exact[:, int(a)])], ref.w[~np.isnan(ref.q_exact[:, int(a)])], o
                )[one(int(a))]
                for o, a in zip(ref.obs, ref.actions)
            ]
        )
        res = {
            "exact_mean": float(h_e.mean()),
            "exact_max": float(h_e.max()),
            "exact_share_zero": float(np.mean(np.isclose(h_e, 0))),
        }
        for seed in SEEDS:
            _, params, entropy, eps = model_games(seed, opponent)
            keys = jax.random.split(jax.random.PRNGKey(seed + 13), len(ref.obs))
            h_m = np.concatenate(
                [
                    np.asarray(entropy(params, jnp.asarray(o), int(a), eps, k))[one(int(a))]
                    for o, a, k in zip(ref.obs, ref.actions, keys)
                ]
            )
            res[seed] = {"share_below_log2": float(np.mean(h_m <= np.log(2) + 0.05)), "model_mean": float(h_m.mean())}
            print("parity", opponent, seed, res[seed], flush=True)
        out[opponent] = res
    return out


def probes() -> dict:
    regions = _module("outcome_regions")
    from xrl.explainability.tictactoe import build_tictactoe_state_distribution

    dist = build_tictactoe_state_distribution(regions.UniformPolicy(), opponent="random")
    obs = np.array([p.observation for p in dist.points], dtype=np.float32)
    rnd, good = ExpectimaxOracle("random"), ExpectimaxOracle("minimax")
    y5 = np.array([regions.outcome_class(good, rnd, p.observation) for p in dist.points])
    y = regions.RESULT3[y5]  # result against a perfect opponent (win / draw / loss), as Figure 2 (a) since 2026-11-09
    m1, m2 = map(np.stack, zip(*(rnd.action_moments(p.observation) for p in dist.points)))
    i, a = np.arange(len(obs)), np.nanargmax(m1, 1)
    e1, e2 = m1[i, a], m2[i, a]
    probs = np.clip(np.stack([(e2 + e1) / 2, 1 - e2, (e2 - e1) / 2], 1), 1e-12, 1)
    outcome_entropy = -(probs * np.log(probs)).sum(1)
    v_exact = np.nanmax(m1, axis=1)
    raw = np.eye(3)[obs.astype(int) + 1].reshape(len(obs), -1)
    out = {
        "raw": {
            "class_balanced_accuracy": float(regions.lda_cv_balanced_accuracy(raw, y)[0]),
            "class5_balanced_accuracy": float(regions.lda_cv_balanced_accuracy(raw, y5)[0]),
            "value_r2": float(regions.ridge_cv_r2(raw, v_exact)),
            "outcome_entropy_r2": float(regions.ridge_cv_r2(raw, outcome_entropy)),
        },
        "chance": 1 / 3,
        "chance5": 1 / len(regions.CLASSES),
    }
    for seed in SEEDS:
        _, model, params = load_checkpoint(RES / "models" / f"td_random_m10_var{TAG}_s{seed}.msgpack")
        log_pi, mu, _ = model.apply({"params": params}, jnp.asarray(obs), method=VAEACQ.full_belief)
        emb = np.asarray(jnp.sum(jnp.exp(log_pi)[..., None] * mu, axis=-2))
        out[seed] = {
            "class_balanced_accuracy": float(regions.lda_cv_balanced_accuracy(emb, y)[0]),
            "class5_balanced_accuracy": float(regions.lda_cv_balanced_accuracy(emb, y5)[0]),
            "value_r2": float(regions.ridge_cv_r2(emb, v_exact)),
            "outcome_entropy_r2": float(regions.ridge_cv_r2(emb, outcome_entropy)),
            "centre_balanced_accuracy": float(regions.lda_cv_balanced_accuracy(emb, obs[:, 4].astype(int))[0]),
        }
        print("probes", seed, out[seed], flush=True)
    return out


def optimal() -> dict:
    out = {}
    for opponent in ("random", "pm05"):
        oracle = ExpectimaxOracle(opponent)
        out[f"tictactoe_{opponent}"] = exact_outcomes(greedy(oracle.q_values), opponent)["win"]
    ms = json.loads((RES / f"minesweeper_4x4_3_var{TAG}_td_s0.json").read_text())
    out["minesweeper"] = ms["play_optimal"]["p_win"]
    out["minesweeper_decision_observations"] = ms["oracle"]["decision_observations"]
    out["minesweeper_mean_v_state_full"] = ms["oracle"]["mean_v_state_full_agent_own_uncertainty"]
    out["minesweeper_scored_observations"] = ms["model_v_mask_vs_oracle"]["scored_observations"]
    print("optimal", out, flush=True)
    return out


SECTIONS = {"chain_rule": chain_rule, "parity": parity, "probes": probes, "optimal": optimal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sections", nargs="*", choices=list(SECTIONS), help="default: all")
    args = parser.parse_args()
    for name in args.sections or list(SECTIONS):
        result = SECTIONS[name]()
        data = json.loads(OUT.read_text()) if OUT.exists() else {}  # re-read: sections may run in parallel
        data[name] = result
        OUT.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
