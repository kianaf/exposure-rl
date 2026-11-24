"""What does the model's belief entropy track? Its phi^H against value-aware exact references, seed by seed.

The certainty game of the belief H(C) = H[e_psi(. | o_C)] is scored in the paper against the exact completion entropy
H[O | o_C], which counts every completion of the hidden cells alike. The latent is trained mainly through the Q
loss, so completions with the same value may sit close together in z. Exact references here (same occupancy,
action-legal view, all coalitions by subset sums; uncertainty-reduction Shapley values throughout):

  completion   H[O | o_C]                             (the paper's reference)
  value_b      H[bin_b(Q*(O, a)) | o_C], equal-width bins of width b on [-1, 1]
  outcome      H[G | o_C, a], G in {-1, 0, 1}         (tic-tac-toe only: recovered from E[G], E[G^2])
  mask         exact v_mask = Var[Q*(O, a) | o_C]     (value uncertainty, not an entropy)

Score: median Spearman between the model's phi^H and each reference's Shapley values (as in Table 1).

Usage: uv run python scripts/entropy_references.py --domain {tictactoe,pm05,minesweeper}
       -> results/entropy_references_xrl256_<domain>.json
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
from exposure_rl.evaluate import OracleReference, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_games, load_oracle, scored_indices
from exposure_rl.oracle import certainty_games
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, make_entropy_game, size_tag

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rep", ROOT / "scripts" / "representation_entropy.py")
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)
WIDTH = 256
BINS = (0.1, 0.25)


def class_entropy(points, weights, observation, probs) -> np.ndarray:
    """Exact ``H[K | O_C = o_C]`` for all coalitions; ``probs[j, k] = p(K = k | point j)`` (rows sum to 1)."""

    n = points.shape[1]
    mismatch = ((points != np.asarray(observation)[None]) << np.arange(n)).sum(1)
    w = weights / weights.sum()
    cols = np.stack(
        [np.bincount(mismatch, weights=w * probs[:, k], minlength=1 << n) for k in range(probs.shape[1])], 1
    )
    mass = coalitions.subset_sums(cols, n)[(1 << n) - 1 - np.arange(1 << n)]
    p = mass / mass.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def binned(q: np.ndarray, width: float) -> np.ndarray:
    k = round(2 / width)
    idx = np.clip(((q + 1) / width).astype(int), 0, k - 1)
    return np.eye(k)[idx]


def score(model, params, obs, actions, points, weights, q_of, legal_of, n, seed, mask_phi, outcome_of=None):
    entropy_fn = make_entropy_game(model, blocks=max(1, (1 << n) // 1024))
    eps = jax.random.normal(jax.random.PRNGKey(seed + 3), (rep.SAMPLES, model.cfg.latent_dim))
    shap = lambda v: coalitions.uncertainty_reduction_shapley(v, n)
    phi_model, refs = [], {"completion": [], **{f"value_{b}": [] for b in BINS}, "mask": []}
    if outcome_of is not None:
        refs["outcome"] = []
    for k, (o, a) in enumerate(zip(obs, actions)):
        a = int(a)
        legal = legal_of(a)
        pts, w = points[legal], weights[legal]
        phi_model.append(shap(np.asarray(entropy_fn(params, jnp.asarray(o, jnp.float32), a, eps))))
        refs["completion"].append(shap(coalitions.completion_entropy(pts, w, o)))
        for b in BINS:
            refs[f"value_{b}"].append(shap(class_entropy(pts, w, o, binned(q_of(a)[legal], b))))
        if outcome_of is not None:
            refs["outcome"].append(shap(class_entropy(pts, w, o, outcome_of(a)[legal])))
        refs["mask"].append(mask_phi(k, o, a))
    phi_model = np.stack(phi_model)
    out = {f"phi_H_vs_{name}": rep.median_spearman(phi_model, np.stack(v)) for name, v in refs.items()}
    out["phi_completion_vs_mask_exact"] = rep.median_spearman(np.stack(refs["completion"]), np.stack(refs["mask"]))
    return out


def tictactoe(opponent: str) -> dict:
    ref = OracleReference(opponent)
    first, second = map(np.stack, zip(*(ref.oracle.action_moments(p.observation) for p in ref.dist.points)))
    legal_of = lambda a: ~np.isnan(first[:, a])

    def outcome_of(a):  # G in {-1, 0, 1}: p(win) - p(loss) = E[G], p(win) + p(loss) = E[G^2]
        m1, m2 = np.nan_to_num(first[:, a]), np.nan_to_num(second[:, a])
        return np.clip(np.stack([(m2 - m1) / 2, 1 - m2, (m2 + m1) / 2], 1), 0, 1)

    def mask_phi(k, o, a):
        g = certainty_games(ref.oracle, ref.dist, ref.dist.points[k].observation, a)
        return coalitions.uncertainty_reduction_shapley(g.mask, 9)

    tag = size_tag(dataclasses.replace(Config(), **{**XRL_MATCHED, "hidden": WIDTH}))
    out = {}
    for seed in range(3):
        _, model, params = load_checkpoint(tictactoe_checkpoint("td", seed, tag, opponent, td_var=True))
        out[seed] = score(
            model,
            params,
            ref.obs,
            ref.actions,
            ref.obs,
            ref.w,
            lambda a: np.nan_to_num(first[:, a]),
            legal_of,
            9,
            seed,
            mask_phi,
            outcome_of,
        )
        print(opponent, seed, json.dumps(out[seed]), flush=True)
    return out


def minesweeper() -> dict:
    g = Geometry(4, 4, 3)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    points, weights, q = oracle["points"], oracle["weights"], oracle["q"]
    actions = np.nanargmax(q, axis=1)
    scored, _ = scored_indices(weights)
    games = load_games(g, oracle, np.unique(scored), cache)
    legal_of = lambda a: points[:, a] == HIDDEN
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
        out[seed] = score(
            model,
            params,
            points[scored],
            actions[scored],
            points,
            weights,
            lambda a: np.nan_to_num(q[:, a]),
            legal_of,
            g.cells,
            seed,
            lambda k, o, a: games["phi_mask"][scored[k]],
        )
        print("minesweeper", seed, json.dumps(out[seed]), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "pm05", "minesweeper"), required=True)
    args = parser.parse_args()
    result = (
        minesweeper() if args.domain == "minesweeper" else tictactoe("random" if args.domain == "tictactoe" else "pm05")
    )
    (ROOT / "results" / f"entropy_references_xrl{WIDTH}_{args.domain}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
