"""Representation-sharpness game: Shapley values of the belief entropy H[e_psi(. | o_C)] vs exact.

Model:  H(C) = entropy of the latent belief for coalition C (Monte Carlo over the mixture; coordinate-free,
        so its Shapley values do not depend on how z is parameterised).
Exact:  H[O | o_C] = entropy of the observations still consistent with the visible cells, under the
        agent's occupancy (action-legal view), computed for all coalitions by the subset-sum transform.

The two are in different units (continuous latent vs discrete completions), so they are compared with
scale-free scores: Spearman over coalitions, median Spearman of the Shapley values, sign agreement.
Also reported: how much the certainty game of the belief differs from the masking game (median Spearman between
the model's phi^H and phi^mask on the same observations).

Usage: uv run python scripts/representation_entropy.py --domain tictactoe
       uv run python scripts/representation_entropy.py --domain minesweeper --mines 4
Writes results/representation_entropy_<domain>[_<r>x<c>_<m>].json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import spearmanr

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_games, load_oracle, scored_indices, tag
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    init_params,
    make_coalition_games,
    make_entropy_game,
    size_tag,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE: dict = {}  # network size from the command line (default: 3 x 256 GELU)
OPPONENT = ["random"]  # tic-tac-toe opponent spec
ALL_TD = [False]  # --all-td: score only the TD agents (variance head on its TD target)
SAMPLES = 32


def median_spearman(a, b):
    rhos = [spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return float(np.median(rhos)) if rhos else float("nan")


def mean_spearman(a, b):
    rhos = [spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return float(np.mean(rhos)) if rhos else float("nan")


def score(model, params, obs, actions, points, weights, legal_of, n, seed):
    blocks = max(1, (1 << n) // 1024)
    entropy_fn = make_entropy_game(model, blocks=blocks)
    games_fn = make_coalition_games(model, blocks=blocks)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 3), (SAMPLES, model.cfg.latent_dim))
    eps_q = jax.random.normal(jax.random.PRNGKey(seed + 1), (64, model.cfg.latent_dim))
    h_m, h_e, v_m = [], [], []
    for o, a in zip(obs, actions):
        legal = legal_of(int(a))
        h_m.append(np.asarray(entropy_fn(params, jnp.asarray(o, jnp.float32), int(a), eps)))
        h_e.append(coalitions.completion_entropy(points[legal], weights[legal], o))
        v_m.append(np.asarray(games_fn(params, jnp.asarray(o, jnp.float32), int(a), eps_q)[1]))
    h_m, h_e, v_m = np.stack(h_m), np.stack(h_e), np.stack(v_m)
    phi_m = coalitions.uncertainty_reduction_shapley(h_m, n)  # positive = the cell sharpens the belief
    phi_e = coalitions.uncertainty_reduction_shapley(h_e, n)
    phi_mask = coalitions.uncertainty_reduction_shapley(v_m, n)
    interior = coalitions.coalition_sizes(n) < n
    big = np.abs(phi_e) > 1e-3
    return {
        "entropy_spearman_interior": float(spearmanr(h_m[:, interior].ravel(), h_e[:, interior].ravel()).statistic),
        "phi_entropy_spearman_median": median_spearman(phi_m, phi_e),
        "phi_entropy_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
        "phi_entropy_vs_phi_mask_spearman_median": median_spearman(phi_m, phi_mask),
        "phi_exact_entropy_vs_phi_mask_spearman_median": median_spearman(phi_e, phi_mask),
        "observations": len(obs),
    }


def tictactoe() -> dict:
    ref = OracleReference(OPPONENT[0])
    legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
    out = {}
    for kind in ("td",) if ALL_TD[0] else ("vaeac_q", "td"):
        for seed in range(3):
            path = tictactoe_checkpoint(
                kind, seed, size_tag(dataclasses.replace(Config(), **SIZE)), OPPONENT[0], td_var=ALL_TD[0]
            )
            name = path.stem
            _, model, params = load_checkpoint(path)
            out[name] = score(model, params, ref.obs, ref.actions, ref.obs, ref.w, legal_of, 9, seed)
            print(name, json.dumps(out[name]), flush=True)
    return out


def minesweeper(mines: int) -> dict:
    g = Geometry(4, 4, mines)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    points, weights = oracle["points"], oracle["weights"]
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(weights)
    load_games(g, oracle, np.unique(scored), cache)
    legal_of = lambda a: points[:, a] == HIDDEN
    out = {}
    for seed in range(3):
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
        name = f"ms{tag(g)}_s{seed}"
        out[name] = score(model, params, points[scored], actions[scored], points, weights, legal_of, g.cells, seed)
        print(name, json.dumps(out[name]), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "minesweeper"), required=True)
    parser.add_argument("--mines", type=int, default=3)
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
    if args.domain == "tictactoe":
        result, suffix = tictactoe(), ""
    else:
        result, suffix = minesweeper(args.mines), f"_4x4_{args.mines}"
    (ROOT / "results" / f"representation_entropy_{args.domain}{suffix}{size_suffix}{size_suffix_opp}.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
