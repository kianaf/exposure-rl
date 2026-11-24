"""Minesweeper: exact certainty games, plus the one-network VAEAC-Q (exact-Q supervision).

Default board is 4x4 with 3 mines, the setting of Philipp's runs (xrl README, runs/minesweeper-4x4-3-div).
Two certainty games per (observation o, greedy action a, coalition C of visible cells):

    v_mask(C)  = Var[Q(O, a) | O_C = o_C]            uncertainty from cells WE hide (0 at C = N)
    v_state(C) = E[Var(G | O, a) | O_C = o_C]        the agent's OWN uncertainty about hidden mines
                                                     and its future (> 0 even at C = N)

All 2^n coalitions are exact (subset-sum transform, ``exposure_rl.coalitions``; 65536 for 16 cells).
Shapley values of v_state answer "which revealed clues make the agent sure this click is safe".
The model scores v_mask from its Q head and v_state from its variance head s_omega(z, a) (trained on
the exact Var(G | o, a); --var-weight 0 disables it). With more than 2000 decision observations the
model is scored on a fixed occupancy-weighted sample of 300 of them.

Usage: uv run python scripts/minesweeper.py [--rows 4 --columns 4 --mines 3] [--seed 0] [--reuse]
Writes results/minesweeper_<r>x<c>_<m>[_var]_s<seed>.json and results/figures/8_minesweeper_<r>x<c>_<m>[_var].png.
The exact oracle and exact games are cached in results/cache/ (the 4x4 oracle takes about 8 minutes).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from exposure_rl import coalitions
from exposure_rl.evaluate import compare, full_information_q
from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle
from exposure_rl.minesweeper_exact import exact_games, load_games, load_oracle, scored_indices
from exposure_rl.minesweeper_exact import tag as geometry_tag
from exposure_rl.td import TDConfig, train_td
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    init_params,
    make_coalition_games,
    size_tag,
    train_supervised,
)

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#a12c2b", "#e34948", "#f4b0ae", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"]
)
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 9,
    }
)
EPS = 1e-12


def wmean(w, x, m=None):
    m = np.ones(len(x), bool) if m is None else m
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def exact_policy_outcome(oracle: MinesweeperOracle, policy) -> dict:
    """Exact P(win) and expected return of a deterministic policy from the all-hidden board."""

    @cache
    def value(obs):
        a = int(policy(np.asarray(obs, np.float32), oracle.legal(obs)))
        layouts = oracle.belief(obs, a)
        win = ret = 0.0
        for L in layouts:
            if a in L:
                ret -= 1.0
                continue
            nxt, r, done = oracle.reveal(obs, a, L)
            w_next, g_next = (1.0, 0.0) if done else value(nxt)
            win += w_next
            ret += r + g_next
        return win / len(layouts), ret / len(layouts)

    p_win, ret = value(tuple([HIDDEN] * oracle.g.cells))
    return {"p_win": p_win, "expected_return": ret}


def score_state(w, vs_m, vs_e, phi_e, n) -> dict:
    """Model ``v_state`` games vs exact, over all coalitions (``v_state(N) > 0`` is part of the target)."""

    phi_m = coalitions.uncertainty_reduction_shapley(vs_m, n)
    rhos = [spearmanr(a, b).statistic for a, b in zip(phi_m, phi_e) if np.std(a) > 1e-9 and np.std(b) > 1e-9]
    big = np.abs(phi_e) > 1e-3
    return {
        "v_state_pearson": float(pearsonr(vs_m.ravel(), vs_e.ravel()).statistic),
        "v_state_mae": float(w @ np.abs(vs_m - vs_e).mean(1)),
        "v_state_full_model_vs_exact": [float(w @ vs_m[:, -1]), float(w @ vs_e[:, -1])],
        "v_state_empty_model_vs_exact": [float(w @ vs_m[:, 0]), float(w @ vs_e[:, 0])],
        "phi_state_mae": float(w @ np.abs(phi_m - phi_e).mean(1)),
        "phi_state_spearman_median": float(np.median(rhos)),
        "phi_state_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
    }


# ---------------------------------------------------------------- figure


def draw(ax, g: Geometry, obs, values, title, vmax):
    r_, c_ = g.rows, g.columns
    ax.set_xlim(0, c_)
    ax.set_ylim(r_, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if values is not None:
        ax.imshow(
            np.asarray(values).reshape(r_, c_), cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), extent=(0, c_, r_, 0)
        )
    else:
        ax.add_patch(plt.Rectangle((0, 0), c_, r_, fc=SURFACE, ec=INK2, lw=1))
    for i in range(1, r_):
        ax.axhline(i, color=SURFACE if values is not None else INK2, lw=2 if values is not None else 0.8)
    for i in range(1, c_):
        ax.axvline(i, color=SURFACE if values is not None else INK2, lw=2 if values is not None else 0.8)
    for c in range(g.cells):
        r, k = divmod(c, c_)
        label = "·" if obs[c] == HIDDEN else str(int(obs[c]))
        dark = values is not None and abs(values[c]) > 0.55 * vmax
        ax.text(
            k + 0.5,
            r + 0.42,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=SURFACE if dark else INK,
        )
        if values is not None:
            ax.text(
                k + 0.5,
                r + 0.82,
                f"{values[c]:+.3f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=SURFACE if dark else INK2,
            )
    ax.set_title(title, pad=3)


def canonical(obs, g: Geometry):
    """Smallest of the board's symmetric images (8 for square boards, 4 otherwise)."""

    b = np.asarray(obs).reshape(g.rows, g.columns)
    if g.rows == g.columns:
        forms = [np.rot90(m, k) for m in (b, b.T) for k in range(4)]
    else:
        forms = [b, b[::-1], b[:, ::-1], b[::-1, ::-1]]
    return min(tuple(f.reshape(-1).tolist()) for f in forms)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--mines", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument(
        "--samples", type=int, default=None, help="latent samples K (default 256 for 9 cells, else 128)"
    )
    parser.add_argument("--reuse", action="store_true", help="load the saved checkpoint instead of training")
    parser.add_argument("--var-weight", type=float, default=50.0, help="weight of the variance head (0 = none)")
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--activation", default=Config.activation, choices=("gelu", "relu"))
    parser.add_argument("--xrl-matched", action="store_true", help="xrl Direct-Q configuration (vaeac_q.XRL_MATCHED)")
    parser.add_argument("--width", type=int, default=0, help="with --xrl-matched: hidden width (0: the xrl 64)")
    parser.add_argument(
        "--td-updates",
        type=int,
        default=0,
        help="train by TD in the game for this many updates (0: exact-Q supervision)",
    )
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim, help="latent size d")
    parser.add_argument("--separate-heads", action="store_true", help="separate Q and variance MLPs (earlier model)")
    args = parser.parse_args()

    g = Geometry(args.rows, args.columns, args.mines)
    with_state = args.var_weight > 0
    n, tag = g.cells, geometry_tag(g) + ("_var" if with_state else "")
    cache_dir = ROOT / "results" / "cache"
    oracle_data = load_oracle(g, cache_dir)
    points, q_exact, g2 = oracle_data["points"], oracle_data["q"], oracle_data["g2"]
    w = oracle_data["weights"] / oracle_data["weights"].sum()
    n_obs = len(points)
    actions = np.nanargmax(q_exact, axis=1)

    scored, w_scored = scored_indices(oracle_data["weights"])
    games = load_games(g, oracle_data, np.unique(scored), cache_dir)
    keep_row = {int(i): k for k, i in enumerate(games["keep"])}
    v_e = games["v_mask_keep"][[keep_row[int(i)] for i in scored]]
    phi_e = games["phi_mask"][scored]
    phi_mask, phi_state, loo_mask = games["phi_mask"], games["phi_state"], games["loo_mask"]
    v_mask0, v_state0, v_stateN = games["summary"].T
    revealed = points != HIDDEN

    # ---------------- exact certainty games (oracle report)
    has = v_mask0 > EPS
    oracle_report = {
        "geometry": dataclasses.asdict(g),
        "decision_observations": n_obs,
        "optimal_first_click_q": np.round(q_exact[np.flatnonzero(~revealed.any(1))[0]], 4)
        .reshape(g.rows, g.columns)
        .tolist(),
        "mean_v_mask_empty": wmean(w, v_mask0),
        "mean_v_state_empty": wmean(w, v_state0),
        "mean_v_state_full_agent_own_uncertainty": wmean(w, v_stateN),
        "share_of_state_uncertainty_removed_by_seeing_board": wmean(
            w, (v_state0 - v_stateN) / np.maximum(v_state0, EPS)
        ),
        "loo_zero_cell_share_v_mask": wmean(w, (loo_mask < EPS).mean(1), has),
        "cells_with_negative_phi_mask": wmean(w, (phi_mask < -EPS).mean(1)),
        "cells_with_negative_phi_state": wmean(w, (phi_state < -EPS).mean(1)),
        "phi_state_on_revealed_vs_hidden_cells": [
            float(np.sum(w[:, None] * phi_state * revealed) / np.sum(w[:, None] * revealed)),
            float(np.sum(w[:, None] * phi_state * ~revealed) / np.sum(w[:, None] * ~revealed)),
        ],
        "mean_v_mask_by_coalition_size": (w @ games["v_mask_by_size"]).tolist(),
    }
    print(json.dumps(oracle_report, indent=2), flush=True)

    # ---------------- one-network VAEAC-Q, exact-Q supervision
    cfg = dataclasses.replace(
        Config(),
        seed=args.seed,
        steps=args.steps,
        n_values=10,
        n_cells=n,
        var_weight=args.var_weight,
        latent_dim=args.latent_dim,
        shared_heads=not args.separate_heads,
        **{
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": args.activation,
            **({**XRL_MATCHED, **({"hidden": args.width} if args.width else {})} if args.xrl_matched else {}),
        },
    )
    tag += (
        size_tag(cfg)
        + ("_td" if args.td_updates else "")
        + (f"_d{cfg.latent_dim}" if cfg.latent_dim != Config.latent_dim else "")
        + ("_separate" if not cfg.shared_heads and cfg.var_weight > 0 else "")
    )
    out = ROOT / "results"
    checkpoint = out / "models" / "minesweeper" / f"vaeac_q_ms{tag}_m{cfg.n_components}_s{args.seed}.msgpack"
    if args.reuse and checkpoint.exists():
        model = VAEACQ(cfg)
        params = serialization.from_bytes(init_params(model, cfg.seed), checkpoint.read_bytes())
    else:
        if args.td_updates:  # TD in the game: Q and (with var_weight) the variance head from bootstrapped targets
            td = dataclasses.replace(
                TDConfig(),
                updates=args.td_updates,
                domain="minesweeper",
                rows=g.rows,
                columns=g.columns,
                mines=g.mines,
            )
            print(f"TD training in Minesweeper {g.rows}x{g.columns}/{g.mines} for {td.updates} updates", flush=True)
            model, params, history = train_td(cfg, td)
            (out / f"minesweeper_{tag}_s{args.seed}_history.json").write_text(json.dumps(history))
        else:
            print(
                f"training VAEAC-Q on {n_obs} Minesweeper observations (M={cfg.n_components}, one network)", flush=True
            )
            model, params = train_supervised(points.astype(np.float32), w, q_exact, cfg, var_exact=g2 - q_exact**2)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(serialization.to_bytes(params))
    samples = args.samples or (256 if n <= 9 else 128)
    eps = jax.random.normal(jax.random.PRNGKey(args.seed + 1), (samples, cfg.latent_dim))
    games_fn = make_coalition_games(model, blocks=max(1, (1 << n) // 2048), with_state=with_state)
    t = time.time()
    rows = [
        [np.asarray(x) for x in games_fn(params, jnp.asarray(o, jnp.float32), int(a), eps)]
        for o, a in zip(points[scored], actions[scored])
    ]
    q_m, v_m = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
    print(f"model games for {len(scored)} observations in {time.time() - t:.0f}s", flush=True)
    exact_scored = [exact_games(oracle_data, int(i)) for i in scored]
    q_e = np.stack([e[0] for e in exact_scored])
    scores, _ = compare(w_scored, q_m, v_m, q_e, v_e, phi_e)
    if with_state:
        vs_m = np.stack([r[2] for r in rows])
        vs_e = np.stack([e[2] for e in exact_scored])
        scores["v_state"] = score_state(w_scored, vs_m, vs_e, phi_state[scored], n)
    full_q = full_information_q(model, params, points.astype(np.float32))
    legal = ~np.isnan(q_exact)
    scores["full_info_q_mae_legal"] = float(
        w @ (np.nansum(np.abs(full_q - np.nan_to_num(q_exact)) * legal, 1) / legal.sum(1))
    )
    chosen = np.argmax(np.where(legal, full_q, -np.inf), axis=1)
    scores["full_info_greedy_action_optimal"] = float(
        w @ np.isclose(q_exact[np.arange(n_obs), chosen], np.nanmax(q_exact, 1)).astype(float)
    )
    scores["scored_observations"] = len(scored)
    scores["latent_samples"] = samples

    game = MinesweeperOracle(g)  # rules only: belief / reveal (no value recursion)
    row_of = {tuple(int(v) for v in p): i for i, p in enumerate(points)}

    def agent(obs, legal_cells):
        qv = full_information_q(model, params, obs[None])[0]
        return int(np.flatnonzero(legal_cells)[np.argmax(qv[legal_cells])])

    def optimal(obs, legal_cells):
        return int(actions[row_of[tuple(int(v) for v in obs)]])

    t = time.time()
    summary = {
        "config": dataclasses.asdict(cfg),
        "oracle": oracle_report,
        "play_agent": exact_policy_outcome(game, agent),
        "play_optimal": exact_policy_outcome(game, optimal),
        "model_v_mask_vs_oracle": scores,
    }
    print(f"exact play evaluation in {time.time() - t:.0f}s", flush=True)
    (out / f"minesweeper_{tag}_s{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "oracle")}, indent=2))

    # ---------------- figure: which clues make the agent certain (three mid-game observations)
    pieces = revealed.sum(1)
    lo, hi = (2, 6) if n <= 9 else (4, 10)
    candidates = np.flatnonzero((pieces >= lo) & (pieces <= hi) & (v_stateN > EPS))
    picks, seen = [], set()
    for i in candidates[np.argsort(-w[candidates])]:
        key = canonical(points[i], g)
        if key not in seen:
            seen.add(key)
            picks.append(int(i))
        if len(picks) == 3:
            break
    pick_rows = [
        [np.asarray(x) for x in games_fn(params, jnp.asarray(points[i], jnp.float32), int(actions[i]), eps)]
        for i in picks
    ]
    phi_pick = np.stack([coalitions.uncertainty_reduction_shapley(r[1], n) for r in pick_rows])
    phi_pick_state = (
        np.stack([coalitions.uncertainty_reduction_shapley(r[2], n) for r in pick_rows]) if with_state else None
    )
    cols = 5 if with_state else 4
    fig, axes = plt.subplots(len(picks), cols, figsize=(2.75 * cols, 3.4 * len(picks)), squeeze=False)
    for row, i in enumerate(picks):
        obs, a = points[i], int(actions[i])
        vmax = max(np.abs(phi_state[i]).max(), np.abs(phi_mask[i]).max(), np.abs(phi_pick[row]).max(), 1e-9)
        if with_state:
            vmax = max(vmax, np.abs(phi_pick_state[row]).max())
        draw(
            axes[row, 0],
            g,
            obs,
            None,
            f"observation; agent clicks cell {a}\nown uncertainty $v_{{inh}}(N)$ = {v_stateN[i]:.3f}",
            vmax,
        )
        r, k = divmod(a, g.columns)
        axes[row, 0].add_patch(plt.Rectangle((k + 0.06, r + 0.06), 0.88, 0.88, fill=False, ec="#eb6834", lw=2.2))
        draw(
            axes[row, 1],
            g,
            obs,
            phi_state[i],
            "exact $\\phi^{state}$: which clues make\nthe agent sure of the outcome",
            vmax,
        )
        draw(axes[row, 2], g, obs, phi_mask[i], "exact $\\phi^{mask}$: certainty from\nseeing vs hiding cells", vmax)
        draw(axes[row, 3], g, obs, phi_pick[row], "model $\\phi^{mask}$ (one network)", vmax)
        if with_state:
            draw(axes[row, 4], g, obs, phi_pick_state[row], "model $\\phi^{state}$ (variance head)", vmax)
    fig.suptitle(
        f"Minesweeper {g.rows}x{g.columns}, {g.mines} mines — blue = the cell makes the agent more certain, red = less certain",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(h_pad=2.5)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"8_minesweeper_{tag}.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
