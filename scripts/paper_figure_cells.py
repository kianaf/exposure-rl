"""Figure 3 of the paper: per-cell Shapley values of the TD-trained EXPOSURE agent on one tic-tac-toe board (own
uncertainty = chance, the opponent). --minesweeper adds a Minesweeper row (own uncertainty = hidden state, the mines),
which the paper no longer shows.

Each row: the observation (pink = the action's cell), then exact and model pairs of phi^Q, phi^state, phi^mask
(phi^state and phi^mask share one colour scale) and phi^H (exact completion entropy vs the model's Shannon entropy
of its belief over the hidden cells, both in nats, one scale). The exact games are those of Table 1: the agent's own
Q for phi^Q and phi^obs, the return variance of its own greedy policy for phi^inh (--reference optimal: all from the
optimal oracle, the figure before 2026-11-15).
Boards are the most frequent mid-game observation with remaining own uncertainty under the exact occupancy.

Usage: uv run python scripts/paper_figure_cells.py [--width 256] [--seed 2] [--ttt-board I] -> ../paper/figures/cells_ttt.pdf
The paper uses --ttt-board 1728 (chosen 2026-11-16 from the top 10 of an 80-board scan under the Table 1
references, results/figures/fig2d_candidates/; Spearman rho phi^Q / phi^inh / phi^obs / phi^H 0.90 / 0.87 / 0.95 / 0.83).
Before: --ttt-board 1344 (chosen 2026-11-12 for the unclipped model from a scan of the 300 most frequent
boards with 3-6 pieces and exact v_inh(N) > 0.05; Spearman rho phi^Q / phi^inh / phi^obs / phi^H 0.95 / 0.98 / 0.72 / 0.83 against the optimal-policy references,
0.92 / 0.35 / 0.90 /
0.83 against the own-policy variance, 0.92 / 0.85 / 0.90 / 0.83 against the Table 1 references (agent's own s), 3 pieces, O to move (blocks X's right column), occupancy rank 54).
Earlier versions used --ttt-board 1269, 1759 and --minesweeper --ms-board 464.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from flax import serialization

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exposure_rl import coalitions
from exposure_rl.evaluate import full_information_q, load_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import exact_games, load_games, load_oracle, scored_indices, tag
from exposure_rl.oracle import ExpectimaxOracle, certainty_games
from exposure_rl.policy_eval import GreedyPolicy, PolicyTicTacToe
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    init_params,
    make_coalition_games,
    make_predictive_entropy_game,
    mixture_q_moments,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ms_script", ROOT / "scripts" / "minesweeper.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)
_spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "replay_matching_eval.py")
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)
OPPONENT = ["random"]  # --opponent: random or pm05 (perturbed minimax, annealed agent pm05a)
REFERENCE = ["table"]  # --reference: exact games as Table 1 (own Q, own-policy variance) or all from the optimal oracle
plt.rcParams.update(
    {
        "font.size": 7,
        "text.usetex": True,  # LaTeX: Helvetica text, LaTeX math
        "font.family": "sans-serif",
        "text.latex.preamble": r"\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{amsmath,amssymb}",
        "axes.titlesize": 4.5,
        "pdf.fonttype": 42,
        "figure.facecolor": "none",  # transparent background
        "savefig.facecolor": "none",
        "axes.facecolor": "none",
    }
)


def draw(ax, rows, cols, labels, values, title, vmax, action=None):
    """Board map: colour = Shapley value (blue = more certain), text = the cell's content."""

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if values is not None:
        ax.imshow(
            np.asarray(values).reshape(rows, cols),
            cmap=ms.DIVERGING,
            norm=matplotlib.colors.TwoSlopeNorm(0, -vmax, vmax),
            extent=(0, cols, rows, 0),
        )
    else:
        ax.add_patch(plt.Rectangle((0, 0), cols, rows, fc=ms.SURFACE, ec=ms.INK2, lw=0.6))
    line = ms.SURFACE if values is not None else ms.INK2
    for i in range(1, rows):
        ax.axhline(i, color=line, lw=1.8 if values is not None else 0.5)
    for i in range(1, cols):
        ax.axvline(i, color=line, lw=1.8 if values is not None else 0.5)
    for c, label in enumerate(labels):
        r, k = divmod(c, cols)
        dark = values is not None and abs(values[c]) > 0.55 * vmax
        ax.text(
            k + 0.5,
            r + 0.5,
            f"\\textbf{{{label}}}" if label and not label.startswith("$") else label,
            ha="center",
            va="center",
            fontsize=4.6,
            fontweight="bold",
            color=ms.SURFACE if dark else ms.INK,
        )
    if action is not None:  # pink square with the mover's mark in pink, on every panel
        r, k = divmod(action, cols)
        ax.add_patch(plt.Rectangle((k + 0.07, r + 0.07), 0.86, 0.86, fill=False, ec="#e0389b", lw=1.5, zorder=4))
        if rows == cols == 3:  # tic-tac-toe: the agent places its mark, the side with fewer marks
            mark = "O" if list(labels).count("X") > list(labels).count("O") else "X"
            ax.text(k + 0.5, r + 0.5, f"\\textbf{{{mark}}}", ha="center", va="center", fontsize=4.6,
                    fontweight="bold", color="#e0389b", zorder=5)
    if title in ("exact", "model", "observation"):  # below the board
        ax.text(0.5, -0.05, title, transform=ax.transAxes, ha="center", va="top", fontsize=6)
    else:
        ax.set_title(title, pad=1.5, fontsize=6)


TITLES = ("observation",) + ("exact", "model") * 3  # sub-titles; the symbol of each pair is a group header
GROUPS = ("$\\phi^{\\mathrm{obs}}$", "$\\phi^{\\mathrm{inh}}$", "$\\phi^{H}$")  # the order of Table 1 (2026-11-16), no phi^Q


SCALE = ["relative"]  # --scale raw: phi^obs and phi^inh on one scale (both return variance), phi^H on its own (nats)
RAW_MAX = [1.0, 1.0]  # the two raw scales of --scale raw, for the colorbars
OWN_MAX = [1.0, 1.0, 1.0]  # --scale own: each explanation's largest |phi| (exact and model), its colorbar's range
V_RANGE = [None]  # the agent's (model) v(empty) and v(N) of obs, inh and H, in parentheses in the headers
Q_HAT = [None]  # the agent's Q(o, a) of the first row, written above the observation


def draw_row(axes, rows, cols, labels, a, q_hat, maps, h_maps, domain, titles, q_maps):
    """``q_maps``: exact/model phi^Q; ``maps``: exact/model phi^inh and phi^obs; ``h_maps``: phi^H. Each explanation is
    divided by its largest |phi| over the exact and model maps (relative scale, one colorbar from -1 to 1,
    2026-11-16), so exact and model share a scale and the explanations, in different units, stay visible. Column titles
    only when ``titles`` (first row)."""

    # obs, inh, H as in Table 1; phi^Q is not our contribution and left out of the figure
    pairs = [[maps[2], maps[3]], [maps[0], maps[1]], list(h_maps)]
    if SCALE[0] == "raw":  # shared scale for the two variances, own scale for the entropy
        m_var = max(float(np.abs(y).max()) for y in pairs[0] + pairs[1]) or 1.0
        m_h = max(float(np.abs(y).max()) for y in pairs[2]) or 1.0
        if titles:
            RAW_MAX[:] = [m_var, m_h]
        pairs = [[np.asarray(x) / m for x in p] for p, m in zip(pairs, (m_var, m_var, m_h))]
    elif SCALE[0] in ("relative", "own"):  # own: the same colors, its colorbar in raw values
        tops = [max(float(np.abs(y).max()) for y in p) or 1.0 for p in pairs]
        if titles:
            OWN_MAX[:] = tops
        pairs = [[np.asarray(x) / m for x in p] for p, m in zip(pairs, tops)]
    else:  # unit: the Shapley values themselves on one fixed scale from -1 to 1
        pairs = [[np.clip(np.asarray(x), -1.0, 1.0) for x in p] for p in pairs]
    t = TITLES if titles else ("",) * 7
    if titles:
        Q_HAT[0] = q_hat
    draw(axes[0], rows, cols, labels, None, t[0], 1.0, a)
    for k, p in enumerate(pairs):
        for j, m in enumerate(p):
            draw(axes[1 + 2 * k + j], rows, cols, labels, m, t[1 + 2 * k + j], 1.0, a)


def _imputation():
    """scripts/imputation_vobs.py: v_obs through the decoder's imputations (exactly 0 at C = N), as in Table 1."""

    spec = importlib.util.spec_from_file_location("imputation_vobs", ROOT / "scripts" / "imputation_vobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_IMPUTATION = []


def _imputation_cached():
    """_imputation() once per process; its decoder (jitted) and Q cache are reused across boards (batch runs)."""

    if not _IMPUTATION:
        mod = _imputation()
        decoders, caches = {}, {}
        make_decoder, q_cache = mod.make_decoder, mod.QCache
        mod.make_decoder = lambda model, *a: decoders.setdefault((id(model), *a), make_decoder(model, *a))
        mod.QCache = lambda model, params, *a: caches.setdefault((id(model), id(params), *a), q_cache(model, params, *a))
        _IMPUTATION.append(mod)
    return _IMPUTATION[0]


_SHARED: dict = {}


def _shared(width, seed):
    """Everything of the tic-tac-toe row that does not depend on the board, built once per process, so
    --ttt-boards renders many boards without rebuilding the oracle, the model and its jitted games (2026-11-24)."""

    key = (OPPONENT[0], REFERENCE[0], width, seed)
    if key in _SHARED:
        return _SHARED[key]
    oracle = ExpectimaxOracle(OPPONENT[0])
    dist = oracle.occupancy()
    pts = np.array([p.observation for p in dist.points], dtype=np.float32)
    q = np.stack([oracle.action_moments(p.observation)[0] for p in dist.points])
    tag_w = "_xrl" if width == 64 else f"_xrl{width}"
    agent = {"random": "random", "pm05": "pm05a"}[OPPONENT[0]]
    _, model, params = load_checkpoint(ROOT / "results" / "models" / f"td_{agent}_m10_var{tag_w}_s{seed}.msgpack")
    sh = types.SimpleNamespace(
        oracle=oracle, dist=dist, pts=pts, w=np.array([p.weight for p in dist.points]), q=q,
        actions=np.nanargmax(q, axis=1), model=model, params=params,
        games_fn=make_coalition_games(model, with_state=True),
        entropy_fn=make_predictive_entropy_game(model, 32, 4),  # Shannon belief entropy over the hidden cells
        q_fn=lambda x: full_information_q(model, params, np.asarray(x, np.float32)),
    )
    if REFERENCE[0] in ("table", "own"):
        sh.q_all = sh.q_fn(pts)
    if REFERENCE[0] == "table":  # the agent's own full-observation variance head on every board
        eps_full = jax.random.normal(jax.random.PRNGKey(12345), (64, model.cfg.latent_dim))
        lp, mu, ls = model.apply({"params": params}, jnp.asarray(pts, jnp.float32), method=VAEACQ.full_belief)
        sh.s_all = np.asarray(mixture_q_moments(model, params, lp, mu, ls, eps_full, head=VAEACQ.variance)[0], np.float64)
    _SHARED[key] = sh
    return sh


def tictactoe_row(axes, width, seed, board=None):
    sh = _shared(width, seed)
    oracle, dist, pts, w, q, actions = sh.oracle, sh.dist, sh.pts, sh.w, sh.q, sh.actions
    model, params, games_fn, q_fn = sh.model, sh.params, sh.games_fn, sh.q_fn
    if board is None:
        # X to move (the agent moved first: as many own as opponent pieces); among the 40 most frequent such
        # mid-game boards with own uncertainty left, the one with the highest exact Q
        pieces = (pts != 0).sum(1)
        best, seen = None, 0
        for i in np.argsort(-w):
            if not 2 <= pieces[i] <= 6 or (pts[i] == 1).sum() != (pts[i] == -1).sum():
                continue
            g = certainty_games(oracle, dist, dist.points[i].observation, int(actions[i]))
            if g.opp[-1] <= 0.05:
                continue
            if best is None or q[i, actions[i]] > best[0]:
                best = (float(q[i, actions[i]]), int(i), g)
            seen += 1
            if seen == 40:
                break
        _, pick, games = best
    else:  # a fixed board (index into the exact occupancy): no scan (it ran before 2026-11-24 and was discarded)
        pick = board
        games = certainty_games(oracle, dist, dist.points[pick].observation, int(actions[pick]))
    a = int(actions[pick])
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (128, model.cfg.latent_dim))
    q_m, v_m, v_s = (np.asarray(t) for t in games_fn(params, jnp.asarray(pts[pick]), a, eps))
    if REFERENCE[0] in ("table", "own"):  # phi^Q, phi^obs: the agent's own full-observation Q
        q_e, mask_e = abl.own_exact_moment_games(pts, w, sh.q_all, pts[pick][None], [a], lambda b: pts[:, b] == 0)
        if REFERENCE[0] == "table":  # phi^inh: E[s(O, a) | o_C] of its own full-observation variance head (Table 1)
            s_all = sh.s_all
            legal = pts[:, a] == 0
            opp = coalitions.conditional_moments(pts[legal], w[legal], s_all[legal, a][:, None], pts[pick])[:, 0]
        else:  # --reference own: the exact return variance of its own greedy policy (2026-11-16 morning version)
            own = PolicyTicTacToe(OPPONENT[0], GreedyPolicy(lambda o: q_fn(np.asarray(o)[None])[0]))
            opp = certainty_games(own, dist, dist.points[pick].observation, a).opp
        games = dataclasses.replace(games, q_mean=q_e[0], mask=mask_e[0], opp=opp)
    v_m = _imputation_cached().vobs_games(model, params, pts[pick][None].astype(np.float32), [a], 9, 8, 4, seed)[0]  # as Table 1
    shap = lambda v: coalitions.uncertainty_reduction_shapley(v, 9)
    q_maps = (coalitions.shapley(games.q_mean, 9), coalitions.shapley(q_m, 9))
    maps = (shap(games.opp), shap(v_s), shap(games.mask), shap(v_m))
    legal = pts[:, a] == 0
    h_exact = coalitions.completion_entropy(pts[legal], w[legal], pts[pick])
    entropy_fn = sh.entropy_fn  # Shannon belief entropy over the hidden cells
    h_model = np.asarray(
        entropy_fn(
            params,
            jnp.asarray(pts[pick]),
            a,
            jax.random.normal(jax.random.PRNGKey(seed + 11), (32, model.cfg.latent_dim)),
            jax.random.PRNGKey(seed + 13),
        )
    )
    # what each map distributes: the agent's uncertainty with no feature and with every feature visible (2026-11-24)
    V_RANGE[0] = [(float(v[0]), float(v[-1])) for v in (v_m, v_s, h_model)]
    # observations are from the agent's view (own pieces +1); draw the real marks: X is the player who moved first,
    # i.e. the agent is X when both have the same number of pieces on its turn, else O
    own, other = int((pts[pick] == 1).sum()), int((pts[pick] == -1).sum())
    agent_mark, opp_mark = ("X", "O") if own == other else ("O", "X")
    labels = [{1: agent_mark, -1: opp_mark, 0: ""}[int(v)] for v in pts[pick]]
    q_hat = float(full_information_q(model, params, pts[pick][None])[0, a])
    draw_row(axes, 3, 3, labels, a, q_hat, maps, (shap(h_exact), shap(h_model)), "tic-tac-toe", True, q_maps)
    return pick


def minesweeper_row(axes, width, seed, board=None):
    g = Geometry(4, 4, 3)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    points, q = oracle["points"], oracle["q"]
    w = oracle["weights"] / oracle["weights"].sum()
    actions = np.nanargmax(q, axis=1)
    scored, _ = scored_indices(oracle["weights"])
    games = load_games(g, oracle, np.unique(scored), cache)
    phi_mask, phi_state, v_state_full = games["phi_mask"], games["phi_state"], games["summary"][:, 2]
    revealed = (points != HIDDEN).sum(1)
    cand = np.flatnonzero((revealed >= 4) & (revealed <= 10) & (v_state_full > 1e-12))
    i = int(cand[np.argmax(w[cand])]) if board is None else board
    a = int(actions[i])
    cfg = dataclasses.replace(Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **XRL_MATCHED)
    cfg = dataclasses.replace(cfg, hidden=width)
    tag_w = "_xrl" if width == 64 else f"_xrl{width}"
    model = VAEACQ(cfg)
    path = ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms{tag(g)}_var{tag_w}_td_m10_s{seed}.msgpack"
    params = serialization.from_bytes(init_params(model, seed), path.read_bytes())
    games_fn = make_coalition_games(model, blocks=(1 << g.cells) // 2048, with_state=True)
    eps = jax.random.normal(jax.random.PRNGKey(seed + 1), (128, cfg.latent_dim))
    q_m, v_m, v_s = (np.asarray(t) for t in games_fn(params, jnp.asarray(points[i], jnp.float32), a, eps))
    v_m = _imputation_cached().vobs_games(model, params, points[i][None].astype(np.float32), [a], g.cells, 8, 4, seed)[0]
    shap = lambda v: coalitions.uncertainty_reduction_shapley(v, g.cells)
    q_exact = np.asarray(exact_games(oracle, i)[0])
    q_maps = (coalitions.shapley(q_exact, g.cells), coalitions.shapley(q_m, g.cells))
    maps = (phi_state[i], shap(v_s), phi_mask[i], shap(v_m))
    legal = points[:, a] == HIDDEN
    h_exact = coalitions.completion_entropy(points[legal], oracle["weights"][legal], points[i])
    entropy_fn = make_predictive_entropy_game(model, 16, 4, blocks=256)  # Shannon belief entropy, hidden cells
    h_model = np.asarray(
        entropy_fn(
            params,
            jnp.asarray(points[i], jnp.float32),
            a,
            jax.random.normal(jax.random.PRNGKey(seed + 11), (16, cfg.latent_dim)),
            jax.random.PRNGKey(seed + 13),
        )
    )
    labels = ["$\\cdot$" if v == HIDDEN else str(int(v)) for v in points[i]]
    q_hat = float(full_information_q(model, params, points[i][None].astype(np.float32))[0, a])
    draw_row(axes, 4, 4, labels, a, q_hat, maps, (shap(h_exact), shap(h_model)), "Minesweeper", False, q_maps)
    return i


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ttt-board", type=int, default=None, help="tic-tac-toe board index (default: selection rule)")
    parser.add_argument("--ms-board", type=int, default=None, help="Minesweeper board index (default: selection rule)")
    parser.add_argument("--minesweeper", action="store_true", help="add the Minesweeper row (not in the paper)")
    parser.add_argument("--out", type=Path, default=None, help="write here instead of ../paper/figures (candidate sheets)")
    parser.add_argument("--reference", choices=("table", "own", "optimal"), default="table", help="exact games of the tic-tac-toe row")
    parser.add_argument("--opponent", choices=("random", "pm05"), default="random", help="tic-tac-toe opponent")
    parser.add_argument("--scale", choices=("relative", "own", "raw", "unit"), default="own",
                        help="relative: each explanation over its largest |phi|; raw: shared variance scale, nats for H")
    parser.add_argument("--ttt-boards", type=int, nargs="+", default=None,
                        help="many boards in one process, one PDF each in --out-dir (candidate sheets)")
    parser.add_argument("--out-dir", type=Path, default=None, help="with --ttt-boards: write b<index>.pdf here")
    args = parser.parse_args()
    REFERENCE[0] = args.reference
    OPPONENT[0] = args.opponent
    SCALE[0] = args.scale
    if args.ttt_boards:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for board in args.ttt_boards:
            figure(args, board, args.out_dir / f"b{board}.pdf")
    else:
        name = "cells_ttt_ms.pdf" if args.minesweeper else "cells_ttt.pdf"
        figure(args, args.ttt_board, args.out or ROOT.parent / "paper" / "figures" / name)


def figure(args, board, out) -> None:
    n_rows = 2 if args.minesweeper else 1
    # 7 panels; a spacer column after the observation and after each exact/model pair, the pairs kept tight
    height = 1.22 if n_rows == 2 else 0.78  # 0.72 before exact/model went below the maps
    k_y = 1.22 / height  # header offsets are figure fractions, tuned for the two-row height
    gap, pair = 0.55, 0.12  # spacer at the dashed lines (0.14, then 0.3 before 2026-11-24), and between exact and model
    bar_gap = 0.85 if SCALE[0] == "own" else gap  # own: room for each pair's colorbar before the dashed line
    ratios = [1, gap, 1, pair, 1, bar_gap, 1, pair, 1, bar_gap, 1, pair, 1]
    # panels as large as with the phi^Q pair (5.15 in for 10.04 ratio units): the figure narrows instead
    # the maps centered on the old 5.15 in width (text width in the paper), "(d)" at its left edge
    fig_w, grid_w = 5.15, 4.738 * sum(ratios) / 10.04
    bars_w = {"raw": 0.95, "own": 0.3}.get(SCALE[0], 0.42)  # the colorbar(s) and their labels right of the grid
    off = (fig_w - grid_w - bars_w) / 2
    fig = plt.figure(figsize=(fig_w, height))
    grid = fig.add_gridspec(
        n_rows,
        len(ratios),
        width_ratios=ratios,
        left=off / fig_w,
        right=(off + grid_w) / fig_w,
        bottom=0.14 if n_rows == 1 else 0.02,  # room for exact/model below the maps
        top=1 - 0.12 * k_y,
        wspace=0.08,
        hspace=0.1,
    )
    panel_cols = [c for c, r in enumerate(ratios) if r == 1]  # 9 panels
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in panel_cols] for r in range(n_rows)])
    t = tictactoe_row(axes[0], args.width, args.seed, board)
    m = minesweeper_row(axes[1], args.width, args.seed, args.ms_board) if args.minesweeper else None
    # one symbol per exact/model pair, centred above its two sub-titles, and the agent's Q(o, a) above the observation
    for j, (k, text) in enumerate(zip((1, 3, 5), GROUPS)):
        if V_RANGE[0] is not None:  # (v(empty) -> v(N)) of the agent
            v0, v1 = V_RANGE[0][j]
            game = ("$\\hat v_{\\mathrm{obs}}$", "$\\hat v_{\\mathrm{inh}}$", "$\\hat H$")[j]  # the agent's game, as in the text
            text += f" ({game}: {abs(v0):.2f} $\\rightarrow$ {abs(v1):.2f})"
        left, right = axes[0, k].get_position(), axes[0, k + 1].get_position()
        fig.text((left.x0 + right.x1) / 2, left.y1 + 1.5 / 72 / height, text, ha="center", va="bottom", fontsize=6)  # title level
    obs = axes[0, 0].get_position()
    fig.text((obs.x0 + obs.x1) / 2, obs.y1 + 1.5 / 72 / height, f"$Q(o,a) = {Q_HAT[0]:.2f}$", ha="center", va="bottom", fontsize=5)  # level set below
    if not args.minesweeper:  # panel (d) of the merged paper figure, the latent UMAPs are (a)-(c)
        obs = axes[0, 0].get_position()
        y_obs = obs.y1 + 1.5 / 72 / height  # level with the "observation" title (pad 1.5 pt)
        y_mid = (obs.y0 + obs.y1) / 2  # level with the middle of the boards
        fig.text(0.0, y_mid, "(d)", ha="left", va="center", fontsize=7)
        # keeps the right margin under bbox_inches="tight", so the maps stay centered
        fig.text(1.0, y_mid, "(d)", ha="right", va="center", fontsize=7, color="none")
    # dashed separators in the spacer columns, between the observation and the pairs (phi^Q, phi^state, ...)
    for k in (0, 2, 4):
        left, right = axes[0, k].get_position(), axes[0, k + 1].get_position()
        x = (left.x1 + right.x0) / 2 if SCALE[0] != "own" or k == 0 else right.x0 - 0.035 / fig_w
        y0, y1 = axes[-1, k].get_position().y0, axes[0, k].get_position().y1 + 0.08 * k_y
        fig.add_artist(
            matplotlib.lines.Line2D([x, x], [y0, y1], transform=fig.transFigure, color=ms.INK2, lw=0.5, ls=(0, (3, 2)))
        )
    # one colorbar for every map: Shapley value / largest |phi| of its explanation, no outline
    top, bot = axes[0, -1].get_position(), axes[-1, -1].get_position()
    if SCALE[0] == "own":  # one colorbar per explanation, right of its model map, in raw values
        for k, vmax in zip((2, 4, 6), OWN_MAX):
            pos = axes[0, k].get_position()
            cax = fig.add_axes([pos.x1 + 0.05 / fig_w, bot.y0, 0.045 / fig_w, pos.y1 - bot.y0])
            sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.TwoSlopeNorm(0, -vmax, vmax), cmap=ms.DIVERGING)
            cb = fig.colorbar(sm, cax=cax)
            end = f"{vmax:.2f}" if round(vmax, 2) != 1 else "1"  # 2 decimals, none for 1
            cb.set_ticks([-vmax, 0, vmax], labels=[f"$-{end}$", "0", f"${end}$"])
            cb.ax.tick_params(labelsize=4.5, length=0, pad=1.5)  # no tick marks
            cb.outline.set_visible(False)
        bars = []
    elif SCALE[0] == "relative":
        bars = [(1.0, "$\\phi$ / largest $|\\phi|$", None)]
    elif SCALE[0] == "unit":
        bars = [(1.0, "$\\phi$", None)]
    else:  # raw: one bar for phi^obs and phi^inh (return variance), one for phi^H (nats)
        bars = [(RAW_MAX[0], "$\\phi^{\\mathrm{obs}}$, $\\phi^{\\mathrm{inh}}$", "{:.2f}"), (RAW_MAX[1], "$\\phi^{H}$ (nats)", "{:.2f}")]
    for j, (vmax, label, fmt) in enumerate(bars):
        cax = fig.add_axes([(off + grid_w + 0.103 + 0.5 * j) / fig_w, bot.y0, 0.062 / fig_w, top.y1 - bot.y0])
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.TwoSlopeNorm(0, -vmax, vmax), cmap=ms.DIVERGING)
        cb = fig.colorbar(sm, cax=cax)
        ticks = [-vmax, 0, vmax]
        cb.set_ticks(ticks, labels=["$-1$", "0", "1"] if fmt is None else [f"${fmt.format(t)}$" for t in ticks])
        cb.ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
        cb.outline.set_visible(False)
        cb.set_label(label, fontsize=4.5, labelpad=1)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"wrote {out}: tic-tac-toe board {t}, Minesweeper board {m}")


if __name__ == "__main__":
    main()
