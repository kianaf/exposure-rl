"""Which parts of the latent embedding encode the game-theoretic outcome classes?

Each learner board gets a class from two exact oracles:
  V_good       game value with both sides optimal (learner vs minimax)      in {+1, 0, -1}
  P(win/loss)  learner playing its best against the uniformly random opponent

  wins even if the opponent plays well    V_good = +1
  wins only if the opponent plays badly   V_good =  0 and P(win  | random) >= 0.5
  draws vs good play, wins < half vs rand V_good =  0 and P(win  | random) <  0.5
  loses only if the opponent plays well   V_good = -1 and P(loss | random) <  0.5
  loses >= half the time, even vs random   V_good = -1 and P(loss | random) >= 0.5

The agent's own occupancy contains almost no losing boards, so a one-network VAEAC-Q (milestone
2a setup, exact Q vs random) is trained on ALL reachable learner boards (uniform weights). The
embedding is the full-information belief mean E[z | x_N]. Outputs (results/figures/):

  7_outcome_regions.png   PCA map, linear-discriminant map, prototype board per class
and a linear-probe table (5-fold CV, LDA classifier) on the latent vs the raw board.

Usage: uv run python scripts/outcome_regions.py [--retrain]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import re
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from xrl.explainability.tictactoe import build_tictactoe_state_distribution

from exposure_rl.evaluate import full_information_q, load_checkpoint
from exposure_rl.oracle import ExpectimaxOracle
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, size_tag, train_supervised

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
SURFACE, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"

CLASSES = [
    ("wins even if the opponent plays well", "#184f95", "o"),
    ("wins only if the opponent plays badly", "#6da7ec", "s"),
    ("draws vs good play, wins < half the time vs random", "#7d7c76", "D"),
    ("loses only if the opponent plays well", "#f08c8b", "^"),
    ("loses at least half the time, even vs random play", "#a12c2b", "v"),
]
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 9,
        "axes.titlesize": 10,
    }
)


class UniformPolicy:
    """Uniform legal play: its occupancy covers every reachable learner-decision board."""

    def q_values(self, observation):
        return np.zeros(9)

    def action_distribution(self, observation, action_mask, epsilon: float = 0.0):
        m = np.asarray(action_mask, float).reshape(-1)
        return m / m.sum()

    def state_value(self, observation, action_mask):
        return 0.0


def outcome_class(good: ExpectimaxOracle, rnd: ExpectimaxOracle, obs) -> int:
    v_good = float(good.q_values(obs).max())
    first, second = rnd.action_moments(obs)
    a = int(np.nanargmax(first))
    win, loss = (second[a] + first[a]) / 2, (second[a] - first[a]) / 2
    if v_good > 0.5:
        return 0
    if v_good > -0.5:
        return 1 if win >= 0.5 else 2
    return 4 if loss >= 0.5 else 3


def lda_fit(x, y, k, reg=1e-3):
    """Fisher LDA: class means, shared covariance, projection onto the top discriminant axes."""

    classes = np.unique(y)
    means = np.stack([x[y == c].mean(0) for c in classes])
    centred = x - means[np.searchsorted(classes, y)]
    within = centred.T @ centred / len(x) + reg * np.eye(x.shape[1])
    between = (means - x.mean(0)).T @ np.diag([np.mean(y == c) for c in classes]) @ (means - x.mean(0))
    evals, evecs = np.linalg.eig(np.linalg.solve(within, between))
    order = np.argsort(-evals.real)
    return classes, means, np.linalg.inv(within), evecs.real[:, order[:k]]


def lda_cv_balanced_accuracy(x, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    pred = np.empty(len(x), int)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        classes, means, inv_w, _ = lda_fit(x[train], y[train], 1)
        prior = np.log(np.array([np.mean(y[train] == c) for c in classes]))
        d = x[test][:, None, :] - means[None]
        score = -0.5 * np.einsum("nkd,de,nke->nk", d, inv_w, d) + prior
        pred[test] = classes[np.argmax(score, 1)]
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls)), {int(c): float(r) for c, r in zip(np.unique(y), recalls)}


VALUE_CMAP = LinearSegmentedColormap.from_list(
    "value", ["#a12c2b", "#e34948", "#f4b0ae", "#a8a6a0", "#9ec5f4", "#2a78d6", "#184f95"]
)  # loss (red) -> draw (mid grey, visible on the light surface) -> win (blue), same poles as the classes


def ridge_cv_r2(x, t, folds=5, lam=1e-2, seed=0):
    """5-fold CV R^2 of a linear (ridge) readout of ``t`` from ``x``."""

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    pred = np.empty(len(x))
    xb = np.hstack([x, np.ones((len(x), 1))])
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        a = xb[train]
        coef = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ t[train])
        pred[test] = xb[test] @ coef
    return float(1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2))


def draw_board(ax, cells, fontsize=10, lw=0.8):
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 3, 3, fc=SURFACE, ec=INK2, lw=lw))
    for i in (1, 2):
        ax.axhline(i, color=INK2, lw=0.75 * lw)
        ax.axvline(i, color=INK2, lw=0.75 * lw)
    for c in range(9):
        r, k = divmod(c, 3)
        ax.text(
            k + 0.5,
            r + 0.5,
            {1: "X", -1: "O", 0: ""}[int(cells[c])],
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
        )


def group_ellipse(ax, xy, color, k=2.0, core=0.7):
    """Outline the core of a group of points: the ``k``-sigma covariance ellipse of the ``core`` share of points
    nearest the group median (so a few outliers do not stretch it). Returns the centre and the ellipse's bounding
    box (x0, x1, y0, y1), for placing a label outside it."""

    from matplotlib.patches import Ellipse

    med = np.median(xy, 0)
    d = np.linalg.norm((xy - med) / xy.std(0), axis=1)
    pts = xy[d <= np.quantile(d, core)]
    mean, cov = pts.mean(0), np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0])
    ax.add_patch(Ellipse(mean, w, h, angle=angle, fill=False, ec=color, lw=0.8, ls="--", zorder=5))
    t = np.linspace(0, 2 * np.pi, 200)
    c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    ring = mean + np.stack(
        [w / 2 * np.cos(t) * c - h / 2 * np.sin(t) * s_, w / 2 * np.cos(t) * s_ + h / 2 * np.sin(t) * c], 1
    )
    return mean, (ring[:, 0].min(), ring[:, 0].max(), ring[:, 1].min(), ring[:, 1].max())


def latent_axes(emb):
    """2-d map of the latent and its axis names: the latent itself when it is 2-d, else its first two PCs."""

    centre = emb.mean(0)
    if emb.shape[1] == 2:
        return emb - centre, ("$z_1$", "$z_2$")
    _, _, vt = np.linalg.svd(emb - centre, full_matrices=False)
    return (emb - centre) @ vt[:2].T, ("PC 1", "PC 2")


RESULT3 = np.array([0, 1, 1, 2, 2])  # five classes -> exact result against a perfect opponent: win / draw / loss
RESULT3_STYLE = (("win", "#184f95"), ("draw", "#7d7c76"), ("loss", "#a12c2b"))  # Figure 2 (a) since 2026-11-09

PAPER_CLASSES = (  # the same five classes: result against a perfect opponent; in brackets, vs a random opponent
    "win",
    "draw (often won)",
    "draw (rarely won)",
    "loss (rarely lost)",
    "loss (often lost)",
)


def tsne_map(emb, perplexity=30.0):
    """2-d t-SNE of the latent (sklearn, PCA initialisation, seed 0), rotated onto its own principal axes so the
    longer extent is horizontal. t-SNE axes carry no units, so the paper map has no ticks."""

    from sklearn.manifold import TSNE

    xy = TSNE(perplexity=perplexity, init="pca", random_state=0).fit_transform(emb)
    xy = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    return xy @ vt.T


def umap_map(emb, n_neighbors=30, min_dist=0.1):
    """2-d UMAP of the latent (seed 0; the settings of scripts/umap_latents.py), rotated onto its own principal axes
    so the longer extent is horizontal. UMAP axes carry no units, so the paper map has no ticks."""

    import umap

    xy = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0).fit_transform(emb)
    xy = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    return xy @ vt.T


TSNE_MAP_ASPECT = 0.55  # height / width of each t-SNE map in the paper figure (t-SNE scale is not metric)
UMAP_MAP_ASPECT = 0.45  # the same for UMAP, which keeps axis ticks and labels (flatter, so the figure height stays)


def plot_regions_paper(emb, y, v_exact, obs, out, centre_cell=None, projection="umap", cache_name="latest"):
    """Paper Figure 2 (2026-11-09 layout, chosen from a roomy preview): three UMAP maps of the latent,
    (a) exact result against a perfect opponent (win / draw / loss), (b) the agent's estimated max_a E_z[Q] (``v_exact`` argument), (c) who holds the centre,
    each legend (colour bar for (b)) beside its map. Drawn 7.5 in wide with 7-8 pt text; the paper includes it at
    text width. Points rasterized, the rest vector."""

    if projection == "umap":  # computed once (one thread, seed 0) and cached per checkpoint, so reruns keep the map
        # keyed by checkpoint, not by a hash of the latent: the latent is not bit-identical across GPU runs, and any
        # bit change gave a new UMAP layout
        nn, md = 100, 0.5  # chosen from a 10-setting sweep (2026-11-12): fills the width, clearest center panel
        cache = ROOT / "results" / "cache" / f"umap_paper_fig2_{cache_name}_nn{nn}_md{md}.npy"
        if cache.exists():
            pc = np.load(cache)
            assert len(pc) == len(emb), f"{cache} has {len(pc)} boards, the latent {len(emb)}"
        else:
            import umap

            xy = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=0, n_jobs=1).fit_transform(emb)
            xy = xy - xy.mean(0)
            _, _, vt = np.linalg.svd(xy, full_matrices=False)
            pc = xy @ vt.T
            np.save(cache, pc)
        axis_names = ("UMAP 1", "UMAP 2")
    else:
        pc, axis_names = latent_axes(emb)
    # maps 0.7 times the 2026-11-09 height, full width, legends inside the maps between the lobes
    # maps 0.8 times as wide, the freed width goes between them
    W, mw0 = 6.9, (6.9 - 0.3 - 2 * 0.2) / 3
    mw, mh, bottom = 0.8 * mw0, 0.7 * 0.66 * mw0, 0.3
    gap = (W - 0.3 - 3 * mw) / 2
    H = bottom + mh + 0.05
    xs = [0.3 + k * (mw + gap) for k in range(3)]
    with plt.rc_context(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "figure.facecolor": "none",  # transparent background (2026-11-10)
            "savefig.facecolor": "none",
            "axes.facecolor": "none",
            "text.usetex": True,  # Helvetica text and LaTeX math, as Figures 1 and 3
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{amsmath,amssymb}",
        }
    ):
        fig = plt.figure(figsize=(W, H))
        axs = [fig.add_axes([x / W, bottom / H, mw / W, mh / H]) for x in xs]
        result = RESULT3[y]  # win / draw / loss against a perfect opponent
        for c in (1, 0, 2):  # draws first, then wins and losses on top
            name, color = RESULT3_STYLE[c]
            sel = result == c
            axs[0].scatter(pc[sel, 0], pc[sel, 1], s=1.8, color=color, lw=0, alpha=0.85, rasterized=True, label=name)
        # below the gap between the lobes, down to the UMAP1 label, smaller text
        side = {"loc": "upper left", "bbox_to_anchor": (0.56, 0.24), "fontsize": 6.5,
                "markerscale": 2.0, "handletextpad": 0.2, "borderaxespad": 0.2, "borderpad": 0.3, "labelspacing": 0.3,
                "frameon": False}  # transparent
        handles, labels = axs[0].get_legend_handles_labels()
        order3 = [labels.index(n) for n, _ in RESULT3_STYLE]  # legend in the order win, draw, loss
        axs[0].legend([handles[k] for k in order3], [labels[k] for k in order3], **side)
        order = np.argsort(-v_exact)  # red on top
        sc = axs[1].scatter(pc[order, 0], pc[order, 1], c=v_exact[order], cmap=VALUE_CMAP,
                            norm=TwoSlopeNorm(0.0, -1.0, 1.0), s=1.8, lw=0, rasterized=True)
        cax = axs[1].inset_axes([0.6, -0.02, 0.34, 0.05], zorder=4)  # horizontal bar above the right lobe, clear of points
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
        cb.outline.set_visible(False)
        cax.patch.set_alpha(0.0)  # transparent background
        cb.ax.tick_params(labelsize=6, length=1.5, pad=1.5, labelcolor=INK, color=INK)
        cb.ax.set_xticklabels(["$-1$", "$0$", "$1$"])
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(r"estimated $\max_a Q$", fontsize=6.5, labelpad=2, color=INK)  # black like the legends  # legends one point bigger
        # empty center in light gray and drawn first, agent and opponent on top
        centre_style = {value: color for value, _, color in CENTRE} | {0: "#d6d5d0"}
        handles = {}
        for value in (0, 1, -1):
            sel = centre_cell == value
            handles[value] = axs[2].scatter(pc[sel, 0], pc[sel, 1], s=1.8, color=centre_style[value], lw=0, alpha=0.85,
                                            rasterized=True, label={1: "agent", 0: "empty", -1: "opponent"}[value])
        # (c): further left and higher, the title on its own with the entries indented below it
        axs[2].legend([handles[v] for v in (1, 0, -1)], ["agent", "empty", "opponent"], **{**side, "bbox_to_anchor": (0.57, 0.17)})
        axs[2].text(0.555, 0.17, "center", transform=axs[2].transAxes, ha="left", va="bottom", fontsize=6.5)
        # no frame, grid or ticks (UMAP coordinates carry no meaning): two small axis arrows in the lower-left corner
        arrow = dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0)
        for k, a in enumerate(axs):
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values():
                spine.set_visible(False)
            # panels (a)-(c) of the merged paper figure (2026-11-10), the Shapley maps are (d)
            a.text(-0.095, 0.94, f"({'abc'[k]})", transform=a.transAxes, ha="left", va="top", fontsize=10)  # flush with UMAP2
            a.annotate("", xy=(0.2, -0.04), xytext=(-0.02, -0.04), xycoords="axes fraction", arrowprops=arrow)
            a.annotate("", xy=(-0.02, 0.24), xytext=(-0.02, -0.04), xycoords="axes fraction", arrowprops=arrow)
            a.text(0.0, -0.08, r"\textbf{" + axis_names[0].replace(" ", "") + "}", transform=a.transAxes, ha="left",
                   va="top", fontsize=5.5, color=INK)
            a.text(-0.05, 0.0, r"\textbf{" + axis_names[1].replace(" ", "") + "}", transform=a.transAxes, ha="right",
                   va="bottom", rotation=90, fontsize=5.5, color=INK)  # axis names a bit smaller
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.01, transparent=True)
        plt.close(fig)


def describe(stem: str) -> str:
    """Plain-language name of a checkpoint for figure titles, e.g. 'TD agent vs perturbed minimax, seed 0'."""

    training = "TD agent" if stem.startswith("td_") else "agent trained on the exact Q"
    opponent = "perturbed minimax (5% random moves)" if "pm05" in stem else "the random opponent"
    seed = re.search(r"_s(\d+)$", stem)
    return f"{training} vs {opponent}" + (f", seed {seed.group(1)}" if seed else "")


def reachable_model(obs, q_exact, retrain, size=None):
    cfg = dataclasses.replace(Config(), seed=0, **(size or {}))
    model = VAEACQ(cfg)
    ckpt = ROOT / "results" / "models" / "reachable" / f"vaeac_q_reachable_m10{size_tag(cfg)}_s0.msgpack"
    if ckpt.exists() and not retrain:
        params = serialization.from_bytes(init_params(model, 0), ckpt.read_bytes())
    else:
        print(f"training one-network VAEAC-Q on all {len(obs)} reachable learner boards ...")
        _, params = train_supervised(obs, np.ones(len(obs)), q_exact, cfg, log_every=0)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(serialization.to_bytes(params))
    return model, params


GLYPH = {"o": "\u25cf", "s": "\u25a0", "D": "\u25c6", "^": "\u25b2", "v": "\u25bc"}  # marker -> text glyph


def wrap(name: str, lines: int = 3) -> str:
    """Split a class name into exactly ``lines`` lines of balanced length (so every example title has the same height)."""

    words = name.split()
    best = None
    for cuts in itertools.combinations(range(1, len(words)), lines - 1):
        parts = [" ".join(words[i:j]) for i, j in zip((0, *cuts), (*cuts, len(words)))]
        width = max(len(p) for p in parts)
        if best is None or width < best[0]:
            best = (width, parts)
    parts = best[1] if best else [name]
    return "\n".join(parts + [""] * (lines - len(parts)))


CENTRE = (  # tic-tac-toe centre cell from the agent's view (+1 = own piece, whichever side the agent plays)
    (1, "agent in the center", "#4aa9b4"),  # cyan / magenta / near-black: categories, kept apart from the
    (0, "center empty", "#2b2b2b"),  # blue-red value scale of panels (a), (b) and Figure 3
    (-1, "opponent in the center", "#b5508f"),
)


def plot_regions(
    emb,
    y,
    v_exact,
    classes,
    draw_proto,
    probe_text,
    value_title,
    readout_title,
    value_label,
    suptitle,
    out,
    centre_cell=None,
):
    """Outcome-region figure: PCA map coloured by class, by value and (tic-tac-toe, ``centre_cell``) by who holds the
    centre; one example board per class, which is the class legend. The probe numbers go to the JSON summary only.

    ``draw_proto(ax, i)`` draws observation ``i``. Returns the class means in latent space.
    """

    pc, axis_names = latent_axes(emb)
    means = np.stack([emb[y == c].mean(0) for c in np.unique(y)])

    pad = 0.04  # data margin around the points
    lo, hi = pc.min(0), pc.max(0)
    span = hi - lo
    xlim, ylim = (lo[0] - pad * span[0], hi[0] + pad * span[0]), (lo[1] - pad * span[1], hi[1] + pad * span[1])

    # fixed layout in inches: every map has the same size and the same scale on both PCs
    n_maps = 2 if centre_cell is None else 3
    map_w = 5.6
    map_h = map_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    legend_w, gap, cbar_gap = 2.7, 0.45, 1.25  # cbar_gap: room for the colour bar and its label after the value map
    left_m, bottom_m, top_m = 0.1, 0.55, 0.55
    fig_w = left_m + legend_w + 0.55 + n_maps * map_w + cbar_gap + (n_maps - 2) * gap + 0.1
    fig_h = bottom_m + map_h + top_m
    fig = plt.figure(figsize=(fig_w, fig_h))

    def place(x, w=map_w):  # an axes at x inches from the left, full map height
        return fig.add_axes([x / fig_w, bottom_m / fig_h, w / fig_w, map_h / fig_h])

    x0 = left_m + legend_w + 0.55
    ax = place(x0)
    for c, (name, color, marker) in enumerate(classes):
        sel = y == c
        ax.scatter(
            pc[sel, 0],
            pc[sel, 1],
            s=13,
            color=color,
            marker=marker,
            lw=0.3,
            ec=SURFACE,
            alpha=0.85,
            zorder=2,
            rasterized=True,
        )

    # the same map coloured by the agent's value
    ax_v = place(x0 + map_w + gap)
    order = np.argsort(-v_exact)  # wins (blue) first, losses (red) last, so red lies on top
    sc = ax_v.scatter(
        pc[order, 0],
        pc[order, 1],
        c=v_exact[order],
        cmap=VALUE_CMAP,
        norm=TwoSlopeNorm(0.0, -1.0, 1.0),
        s=11,
        lw=0.3,
        ec=SURFACE,
        rasterized=True,
    )
    cb = fig.colorbar(sc, cax=place(x0 + 2 * map_w + gap + 0.12, 0.18))
    cb.outline.set_visible(False)
    cb.set_label(value_label, color=INK2)
    maps = [ax, ax_v]
    if centre_cell is not None:
        ax_c = place(x0 + 2 * map_w + gap + cbar_gap)
        for value, name, color in CENTRE:
            sel = centre_cell == value
            ax_c.scatter(
                pc[sel, 0],
                pc[sel, 1],
                s=11,
                color=color,
                lw=0.3,
                ec=SURFACE,
                alpha=0.85,
                label=f"{name} ({sel.sum()})",
                rasterized=True,
            )
        ax_c.legend(frameon=False, fontsize=8, loc="lower left", markerscale=1.6)
        maps.append(ax_c)
    for a in maps:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.set_xlabel(axis_names[0])
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
    for letter, a in zip("abc", maps):  # panel letters, top left inside each map
        a.text(
            0.015,
            0.975,
            f"({letter})",
            transform=a.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylabel(axis_names[1])  # same axes everywhere: PC 2 label and tick numbers on the left map only
    for a in maps[1:]:
        a.tick_params(labelleft=False)

    # class legend: one example board per class (nearest the class centre) with its marker and label beside it
    row_h = map_h / len(classes)
    board_w = min(0.72, 0.8 * row_h)
    for c, (name, color, marker) in enumerate(classes):
        y_top = bottom_m + map_h - c * row_h
        board = fig.add_axes(
            [left_m / fig_w, (y_top - (row_h + board_w) / 2) / fig_h, board_w / fig_w, board_w / fig_h]
        )
        fig.text(
            (left_m + board_w + 0.1) / fig_w,
            (y_top - row_h / 2) / fig_h,
            f"{GLYPH[marker]} {wrap(name)}",
            fontsize=9.5,
            color=color,
            va="center",
            ha="left",
        )
        sel = np.flatnonzero(y == c)
        if len(sel) == 0:
            board.axis("off")
            board.text(0.5, 0.5, "no\nobservations", fontsize=7, color=INK2, ha="center", va="center")
            continue
        draw_proto(board, sel[np.argmin(np.linalg.norm(emb[sel] - emb[sel].mean(0), axis=1))])
    message, _ = suptitle  # the takeaway only; the model and data are in the file name and the JSON summary
    fig.suptitle(message, x=0.5, y=1 - 0.12 / fig_h, va="top", ha="center", color=INK, fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--paper-out", type=Path, default=None, help="also write the compact paper figure here (PDF)")
    parser.add_argument(
        "--title", default=None, help="the figure's message (its takeaway); a neutral default if omitted"
    )
    parser.add_argument(
        "--xrl-matched", action="store_true", help="reachable-boards model in the xrl Direct-Q configuration"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="embed with a sweep checkpoint (e.g. results/models/td_random_m10_s0.msgpack) instead of the "
        "model trained on all reachable boards",
    )
    parser.add_argument(
        "--projection", choices=("pca", "tsne", "umap"), default="pca", help="map of the latent in the paper figure"
    )
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    opp = "pm05" if args.checkpoint is not None and "pm05" in args.checkpoint.stem else "random"
    rnd, good = ExpectimaxOracle("random"), ExpectimaxOracle("minimax")
    agent_oracle = rnd if opp == "random" else ExpectimaxOracle(opp)  # the value the agent was trained on
    dist = build_tictactoe_state_distribution(UniformPolicy(), opponent="random")
    obs = np.array([p.observation for p in dist.points], dtype=np.float32)
    y = np.array([outcome_class(good, rnd, p.observation) for p in dist.points])
    q_exact = np.stack([agent_oracle.action_moments(p.observation)[0] for p in dist.points])
    agent_occ = {p.observation for p in agent_oracle.occupancy().points}
    in_occ = np.array([p.observation in agent_occ for p in dist.points])

    if args.checkpoint is not None:
        _, model, params = load_checkpoint(args.checkpoint)
        label, suffix = describe(args.checkpoint.stem), f"_{args.checkpoint.stem}"
    else:
        size = XRL_MATCHED if args.xrl_matched else None
        model, params = reachable_model(obs, q_exact, args.retrain, size)
        label = "model trained on all reachable boards"
        suffix = "_reachable_xrl" if size else ""

    log_pi, mu, _ = model.apply({"params": params}, jnp.asarray(obs), method=VAEACQ.full_belief)
    emb = np.asarray(jnp.sum(jnp.exp(log_pi)[..., None] * mu, axis=-2))  # (N, 16)
    raw = np.eye(3)[obs.astype(int) + 1].reshape(len(obs), -1)  # one-hot board baseline (27-d)
    v_exact_all = np.nanmax(q_exact, axis=1)
    np.savez_compressed(  # latents and labels for scripts/umap_latents.py
        ROOT / "results" / "cache" / f"latents_tictactoe{suffix or '_reachable'}.npz",
        emb=emb,
        y=y,
        value=v_exact_all,
        obs=obs,
        in_occupancy=in_occ,
    )

    v_exact = np.nanmax(q_exact, axis=1)  # exact V(x) vs the agent's opponent (the training target)
    q_model = full_information_q(model, params, obs)
    v_model = np.max(np.where(np.isnan(q_exact), -np.inf, q_model), axis=1)
    r2_latent, r2_raw = ridge_cv_r2(emb, v_exact), ridge_cv_r2(raw, v_exact)
    acc_latent, rec_latent = lda_cv_balanced_accuracy(emb, y)
    acc_raw, rec_raw = lda_cv_balanced_accuracy(raw, y)

    plot_regions(
        emb,
        y,
        np.clip(v_model, -1, 1),  # coloured by the agent's own value, from the same network as the latent
        CLASSES,
        lambda ax, i: draw_board(ax, obs[i]),
        f"latent $\\mathbb{{E}}[z \\mid x_N]$ (16-d): {acc_latent:.0%}\nraw board one-hot (27-d): {acc_raw:.0%}\n"
        f"chance: {1 / len(CLASSES):.0%}",
        "Coloured by the agent's estimated $\\max_a Q(x, a)$",
        f"Linear readout of $V$: $R^2$ = {r2_latent:.2f} from the latent, {r2_raw:.2f} from the raw board",
        "estimated $\\max_a Q$ (win +1, draw 0, loss -1)",
        (
            args.title or "Where the latent places each outcome",
            f"Tic-tac-toe, {label}; all 4520 reachable boards (X = learner to move)",
        ),
        FIG / f"7_outcome_regions{suffix}.png",
        centre_cell=obs[:, 4].astype(int),
    )
    if args.paper_out is not None:
        plot_regions_paper(
            emb,
            y,
            np.clip(v_model, -1, 1),
            obs,
            args.paper_out,
            centre_cell=obs[:, 4].astype(int),
            projection=args.projection,
            cache_name=args.checkpoint.stem if args.checkpoint is not None else "latest",
        )
        print(f"wrote {args.paper_out}")

    summary = {
        "boards": len(obs),
        "value_readout_r2": {"latent": r2_latent, "raw_board": r2_raw},
        "model_value_vs_exact": {
            "mae": float(np.mean(np.abs(v_model - v_exact))),
            "pearson": float(np.corrcoef(v_model, v_exact)[0, 1]),
        },
        "class_counts": {CLASSES[c][0]: int(np.sum(y == c)) for c in range(len(CLASSES))},
        "class_counts_in_agent_occupancy": {CLASSES[c][0]: int(np.sum((y == c) & in_occ)) for c in range(len(CLASSES))},
        "probe_balanced_accuracy": {"latent": acc_latent, "raw_board": acc_raw},
        "probe_recall_by_class": {
            "latent": {CLASSES[c][0]: r for c, r in rec_latent.items()},
            "raw_board": {CLASSES[c][0]: r for c, r in rec_raw.items()},
        },
    }
    (ROOT / "results" / f"outcome_regions{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
