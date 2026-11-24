"""UMAP of the agent's full-information belief mean E[z | o_N], tic-tac-toe and Minesweeper.

Reads the latents saved by scripts/outcome_regions.py and scripts/minesweeper_outcome_regions.py
(results/cache/latents_*.npz). One row per model, three panels each:

  tic-tac-toe (all 4520 reachable learner boards)   outcome class | exact V vs random | centre cell
  Minesweeper 4x4 / 4 mines (optimal occupancy)     outcome class | exact inherent uncertainty
                                                    v_state(N) | revealed cells (game stage)

Classes and colours are labels only (never model input). UMAP seed 0; tic-tac-toe 30 neighbours, min_dist
0.1; Minesweeper 500 neighbours, min_dist 0.9 (chosen by a sweep; smaller neighbourhoods fragment the map).
Usage: uv run python scripts/umap_latents.py   ->   results/figures/10_umap_latents.png
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import umap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
FIG = ROOT / "results" / "figures"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regions, ms_regions = _load("outcome_regions"), _load("minesweeper_outcome_regions")
SURFACE, INK, INK2, GRID = regions.SURFACE, regions.INK, regions.INK2, regions.GRID


UMAP_PARAMS = {  # per domain; Minesweeper has 119k discrete boards and fragments at small neighbourhoods
    "tictactoe": {"n_neighbors": 30, "min_dist": 0.1},
    "minesweeper": {"n_neighbors": 500, "min_dist": 0.9},
}


def embed(emb, domain):
    return umap.UMAP(**UMAP_PARAMS[domain], random_state=0).fit_transform(emb)


def scatter_classes(ax, xy, y, classes):
    counts = np.bincount(y, minlength=len(classes))
    for c in np.argsort(-counts):  # largest class first, minorities on top
        sel = y == c
        if not sel.any():
            continue
        name, color, marker = classes[c]
        ax.scatter(
            xy[sel, 0],
            xy[sel, 1],
            s=3 if counts[c] > 2000 else 10,
            color=color,
            marker=marker,
            lw=0,
            alpha=0.8,
            label=f"{name} ({counts[c]})",
            rasterized=True,
        )
    ax.legend(frameon=False, fontsize=7, loc="best", markerscale=2.5)


def scatter_values(fig, ax, xy, values, cmap, norm, label):
    order = np.argsort(np.abs(values - (norm.vcenter if hasattr(norm, "vcenter") else 0)))
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], cmap=cmap, norm=norm, s=3, lw=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(label, color=INK2, fontsize=8)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xrl-matched", action="store_true", help="tic-tac-toe models in the xrl Direct-Q configuration"
    )
    args = parser.parse_args()
    if args.xrl_matched:
        return main_xrl()
    rows = [
        ("tictactoe", CACHE / "latents_tictactoe_td_random_m10_s0.npz", "tic-tac-toe, TD agent (one network)"),
        ("tictactoe", CACHE / "latents_tictactoe_reachable.npz", "tic-tac-toe, model trained on all reachable boards"),
        (
            "minesweeper",
            CACHE / "latents_minesweeper_optimal_4x4_4_var_s0.npz",
            "Minesweeper 4x4, 4 mines (optimal occupancy)",
        ),
    ]
    plot_rows(rows, FIG / "10_umap_latents.png")


def main_xrl() -> None:
    rows = [
        (
            "tictactoe",
            CACHE / "latents_tictactoe_td_random_m10_xrl_s0.npz",
            "tic-tac-toe, TD agent",
        ),
        (
            "tictactoe",
            CACHE / "latents_tictactoe_reachable_xrl.npz",
            "tic-tac-toe, model trained on all reachable boards",
        ),
    ]
    plot_rows(rows, FIG / "10_umap_latents_xrl.png")


def plot_rows(rows, out) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.6 * len(rows)), squeeze=False)
    for r, (domain, path, title) in enumerate(rows):
        d = np.load(path)
        xy = embed(d["emb"], domain)
        ax0, ax1, ax2 = axes[r]
        if domain == "tictactoe":
            scatter_classes(ax0, xy, d["y"], regions.CLASSES)
            scatter_values(
                fig, ax1, xy, d["value"], regions.VALUE_CMAP, TwoSlopeNorm(0, -1, 1), "exact $V$ vs random opponent"
            )
            centre = d["obs"][:, 4]
            for v, (name, color) in {
                1: ("learner holds centre", "#2a78d6"),
                0: ("centre empty", "#a9a8a0"),
                -1: ("opponent holds centre", "#e34948"),
            }.items():
                sel = centre == v
                ax2.scatter(
                    xy[sel, 0], xy[sel, 1], s=3, color=color, lw=0, label=f"{name} ({sel.sum()})", rasterized=True
                )
            ax2.legend(frameon=False, fontsize=7, markerscale=2.5)
            titles = ("outcome class", "value", "centre cell")
        else:
            scatter_classes(ax0, xy, d["y"], ms_regions.CLASSES)
            v = d["v_inherent_full"]
            zero = v < 1e-12  # sure outcomes: no inherent uncertainty left
            ax1.scatter(xy[zero, 0], xy[zero, 1], s=2, color=GRID, lw=0, rasterized=True, label=f"none ({zero.sum()})")
            sc = ax1.scatter(
                xy[~zero, 0],
                xy[~zero, 1],
                c=v[~zero],
                cmap="Blues",
                norm=matplotlib.colors.Normalize(0, float(v.max())),
                s=8,
                lw=0,
                rasterized=True,
                label=f"> 0 ({(~zero).sum()})",
            )
            cb = fig.colorbar(sc, ax=ax1, fraction=0.045, pad=0.02)
            cb.outline.set_visible(False)
            cb.set_label("exact inherent uncertainty $v_{\\mathrm{inh}}(N)$", color=INK2, fontsize=8)
            ax1.legend(frameon=False, fontsize=7, markerscale=2.5)
            stage = (d["obs"] != -1).sum(1)
            scatter_values(
                fig,
                ax2,
                xy,
                stage.astype(float),
                "Purples",
                matplotlib.colors.Normalize(0, float(stage.max())),
                "revealed cells",
            )
            titles = ("outcome class (exact P(win))", "agent's own uncertainty", "game stage")
        for ax, t in zip(axes[r], titles):
            ax.set_title(f"{title}\n{t}", loc="left", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
        print(f"embedded {path.name}: {len(xy)} points", flush=True)
    fig.suptitle(
        "UMAP of the full-information belief mean $\\mathbb{E}[z \\mid o_N]$ (labels are colour coding only)",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
