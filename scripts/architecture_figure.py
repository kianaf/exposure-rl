"""Schematic of the VAEAC-Q architecture.

Full version (slides) → results/figures/architecture.{svg,pdf,png}; compact one-row version for the
ESANN paper (symbols only, explanations in the caption) → ../paper/figures/architecture.pdf.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parents[1] / "results" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
BLUE, BLUE_FILL = "#2a78d6", "#e3eefb"
ORANGE, ORANGE_FILL = "#eb6834", "#fdebe3"
AQUA, AQUA_FILL = "#1baf7a", "#e0f4ec"
GRAY_FILL = "#f0efec"
VIOLET, VIOLET_FILL = "#6250d6", "#ecE9fb"
CERTAINTY_FILL, ENTROPY_FILL = "#fbe7f3", "#e8d5f6"  # box fills: v_mask / v_state and H (icons: recolor_icons.py)
PURPLE, PINK = "#8a3fc0", "#b8458c"  # edges of the parts EXPOSURE adds: entropy, certainty games
CYAN, CYAN_FILL = "#1b9aaa", "#ddf3f5"  # variance and return-variance boxes in compact(); pink is the action there
ACTION = "#d4147e"  # a bold magenta for the action in compact()

plt.rcParams.update({"font.size": 10, "text.color": INK, "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, title, body="", edge=BLUE, fill=BLUE_FILL, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fill,
            ec=edge,
            lw=1.6,
            ls=(0, (4, 3)) if dashed else "-",
        )
    )
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", fontsize=10, color=INK)


def trap(ax, x, y, w, h, title, body="", edge=BLUE, fill=BLUE_FILL, dashed=False, narrow="right", inset=0.45):
    """Trapezoid block (encoder narrows towards the latent, decoder widens away from it)."""

    if narrow == "right":
        corners = [(x, y), (x + w, y + inset), (x + w, y + h - inset), (x, y + h)]
    else:  # narrow at the top
        corners = [(x, y), (x + w, y), (x + w - inset, y + h), (x + inset, y + h)]
    ax.add_patch(
        plt.Polygon(
            corners, closed=True, fc=fill, ec=edge, lw=1.6, ls=(0, (4, 3)) if dashed else "-", joinstyle="round"
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.32 - (inset if narrow != "right" else 0),
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.25, body, ha="center", va="center", fontsize=10, color=INK)


def arrow(ax, start, end, color=INK2, dashed=False, rad=0.0, label="", label_xy=None, lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            lw=lw,
            ls=(0, (4, 3)) if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=9, color=color)


def board(ax, x, y, cells, masked=(), size=0.36):
    for c in range(9):
        r, k = divmod(c, 3)
        cx, cy = x + k * size, y + (2 - r) * size
        hidden = c in masked
        ax.add_patch(plt.Rectangle((cx, cy), size, size, fc=GRAY_FILL if hidden else SURFACE, ec=INK2, lw=0.8))
        mark = "?" if hidden else {1: "X", -1: "O", 0: ""}[cells[c]]
        ax.text(
            cx + size / 2,
            cy + size / 2,
            mark,
            ha="center",
            va="center",
            fontsize=9,
            color=INK2 if hidden else INK,
            fontweight="bold",
        )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 23)
    ax.set_ylim(-0.7, 10.4)
    ax.axis("off")
    cells = [-1, 0, -1, 0, 0, 0, 1, 0, 1]

    # Row labels
    ax.text(0.2, 9.9, "Agent at full information — acting and training", fontsize=12, fontweight="bold", color=ORANGE)
    ax.text(
        0.2,
        3.55,
        "Explanation path — any coalition $C$ of visible cells",
        fontsize=12,
        fontweight="bold",
        color=BLUE,
        bbox={"fc": SURFACE, "ec": "none", "pad": 3},
        zorder=5,
    )
    # Level labels above the columns
    ax.text(7.75, 9.25, "level 1: latent belief", ha="center", fontsize=10, color=INK2, style="italic")
    ax.text(
        6.1,
        8.85,
        "VAEAC = belief network (encoder) + $z$ + decoder, trained with KL + reconstruction",
        ha="center",
        fontsize=8.6,
        color=VIOLET,
    )
    ax.text(11.4, 9.25, "level 2: predicted return", ha="center", fontsize=10, color=INK2, style="italic")

    def heads(y, note):
        box(ax, 10.0, y + 1.25, 2.8, 1.05, "Q head", "$Q_\\omega(z, a)$", edge=AQUA, fill=AQUA_FILL)
        box(ax, 10.0, y, 2.8, 1.05, "variance head", "$s_\\omega(z, a) \\geq 0$", edge=AQUA, fill=AQUA_FILL)
        ax.text(11.4, y - 0.12, note, ha="center", va="top", fontsize=8.6, color=INK2)

    # ---------------- top row: full information
    board(ax, 0.55, 6.55, cells)
    ax.text(1.09, 6.25, "observation $o$", ha="center", va="top", fontsize=10, color=INK2)
    trap(
        ax,
        2.6,
        6.4,
        3.0,
        2.3,
        "belief network",
        "$e_\\psi(z \\mid o_C, C)$\nall cells visible ($C = N$)",
        edge=VIOLET,
        fill=VIOLET_FILL,
        narrow="right",
    )
    box(
        ax,
        6.5,
        6.4,
        2.5,
        2.3,
        "belief over $z$",
        "mixture of\n$\\mathcal{N}(\\mu_m, \\sigma_m^2)$,\n$z \\sim e_\\psi(\\cdot \\mid o_N)$",
        edge=INK2,
        fill=GRAY_FILL,
    )
    heads(6.4, "")
    box(
        ax,
        13.8,
        7.45,
        3.9,
        1.25,
        "act",
        "$\\arg\\max_{a\\ \\mathrm{legal}} \\mathbb{E}_z[Q_\\omega(z, a)]$",
        edge=ORANGE,
        fill=SURFACE,
    )
    box(
        ax,
        13.8,
        4.75,
        3.9,
        2.45,
        "head targets (training)",
        "$Q_\\omega \\leftarrow$ exact $Q$, or TD target\n$r + \\max_{a'} \\mathbb{E}_z[Q^-_\\omega(z, a')]$\n$s_\\omega \\leftarrow$ exact $\\mathrm{Var}(G \\mid o, a)$\n(supervised runs only)",
        edge=GRID,
        fill=SURFACE,
        dashed=True,
    )
    arrow(ax, (1.65, 7.55), (2.6, 7.55))
    arrow(ax, (5.6, 7.55), (6.5, 7.55))
    arrow(ax, (9.0, 7.9), (10.0, 8.2))
    arrow(ax, (9.0, 7.2), (10.0, 6.95))
    arrow(ax, (12.8, 8.2), (13.8, 8.1), color=ORANGE)
    arrow(ax, (12.8, 6.95), (13.8, 6.2), color=GRID, dashed=True)

    # ---------------- bottom row: coalition / explanation
    board(ax, 0.55, 0.75, cells, masked=(1, 3, 4, 5))
    ax.text(
        1.09,
        0.45,
        "visible cells $o_C$ + mask $C$\n(action cell always visible)",
        ha="center",
        va="top",
        fontsize=9,
        color=INK2,
    )
    trap(
        ax,
        2.6,
        0.6,
        3.0,
        2.3,
        "belief network",
        "$e_\\psi(z \\mid o_C, C)$\n$M$-component mixture",
        edge=VIOLET,
        fill=VIOLET_FILL,
        narrow="right",
    )
    box(
        ax,
        6.5,
        0.6,
        2.5,
        2.3,
        "belief over $z$",
        "wider when cells\nare hidden,\n$z \\sim e_\\psi(\\cdot \\mid o_C)$",
        edge=INK2,
        fill=GRAY_FILL,
    )
    heads(0.6, "same weights as above")
    box(
        ax,
        13.8,
        -0.35,
        4.3,
        3.25,
        "coalition games",
        "$K$ shared samples $\\varepsilon$\n\n$\\hat Q(C) = \\mathbb{E}_z[Q_\\omega]$\n"
        "$v_{\\mathrm{obs}}(C) = \\mathrm{Var}_z[Q_\\omega]$\n"
        "$v_{\\mathrm{inh}}(C) = \\mathbb{E}_z[s_\\omega]$\n\n"
        "$\\mathrm{Var}(G \\mid o_C) = v_{\\mathrm{obs}} + v_{\\mathrm{inh}}$",
        edge=BLUE,
        fill=SURFACE,
    )
    box(
        ax,
        18.9,
        -0.35,
        3.9,
        3.25,
        "Shapley over cells",
        "all $2^n$ coalitions, signed\n\n"
        "$\\phi^{Q}_i$: return shift\n"
        "$\\phi^{\\mathrm{obs}}_i$, $\\phi^{\\mathrm{inh}}_i$: certainty\n\n"
        "$\\phi^{\\mathrm{tot}} = \\phi^{\\mathrm{obs}} + \\phi^{\\mathrm{inh}}$",
        edge=BLUE,
    )
    arrow(ax, (1.65, 1.75), (2.6, 1.75), color=BLUE)
    arrow(ax, (5.6, 1.75), (6.5, 1.75), color=BLUE)
    arrow(ax, (9.0, 2.1), (10.0, 2.4), color=BLUE)
    arrow(ax, (9.0, 1.4), (10.0, 1.15), color=BLUE)
    arrow(ax, (12.8, 2.4), (13.8, 2.0), color=BLUE)
    arrow(ax, (12.8, 1.15), (13.8, 0.95), color=BLUE)
    arrow(ax, (18.1, 1.25), (18.9, 1.25), color=BLUE)

    # shared heads and belief network
    ax.plot([12.95, 12.95], [2.9, 6.4], color=AQUA, lw=1.6, ls=(0, (1.5, 2.5)))
    ax.text(12.8, 4.65, "one set of\nweights", fontsize=9, color=AQUA, va="center", ha="right")
    ax.plot([4.6, 4.6], [2.9, 6.4], color=VIOLET, lw=1.6, ls=(0, (1.5, 2.5)))
    ax.text(4.75, 4.65, "one set of\nweights", fontsize=9, color=VIOLET, va="center")

    # ---------------- training-only couplings
    arrow(ax, (3.3, 6.4), (3.3, 2.9), color=GRID, dashed=True)
    ax.text(
        3.15,
        4.65,
        "$\\mathrm{KL}(e_\\psi(\\cdot \\mid o_N) \\,\\|\\, e_\\psi(\\cdot \\mid o_C))$\ncoalition belief must\ncover every completion",
        fontsize=8.6,
        color=INK2,
        va="center",
        ha="right",
    )
    trap(
        ax,
        6.0,
        3.85,
        3.6,
        2.1,
        "decoder (training)",
        "$p_\\theta(o_{N \\setminus C} \\mid z, o_C)$\nreconstructs hidden cells",
        edge=GRID,
        fill=SURFACE,
        dashed=True,
        narrow="top",
    )
    arrow(ax, (7.75, 6.4), (7.75, 5.95), color=GRID, dashed=True)

    # ---------------- legend / notes
    notes = [
        (ORANGE, "—", "agent acts (full observation)"),
        (BLUE, "—", "explanation only (any coalition);\nmasked games never trained directly"),
        (GRID, "- -", "training only (head targets, KL,\nreconstruction); ADR 0015 masks"),
    ]
    for i, (color, style, text) in enumerate(notes):
        y = 9.55 - i * 0.66
        ax.plot([18.3, 18.8], [y, y], color=color, lw=2, ls=(0, (4, 3)) if style == "- -" else "-")
        ax.text(18.95, y, text, fontsize=8.6, color=INK2, va="center")
    ax.text(
        18.3,
        7.35,
        "Grand coalition $C = N$ is the agent's own\nbelief: same network, full mask.\n\n"
        "$v_{\\mathrm{inh}}$ = hidden state (Minesweeper)\n      + randomness (tic-tac-toe opponent).\n\n"
        "Training $z$: one reparameterised sample per\nfull-belief component, weighted by $w_m$.\n\n"
        "Latent $\\mu_m, \\sigma_m$ are not explained per\ncoordinate (not identifiable); they act\n"
        "through $Q_\\omega$ as $v_{\\mathrm{obs}}$.",
        fontsize=8.6,
        color=INK2,
        va="top",
    )

    fig.savefig(FIG / "architecture.svg", bbox_inches="tight")
    fig.savefig(FIG / "architecture.pdf", bbox_inches="tight")
    fig.savefig(FIG / "architecture.png", dpi=180, bbox_inches="tight")


XMIN, XMAX = 0.9, 22.13  # drawing area cropped to the content, same scale (2026-11-17)
TOP = 6.1  # room for the column headers above the outputs (2026-11-17)
TEXT_SCALE = 0.78  # LaTeX math is wider than mathtext: shrink all compact-figure text alike


def compact(out: Path | None = None) -> None:
    """One-row paper schematic, 12.2 cm wide; details live in the LaTeX caption."""

    out = out or FIG.parents[2] / "paper" / "figures" / "architecture.pdf"
    with plt.rc_context(
        {
            "font.size": 7,
            "pdf.fonttype": 42,
            "text.usetex": True,  # LaTeX: Helvetica text, LaTeX math
            "font.family": "sans-serif",
            "text.latex.preamble": r"\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{amsmath,amssymb}",
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8 * (XMAX - XMIN) / 24.0, 1.6 * 5.9 / 7.4 * (TOP - 1.65) / 5.95))  # same scale as before, shorter  # same scale, shorter: z and the decoder are lower
        fig.patch.set_facecolor("white")
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(1.65, TOP)
        ax.axis("off")

        units_xy = (1.6 * 5.9 / 7.4 / 5.95) / (4.8 / 24.0)  # inches per y unit / inches per x unit

        def node(x, y, w, h, text, edge, fill, size=6.5, dashed=False):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.25",
                    fc=fill,
                    ec=edge,
                    lw=0.9,
                    ls=(0, (3, 2)) if dashed else "-",
                )
            )
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size * TEXT_SCALE, color=INK)

        def trapezoid(corners, text, edge, fill, size=6.5, dashed=False, centre=None):
            ax.add_patch(
                plt.Polygon(
                    corners, closed=True, fc=fill, ec=edge, lw=0.9, ls=(0, (3, 2)) if dashed else "-", joinstyle="round"
                )
            )
            cx, cy = centre or np.mean(np.asarray(corners), axis=0)
            ax.text(cx, cy, text, ha="center", va="center", fontsize=size * TEXT_SCALE, color=INK)

        def network_glyph(cx, cy, w=0.44, h=0.34, vertical=False):
            """Small layered-network icon marking a trainable part (nodes 2-3-2, fully connected); ``vertical``:
            rotated by 90 degrees (layers stacked bottom to top), with the same on-paper size."""
            layers = [np.linspace(-0.3, 0.3, 2), np.linspace(-0.5, 0.5, 3), np.linspace(-0.3, 0.3, 2)]
            if vertical:  # the axes are not equal-scaled: keep the icon's printed aspect when rotating
                w, h = h * units_xy, w / units_xy
            xs = [cx - w / 2, cx, cx + w / 2]

            def pt(x, y):
                return (cx + (y - cy) * (w / h) * 0 + (y - cy) / h * w / 1, cy + (x - cx) / w * h) if vertical else (x, y)

            for (x0, ys0), (x1, ys1) in zip(zip(xs, layers), zip(xs[1:], layers[1:])):
                for y0 in ys0:
                    for y1 in ys1:
                        if vertical:
                            ax.plot([cx + y0 * w, cx + y1 * w], [cy - h / 2 + (x0 - xs[0]) / (xs[2] - xs[0]) * h,
                                    cy - h / 2 + (x1 - xs[0]) / (xs[2] - xs[0]) * h], color=INK2, lw=0.35, zorder=4)
                        else:
                            ax.plot([x0, x1], [cy + y0 * h, cy + y1 * h], color=INK2, lw=0.35, zorder=4)
            for x, ys in zip(xs, layers):
                if vertical:
                    ax.scatter(cx + ys * w, [cy - h / 2 + (x - xs[0]) / (xs[2] - xs[0]) * h] * len(ys), s=2.2,
                               color=INK, zorder=5, linewidths=0)
                else:
                    ax.scatter([x] * len(ys), cy + ys * h, s=1.5, color=INK, zorder=5, linewidths=0)

        def link(a, b, color=INK2, dashed=False, dotted=False):
            ax.add_patch(
                FancyArrowPatch(
                    a,
                    b,
                    arrowstyle="-|>",
                    mutation_scale=6,
                    color=color,
                    lw=0.8,
                    shrinkA=0,
                    shrinkB=0,
                    ls=(0, (1, 1.2)) if dotted else (0, (3, 2)) if dashed else "-",
                    zorder=0.8,  # arrows behind the shapes
                )
            )

        # inputs: the full observation o_N (training, top, dashed) and the masked o_C (explanations, bottom); both go
        # through the same encoder. Training draws z from the full belief and pulls the coalition belief to it (KL),
        # explanations draw z from the coalition belief (2026-11-13, so the two routes are visible).
        size = 0.42
        A, B = -0.5, -0.75
        C = B - 0.27  # right of a narrower Q-head
        D = 0.14  # right of the VAE: lower, header tops level with the VAE background  # moved left for the map legend
        # the board of Figure 2d (1728) and EXPOSURE's maps for it (scripts/architecture_maps.py, 2026-11-17)
        maps = np.load(FIG / "architecture_maps.npz")
        full_board = [str(v) for v in maps["labels"]]

        def draw_board(xc, yc, cells, masked, edge=INK2):
            for c, v in enumerate(cells):
                r, k = divmod(c, 3)
                hidden = c in masked
                x0, y0 = xc - 1.5 * size + k * size, yc - 1.5 * size + (2 - r) * size
                ax.add_patch(plt.Rectangle((x0, y0), size, size, fc=GRAY_FILL if hidden else "white", ec=edge, lw=0.5))
                mark = "?" if hidden else v
                ax.text(x0 + size / 2, y0 + size / 2, mark, ha="center", va="center", fontsize=3.8 * TEXT_SCALE,
                        color=INK2 if hidden else INK)

        # o_C is made from o_N by masking: o_N below, an arrow up to o_C; the action cell stays visible
        # closer to the encoder, no o_N / o_C labels: the mask arrow says which is which
        draw_board(1.75, 5.25, full_board, (1, 4, 5, 8), edge=BLUE)  # coalition: blue, like its paths
        draw_board(1.75, 3.05, full_board, ())
        link((1.75, 3.72), (1.75, 4.58), BLUE)
        ax.text(1.75, 2.3, "agent O\nto move", ha="center", va="top", linespacing=1.0, fontsize=4.6 * TEXT_SCALE,
                color=INK2)  # whose turn
        ax.text(1.66, 4.15, "mask", ha="right", va="center", fontsize=4.2 * TEXT_SCALE, color=BLUE)

        # VAEAC: encoder (belief network) -> z -> decoder
        ax.add_patch(
            FancyBboxPatch(
                (2.55, 1.75),
                9.8 + A,
                4.15,
                boxstyle="round,pad=0.02,rounding_size=0.3",
                fc="#f4f3f0",  # a pale background instead of the dashed box
                ec="none",
                zorder=0.3,
            )
        )
        ax.text(
            2.8,
            5.55,
            "VAE with arbitrary conditioning",
            ha="left",
            va="center",
            fontsize=5.0 * TEXT_SCALE,
            color=INK,
        )
        # encoder narrows towards the bottleneck (wide input side on the left)
        trapezoid(
            [(3.35, 2.95), (4.95, 3.2), (4.95, 4.0), (3.35, 4.25)],  # smaller (2026-11-17)
            "encoder",
            INK,
            "white",
            centre=(4.15, 3.4),
            size=5.2,
        )
        # the two beliefs of the same encoder (2026-11-13): training samples z from the full belief (dashed) and pulls
        # the coalition belief to it (KL); explanations sample z from the coalition belief (solid)
        node(5.55, 3.855, 1.95, 0.75, "", BLUE, BLUE_FILL)  # name over symbol (2026-11-17)
        ax.text(5.68, 4.23, "masked\nbelief", ha="left", va="center", linespacing=1.1, fontsize=4.6 * TEXT_SCALE, color=INK)
        # each belief drawn as a small density instead of its symbol: masked = wide, several modes,
        # full = one narrow peak
        # small filled densities (2026-11-17): both Gaussian mixtures, masked wide with several modes, full one narrow
        # skewed peak; drawn only where the density is visible, so no flat tails
        def density(xc, y0, comps, color, height=0.22):
            t = np.linspace(-1, 1, 400)
            d = sum(w * np.exp(-0.5 * ((t - m) / sd) ** 2) for w, m, sd in comps)
            d = d / d.max()
            keep = d > 0.004
            xs, ys = xc + 0.4 * t[keep], y0 + height * d[keep]
            ax.fill_between(xs, y0, ys, color=color, alpha=0.25, lw=0)
            ax.plot(xs, ys, color=color, lw=0.7, solid_capstyle="round")

        density(6.98, 4.09, [(0.7, -0.5, 0.17), (1.0, 0.0, 0.18), (0.6, 0.5, 0.16)], BLUE)
        node(5.55, 2.615, 1.95, 0.75, "", INK, GRAY_FILL)
        ax.text(5.68, 2.99, "full\nbelief", ha="left", va="center", linespacing=1.1, fontsize=4.6 * TEXT_SCALE, color=INK)
        density(6.98, 2.82, [(1.0, -0.08, 0.15), (0.55, 0.18, 0.13)], INK, height=0.3)  # bigger (2026-11-17)
        ax.annotate("", xy=(6.525, 3.83), xytext=(6.525, 3.39),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0))
        ax.text(6.7, 3.57, "KL", ha="left", va="center", fontsize=4.0 * TEXT_SCALE, color=INK2)
        node(8.2, 3.27, 0.8, 0.66, "$z$", INK, GRAY_FILL, 6.2)  # smaller (2026-11-17)
        # decoder right of z: the encoder's mirror image (narrow side at z, widening to the right, towards the cells)
        trapezoid(
            [(9.9 + A, 2.6), (11.5 + A, 2.35), (11.5 + A, 3.65), (9.9 + A, 3.4)],  # right and lower (2026-11-17)
            "decoder",
            INK,
            "white",
            centre=(10.7 + A, 2.82),
            size=5.2,
        )
        # the KL term is a training loss (Eq. loss in the paper), not drawn
        link((4.95, 3.85), (5.55, 4.23), BLUE)
        link((4.95, 3.35), (5.55, 2.99))
        link((7.5, 4.23), (8.2, 3.85), BLUE)  # explanations: z from the coalition belief
        link((7.5, 2.99), (8.2, 3.35))
        link((9.0, 3.45), (9.9 + A, 3.0))  # into the middle of the decoder (2026-11-17)

        # heads on z, then one row per coalition game with its Shapley values:
        # Q head -> mean and spread of Q over the belief; variance head -> mean of s; the belief itself -> entropy
        # Q-head: one network on z with two outputs, the mean Q and the variance s
        # a square-cornered block (an MLP head), unlike the trapezoids of the encoder and decoder
        ax.add_patch(plt.Rectangle((12.9 + B, 3.575 - D), 1.33, 1.5, fc="white", ec=INK, lw=0.9))
        ax.text(13.565 + B, 4.03 - D, "Q-head", ha="center", va="center", fontsize=4.8 * TEXT_SCALE, color=INK)
        # trainable parts carry the network glyph; inputs, latents and outputs do not
        network_glyph(4.15, 3.8)
        network_glyph(10.7 + A, 3.22)  # identical icon in every trainable part
        network_glyph(13.565 + B, 4.62 - D)
        # the Q-head's outputs in the paper's notation, boxes the size of the targets they align with (2026-11-17)
        node(15.2 + C, 4.935 - D, 1.7, 0.68, "mean\n$Q_\\omega(z,a)$", INK, GRAY_FILL, 4.2)  # level with the expected return target
        node(15.2 + C, 3.035 - D, 1.7, 0.68, "variance\n$s_\\omega(z,a)$", CYAN, CYAN_FILL, 4.2)  # level with the inherent target
        # the action is the argmax of the mean: a dotted flag under the mean box; not on the input,
        # which has no action yet, but on the maps
        ax.add_patch(FancyArrowPatch((15.6 + C, 4.935 - D), (15.98 + C, 4.4 - D), arrowstyle="-|>", mutation_scale=4,
                                     color=ACTION, lw=0.5, connectionstyle="arc3,rad=0.35", shrinkA=0, shrinkB=0,
                                     zorder=0.8))  # solid, thin, curved
        ax.text(16.02 + C, 4.4 - D, "action", ha="left", va="center", fontsize=4.2 * TEXT_SCALE, color=ACTION)  # from the arrow head
        link((14.5 + C, 4.85 - D), (15.2 + C, 5.275 - D))
        link((14.5 + C, 3.8 - D), (15.2 + C, 3.375 - D))
        # colour by novelty: black = existing (VAEAC, Q head and mean game, Shapley values), violet = added by EXPOSURE
        # (variance head and the three new games); blue/red stay reserved for value signs in Figures 2 and 3
        SLATE, PHI_FILL = "#4a4f57", "#f3f3f1"
        rows = (  # (y of the row, target, edge, fill); the games are named, their symbols are defined in the text
            (4.875 - D, "expected\nreturn", INK, GRAY_FILL),
            (3.925 - D, "observational\nuncertainty", CYAN, CYAN_FILL),
            (2.975 - D, "inherent\nuncertainty", CYAN, CYAN_FILL),
            (2.025 - D, "belief\nuncertainty", PURPLE, ENTROPY_FILL),
        )
        # Shapley values as EXPOSURE's maps of the board (Figure 2d's colors and relative scale, no marks), 2026-11-17
        diverging = matplotlib.colors.LinearSegmentedColormap.from_list(
            "div", ["#a12c2b", "#e34948", "#f4b0ae", "#f0efec", "#9ec5f4", "#2a78d6", "#184f95"]
        )  # scripts/minesweeper.py DIVERGING
        light = lambda u: 0.6 * np.asarray(diverging(u)) + 0.4  # lighter map colors, as if transparent on white
        units = (1.6 * 5.9 / 7.4 / 5.95) / (4.8 / 24.0)  # inches per y unit / inches per x unit
        for k, (y, name, edge, fill) in enumerate(rows):
            node(17.6 + C, y + 0.06, 2.15, 0.68, name, edge, fill, 4.5)  # a bit shorter (2026-11-17)
            h = 0.8
            w = h * units  # square on paper
            x0, y0 = 20.45 + C, y + 0.4 - h / 2  # targets and maps moved left (2026-11-17)
            norm = matplotlib.colors.TwoSlopeNorm(0, -1, 1)
            for c, v in enumerate(maps["model"][k]):  # one square per cell (an image left white slivers at the lines)
                r, j = divmod(c, 3)
                acted = c == int(maps["action"])  # the action cell has no Shapley value: pink with a
                ax.add_patch(plt.Rectangle((x0 + j * w / 3, y0 + (2 - r) * h / 3), w / 3, h / 3,
                                           fc=ACTION if acted else light(norm(v)), ec="none", zorder=3))
                if acted:
                    ax.text(x0 + (j + 0.5) * w / 3, y0 + (2.5 - r) * h / 3, "$a$", ha="center", va="center",
                            fontsize=3.8 * TEXT_SCALE, color="white", zorder=6)
                # no X/O marks on the maps
            for t in (1, 2):
                ax.plot([x0 + t * w / 3] * 2, [y0, y0 + h], color=INK2, lw=0.5, zorder=4)
                ax.plot([x0, x0 + w], [y0 + t * h / 3] * 2, color=INK2, lw=0.5, zorder=4)
            ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=INK2, lw=0.5, zorder=5))  # outline like the input boards
            link((19.75 + C, y + 0.4), (x0 - 0.05, y + 0.4), INK2)
        # one legend for every map instead of the puzzle pieces: Figure 2d's colors, blue raises
        # two colorbars, each as tall as its maps: return (top map), uncertainty (the three below)
        lx, lw_ = x0 + w + 0.2, 0.16
        pos, neg = "$+$", "$-$"  # the sign of the contribution, instead of arrows
        for b0, b1, word, top, bottom in ((4.875 - D, 5.675 - D, "return", pos, neg),
                                          (2.025 - D, 4.725 - D, "uncertainty", neg, pos)):
            for t in np.linspace(0, 1, 80, endpoint=False):
                ax.add_patch(plt.Rectangle((lx, b0 + t * (b1 - b0)), lw_, (b1 - b0) / 80 + 0.002,
                                           fc=light(t + 0.5 / 80), ec="none", zorder=3))
            ax.add_patch(plt.Rectangle((lx, b0), lw_, b1 - b0, fill=False, ec=INK2, lw=0.4, zorder=4))
            ax.text(lx + lw_ + 0.05, b1, top, ha="left", va="top", fontsize=4.2 * TEXT_SCALE, color=INK2)
            ax.text(lx + lw_ + 0.05, b0, bottom, ha="left", va="bottom", fontsize=4.2 * TEXT_SCALE, color=INK2)
            ax.text(lx + lw_ + 0.4, (b0 + b1) / 2, word, ha="left", va="center", rotation=90, fontsize=3.7 * TEXT_SCALE,
                    color=INK2)
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(1.65, TOP)
        # column headers
        for x, y, text, ha in ((16.05 + C, 5.82 - D, "model outputs", "center"), (18.675 + C, 5.82 - D, "targets", "center"),
                               (20.45 + C + 0.43, 5.82 - D, "explanations", "center")):  # over the maps and the puzzle pieces
            ax.text(x, y, text, ha=ha, va="bottom", fontsize=4.8 * TEXT_SCALE, color=INK2, linespacing=1.0)
        link((2.4, 3.55), (3.35, 3.55))  # training and acting: the full observation
        link((2.4, 5.2), (3.35, 4.1), BLUE)  # explanations: the masked observation  # training: the full observation, same encoder
        # z -> Q-head from the right side of z (2026-11-17), under the UMAP
        # mirror image of the arrow to the decoder (slope 0.45 / 0.9 from 0.15 above the centre of z), then level
        ax.plot([9.0, 9.9 + A, 12.8 + B], [3.75, 4.2 - D, 4.2 - D], color=INK2, lw=0.8, solid_joinstyle="round", zorder=0.8)
        link((12.8 + B, 4.2 - D), (12.9 + B, 4.2 - D))  # straight (2026-11-17)  # into the middle of the Q-head (4.0-5.9)  # ... and right into the Q-head, above the decoder
        link((16.9 + C, 5.275 - D), (17.6 + C, 5.275 - D), INK2)
        link((16.9 + C, 5.0 - D), (17.6 + C, 4.325 - D), INK2)
        link((16.9 + C, 3.375 - D), (17.6 + C, 3.375 - D), INK2)
        # the belief over the hidden cells (decoder imputation) gives the entropy game: straight along the bottom
        ax.plot([11.5 + A, 12.65 + A, 12.65 + A], [3.1, 3.1, 2.425 - D], color=INK2, lw=0.8, solid_joinstyle="round", zorder=0.8)
        link((12.65 + A, 2.425 - D), (17.6 + C, 2.425 - D), INK2)  # from the decoder's wide side down to the H row
        # the decoder's completed boards go back through the encoder (full observation) to the Q-head:
        # v_obs is the variance of the agent's Q over them (2026-11-09; replaces the formulas e_psi, p_theta)
        # out of the decoder's output side (right), below its main output, then back along the bottom (dotted)
        dotted = (0, (1, 1.5))
        # full observations in gray (o_N, completed boards for v_obs), the coalition route in blue (2026-11-13)
        ax.plot([11.5 + A, 12.12 + A, 12.12 + A, 2.75, 2.75, 3.17], [2.55, 2.55, 1.9, 1.9, 3.15, 3.15], color=BLUE, lw=0.8,
                solid_joinstyle="round", zorder=0.8)  # blue: completions of a coalition
        link((3.15, 3.15), (3.35, 3.15), BLUE)
        # the latent: a schematic 2-d map coloured by value, in Figure 2b's colors (not its layout)
        value_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "value", ["#a12c2b", "#e34948", "#f4b0ae", "#a8a6a0", "#9ec5f4", "#2a78d6", "#184f95"]
        )  # scripts/outcome_regions.py VALUE_CMAP
        # the latent belief: the encoder's two beliefs and z on a transparent tint, named
        ax.add_patch(FancyBboxPatch((5.2, 2.25), 4.1, 3.07, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=ENTROPY_FILL, ec="none", alpha=0.6, zorder=0.5))  # the belief box colors
        ax.text(5.37, 5.07, "latent belief", ha="left", va="center", fontsize=4.9 * TEXT_SCALE, color=PURPLE)
        ax.add_patch(FancyBboxPatch((10.3 + A, 4.5), 1.5, 1.28, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=ENTROPY_FILL, ec="none", alpha=0.6, zorder=0.5))  # the same tint as the latent belief
        uax = ax.inset_axes([10.42 + A, 4.58, 1.3, 1.12], transform=ax.transData)  # about square on paper
        rng = np.random.default_rng(0)
        centres = np.array([[-0.9, 0.7], [0.75, 0.55], [0.1, -0.85]])  # win-like, mixed and loss-like regions
        values = np.array([0.9, 0.1, -0.8])
        k = rng.integers(0, 3, 60)
        xy = centres[k] + rng.normal(0, 0.32, (60, 2))
        q = np.clip(values[k] + rng.normal(0, 0.25, 60), -1, 1)
        order = np.argsort(-q)  # red on top, as Figure 2b
        uax.scatter(xy[order, 0], xy[order, 1], c=q[order], cmap=value_cmap,
                    norm=matplotlib.colors.TwoSlopeNorm(0.0, -1.0, 1.0), s=1.4, lw=0)
        uax.set_xlim(-1.75, 1.6)
        uax.set_ylim(-1.6, 1.6)
        uax.axis("off")
        axis_arrow = dict(arrowstyle="-|>", color=INK2, lw=0.5, mutation_scale=3.5, shrinkA=0, shrinkB=0)
        uax.annotate("", xy=(0.32, 0.0), xytext=(-0.01, 0.0), xycoords="axes fraction", arrowprops=axis_arrow)  # 2-d map:
        uax.annotate("", xy=(-0.01, 0.36), xytext=(-0.01, 0.0), xycoords="axes fraction", arrowprops=axis_arrow)  # two axes
        ax.add_patch(FancyArrowPatch((8.6, 5.15), (10.26 + A, 5.15), arrowstyle="-|>", mutation_scale=4, color=PURPLE, lw=0.5,
                                     connectionstyle="arc3,rad=-0.35", shrinkA=0, shrinkB=0))  # solid, thin
        ax.text(2.92, 2.27, "completed\nboards", ha="left", va="center", multialignment="left", linespacing=1.0, fontsize=4.6 * TEXT_SCALE, color=BLUE)

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.01, dpi=600)  # the rasterized UMAP points
        plt.close(fig)


if __name__ == "__main__":
    main()
    compact()
