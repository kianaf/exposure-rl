"""The compared networks side by side (width 256), layer by layer, with exact parameter counts.

Layer shapes are read from the trained checkpoints (seed 0), not restated: EXPOSURE (vaeac_q.VAEACQ), the
mask-aware DQN (xrl Direct-Q), and DQN + surrogate (standard xrl DQN, exposure_rl.scalar_surrogate). Drawn for
tic-tac-toe (wide layers elided); the second page lists every block in both domains.

Usage: uv run python scripts/architecture_detail.py -> results/figures/architectures_detail.pdf
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from flax.traverse_util import flatten_dict
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

from exposure_rl.evaluate import load_checkpoint
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
P = RES / "philipp"
_spec = importlib.util.spec_from_file_location("ablations", ROOT / "scripts" / "replay_matching_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#7d7c76"
SLATE = "#4f5d6b"
AQUA = "#1baf7a"
VIOLET, VIOLET_LIGHT = "#6250d6", "#a99ff0"
ORANGE = "#eb6834"
BLUE = "#2a78d6"
RC = {
    "font.size": 9,
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage[scaled]{helvet}\renewcommand{\familydefault}{\sfdefault}\usepackage{amsmath,amssymb}",
}


# ---------------------------------------------------------------- parameter trees


def dense_layers(params: dict, prefix: str) -> list[tuple[int, int]]:
    """``(fan_in, fan_out)`` of each Dense under ``prefix``, in order."""

    flat = flatten_dict(params)
    kernels = sorted((k, v.shape) for k, v in flat.items() if k[-1] == "kernel" and "/".join(k).startswith(prefix))
    return [tuple(int(s) for s in shape) for _, shape in kernels]


def n_params(layers: list[tuple[int, int]]) -> int:
    return sum(i * o + o for i, o in layers)


def load_all() -> dict:
    """Per domain and network: the list of Dense ``(in, out)`` shapes, straight from the checkpoints."""

    out = {}
    _, _, p = load_checkpoint(RES / "models" / "td_random_m10_var_xrl256_s0.msgpack")
    cfg = dataclasses.replace(Config(), n_values=10, n_cells=16, var_weight=50.0, **{**XRL_MATCHED, "hidden": 256})
    model = VAEACQ(cfg)
    p_ms = serialization.from_bytes(
        init_params(model, 0),
        (RES / "models" / "minesweeper" / "vaeac_q_ms4x4_3_var_xrl256_td_m10_s0.msgpack").read_bytes(),
    )
    for dom, params, pre in (("tictactoe", p, "ttt"), ("minesweeper", p_ms, "ms4x4_3")):
        out[dom] = {
            "belief": dense_layers(params, "belief_net"),
            "decoder": dense_layers(params, "decoder"),
            # shared heads (default): the Q-head trunk and its Q output; the variance output layer
            "q_head": dense_layers(params, "q_head") or dense_layers(params, "head_trunk") + dense_layers(params, "q_out"),
            "var_head": dense_layers(params, "var_head") or dense_layers(params, "var_out"),
            "direct_q": dense_layers(
                ev.DirectQGames(P / f"{pre}_directq_scalar_w256" / "seed_0").policy.params, "params"
            ),
            "dqn": dense_layers(ev.StandardDQN(P / f"{pre}_dqn_scalar_w256" / "seed_0").policy.params, "params"),
            "surrogate": dense_layers(
                ev.ScalarSurrogateGames(
                    P
                    / (
                        f"{pre}_dqn_surrogate_scalar_eps10_w256"
                        if dom == "tictactoe"
                        else f"{pre}_dqn_surrogate_scalar_w256"
                    )
                    / "seed_0"
                ).surrogate.params,
                "params",
            ),
        }
    return out


# ---------------------------------------------------------------- drawing

SHOW = 7  # nodes drawn per layer: wide layers show the first 5, an ellipsis, and the last 2
ROW = 1.25  # vertical distance between layers (same in every panel, so heights compare)


def layer(ax, xc, y, n, colours=SLATE, width=2.6, label="", label_side="right", size=26):
    """One layer as a row of nodes; layers wider than SHOW are elided in the middle. Returns the node positions."""

    k = min(n, SHOW)
    xs = np.linspace(xc - width / 2, xc + width / 2, k + (1 if n > SHOW else 0))
    if n > SHOW:
        ax.text(xs[5], y, r"$\cdots$", ha="center", va="center", fontsize=10, color=INK2)
        xs = np.delete(xs, 5)
    cols = colours if isinstance(colours, list) else [colours] * k
    ax.scatter(xs, np.full(k, y), s=size, c=cols, edgecolors="white", linewidths=0.6, zorder=3)
    if label:
        x = xc + width / 2 + 0.25 if label_side == "right" else xc - width / 2 - 0.25
        ax.text(
            x,
            y,
            label,
            ha="left" if label_side == "right" else "right",
            va="center",
            fontsize=7.5,
            color=INK2,
            bbox={"fc": "white", "ec": "none", "pad": 0.5},
            zorder=5,
        )
    return np.stack([xs, np.full(k, y)], axis=1)


def connect(ax, a, b, params=None, xlab=None):
    """All edges between the drawn nodes of two layers (the weights of one Dense layer)."""

    seg = np.stack([np.repeat(a, len(b), axis=0), np.tile(b, (len(a), 1))], axis=1)
    ax.add_collection(LineCollection(seg, colors=EDGE, linewidths=0.35, alpha=0.45, zorder=1))
    if params is not None:
        x = xlab if xlab is not None else a[:, 0].min() - 0.3
        ax.text(x, (a[0, 1] + b[0, 1]) / 2, f"{params:,}", ha="right", va="center", fontsize=6.5, color=EDGE)


def stack(ax, xc, y0, layers, in_label, out_label, in_colours=BLUE, out_colours=AQUA, width=2.6, first=None):
    """An MLP drawn bottom-up: input row (or ``first``, already drawn), hidden ReLU rows, output row."""

    prev = first if first is not None else layer(ax, xc, y0, layers[0][0], in_colours, width, in_label)
    for i, (fan_in, fan_out) in enumerate(layers):
        last = i == len(layers) - 1
        row = layer(
            ax,
            xc,
            y0 + (i + 1) * ROW,
            fan_out,
            out_colours if last else SLATE,
            width,
            out_label if last else f"{fan_out}, ReLU",
        )
        connect(ax, prev, row, fan_in * fan_out + fan_out)
        prev = row
    return prev


def title(ax, x, y, text, count, colour=INK):
    ax.text(x, y + 0.28, rf"\textbf{{{text}}}", ha="center", va="bottom", fontsize=9, color=colour)
    ax.text(x, y, f"{count:,} parameters", ha="center", va="bottom", fontsize=7, color=INK2)


def panel_exposure(ax, L):
    b, dec, qh, vh = L["belief"], L["decoder"], L["q_head"], L["var_head"]
    n, d = qh[-1][1], qh[0][0]
    m = b[-1][1] // (2 * d + 1)
    n_values = dec[-1][1] // n
    xc = 6.0
    # belief network, bottom
    head_cols = [ORANGE, ORANGE, VIOLET, VIOLET, VIOLET, VIOLET_LIGHT, VIOLET_LIGHT]
    top = stack(
        ax,
        xc,
        0.0,
        b,
        rf"{b[0][0]}: $[x\odot s,\ s]$",
        rf"{b[-1][1]}: $\log\pi$ ({m}), $\mu$, $\log\sigma$ (${m}\times{d}$ each)",
        out_colours=head_cols,
    )
    ax.text(
        xc - 2.5,
        1.5 * ROW,
        r"\textbf{Belief network}" + "\n" + r"$e_\psi(z \mid o_C)$",
        ha="right",
        va="center",
        fontsize=8.5,
        color=VIOLET,
    )
    ax.text(xc - 2.5, 1.5 * ROW - 0.75, f"{n_params(b):,} parameters", ha="right", va="center", fontsize=7, color=INK2)
    # latent: sampled, no weights
    yz = 4 * ROW + 0.4
    z = layer(ax, xc, yz, d, VIOLET, 2.0, rf"$z$ ({d}): $\mu_k + \sigma_k \odot \varepsilon$, no weights", size=34)
    for p in top:
        ax.annotate(
            "", xy=(xc, yz - 0.12), xytext=p, arrowprops={"arrowstyle": "-", "color": VIOLET, "lw": 0.5, "ls": ":"}
        )
    # three heads on top of z
    y_heads = yz
    for x, layers, name, out_label, colour in (
        (2.0, dec, r"Decoder $p_\theta(x \mid z, o_C)$", rf"{dec[-1][1]}: ${n}\times{n_values}$ logits", BLUE),
        (6.0, qh, r"Q head $Q_\omega(z,a)$", rf"{n}: $Q$", AQUA),
        (10.0, vh, r"Variance head $s_\omega(z,a)$", rf"{n}: $s$ (softplus)", AQUA),
    ):
        w = 2.2
        h1 = layer(ax, x, y_heads + ROW, layers[0][1], SLATE, w, "")
        connect(ax, z, h1)
        if layers is dec:  # the decoder also sees the observed cells and the mask
            ax.annotate(
                "",
                xy=(x - w / 2 - 0.1, y_heads + ROW),
                xytext=(xc - 1.4, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": BLUE,
                    "lw": 0.8,
                    "ls": "--",
                    "connectionstyle": "arc3,rad=-0.3",
                },
            )
            ax.text(
                0.0,
                2.2,
                r"$+\,[x\odot s,\ s]$" + "\n" + rf"(input {layers[0][0]})",
                ha="center",
                fontsize=7,
                color=BLUE,
            )
        h2 = layer(ax, x, y_heads + 2 * ROW, layers[1][1], SLATE, w, "")
        connect(ax, h1, h2)
        out = layer(ax, x, y_heads + 3 * ROW, layers[2][1], colour, w, "")
        connect(ax, h2, out)
        ax.text(x, y_heads + 3 * ROW + 0.25, out_label, ha="center", va="bottom", fontsize=7, color=INK2)
        title(ax, x, y_heads + 3 * ROW + 0.75, name, n_params(layers), colour)
    ax.text(12.0, y_heads + 1.5 * ROW, "256, ReLU\n\n256, ReLU", ha="left", va="center", fontsize=7.5, color=INK2)
    total = sum(n_params(v) for v in (b, dec, qh, vh))
    ax.set_title(
        rf"\textbf{{EXPOSURE}}: {total:,} parameters"
        + "\n"
        + rf"\small acts with belief network + Q head ({n_params(b) + n_params(qh):,})",
        fontsize=12,
        pad=8,
    )
    ax.set_xlim(-1.2, 14.5)


def panel_masked_dqn(ax, L):
    dq = L["direct_q"]
    stack(ax, 2.0, 0.0, dq, rf"{dq[0][0]}: $[x\odot s,\ s]$", rf"{dq[-1][1]}: $Q(o_C, a)$")
    ax.set_title(
        rf"\textbf{{Mask-aware DQN}}: {n_params(dq):,} parameters" + "\n" + r"\small one network; acts at $s = 1$",
        fontsize=12,
        pad=8,
    )
    ax.set_xlim(-1.5, 6.5)


def panel_surrogate(ax, L):
    dqn, sur = L["dqn"], L["surrogate"]
    stack(ax, 2.0, 0.0, dqn, rf"{dqn[0][0]}: $x$", "", width=2.4)
    ax.text(2.0, 3 * ROW + 0.25, rf"{dqn[-1][1]}: $Q(o, a)$", ha="center", va="bottom", fontsize=7, color=INK2)
    title(ax, 2.0, 3 * ROW + 0.75, "(1) DQN, acts", n_params(dqn), AQUA)
    stack(ax, 8.4, 0.0, sur, "", rf"{sur[-1][1]}: $\hat Q(o_C, a)$", width=2.4)
    ax.text(8.4, -0.3, rf"{sur[0][0]}: $[x\odot s,\ s]$", ha="center", va="top", fontsize=7.5, color=INK2)
    title(ax, 8.4, 3 * ROW + 0.75, "(2) masked surrogate", n_params(sur), AQUA)
    ax.annotate(
        "",
        xy=(6.6, 3 * ROW),
        xytext=(3.6, 3 * ROW),
        arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.9, "ls": "--"},
    )
    ax.text(
        5.1,
        3 * ROW - 0.2,
        "distilled after\ntraining on\n" + r"$\varepsilon$-greedy boards",
        ha="center",
        va="top",
        fontsize=7,
        color=INK2,
    )
    ax.set_title(
        rf"\textbf{{DQN + surrogate}}: {n_params(dqn) + n_params(sur):,} parameters"
        + "\n"
        + r"\small DQN acts; surrogate explains",
        fontsize=12,
        pad=8,
    )
    ax.set_xlim(-1.0, 12.3)


def page_side_by_side(pdf, L):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), gridspec_kw={"width_ratios": [15.7, 8.0, 12.5], "wspace": 0.04})
    panel_exposure(axes[0], L)
    panel_masked_dqn(axes[1], L)
    panel_surrogate(axes[2], L)
    for ax in axes:
        ax.set_ylim(-0.9, 4 * ROW + 0.4 + 3 * ROW + 1.4)
        ax.axis("off")
    fig.text(
        0.5,
        0.02,
        r"Tic-tac-toe, width 256. Each row is one layer (wide layers elided: $\cdots$); every drawn edge stands for the "
        r"Dense weights between two rows, grey numbers = weights + biases of that layer. "
        r"$[x\odot s,\ s]$: cell values zeroed where hidden, plus the mask $s$. TD target copies not counted.",
        ha="center",
        fontsize=8,
        color=INK2,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_table(pdf, A):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    def shapes(layers):
        return r" $\to$ ".join([str(layers[0][0])] + [str(o) for _, o in layers])

    rows = []
    for dom in ("tictactoe", "minesweeper"):
        L = A[dom]
        blocks = [
            ("EXPOSURE", "belief network", L["belief"]),
            ("", "Q head", L["q_head"]),
            ("", "variance head", L["var_head"]),
            ("", "decoder", L["decoder"]),
            ("Mask-aware DQN", "Q network", L["direct_q"]),
            ("DQN + surrogate", "DQN", L["dqn"]),
            ("", "masked surrogate", L["surrogate"]),
        ]
        total = {
            "EXPOSURE": sum(n_params(L[k]) for k in ("belief", "q_head", "var_head", "decoder")),
            "Mask-aware DQN": n_params(L["direct_q"]),
            "DQN + surrogate": n_params(L["dqn"]) + n_params(L["surrogate"]),
        }
        rows.append(
            (rf"\textbf{{{'Tic-tac-toe' if dom == 'tictactoe' else 'Minesweeper 4x4, 3 mines'}}}", "", "", "", "")
        )
        for method, block, layers in blocks:
            rows.append(
                (method, block, shapes(layers), f"{n_params(layers):,}", f"{total[method]:,}" if method else "")
            )
        rows.append(("", "", "", "", ""))
    y = 0.95
    heads = ("Method", "Block", "Dense layers (ReLU between)", "Parameters", "Method total")
    xs = (0.02, 0.2, 0.38, 0.75, 0.88)
    for x, t in zip(xs, heads):
        ax.text(x, y, rf"\textbf{{{t}}}", transform=ax.transAxes, fontsize=10, va="top")
    y -= 0.045
    ax.plot([0.02, 0.98], [y + 0.012, y + 0.012], transform=ax.transAxes, color=INK2, lw=0.6)
    for r in rows:
        for x, t in zip(xs, r):
            ax.text(x, y, t, transform=ax.transAxes, fontsize=9, va="top")
        y -= 0.034
    ax.text(
        0.02,
        y - 0.01,
        "Parameters = sum over Dense layers of fan-in x fan-out + fan-out (weights + biases), read from the seed-0 "
        "checkpoints.\nTarget-network copies used for TD bootstrapping are synced, not trained, and not counted.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=INK2,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    A = load_all()
    out = RES / "figures" / "architectures_detail.pdf"
    with plt.rc_context(RC), PdfPages(out) as pdf:
        page_side_by_side(pdf, A["tictactoe"])
        page_table(pdf, A)
    print("wrote", out)


if __name__ == "__main__":
    main()
