"""EXPOSURE's four maps of the Figure 2d board for the overview figure (2026-11-17).

Runs the tic-tac-toe row of scripts/paper_figure_cells.py with the drawing replaced by a recorder and keeps the model
maps on Figure 2d's relative scale (each explanation divided by its largest |phi| over exact and model), so the colors
match Figure 2d. Order Q, obs, inh, H as in Table 1.

Usage: uv run python scripts/architecture_maps.py [--board 1728] [--seed 2] -> results/figures/architecture_maps.npz
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cells", ROOT / "scripts" / "paper_figure_cells.py")
cells = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cells)

RECORD: dict = {}


def recorder(axes, rows, cols, labels, a, q_hat, maps, h_maps, domain, titles, q_maps):
    pairs = [list(q_maps), [maps[2], maps[3]], [maps[0], maps[1]], list(h_maps)]  # as draw_row
    scale = [max(float(np.abs(y).max()) for y in p) or 1.0 for p in pairs]
    RECORD.update(model=np.stack([np.asarray(p[1]) / s for p, s in zip(pairs, scale)]), labels=np.array(labels),
                  action=int(a))


cells.draw_row = recorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=int, default=1728)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    cells.tictactoe_row(None, 256, args.seed, args.board)
    out = ROOT / "results" / "figures" / "architecture_maps.npz"
    np.savez(out, **RECORD)
    print(out, RECORD["labels"], RECORD["action"], np.round(RECORD["model"], 2))


if __name__ == "__main__":
    main()
