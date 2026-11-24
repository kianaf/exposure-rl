"""Why is EXPOSURE's v_obs weaker than replay matching on the same Q? (2026-11-06)

Both are scored against the exact v_obs game of the agent's OWN full-information Q (Table 1, "own" and "replay"),
so Q error is already removed: the gap is the coalition belief. This script asks what kind of error it is:

- level: is the belief's v_obs too large (belief too broad) or too small, per coalition size?
- shape: Pearson r per coalition size (scale-free), so a global over-breadth alone would not lower it;
- floor: v_obs at the full coalition, which is 0 exactly; the model's value there is pure belief noise.

Usage: uv run python scripts/vobs_diagnostic.py [--domains random pm05 ms]
Writes results/vobs_diagnostic.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
from flax import serialization
from scipy.stats import pearsonr

from exposure_rl import coalitions
from exposure_rl.evaluate import OracleReference, full_information_q, load_checkpoint, tictactoe_checkpoint
from exposure_rl.minesweeper import HIDDEN, Geometry
from exposure_rl.minesweeper_exact import load_oracle, scored_indices, tag
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, size_tag

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "replay_matching_eval.py")
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)

SIZE = {**XRL_MATCHED, "hidden": 256}
REPLAY = 10_000


def setting(name: str, m: int = 10, beta: float = 1.0):
    """(points, weights, scored obs, actions, legal_of, n, model loader) for one Table 1 block."""

    suffix = size_tag(dataclasses.replace(Config(), **SIZE))
    if name in ("random", "pm05"):
        ref = OracleReference(name)

        def load(seed):
            cfg = dataclasses.replace(Config(), seed=seed, **SIZE)
            path = tictactoe_checkpoint("td", seed, size_tag(cfg), name, td_var=True)
            path = path.with_name(path.name.replace("_m10_", f"_m{m}_"))
            if beta == 1.0:
                _, model, params = load_checkpoint(path)
                return model, params
            cfg = dataclasses.replace(cfg, n_components=m, var_weight=50.0, beta=beta)
            model = VAEACQ(cfg)
            path = path.with_name(path.name.replace(f"_s{seed}.", f"_b{beta:g}_s{seed}."))
            return model, serialization.from_bytes(init_params(model, seed), path.read_bytes())

        legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
        return ref.obs, ref.w, ref.obs, ref.actions, legal_of, 9, load
    g = Geometry(4, 4, 3)
    oracle = load_oracle(g, ROOT / "results" / "cache")
    points = oracle["points"]
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(oracle["weights"])

    def load(seed):
        cfg = dataclasses.replace(Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **SIZE)
        model = VAEACQ(cfg)
        path = ROOT / "results" / "models" / "minesweeper" / f"vaeac_q_ms{tag(g)}_var{suffix}_td_m10_s{seed}.msgpack"
        return model, serialization.from_bytes(init_params(model, seed), path.read_bytes())

    legal_of = lambda a: points[:, a] == HIDDEN
    return points.astype(np.float32), oracle["weights"], points[scored].astype(np.float32), actions[scored], legal_of, 16, load


def by_size(v, sizes, n, w):
    """Occupancy-weighted mean of a game per coalition size (sizes 0..n)."""

    return [float(w @ v[:, sizes == k].mean(1)) for k in range(n + 1)]


def run(name: str, m: int = 10, beta: float = 1.0) -> dict:
    points, weights, obs, actions, legal_of, n, load = setting(name, m, beta)
    sizes = coalitions.coalition_sizes(n)
    interior = sizes < n
    # occupancy weight of each scored observation (Minesweeper: scored subset, uniform as in the paper's pooled r)
    w_obs = np.ones(len(obs)) / len(obs)
    out = {}
    for seed in range(3):
        model, params = load(seed)
        q_agent = full_information_q(model, params, points)
        _, v_exact = abl.own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
        v_model = abl.exposure_vmask_games(model, params, obs, actions, n)
        _, v_rep = abl.replay_games(points, weights, q_agent, obs, actions, legal_of, n, REPLAY, seed)
        row = {}
        for label, v in (("belief", v_model), ("replay", v_rep)):
            per_size_r = {}
            for k in range(1, n):
                m = sizes == k
                a, b = v[:, m].ravel(), v_exact[:, m].ravel()
                per_size_r[k] = float(pearsonr(a, b).statistic) if a.std() > 1e-12 and b.std() > 1e-12 else None
            row[label] = {
                "r_interior": float(pearsonr(v[:, interior].ravel(), v_exact[:, interior].ravel()).statistic),
                "mean_by_size": by_size(v, sizes, n, w_obs),
                "r_by_size": per_size_r,
                "ratio_to_exact_interior": float(v[:, interior].mean() / v_exact[:, interior].mean()),
                "full_coalition": float(v[:, -1].mean()),
            }
        row["exact_mean_by_size"] = by_size(v_exact, sizes, n, w_obs)
        out[seed] = row
        b, r = row["belief"], row["replay"]
        print(
            f"{name} M={m} beta={beta:g} s{seed}: r belief {b['r_interior']:.3f} replay {r['r_interior']:.3f} | "
            f"level belief/exact {b['ratio_to_exact_interior']:.2f} replay/exact {r['ratio_to_exact_interior']:.2f} | "
            f"belief at C=N {b['full_coalition']:.4f}",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["random", "pm05", "ms"])
    parser.add_argument("--components", type=int, default=10, help="mixture size M of the checkpoints (tic-tac-toe)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight of the checkpoints (tic-tac-toe)")
    args = parser.parse_args()
    path = ROOT / "results" / "vobs_diagnostic.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for name in args.domains:
        key = name + (f"_m{args.components}" if args.components != 10 else "") + (f"_b{args.beta:g}" if args.beta != 1.0 else "")
        result[key] = run(name, args.components, args.beta)
        path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
