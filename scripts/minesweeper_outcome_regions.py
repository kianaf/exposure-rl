"""Which parts of the Minesweeper latent encode the outcome? (Analogue of figure 7 for tic-tac-toe.)

There is no opponent; the luck is the hidden mines. Each decision observation of Minesweeper 4x4 / 3
mines gets a class from its exact win probability under belief-optimal play, P(win | o):

  wins for sure            P = 1        (the clues certify a safe path to the end)
  wins unless unlucky      1/2 <= P < 1
  loses unless lucky       0 < P < 1/2
  loses for sure           P = 0

Classes are colour coding only (never model input or target). The embedding is the full-information
belief mean E[z | o_N] of a trained one-network model; the value row is coloured by
P(win) - P(loss) = 2 P(win) - 1, the analogue of tic-tac-toe's V against the random opponent.

--boards optimal   the belief-optimal occupancy (46,763 observations) and a trained checkpoint. Under
                   optimal play almost every observation is a sure win (risk sits in 95 openings).
--boards random    observations reached by uniformly random clicking (20,000 episodes, seed 0), with
                   exact Q and P(win) under optimal continuation, and a model trained on them (exact Q,
                   uniform weights), as for tic-tac-toe's figure 7 on all reachable boards.

Usage: uv run python scripts/minesweeper_outcome_regions.py [--mines 4] [--boards optimal] [--checkpoint-tag <r>x<c>_<m>_var]
Writes results/figures/7_outcome_regions_minesweeper_<boards>_<tag>.png and a matching JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import serialization

from exposure_rl.evaluate import full_information_q
from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle
from exposure_rl.minesweeper_exact import load_oracle
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, init_params, train_supervised

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regions, ms = _load("outcome_regions"), _load("minesweeper")
CLASSES = [
    ("wins for sure", "#184f95", "o"),
    ("wins unless unlucky", "#6da7ec", "s"),
    ("loses unless lucky", "#f08c8b", "^"),
    ("loses for sure", "#a12c2b", "v"),
]


def win_probabilities(oracle: MinesweeperOracle, points: np.ndarray) -> np.ndarray:
    """Exact P(win | o) under the belief-optimal policy (ties averaged, as in the oracle's values)."""

    sys.setrecursionlimit(10_000)

    @cache
    def p_win(obs):
        q = oracle.q_values(obs)
        tied = np.flatnonzero(np.isclose(q, q.max()))
        out = 0.0
        for a in tied:
            layouts = oracle.belief(obs, int(a))
            total = 0.0
            for L in layouts:
                if a in L:
                    continue
                nxt, _, done = oracle.reveal(obs, int(a), L)
                total += 1.0 if done else p_win(nxt)
            out += total / len(layouts)
        return out / len(tied)

    return np.array([p_win(tuple(int(v) for v in p)) for p in points])


def random_play_boards(g: Geometry, cache_dir: Path, episodes: int = 20_000) -> dict:
    """Observations reached by uniformly random clicking, with exact Q, E[G^2] and P(win) (cached)."""

    path = cache_dir / f"minesweeper_{g.rows}x{g.columns}_{g.mines}_random_play.npz"
    if path.exists():
        return dict(np.load(path))
    oracle, rng, seen = MinesweeperOracle(g), np.random.default_rng(0), {}
    start = tuple([HIDDEN] * g.cells)
    for _ in range(episodes):
        obs = start
        while True:
            seen[obs] = seen.get(obs, 0) + 1
            a = int(rng.choice(np.flatnonzero(oracle.legal(obs))))
            layouts = oracle.belief(obs, a)
            layout = layouts[rng.integers(len(layouts))]
            if a in layout:
                break
            obs, _, done = oracle.reveal(obs, a, layout)
            if done:
                break
    points = np.array(list(seen), np.int8)
    moments = [oracle.action_moments(p) for p in points]
    data = {
        "points": points,
        "visits": np.array(list(seen.values())),
        "q": np.stack([m[0] for m in moments]),
        "g2": np.stack([m[1] for m in moments]),
        "p_win": win_probabilities(oracle, points),
    }
    np.savez_compressed(path, **data)
    return data


def outcome_class(p: np.ndarray) -> np.ndarray:
    return np.select([p > 1 - 1e-9, p >= 0.5, p > 1e-9], [0, 1, 2], default=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", choices=("optimal", "random"), default="random")
    parser.add_argument("--mines", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--title", default=None, help="the figure's message (its takeaway); a neutral default if omitted"
    )
    parser.add_argument("--checkpoint-tag", default=None, help="--boards optimal only; default 4x4_<mines>_var")
    args = parser.parse_args()

    g = Geometry(4, 4, args.mines)
    gtag = f"4x4_{args.mines}"
    args.checkpoint_tag = args.checkpoint_tag or f"{gtag}_var"
    cache_dir = ROOT / "results" / "cache"
    if args.boards == "optimal":
        data = load_oracle(g, cache_dir)
        pwin_file = cache_dir / f"minesweeper_{gtag}_pwin.npy"
        if not pwin_file.exists():
            np.save(pwin_file, win_probabilities(MinesweeperOracle(g), data["points"]))
        p, mass = np.load(pwin_file), data["weights"]
        tag = f"optimal_{args.checkpoint_tag}_s{args.seed}"
        var_weight = 50.0 if "_var" in args.checkpoint_tag else 0.0
        ckpt = f"vaeac_q_ms{args.checkpoint_tag}_m10_s{args.seed}.msgpack"
        boards_text = "belief-optimal occupancy"
    else:
        data = random_play_boards(g, cache_dir)
        p, mass = data["p_win"], data["visits"].astype(float)
        tag, var_weight = f"random_{gtag}_s{args.seed}", 0.0
        ckpt = f"vaeac_q_ms{gtag}_randomplay_m10_s{args.seed}.msgpack"
        boards_text = "observations reached by random clicking"
    points, q_exact = data["points"], data["q"]
    obs = points.astype(np.float32)
    y = outcome_class(p)
    v_exact = 2 * p - 1  # P(win) - P(loss): no draws in Minesweeper

    cfg = dataclasses.replace(Config(), seed=args.seed, n_values=10, n_cells=g.cells, var_weight=var_weight)
    if args.checkpoint_tag and args.checkpoint_tag.endswith("_xrl"):  # xrl Direct-Q configuration
        cfg = dataclasses.replace(cfg, **XRL_MATCHED)
    model = VAEACQ(cfg)
    path = ROOT / "results" / "models" / "minesweeper" / ckpt
    if path.exists():
        params = serialization.from_bytes(init_params(model, args.seed), path.read_bytes())
    elif args.boards == "random":
        print(f"training one-network VAEAC-Q on {len(obs)} random-play observations ...", flush=True)
        _, params = train_supervised(obs, np.ones(len(obs)), q_exact, cfg, log_every=0)
        path.write_bytes(serialization.to_bytes(params))
    else:
        raise FileNotFoundError(path)
    emb = []
    for chunk in np.array_split(obs, max(1, len(obs) // 4096)):
        log_pi, mu, _ = model.apply({"params": params}, jnp.asarray(chunk), method=VAEACQ.full_belief)
        emb.append(np.asarray(jnp.sum(jnp.exp(log_pi)[..., None] * mu, axis=-2)))
    emb = np.concatenate(emb)
    raw = np.eye(10)[points.astype(int) + 1].reshape(len(points), -1)  # one-hot board baseline (160-d)
    extra = {}
    if args.boards == "optimal":
        games = np.load(cache_dir / f"minesweeper_{gtag}_games.npz")
        extra["v_inherent_full"] = games["summary"][:, 2]  # exact agent's own uncertainty v_state(N)
    np.savez_compressed(  # latents and labels for scripts/umap_latents.py
        cache_dir / f"latents_minesweeper_{tag}.npz", emb=emb, y=y, p_win=p, obs=points, mass=mass, **extra
    )

    v_star = np.nanmax(q_exact, axis=1)
    q_model = full_information_q(model, params, obs)
    v_model = np.max(np.where(np.isnan(q_exact), -np.inf, q_model), axis=1)
    r2_latent, r2_raw = regions.ridge_cv_r2(emb, v_exact), regions.ridge_cv_r2(raw, v_exact)
    acc_latent, rec_latent = regions.lda_cv_balanced_accuracy(emb, y)
    acc_raw, rec_raw = regions.lda_cv_balanced_accuracy(raw, y)
    present = np.unique(y)

    regions.plot_regions(
        emb,
        y,
        np.clip(v_model, -1, 1),  # coloured by the agent's own value, from the same network as the latent
        CLASSES,
        lambda ax, i: ms.draw(ax, g, points[i], None, "", 1.0),
        f"latent $\\mathbb{{E}}[z \\mid o_N]$ (16-d): {acc_latent:.0%}\nraw board one-hot (160-d): {acc_raw:.0%}\n"
        f"chance: {1 / len(present):.0%}",
        "Coloured by the agent's estimated $\\max_a Q(o, a)$",
        f"Linear readout: $R^2$ = {r2_latent:.2f} from the latent, {r2_raw:.2f} from the raw board",
        "estimated $\\max_a Q$ (win +1, loss -1)",
        (
            args.title or "Where the latent places each outcome",
            f"Minesweeper 4x4, {args.mines} mines; {len(obs)} {boards_text}; classes from the exact P(win)",
        ),
        ROOT / "results" / "figures" / f"7_outcome_regions_minesweeper_{tag}.png",
    )
    names = [c[0] for c in CLASSES]
    summary = {
        "observations": len(points),
        "class_counts": {names[c]: int(np.sum(y == c)) for c in range(len(CLASSES))},
        "class_mass": {names[c]: float(mass[y == c].sum() / mass.sum()) for c in range(len(CLASSES))},
        "value_readout_r2": {"latent": r2_latent, "raw_board": r2_raw},
        "model_value_vs_exact_v_star": {
            "mae": float(np.mean(np.abs(v_model - v_star))),
            "pearson": float(np.corrcoef(v_model, v_star)[0, 1]),
        },
        "probe_balanced_accuracy": {"latent": acc_latent, "raw_board": acc_raw},
        "probe_recall_by_class": {
            "latent": {names[c]: r for c, r in rec_latent.items()},
            "raw_board": {names[c]: r for c, r in rec_raw.items()},
        },
        "revealed_cells_by_class": {names[c]: float(np.mean((points[y == c] != HIDDEN).sum(1))) for c in present},
    }
    (ROOT / "results" / f"outcome_regions_minesweeper_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
