"""DQN + surrogate in tic-tac-toe with the surrogate distilled on the DQN's epsilon-greedy occupancy, seed by seed.

The Table 1 surrogate is distilled on its DQN's greedy occupancy (xrl recipe), which covers only part of the
scored boards (the optimal policy's occupancy); the Minesweeper surrogate already uses epsilon-greedy (0.1)
rollouts. Here the tic-tac-toe surrogate is distilled on the exact epsilon-greedy occupancy of the same DQN
(everything else as in Table 1: width 256, 16k updates) and scored like the Table 1 mean game, against the exact
game of the DQN's own Q. Also reports how much of the scored occupancy the training boards cover.

Usage: uv run python scripts/score_surrogate_occupancy.py --opponent {random,pm05} --seed S [--epsilon 0.1]
       -> results/surrogate_eps<e>_<opponent>_s<S>.json (surrogate in results/philipp/<...>_eps<e>_w256/seed_S)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from xrl.explainability.adapters import load_explanation_run

from exposure_rl import scalar_surrogate
from exposure_rl.evaluate import OracleReference

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "philipp"
_spec = importlib.util.spec_from_file_location("ablations", ROOT / "scripts" / "replay_matching_eval.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
WIDTH = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=("random", "pm05"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()
    prefix = "ttt" if args.opponent == "random" else "ttt_pm05"
    dqn_dir = P / f"{prefix}_dqn_scalar_w{WIDTH}" / f"seed_{args.seed}"
    tag = f"eps{round(100 * args.epsilon):02d}"
    out_dir = P / f"{prefix}_dqn_surrogate_scalar_{tag}_w{WIDTH}" / f"seed_{args.seed}"
    cfg = scalar_surrogate.ScalarSurrogateConfig(
        base_run_dir=str(dqn_dir),
        opponent=args.opponent,
        seed=args.seed,
        hidden=(WIDTH, WIDTH),
        occupancy_epsilon=args.epsilon,
    )
    if not (out_dir / "scalar_surrogate.msgpack").exists():
        scalar_surrogate.train(cfg, out_dir)

    ref = OracleReference(args.opponent)
    train_obs = scalar_surrogate._dataset(cfg, load_explanation_run(dqn_dir))[0]
    keys = {o.tobytes() for o in train_obs.astype(np.float32)}
    covered = np.array([o.astype(np.float32).tobytes() in keys for o in ref.obs])
    legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
    dqn = ev.StandardDQN(dqn_dir)
    surrogate = ev.ScalarSurrogateGames(out_dir)
    q_e = ev.own_exact_mean_games(ref.obs, ref.w, ev.full_q(dqn, ref.obs), ref.obs, ref.actions, legal_of)
    q_m = np.stack([ev.method_games(surrogate, o.astype(np.float32), int(a), 9) for o, a in zip(ref.obs, ref.actions)])
    result = ev.score_mean_game(q_m, q_e, 9) | {
        "epsilon": args.epsilon,
        "train_boards": len(train_obs),
        "score_boards_covered": float(covered.mean()),
        "score_mass_covered": float(ref.w[covered].sum() / ref.w.sum()),
    }
    out = ROOT / "results" / f"surrogate_{tag}_{args.opponent}_s{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    print(args.opponent, args.seed, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
