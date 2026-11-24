"""Train the scalar masked surrogate (exposure_rl.scalar_surrogate) for one scalar Direct-Q run.

Usage: uv run python scripts/train_scalar_surrogate.py --base-run <xrl run dir> --out <dir> [--seed 0] [--opponent pm05]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from exposure_rl.scalar_surrogate import ScalarSurrogateConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--updates", type=int, default=ScalarSurrogateConfig.updates)
    parser.add_argument("--width", type=int, default=64, help="surrogate MLP width (2 hidden layers; xrl: 64)")
    parser.add_argument("--occupancy-epsilon", type=float, default=0.0, help="tic-tac-toe: epsilon-greedy occupancy")
    args = parser.parse_args()
    cfg = ScalarSurrogateConfig(
        base_run_dir=args.base_run,
        opponent=args.opponent,
        seed=args.seed,
        updates=args.updates,
        hidden=(args.width, args.width),
        occupancy_epsilon=args.occupancy_epsilon,
    )
    train(cfg, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
