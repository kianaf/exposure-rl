"""Milestone 2b: TD-trained VAEAC-Q, agent acting through its latent; scored against the exact oracle.

Usage: uv run python scripts/train_td.py [--components 10] [--seed 0] [--updates 150000] [--samples 256]
Writes results/td_random_m<M>_s<seed>.json and results/models/td_random_m<M>_s<seed>.msgpack.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from exposure_rl.evaluate import OracleReference, compare, full_information_q, greedy_optimal_rate, model_games
from exposure_rl.td import TDConfig, TicTacToeVsOpponent, train_td
from exposure_rl.vaeac_q import XRL_MATCHED, Config, make_coalition_games, size_tag


def play(policy, opponent: str, games: int, seed: int) -> dict:
    """Greedy evaluation games; ``policy(obs, legal) -> action``."""

    env = TicTacToeVsOpponent(opponent, np.random.default_rng(seed))
    results = []
    for _ in range(games):
        obs, legal = env.reset()
        while True:
            obs, legal, r, done = env.step(policy(obs, legal))
            if done:
                results.append(r)
                break
    results = np.asarray(results)
    return {
        "mean_return": float(results.mean()),
        "win": float(np.mean(results > 0)),
        "draw": float(np.mean(results == 0)),
        "loss": float(np.mean(results < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=Config.n_components)
    parser.add_argument("--untied", action="store_true", help="two-network ablation (separate full encoder)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=TDConfig.updates)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--games", type=int, default=4000)
    parser.add_argument("--opponent", default="random", help="random or pm05 (perturbed minimax, 5% random)")
    parser.add_argument("--anneal-opponent", action="store_true", help="anneal the random-move rate 1 -> target")
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--depth", type=int, default=Config.depth)
    parser.add_argument("--activation", default=Config.activation, choices=("gelu", "relu"))
    parser.add_argument("--xrl-matched", action="store_true", help="xrl Direct-Q configuration (vaeac_q.XRL_MATCHED)")
    parser.add_argument("--width", type=int, default=0, help="with --xrl-matched: hidden width (0: the xrl 64)")
    parser.add_argument("--var-weight", type=float, default=0.0, help="variance head, trained on its TD target")
    parser.add_argument("--separate-heads", action="store_true", help="separate Q and variance MLPs (earlier model)")
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim, help="latent size d")
    parser.add_argument("--beta", type=float, default=Config.beta, help="KL weight (checkpoint suffix _b<beta>)")
    args = parser.parse_args()
    cfg = dataclasses.replace(
        Config(),
        n_components=args.components,
        seed=args.seed,
        tied_encoder=not args.untied,
        var_weight=args.var_weight,
        latent_dim=args.latent_dim,
        beta=args.beta,
        shared_heads=not args.separate_heads,
        **{
            "hidden": args.hidden,
            "depth": args.depth,
            "activation": args.activation,
            **({**XRL_MATCHED, **({"hidden": args.width} if args.width else {})} if args.xrl_matched else {}),
        },
    )
    td = dataclasses.replace(
        TDConfig(),
        updates=args.updates,
        opponent=args.opponent,
        anneal_opponent=args.anneal_opponent,
    )

    ref = OracleReference(td.opponent)

    def on_log(model, params):
        return {"greedy_optimal": greedy_optimal_rate(model, params, ref)}

    print(f"TD training (M={cfg.n_components}, seed={cfg.seed}, updates={td.updates})")
    model, params, history = train_td(cfg, td, on_log=on_log)

    out = Path(__file__).resolve().parents[1] / "results"
    opp = args.opponent + ("a" if args.anneal_opponent else "")
    tag = f"m{cfg.n_components}{'_untied' * args.untied}{'_var' * (cfg.var_weight > 0)}{size_tag(cfg)}{f'_d{cfg.latent_dim}' * (cfg.latent_dim != Config.latent_dim)}{'_separate' * (not cfg.shared_heads and cfg.var_weight > 0)}{f'_b{cfg.beta:g}' * (cfg.beta != Config.beta)}_s{cfg.seed}"
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "models" / f"td_{opp}_{tag}.msgpack").write_bytes(serialization.to_bytes(params))

    eps = jax.random.normal(jax.random.PRNGKey(cfg.seed + 1), (args.samples, cfg.latent_dim))
    q_m, v_m = model_games(make_coalition_games(model), params, ref.obs, ref.actions, eps)
    scores, _ = compare(ref.w, q_m, v_m, ref.q_e, ref.v_e, ref.phi_e)

    def agent_policy(obs, legal):
        q = full_information_q(model, params, obs[None])[0]
        return int(np.flatnonzero(legal)[np.argmax(q[legal])])

    def expectimax_policy(obs, legal):
        return int(np.argmax(ref.oracle.q_values(obs)))

    q_full = full_information_q(model, params, ref.obs)
    legal = ~np.isnan(ref.q_exact)
    summary = {
        "config": dataclasses.asdict(cfg),
        "td_config": dataclasses.asdict(td),
        "latent_samples": args.samples,
        "full_info_greedy_action_optimal": greedy_optimal_rate(model, params, ref),
        "full_info_q_mae_legal": float(
            ref.w @ (np.where(legal, np.abs(q_full - np.nan_to_num(ref.q_exact)), 0).sum(1) / legal.sum(1))
        ),
        "play_agent": play(agent_policy, td.opponent, args.games, seed=123),
        "play_expectimax": play(expectimax_policy, td.opponent, args.games, seed=123),
        "certainty_vs_oracle": scores,
        "history": history,
    }
    (out / f"td_{opp}_{tag}.json").write_text(json.dumps(summary, indent=2))
    for key in ("full_info_greedy_action_optimal", "full_info_q_mae_legal", "play_agent", "play_expectimax"):
        print(f"  {key}: {summary[key]}")
    print("  certainty vs oracle: " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
