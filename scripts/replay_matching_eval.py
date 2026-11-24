"""Mean game phi^Q of Philipp's Direct-Q and distilled surrogate (xrl, read-only) vs EXPOSURE.

Philipp's methods provide only the mean game Q(C) (their categorical heads model the one-step TD target,
not Var(G), so no certainty game is derived from them). Every method is scored against the exact mean
game of ITS OWN agent: E[Q_agent(O, a) | o_C] over the occupancy (action-legal view), where Q_agent is
the agent's full-information Q (for the surrogate: the Direct-Q it was distilled from). This separates
explanation quality from playing strength (cf. xrl ADR 0022). Shapley values are exact over all 2^n
coalitions for every method. The explained action is the oracle's greedy action, as in the other games.
Each agent's exact win probability is reported alongside.

Usage: uv run python scripts/replay_matching_eval.py --domain tictactoe|minesweeper
Writes results/philipp_ablations_<domain>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, spearmanr
from xrl.explainability.adapters import load_explanation_run
from xrl.explainability.distributional_q_masks import masked_surrogate_inputs
from xrl.explainability.distributional_q_trainer import load_distributional_q_surrogate

from exposure_rl import coalitions, scalar_surrogate
from exposure_rl.evaluate import OracleReference, full_information_q, load_checkpoint, tictactoe_checkpoint
from exposure_rl.exact_eval import exact_outcomes, greedy
from exposure_rl.minesweeper import HIDDEN, Geometry, MinesweeperOracle
from exposure_rl.minesweeper_exact import load_oracle, scored_indices, tag
from exposure_rl.vaeac_q import (
    VAEACQ,
    XRL_MATCHED,
    Config,
    init_params,
    make_coalition_games,
    size_tag,
    train_supervised,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ms_script", ROOT / "scripts" / "minesweeper.py")
ms_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms_script)
PHILIPP = ROOT / "results" / "philipp"
# Per head (and opponent): the Direct-Q run, and, as in xrl's explainable_q_comparison_walkthrough, a standard
# DQN ("dqn": no mask input) with the separate masked surrogate distilled from it. Without "dqn" the surrogate
# explains the Direct-Q agent (the older HL-Gauss runs).
PHILIPP_DIRS = {
    "tictactoe": {
        "scalar": {
            "direct_q": PHILIPP / "ttt_directq_scalar",
            "dqn": PHILIPP / "ttt_dqn_scalar",
            "surrogate": PHILIPP / "ttt_dqn_surrogate_scalar",
        },
        "scalar-pm05": {
            "direct_q": PHILIPP / "ttt_pm05_directq_scalar",
            "dqn": PHILIPP / "ttt_pm05_dqn_scalar",
            "surrogate": PHILIPP / "ttt_pm05_dqn_surrogate_scalar",
        },
        "hl-gauss": {"direct_q": PHILIPP / "ttt_directq_hlg", "surrogate": PHILIPP / "ttt_surrogate_hlg"},
        "hl-gauss-pm05": {
            "direct_q": PHILIPP / "ttt_pm05_directq_hlg",
            "surrogate": PHILIPP / "ttt_pm05_surrogate_hlg",
        },
    },
    "minesweeper": {
        "scalar": {
            "direct_q": PHILIPP / "ms4x4_3_directq_scalar",
            "dqn": PHILIPP / "ms4x4_3_dqn_scalar",
            "surrogate": PHILIPP / "ms4x4_3_dqn_surrogate_scalar",
        },
        "hl-gauss": {"direct_q": PHILIPP / "ms4x4_3_directq_hlg", "surrogate": PHILIPP / "ms4x4_3_surrogate_hlg"},
    },
}
HEAD = ["scalar"]
OPPONENT = ["random"]  # tic-tac-toe opponent spec (exposure_rl.opponents)
ALL_TD = [False]  # --all-td: only the TD agents (variance head on its TD target)
REPLAY = [0]  # --replay R: also score replay matching with R boards on every agent's own Q


# ---------------------------------------------------------------- categorical-head games (Philipp's models)


class DirectQGames:
    """Direct-Q (mask-aware DQN) mean games via its own batched forward pass (identity encoder: env actions).

    Scalar head: the network output is Q; categorical head: Q is the mean over its support.
    """

    def __init__(self, run_dir: Path) -> None:
        self.policy = load_explanation_run(run_dir).policy
        self.scalar = self.policy.config.value_head == "scalar"
        self.support = None if self.scalar else np.asarray(self.policy.support, dtype=np.float64)

    def q_mean(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        inputs = jnp.asarray(self.policy._model_inputs(states.astype(np.float32), masks.astype(np.float32)))
        out = np.asarray(self.policy.model.apply(self.policy.params, inputs), np.float64)
        return out if self.scalar else np.asarray(jax.nn.softmax(out, axis=-1)) @ self.support

    def check(self, obs: np.ndarray, mask: np.ndarray) -> None:
        single = np.asarray(self.policy.masked_q_values(obs, mask), np.float64)
        assert np.allclose(single, self.q_mean(obs[None], mask[None])[0], atol=1e-3), "batched != xrl path"


class StandardDQN:
    """A standard (not mask-aware) xrl DQN: the agent a separate masked surrogate explains. Full information only."""

    def __init__(self, run_dir: Path) -> None:
        self.policy = load_explanation_run(run_dir).policy
        assert not self.policy.config.feature_mask_input, f"{run_dir} is mask-aware, not a standard DQN"
        assert self.policy.config.value_head == "scalar"

    def q_mean(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        assert np.all(masks == 1), "a standard DQN has no masked queries"
        inputs = jnp.asarray(self.policy._model_inputs(states.astype(np.float32)))
        return np.asarray(self.policy.model.apply(self.policy.params, inputs), np.float64)


class SurrogateGames:
    """Masked distributional surrogate mean games via its own network on batched masked inputs."""

    def __init__(self, run_dir: Path) -> None:
        self.surrogate = load_distributional_q_surrogate(run_dir)
        self.support = np.asarray(self.surrogate.support, dtype=np.float64)

    def q_mean(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        inputs = jnp.asarray(masked_surrogate_inputs(states.astype(np.float32), masks.astype(np.float32)))
        logits = self.surrogate.model.apply(self.surrogate.params, inputs)
        return np.asarray(jax.nn.softmax(logits, axis=-1), np.float64) @ self.support

    def check(self, obs: np.ndarray, mask: np.ndarray) -> None:
        single = self.surrogate.q_distributions(obs, mask) @ self.support
        assert np.allclose(single, self.q_mean(obs[None], mask[None])[0], atol=1e-3), "batched != xrl path"


class ScalarSurrogateGames:
    """Scalar masked surrogate (exposure_rl.scalar_surrogate, the xrl recipe with a scalar target)."""

    def __init__(self, run_dir: Path) -> None:
        self.surrogate = scalar_surrogate.load(run_dir)

    def q_mean(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return self.surrogate.q_batch(states, masks)

    def check(self, obs: np.ndarray, mask: np.ndarray) -> None:
        assert np.allclose(self.surrogate.q_values(obs, mask), self.q_mean(obs[None], mask[None])[0], atol=1e-3)


def method_games(method, obs: np.ndarray, action: int, n: int, chunk: int = 8192) -> np.ndarray:
    """Mean game Q(C) of one xrl method for every coalition (the action's cell always known)."""

    masks = coalitions_masks(n)
    masks[:, action] = 1.0
    return np.concatenate(
        [
            method.q_mean(np.broadcast_to(obs, m.shape), m)[:, action]
            for m in np.split(masks, max(1, len(masks) // chunk))
        ]
    )


def coalitions_masks(n: int) -> np.ndarray:
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float32)


# ---------------------------------------------------------------- scoring


def median_spearman(a, b):
    rhos = [spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return float(np.median(rhos)) if rhos else float("nan")


def mean_spearman(a, b):
    """Mean over boards of the per-board Spearman rho (2026-11-08: the median of few-player rhos is quantized)."""

    rhos = [spearmanr(x, y).statistic for x, y in zip(a, b) if np.std(x) > 1e-9 and np.std(y) > 1e-9]
    return float(np.mean(rhos)) if rhos else float("nan")


def own_exact_mean_games(points, weights, q_agent, obs, actions, legal_of):
    """Exact E[Q_agent(O, a) | o_C] for every coalition of every scored observation."""

    rows = []
    for o, a in zip(obs, actions):
        legal = legal_of(int(a))
        rows.append(
            coalitions.conditional_moments(points[legal], weights[legal], q_agent[legal, int(a)][:, None], o)[:, 0]
        )
    return np.stack(rows)


def score_mean_game(q_m, q_e, n):
    interior = coalitions.coalition_sizes(n) < n
    phi_m, phi_e = coalitions.shapley(q_m, n), coalitions.shapley(q_e, n)
    big = np.abs(phi_e) > 1e-3
    return {
        "mean_game_pearson_interior": float(pearsonr(q_m[:, interior].ravel(), q_e[:, interior].ravel()).statistic),
        "phi_Q_spearman_median": median_spearman(phi_m, phi_e),
        "phi_Q_spearman_mean": mean_spearman(phi_m, phi_e),
        "phi_Q_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
        "phi_Q_exact_positive_share": float((phi_e[big] > 0).mean()),  # base rate for the sign score
    }


def own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of):
    """Exact ``E[Q_agent(O, a) | o_C]`` and ``Var[Q_agent(O, a) | o_C]`` for every coalition of every observation."""

    means, variances = [], []
    for o, a in zip(obs, actions):
        legal = legal_of(int(a))
        q = q_agent[legal, int(a)]
        m = coalitions.conditional_moments(points[legal], weights[legal], np.stack([q, q**2], 1), o)
        means.append(m[:, 0])
        variances.append(np.maximum(m[:, 1] - m[:, 0] ** 2, 0.0))
    return np.stack(means), np.stack(variances)


def replay_games(points, weights, q_agent, obs, actions, legal_of, n, replay, seed):
    """Replay matching on the agent's own full-information Q: ``replay`` boards drawn from the occupancy; for each
    coalition, the mean and variance of Q over the replay boards that match ``o`` on it (the empirical conditional).
    Fewer than two matches: the mean falls back to the mean over all legal replay boards, the variance to 0."""

    idx = np.random.default_rng(seed).choice(len(points), size=replay, p=weights / weights.sum())
    rep = points[idx]
    means, variances = [], []
    for o, a in zip(obs, actions):
        legal = legal_of(int(a))[idx]
        pts, q = rep[legal], q_agent[idx][legal, int(a)]
        m = coalitions.conditional_moments(pts, np.ones(len(pts)), np.stack([q, q**2], 1), o)
        counts = coalitions.subset_sums(
            np.bincount(((pts != np.asarray(o)[None]) << np.arange(n)).sum(1), minlength=1 << n).astype(float), n
        )[(1 << n) - 1 - np.arange(1 << n)]
        few = counts < 2
        mean = np.where(few, q.mean() if len(q) else 0.0, m[:, 0])
        var = np.where(few, 0.0, np.maximum(m[:, 1] - m[:, 0] ** 2, 0.0))
        var[-1] = 0.0
        means.append(mean)
        variances.append(var)
    return np.stack(means), np.stack(variances)


def score_vmask(v_m, v_e, n):
    """Pearson r over interior coalitions and Shapley sign agreement (|phi| > 1e-3) of a v_mask game."""

    interior = coalitions.coalition_sizes(n) < n
    phi_m, phi_e = coalitions.shapley(v_m, n), coalitions.shapley(v_e, n)
    big = np.abs(phi_e) > 1e-3
    return {
        "v_mask_pearson_interior": float(pearsonr(v_m[:, interior].ravel(), v_e[:, interior].ravel()).statistic),
        "phi_mask_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
        "phi_mask_spearman_median": median_spearman(phi_m, phi_e),
        "phi_mask_spearman_mean": mean_spearman(phi_m, phi_e),
    }


def replay_scores(points, weights, q_agent, obs, actions, legal_of, n, q_e, seed):
    """``replay_*`` scores of replay matching (``REPLAY[0]`` boards) on the agent's own Q, against its own exact games."""

    if not REPLAY[0]:
        return {}
    _, v_e = own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
    q_r, v_r = replay_games(points, weights, q_agent, obs, actions, legal_of, n, REPLAY[0], seed)
    scores = score_mean_game(q_r, q_e, n) | score_vmask(v_r, v_e, n)
    return {f"replay_{k}": v for k, v in scores.items()}


def exposure_mean_games(model, params, obs, actions, n):
    games_fn = make_coalition_games(model, blocks=max(1, (1 << n) // 2048))
    eps = jax.random.normal(jax.random.PRNGKey(model.cfg.seed + 1), (128, model.cfg.latent_dim))
    return np.stack(
        [np.asarray(games_fn(params, jnp.asarray(o, jnp.float32), int(a), eps)[0]) for o, a in zip(obs, actions)]
    )


def exposure_vmask_games(model, params, obs, actions, n):
    games_fn = make_coalition_games(model, blocks=max(1, (1 << n) // 2048))
    eps = jax.random.normal(jax.random.PRNGKey(model.cfg.seed + 1), (128, model.cfg.latent_dim))
    return np.stack(
        [np.asarray(games_fn(params, jnp.asarray(o, jnp.float32), int(a), eps)[1]) for o, a in zip(obs, actions)]
    )


def full_q(method, points, chunk=8192):
    """A method's full-information Q for every point (all cells known)."""

    return np.concatenate(
        [
            method.q_mean(points[k : k + chunk], np.ones_like(points[k : k + chunk]))
            for k in range(0, len(points), chunk)
        ]
    )


def run_methods(name, points, weights, obs, actions, legal_of, n, exposures, philipp_dirs, play):
    out = {}
    for seed in range(3):
        for label_e, exposure in exposures.items():
            model, params = exposure(seed)
            q_agent = full_information_q(model, params, points.astype(np.float32))
            q_e = own_exact_mean_games(points, weights, q_agent, obs, actions, legal_of)
            q_m = exposure_mean_games(model, params, obs, actions, n)
            q_fn = lambda o, model=model, params=params: full_information_q(
                model, params, np.asarray(o, np.float32)[None]
            )[0]
            extra = replay_scores(points, weights, q_agent, obs, actions, legal_of, n, q_e, seed)
            if REPLAY[0]:  # our own v_mask, against the exact v_mask of our own Q
                _, v_e = own_exact_moment_games(points, weights, q_agent, obs, actions, legal_of)
                v_m = exposure_vmask_games(model, params, obs, actions, n)
                extra |= {f"own_{k}": v for k, v in score_vmask(v_m, v_e, n).items()}
            out.setdefault(label_e, []).append(score_mean_game(q_m, q_e, n) | play(q_fn) | extra)
        direct = DirectQGames(philipp_dirs["direct_q"] / f"seed_{seed}")
        surrogate = None
        if philipp_dirs.get("surrogate"):
            surrogate_cls = ScalarSurrogateGames if direct.scalar else SurrogateGames
            surrogate = surrogate_cls(philipp_dirs["surrogate"] / f"seed_{seed}")
        mask = np.ones(n, np.float32)
        mask[: n // 2] = 0
        mask[int(actions[0])] = 1
        direct.check(obs[0].astype(np.float32), mask)
        if surrogate is not None:
            surrogate.check(obs[0].astype(np.float32), mask)
        agents = {"direct_q": direct}
        if philipp_dirs.get("dqn"):  # the surrogate explains its own standard DQN, not the Direct-Q
            agents["surrogate"] = StandardDQN(philipp_dirs["dqn"] / f"seed_{seed}")
        else:
            agents["surrogate"] = direct
        for label, method in (("direct_q", direct), ("surrogate", surrogate)):
            if method is None:
                continue
            agent = agents[label]
            q_agent = full_q(agent, points)
            q_e = own_exact_mean_games(points, weights, q_agent, obs, actions, legal_of)
            win = play(lambda o, d=agent: d.q_mean(np.asarray(o, np.float32)[None], np.ones((1, n), np.float32))[0])
            q_m = np.stack([method_games(method, o.astype(np.float32), int(a), n) for o, a in zip(obs, actions)])
            extra = replay_scores(points, weights, q_agent, obs, actions, legal_of, n, q_e, seed)
            out.setdefault(label, []).append(score_mean_game(q_m, q_e, n) | win | extra)
        print(name, seed, json.dumps({k: v[-1] for k, v in out.items()}), flush=True)
    return {
        k: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        | {f"{m}_sd": float(np.std([r[m] for r in rows], ddof=1)) for m in rows[0] if len(rows) > 1}
        | {"seeds": len(rows)}
        for k, rows in out.items()
    }


# ---------------------------------------------------------------- domains


def tictactoe(size: dict) -> dict:
    ref = OracleReference(OPPONENT[0])
    moments = [ref.oracle.action_moments(p.observation) for p in ref.dist.points]
    var_exact = np.stack([m[1] - m[0] ** 2 for m in moments])

    def exposure(seed):
        cfg = dataclasses.replace(Config(), seed=seed, var_weight=50.0, **size)
        model = VAEACQ(cfg)
        path = ROOT / "results" / "models" / f"vaeac_q_{OPPONENT[0]}_m10_var{size_tag(cfg)}_s{seed}.msgpack"
        if path.exists():
            return model, serialization.from_bytes(init_params(model, seed), path.read_bytes())
        print(f"training EXPOSURE tic-tac-toe with variance head, seed {seed}", flush=True)
        _, params = train_supervised(ref.obs, ref.w, ref.q_exact, cfg, log_every=0, var_exact=var_exact)
        path.write_bytes(serialization.to_bytes(params))
        return model, params

    def play(q_fn):
        o = exact_outcomes(greedy(q_fn), OPPONENT[0])
        return {"agent_" + k: float(v) for k, v in o.items() if isinstance(v, (int, float))}

    legal_of = lambda a: ~np.isnan(ref.q_exact[:, a])
    head = HEAD[0] + ("" if OPPONENT[0] == "random" else f"-{OPPONENT[0]}")
    dirs = PHILIPP_DIRS["tictactoe"][head]

    def exposure_td(seed):
        cfg = dataclasses.replace(Config(), seed=seed, **size)
        _, model, params = load_checkpoint(
            tictactoe_checkpoint("td", seed, size_tag(cfg), OPPONENT[0], td_var=ALL_TD[0])
        )
        return model, params

    exposures = (
        {"exposure_rl_td": exposure_td} if ALL_TD[0] else {"exposure_rl": exposure, "exposure_rl_td": exposure_td}
    )
    return run_methods("tictactoe", ref.obs, ref.w, ref.obs, ref.actions, legal_of, 9, exposures, dirs, play)


def minesweeper(size: dict) -> dict:
    g = Geometry(4, 4, 3)
    cache = ROOT / "results" / "cache"
    oracle = load_oracle(g, cache)
    points = oracle["points"]
    actions = np.nanargmax(oracle["q"], axis=1)
    scored, _ = scored_indices(oracle["weights"])
    rules = MinesweeperOracle(g)

    def exposure(seed):
        cfg = dataclasses.replace(Config(), seed=seed, n_values=10, n_cells=g.cells, var_weight=50.0, **size)
        model = VAEACQ(cfg)
        path = (
            ROOT
            / "results"
            / "models"
            / "minesweeper"
            / f"vaeac_q_ms{tag(g)}_var{size_tag(cfg)}{'_td' * ALL_TD[0]}_m10_s{seed}.msgpack"
        )
        return model, serialization.from_bytes(init_params(model, seed), path.read_bytes())

    def play(q_fn):
        def policy(obs, legal):
            q = np.asarray(q_fn(obs))
            return int(np.flatnonzero(legal)[np.argmax(q[legal])])

        return {"agent_" + k: v for k, v in ms_script.exact_policy_outcome(rules, policy).items()}

    legal_of = lambda a: points[:, a] == HIDDEN
    dirs = PHILIPP_DIRS["minesweeper"][HEAD[0]]
    obs = points[scored].astype(np.float32)
    return run_methods(
        "minesweeper",
        points.astype(np.float32),
        oracle["weights"],
        obs,
        actions[scored],
        legal_of,
        16,
        {"exposure_rl": exposure},
        dirs,
        play,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("tictactoe", "minesweeper"), required=True)
    parser.add_argument("--hidden", type=int, default=Config.hidden, help="EXPOSURE MLP width (xrl default: 64)")
    parser.add_argument("--depth", type=int, default=Config.depth, help="EXPOSURE MLP depth (xrl default: 2)")
    parser.add_argument("--activation", default=Config.activation, choices=("gelu", "relu"))
    parser.add_argument("--xrl-matched", action="store_true", help="xrl Direct-Q configuration (vaeac_q.XRL_MATCHED)")
    parser.add_argument("--width", type=int, default=0, help="with --xrl-matched: hidden width (0: the xrl 64)")
    parser.add_argument("--all-td", action="store_true", help="only the TD agents with a TD-trained variance head")
    parser.add_argument("--replay", type=int, default=0, help="also score replay matching with this many boards")
    parser.add_argument("--philipp-head", default="scalar", choices=("scalar", "hl-gauss"))
    parser.add_argument("--opponent", default="random", help="tic-tac-toe opponent: random or pm05")
    args = parser.parse_args()
    HEAD[0] = args.philipp_head
    OPPONENT[0] = args.opponent
    ALL_TD[0] = args.all_td
    REPLAY[0] = args.replay
    if args.width and args.width != 64:  # xrl runs of the network-size sweep live in <dir>_w<width>
        for domain in PHILIPP_DIRS.values():
            for head in domain.values():
                for key, path in head.items():
                    head[key] = path.with_name(f"{path.name}_w{args.width}")
    size = (
        {**XRL_MATCHED, **({"hidden": args.width} if args.width else {})}
        if args.xrl_matched
        else {"hidden": args.hidden, "depth": args.depth, "activation": args.activation}
    )
    result = tictactoe(size) if args.domain == "tictactoe" else minesweeper(size)
    suffix = size_tag(dataclasses.replace(Config(), **size))
    (
        ROOT
        / "results"
        / f"philipp_ablations_{args.domain}{suffix}_{args.philipp_head}{'' if args.opponent == 'random' else '_' + args.opponent}.json"
    ).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
