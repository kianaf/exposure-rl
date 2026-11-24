"""Shared scoring of a trained VAEAC-Q against the exact oracle (used by milestones 2a and 2b)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.stats import pearsonr, spearmanr

from exposure_rl import coalitions
from exposure_rl.oracle import ExpectimaxOracle, certainty_games, uncertainty_reduction_shapley
from exposure_rl.vaeac_q import VAEACQ, XRL_MATCHED, Config, belief_q, init_params


class OracleReference:
    """Exact games for every learner-decision state under the expectimax occupancy (greedy action per state)."""

    def __init__(self, opponent: str = "random", oracle=None) -> None:
        """Tic Tac Toe expectimax oracle by default; any oracle with ``occupancy()`` / ``action_moments()`` works."""

        self.oracle = oracle if oracle is not None else ExpectimaxOracle(opponent)
        self.dist = self.oracle.occupancy()
        self.obs = np.array([p.observation for p in self.dist.points], dtype=np.float32)
        weights = np.array([p.weight for p in self.dist.points])
        self.w = weights / weights.sum()
        self.q_exact = np.stack([self.oracle.action_moments(p.observation)[0] for p in self.dist.points])
        self.actions = np.nanargmax(self.q_exact, axis=1)
        exact = [
            certainty_games(self.oracle, self.dist, p.observation, int(a))
            for p, a in zip(self.dist.points, self.actions)
        ]
        self.v_e = np.stack([g.mask for g in exact])
        self.q_e = np.stack([g.q_mean for g in exact])
        self.phi_e = np.stack([uncertainty_reduction_shapley(v) for v in self.v_e])


def model_games(games_fn, params, obs, actions, eps):
    rows = [[np.asarray(t) for t in games_fn(params, jnp.asarray(x), int(a), eps)] for x, a in zip(obs, actions)]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def compare(w, q_m, v_m, q_e, v_e, phi_e) -> tuple[dict, np.ndarray]:
    n = int(np.log2(v_m.shape[1]))
    sizes = coalitions.coalition_sizes(n)
    interior = sizes < n
    phi_m = coalitions.uncertainty_reduction_shapley(v_m, n)
    rhos = [spearmanr(pm, pe).statistic for pm, pe in zip(phi_m, phi_e) if np.std(pe) > 1e-9 and np.std(pm) > 1e-9]
    big = np.abs(phi_e) > 1e-3
    return {
        "v_mask_pearson_interior": float(pearsonr(v_m[:, interior].ravel(), v_e[:, interior].ravel()).statistic),
        "v_mask_spearman_interior": float(spearmanr(v_m[:, interior].ravel(), v_e[:, interior].ravel()).statistic),
        "v_mask_mae_interior": float(w @ np.abs(v_m - v_e)[:, interior].mean(1)),
        "v_mask_empty_model_vs_exact": [float(w @ v_m[:, 0]), float(w @ v_e[:, 0])],
        "v_mask_full_model_belief": float(w @ v_m[:, -1]),
        "calibration_by_coalition_size_model_vs_exact": {
            int(k): [float(w @ v_m[:, sizes == k].mean(1)), float(w @ v_e[:, sizes == k].mean(1))] for k in range(n)
        },
        "q_coalition_mae": float(w @ np.abs(q_m - q_e).mean(1)),
        "phi_mask_mae": float(w @ np.abs(phi_m - phi_e).mean(1)),
        "phi_mask_mean_abs_exact": float(w @ np.abs(phi_e).mean(1)),
        "phi_mask_spearman_median": float(np.median(rhos)),
        "phi_mask_sign_agreement": float((np.sign(phi_m) == np.sign(phi_e))[big].mean()),
    }, phi_m


_EVAL_EPS = None


def full_information_q(model: VAEACQ, params, obs: np.ndarray, samples: int = 64) -> np.ndarray:
    """The agent's Q at full information, ``E_{z ~ full belief}[Q_omega(z, .)]`` (what it acts on).

    Fixed common random numbers, so repeated calls are deterministic.
    """

    global _EVAL_EPS
    if _EVAL_EPS is None or _EVAL_EPS.shape != (samples, model.cfg.latent_dim):
        _EVAL_EPS = jax.random.normal(jax.random.PRNGKey(12345), (samples, model.cfg.latent_dim))
    obs = np.asarray(obs, np.float32)
    return np.concatenate(
        [
            np.asarray(belief_q(model, params, jnp.asarray(c), _EVAL_EPS))
            for c in np.array_split(obs, max(1, len(obs) // 256))
        ]
    )


def greedy_optimal_rate(model: VAEACQ, params, ref: OracleReference) -> float:
    q_full = full_information_q(model, params, ref.obs)
    chosen = np.argmax(np.where(~np.isnan(ref.q_exact), q_full, -np.inf), axis=1)
    ok = np.isclose(ref.q_exact[np.arange(len(ref.obs)), chosen], np.nanmax(ref.q_exact, axis=1))
    return float(ref.w @ ok.astype(float))


CHECKPOINT = re.compile(
    r"(?P<kind>vaeac_q|td)_(?P<opp>random|pm\d+a?)_m(?P<m>\d+)(?P<untied>_untied)?(?P<var>_var)?"
    r"(?:_h(?P<hidden>\d+)x(?P<depth>\d+)(?P<act>gelu|relu)|(?P<xrl>_xrl)(?P<xrl_width>\d+)?)?(?:_d(?P<latent>\d+))?(?P<separate>_separate)?_s(?P<seed>\d+)$"
)


def load_checkpoint(path: Path):
    """``(kind, model, params)`` from a ``{vaeac_q|td}_random_m<M>[_untied]_s<seed>.msgpack`` checkpoint."""

    m = CHECKPOINT.match(path.stem)
    if m is None:
        raise ValueError(f"unrecognised checkpoint name {path.name}")
    cfg = Config(n_components=int(m["m"]), seed=int(m["seed"]), tied_encoder=m["untied"] is None)
    if m["var"]:
        cfg = dataclasses.replace(cfg, var_weight=50.0)
    if m["xrl"]:
        cfg = dataclasses.replace(cfg, **XRL_MATCHED)
        if m["xrl_width"]:
            cfg = dataclasses.replace(cfg, hidden=int(m["xrl_width"]))
    if m["latent"]:
        cfg = dataclasses.replace(cfg, latent_dim=int(m["latent"]))
    if m["separate"]:
        cfg = dataclasses.replace(cfg, shared_heads=False)
    if m["hidden"]:
        cfg = dataclasses.replace(cfg, hidden=int(m["hidden"]), depth=int(m["depth"]), activation=m["act"])
    model = VAEACQ(cfg)
    params = serialization.from_bytes(init_params(model, 0), path.read_bytes())
    return m["kind"], model, params


def tictactoe_checkpoint(
    kind: str, seed: int, size_suffix: str = "", opponent: str = "random", td_var: bool = False
) -> Path:
    """Tic-tac-toe checkpoint path. At a non-default size the exact-Q model is the variance-head one; a TD agent
    against a perturbed-minimax opponent is the annealed one (``pm05a``); ``td_var``: the TD agent whose variance
    head is trained on its TD target."""

    var = "_var" if (kind == "vaeac_q" and size_suffix) or (kind == "td" and td_var) else ""
    opp = opponent + "a" if kind == "td" and opponent != "random" else opponent
    return (
        Path(__file__).resolve().parents[2]
        / "results"
        / "models"
        / f"{kind}_{opp}_m10{var}{size_suffix}_s{seed}.msgpack"
    )
