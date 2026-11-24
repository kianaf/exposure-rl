"""Every number of the ESANN paper, recomputed from the result files, and checked against main.tex.

Table 1 rows come from paper_table.py; text numbers from the result files of predictive_entropy.py,
score_own_state.py and paper_analyses.py, parameter counts from the checkpoints (architecture_detail.py). Each text number is rendered into the exact LaTeX snippet the paper uses
and looked up in main.tex, so a changed result or a stale sentence shows up as a mismatch.

Usage: uv run python scripts/paper_numbers.py [--tex ../paper/main.tex]   (exit code 1 if anything mismatches)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAG = "_xrl256"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def mean(values) -> float:
    return float(np.mean(list(values)))


def seeds(d: dict, key: str, sub: str | None = None) -> float:
    """Mean over the seed entries ("0", "1", "2") of ``d[seed][key]`` (or ``d[seed][key][sub]``)."""

    vals = [v[key] if sub is None else v[key][sub] for k, v in d.items() if k.isdigit()]
    return mean(vals)


def _certainty_means() -> dict:
    """Mean over seeds of the certainty-head baselines (results/certainty_baselines/<domain>_<variant>_s<seed>.json)."""

    import glob
    import os

    rows: dict = {}
    for f in glob.glob(str(RES / "certainty_baselines" / "*_s[012].json")):
        key = os.path.basename(f)[:-5].rsplit("_s", 1)[0]
        r = json.loads(Path(f).read_text())
        if "rho_mean" not in r["predicted_outcome"]:  # the diverged first surrogate version, not rescored
            continue
        rows.setdefault(key, []).append(
            {"q": r["predicted_outcome"]["rho_mean"], "obs": r["v_obs"]["rho_mean"], "inh": r["v_inh"]["rho_mean"], "neg": r["v_obs_negative_share"]}
        )
    return {k: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for k, v in rows.items()}


def text_numbers() -> list[tuple[str, str]]:
    """``(what, LaTeX snippet that must occur in main.tex)``."""

    tab = load("paper_table.json")
    cell = lambda block, agent, col: tab[f"{block}_{agent}"][col][0]
    an = load("paper_analyses.json")
    pr, op = an["probes"], an["optimal"]
    f2 = lambda x: f"{x:.2f}"
    pct = lambda x: f"{100 * x:.0f}"
    e, d, s = "exposure_rl_td", "direct_q", "surrogate"
    arch = _module("architecture_detail").load_all()["tictactoe"]  # Dense shapes read from the seed-0 checkpoints
    n_params = lambda layers: sum(i * o + o for i, o in layers)
    ours = sum(n_params(arch[k]) for k in ("belief", "decoder", "q_head", "var_head"))
    dq = arch["direct_q"]  # xrl MA-DQN layers; the table's MA-DQN adds s and m2 heads (3 outputs per action)
    ma = n_params([(dq[0][0], 256), (256, 256), (256, 3 * dq[-1][1])])
    ma512 = n_params([(dq[0][0], 512), (512, 512), (512, 3 * dq[-1][1])])
    cb = _certainty_means()
    mc = load("mc_noise_minesweeper.json")  # three sampling repeats, one checkpoint
    rng = lambda gs: max(max(mc[r_][g] for r_ in ("0", "1", "2")) - min(mc[r_][g] for r_ in ("0", "1", "2")) for g in gs)
    rk = load("rank_columns_xrl256.json")  # latent v_obs (own Q), scripts/score_rank_columns.py
    lat = lambda b_: mean(v["own_phi_mask_spearman_mean"] for v in rk[b_].values())
    _exposure = lambda b_, c: tab[f"{b_}_exposure_rl_td"][c][0]
    k = lambda n: f"${round(n / 1000)}$k"
    own_inh = lambda d_, m_: mean(load(f"own_inherent/{d_}_{m_}_s{s_}.json")["v_inh"]["rho_mean"] for s_ in range(3))
    w512 = lambda d_, g: cb[f"{d_}_ma_w512"]["obs"] if g == "obs" else mean(
        load(f"own_inherent/{d_}_ma_w512_s{s_}_agent.json")["v_inh"]["rho_mean"] for s_ in range(3))
    n_ttt = {d_: len(_module("vobs_diagnostic").setting(d_)[2]) for d_ in ("random", "pm05")}  # scored boards = the occupancy
    spread_rep = round(max(  # largest gap between the methods under replay matching on their own Q
        max(cell(b_, x, c_) for x in (e, d, s)) - min(cell(b_, x, c_) for x in (e, d, s))
        for b_ in ("ttt_random", "ttt_pm05", "ms") for c_ in ("q_rep_rho", "vm_rep_rho")
    ), 2)
    return [
        (
            "setup: parameters",
            f"(EXPOSURE has about ${round(ours / ma)}\\times$ the parameters of the MA-DQN, but a width-512 MA-DQN of similar "
            "size does not close the gap)"
            # similar size: the width-512 MA-DQN within 15% of EXPOSURE's parameters
            if abs(ma512 - ours) / ours <= 0.15 and
            # wherever EXPOSURE leads the width-256 MA-DQN on a certainty game, the width-512 one stays below it too
            # (v_inh: the agent's own s, results/own_inherent/<domain>_ma_w512_s<seed>_agent.json)
            all(
                w512(d_, g) < _exposure(b_, c)
                for d_, b_ in (("random", "ttt_random"), ("pm05", "ttt_pm05"), ("ms", "ms"))
                for g, c in (("obs", "vm_rho"), ("inh", "vs_rho"))
                if _exposure(b_, c) > cell(b_, d, c)
            )
            else "WIDTH 512 CLOSES THE GAP",
        ),
        (
            "setup: Minesweeper scored observations",
            f"Tic-tac-toe is scored on all ${n_ttt['random']}$ and ${n_ttt['pm05']}$ observations of the occupancy, Minesweeper on "
            f"${op['minesweeper_scored_observations']}$ occupancy-sampled ones.",
        ),
        (
            "Table 1 caption: optimal wins",
            f"and the optimal agent reaches ${100 * op['tictactoe_random']:.1f}$, ${100 * op['tictactoe_pm05']:.1f}$ and "
            f"${100 * op['minesweeper']:.1f}\\%$.",
        ),
        (
            "probe: outcome class accuracy and optimal expected return R^2 (raw board)",
            (
                f"recovers the result class with ${pct(seeds(pr, 'class_balanced_accuracy'))}\\%$ balanced accuracy versus "
                f"${pct(pr['raw']['class_balanced_accuracy'])}\\%$ for the same probe on the raw board, and the expected return with $R^2="
                f"{f2(seeds(pr, 'value_r2'))}$ versus ${f2(pr['raw']['value_r2'])}$."
            ),
        ),
        (
            "results: predicted outcome, best everywhere, trivial gap vs random (within ~1 seed SD)",
            "For the expected return, EXPOSURE is best in all domains, but the gap is trivial in tic-tac-toe with a random "
            "opponent."
            if 0 < cell("ttt_random", e, "q_rho") - cell("ttt_random", d, "q_rho") <= 0.015
            and all(cell(b_, e, "q_rho") > max(cell(b_, x, "q_rho") for x in (d, s)) for b_ in ("ttt_random", "ttt_pm05", "ms"))
            else "PREDICTED-OUTCOME CLAIM NO LONGER HOLDS",
        ),
        (
            "results: v_obs and v_inh (agent's own s) best in every domain",
            "It also leads on the observational and inherent uncertainty in every domain."
            if all(cell(b_, e, c_) > max(cell(b_, x, c_) for x in (d, s))
                   for b_ in ("ttt_random", "ttt_pm05", "ms") for c_ in ("vm_rho", "vs_rho"))
            else "V_OBS / V_INH CLAIM NO LONGER HOLDS",
        ),
        # own-policy v_inh sentence commented out of the paper (2026-11-16); results/own_inherent keeps the numbers
        # latent-spread v_obs sentence cut from the Results (2026-11-16); rank_columns_xrl256.json keeps the numbers
        (
            "results: replay matching on every method's own Q (shared coalition values)",
            (
                f"Correlations under this common estimation procedure differ by at most ${spread_rep:.2f}$, consistent with "
                "coalition estimation contributing to the main gaps."
            )
            if spread_rep <= 0.07
            else "REPLAY SPREAD ABOVE 0.07",
        ),
        # Monte Carlo noise sentence commented out of the paper (2026-11-16); scripts/mc_noise.py keeps the numbers
        # "In Minesweeper, it also wins the most games." removed from the paper (2026-11-16)
        # "...best against the random opponent" (phi^H) cut from the paper (2026-11-16)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=ROOT.parent / "paper" / "main.tex")
    args = parser.parse_args()
    tex = args.tex.read_text()
    flat = re.sub(r"\s+", " ", tex)
    bad = 0

    # Table 1: every generated row must occur verbatim
    table = _module("paper_table")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["paper_table.py", "--stat", "mean"]  # the paper reports the mean over boards since 2026-11-08
        table.main()
    print("Table 1")
    for line in buf.getvalue().splitlines():
        row = line.split("  %")[0].strip()
        ok = row in flat
        bad += not ok
        print(f"  {'ok ' if ok else 'MISMATCH'}  {row[:110]}")

    print("Text")
    for what, snippet in text_numbers():
        ok = snippet in flat
        bad += not ok
        print(f"  {'ok ' if ok else 'MISMATCH'}  {what}: {snippet}")
    print(f"\n{'all numbers match' if not bad else f'{bad} mismatches'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
