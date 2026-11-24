"""Table 1 of the paper: TD-trained agents only, one network width (2 hidden layers, ReLU, xrl-matched otherwise).

Rows per domain: EXPOSURE, the mask-aware DQN and DQN + surrogate. Columns (as in main.tex, since 2026-11-07 rho only):
  predicted outcome | v_obs | v_inh | H | wins (replay matching is quoted in the text, not the table)
rho = median over boards of the Spearman rho of the Shapley values (Pearson r of coalition values stays in the text
only). Every cell is the mean $\\pm$ seed SD over 3 seeds; within each domain the best learned method
of a column is bold (when two or more methods report it; replay columns are a reference and never bold). Missing
cells print as \\pending{}.

Per-seed values come from the result files where they are stored per seed, else from the per-seed lines that
replay_matching_eval.py prints ("<domain> <seed> {json}") in results/logs/fix/eval[_replay]_<domain>_w<W>.txt.
v_inh: against the exact return variance of each method's own greedy policy (score_own_inherent.py; --vinh optimal
for the optimal policy's, the reference before 2026-11-15). H: Shannon belief entropy (predictive_entropy.py). DQN + surrogate, tic-tac-toe mean game: the surrogate distilled on
its DQN's epsilon-greedy occupancy (score_surrogate_occupancy.py); its replay column is replay matching on the DQN's
own Q (eval log). Wins in percent (the % sign is in the column header).

Usage: uv run python scripts/paper_table.py [--width 256] -> prints the LaTeX rows, writes results/paper_table.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LOGS = RES / "logs" / "fix"
PENDING = "\\pending{}"
DOMAINS = (  # table block, eval domain, opponent suffix, log stem, wins metric
    ("ttt_random", "tictactoe", "", "ttt_random", "agent_win"),
    ("ttt_pm05", "tictactoe", "_pm05", "ttt_pm05", "agent_win"),
    ("ms", "minesweeper", "", "ms", "agent_p_win"),
)
AGENTS = (("exposure_rl_td", "EXPOSURE"), ("direct_q", "MA-DQN"), ("surrogate", "DQN + surr."))  # short names defined in the setup
COLUMNS = ("wins", "q_rho", "vm_rho", "vs_rho", "h_rho")  # replay rho: text only since 2026-11-07
OURS_ONLY = {"vm_r", "vs_r", "h_rho"}  # since 2026-11-08 the baselines have q, s and m2 heads: only H is ours alone
# baselines with certainty heads (scripts/certainty_baselines.py): MA-DQN + heads, DQN + surrogate (fixed version)
CERTAINTY = {"direct_q": "ma", "surrogate": "surr2"}
CERT_DOMAIN = {"ttt_random": "random", "ttt_pm05": "pm05", "ms": "ms"}
REFERENCE = {"q_rep_rho", "vm_rep_rho"}  # replay matching: never bold
EVAL_METRIC = {  # column -> metric in replay_matching_eval.py output
    "vm_r": "own_v_mask_pearson_interior",
    "vm_rep": "replay_v_mask_pearson_interior",
    "q_r": "mean_game_pearson_interior",
    "q_rho": "phi_Q_spearman_median",
    "q_rep": "replay_mean_game_pearson_interior",
    "q_rep_rho": "replay_phi_Q_spearman_median",
    "vm_rep_rho": "replay_phi_mask_spearman_median",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def eval_seeds(log_stem: str, width: int, domain: str) -> dict:
    """``{agent key: [per-seed metric dict, ...]}`` from the eval log (the replay log when it exists)."""

    for name in (f"eval_replay_{log_stem}_w{width}.txt", f"eval_{log_stem}_w{width}.txt"):
        path = LOGS / name
        if not path.exists():
            continue
        out: dict[str, list] = {}
        for line in path.read_text().splitlines():
            m = re.match(rf"^{domain} (\d+) (\{{.*\}})$", line)
            if m:
                for key, metrics in json.loads(m.group(2)).items():
                    out.setdefault(key, []).append(metrics)
        if out:
            return out
    return {}


def agent_seeds(seeds: dict, key: str) -> list:
    return seeds.get(key) or (seeds.get("exposure_rl", []) if key == "exposure_rl_td" else [])


STAT = ["mean"]  # --stat: per-board Spearman rho summarized by the mean over boards (paper) or the median
VOBS = ["imputation"]  # --vobs: EXPOSURE's v_obs through the decoder's completions (paper) or the latent spread
VINH = ["agent"]  # --vinh: phi^inh against each method's own greedy policy (paper, score_own_inherent.py) or the optimal one
OWN_INH = {"exposure_rl_td": "exposure", "direct_q": "ma", "surrogate": "surr"}


def S(prefix: str) -> str:
    return f"{prefix}_spearman_{STAT[0]}"


def RHO() -> str:
    return "rho" if STAT[0] == "median" else "rho_mean"


def cells(width: int) -> dict:
    tag = "_xrl" if width == 64 else f"_xrl{width}"
    table = {}
    for block, domain, opp, stem, win_key in DOMAINS:
        seeds = eval_seeds(stem, width, domain)
        for key, label in AGENTS:
            per = agent_seeds(seeds, key)
            metric = {c: m.replace("_spearman_median", f"_spearman_{STAT[0]}") for c, m in EVAL_METRIC.items()}
            col = {c: [r[m] for r in per if m in r] for c, m in metric.items()}
            col["wins"] = [r[win_key] for r in per if win_key in r]
            if key == "exposure_rl_td":
                if domain == "tictactoe":
                    files = [load(RES / f"baselines_{domain}{tag}{opp}_s{s}.json") for s in range(3)]
                    st = [f["td"]["exposure_rl"] for f in files if f and "td" in f]
                    col["vs_r"] = [x["v_state_pearson"] for x in st if "v_state_pearson" in x]
                    col["vs_rho"] = [x[S("phi_state")] for x in st if S("phi_state") in x]
                    ent = load(RES / f"predictive_entropy{tag}_{'pm05' if opp else 'tictactoe'}.json") or {}
                else:
                    ms = [load(RES / f"minesweeper_4x4_3_var{tag}_td_s{s}.json") for s in range(3)]
                    st = [f["model_v_mask_vs_oracle"]["v_state"] for f in ms if f]
                    col["vs_r"] = [x["v_state_pearson"] for x in st]
                    col["vs_rho"] = [x[S("phi_state")] for x in st if S("phi_state") in x]
                    ent = load(RES / f"predictive_entropy{tag}_minesweeper.json") or {}
                col["h_rho"] = [v[S("predictive_phi")] for v in ent.values()]
                col["vm_rho"] = [r[S("own_phi_mask")] for r in per if S("own_phi_mask") in r]
                ranks = load(RES / f"rank_columns{tag}.json") or {}  # scripts/score_rank_columns.py
                seeds_r = list((ranks.get(block) or {}).values())
                if not col["vm_rho"]:
                    col["vm_rho"] = [x[S("own_phi_mask")] for x in seeds_r]
                imp = (load(RES / "imputation_vobs.json") or {}).get(CERT_DOMAIN[block])  # v_obs via imputations
                if imp and len(imp) == 3 and VOBS[0] == "imputation":  # review point 1; scripts/imputation_vobs.py
                    col["vm_rho"] = [x[S("phi_obs")] for x in imp.values()]
                if not col.get("vs_rho") or STAT[0] == "mean":
                    col["vs_rho"] = [
                        x[S("phi_state")] for x in seeds_r if S("phi_state") in x
                    ]
            if key in CERTAINTY:  # all baseline columns from the runs with certainty heads
                runs = [load(RES / "certainty_baselines" / f"{CERT_DOMAIN[block]}_{CERTAINTY[key]}_s{s}.json") for s in range(3)]
                runs = [r for r in runs if r]
                col["wins"] = [r[win_key] for r in runs]
                col["q_rho"] = [r["predicted_outcome"][RHO()] for r in runs]
                col["vm_rho"] = [r["v_obs"][RHO()] for r in runs]
                col["vs_rho"] = [r["v_inh"][RHO()] for r in runs]
            if VINH[0] in ("own", "agent"):  # 2026-11-15: own policy's exact variance; 2026-11-16: agent's own full-obs s
                suffix = "_agent" * (VINH[0] == "agent")
                own = [load(RES / "own_inherent" / f"{CERT_DOMAIN[block]}_{OWN_INH[key]}_s{s}{suffix}.json") for s in range(3)]
                col["vs_rho"] = [r["v_inh"][RHO()] for r in own if r]
            table[(block, key)] = {
                c: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None) if v else None
                for c, v in col.items()
            }
            table[(block, key)]["label"] = label
    return table


def fmt(cell, col, bold):
    if cell is None:
        return PENDING
    mean, sd = cell
    if col == "wins":
        text = f"{100 * mean:.1f}" + (f" \\pm {100 * sd:.1f}" if sd is not None else "")
    else:
        text = f"{mean:.2f}" + (f" \\pm {sd:.2f}" if sd is not None else "")
    return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--stat", choices=("mean", "median"), default="mean")
    parser.add_argument("--vobs", choices=("imputation", "latent"), default="imputation",
                        help="EXPOSURE's v_obs: decoder completions (scripts/imputation_vobs.py) or the latent spread")
    parser.add_argument("--vinh", choices=("own", "optimal", "agent"), default="agent",
                        help="phi^inh reference: the agent's own full-observation s (paper), its own greedy policy's exact variance, or the optimal policy's")
    parser.add_argument("--exposure-snapshot", type=Path, default=None,
                        help="take the EXPOSURE cells from a saved paper_table.json (while its sources are re-scored)")
    args = parser.parse_args()
    STAT[0], VOBS[0], VINH[0] = args.stat, args.vobs, args.vinh
    table = cells(args.width)
    if args.exposure_snapshot:
        snap = json.loads(args.exposure_snapshot.read_text())
        for block, *_ in DOMAINS:
            table[(block, "exposure_rl_td")] = {k: tuple(v) if isinstance(v, list) else v for k, v in snap[f"{block}_exposure_rl_td"].items()}
    (RES / "paper_table.json").write_text(json.dumps({f"{b}_{k}": v for (b, k), v in table.items()}, indent=2))
    for block, *_ in DOMAINS:
        best = {}
        for c in COLUMNS:
            if c in REFERENCE:
                continue
            vals = [table[(block, k)].get(c) for k, _ in AGENTS if c not in OURS_ONLY or k == "exposure_rl_td"]
            vals = [v[0] for v in vals if v is not None]
            if len(vals) > 1:
                best[c] = round(max(vals), 3 if c == "wins" else 2)
        for key, _ in AGENTS:
            row = table[(block, key)]
            out = []
            for c in COLUMNS:
                if c in OURS_ONLY and key != "exposure_rl_td":
                    out.append("--")
                    continue
                cell = row.get(c)
                is_best = cell is not None and c in best and round(cell[0], 3 if c == "wins" else 2) == best[c]
                out.append(fmt(cell, c, is_best))
            print(f"{row['label']} & " + " & ".join(out) + f" \\\\  % {block}")


if __name__ == "__main__":
    main()
