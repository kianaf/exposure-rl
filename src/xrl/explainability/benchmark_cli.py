"""``xrl benchmark`` — compare the two masked-Q arms at matched compute.

Takes two finished runs, not a training configuration: the surrogate-backed arm
(an ordinary DQN plus a distilled masked-Q surrogate) and the direct arm (one
network trained with a feature-mask input and positive masked batch ratio). Both must have been trained with
``--eval-interval > 0``, because the comparison is read off their evaluation
curves.

Two numbers come out of the FLOP curves:

* **iso-FLOP quality** — each arm's evaluation metric at the largest compute
  both actually reached;
* **FLOPs to parity** — the compute at which each arm first reached a target
  and held it.

``--surrogate-arm`` and ``--q-surrogate`` are two artifacts of the same arm, not
alternatives: the first is the base DQN run supplying that arm's FLOP curve and
policy performance, the second is the masked-Q model distilled *from* it, which
supplies its coalition values and adds its distillation FLOPs to the arm's
total.

Attribution quality runs as one :func:`run_comparison` study per arm, each
against its own policy. The arms are independently trained networks, so a
surrogate distilled from one approximates a Q-function the other does not have.

The environments are not symmetric and the command does not pretend otherwise.
Minesweeper is single-agent, so ``--opponent`` is rejected for it. Tic Tac Toe
records several opponent streams and needs one named — but not ``minimax``,
whose win rate is zero by construction, so a parity target on it is
unreachable by either arm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from xrl.explainability.benchmark import (
    ArmCurve,
    common_flop_budget,
    flops_to_parity,
    load_arm_curve,
    quality_at_budget,
)
from xrl.training.metrics_io import DEFAULT_ENV_ID
from xrl.training.progress import announce

__all__ = ["benchmark"]

SURROGATE_ARM = "surrogate-backed"
DIRECT_ARM = "direct-masked-q"

# Perfect play cannot lose, so win_rate against it is identically zero and no
# amount of compute reaches a positive target.
_UNREACHABLE_OPPONENT = "minimax"


def benchmark(
    surrogate_arm: Annotated[Path, typer.Option("--surrogate-arm", help="Base DQN run for the surrogate-backed arm.")],
    direct_arm: Annotated[
        Path,
        typer.Option(
            "--direct-arm",
            help="Run trained with feature_mask_input=true and masked_batch_ratio>0.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for the benchmark's artifacts.")],
    q_surrogate: Annotated[
        Path | None,
        typer.Option("--q-surrogate", help="Distilled surrogate for the surrogate-backed arm's attributions."),
    ] = None,
    metric: Annotated[str, typer.Option(help="Evaluation column defining performance.")] = "win_rate",
    opponent: Annotated[
        str | None,
        typer.Option(help="Evaluation opponent (Tic Tac Toe only; Minesweeper has none)."),
    ] = None,
    parity_threshold: Annotated[
        float, typer.Option(help="Metric value an arm must reach and hold to count as at parity.")
    ] = 0.8,
    parity_hold: Annotated[
        int, typer.Option(help="Consecutive checkpoints the target must hold, so one lucky eval does not count.")
    ] = 2,
    states: Annotated[int, typer.Option(help="Held-out probe states for the attribution comparison.")] = 32,
    num_permutations: Annotated[int, typer.Option(help="Permutation budget for sampled masked-q Shapley.")] = 64,
    attributions: Annotated[
        bool,
        typer.Option("--attributions/--no-attributions", help="Run the attribution-quality comparison."),
    ] = True,
    plots: Annotated[bool, typer.Option("--plots/--no-plots", help="Render figures (needs the monitoring extra).")] = (
        True
    ),
) -> None:
    """Compare the surrogate-backed and direct masked-Q arms at matched compute."""

    if opponent == _UNREACHABLE_OPPONENT:
        raise typer.BadParameter(
            f"{metric} against {_UNREACHABLE_OPPONENT} is zero by construction, so no arm can reach a "
            "parity target on it; use --opponent random or perturbed_minimax, and read draw_rate separately"
        )
    if parity_hold < 1:
        raise typer.BadParameter("--parity-hold must be positive")

    try:
        curves = [
            load_arm_curve(surrogate_arm, name=SURROGATE_ARM, metric=metric, opponent=opponent),
            load_arm_curve(direct_arm, name=DIRECT_ARM, metric=metric, opponent=opponent),
        ]
        budget = common_flop_budget(curves)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    out.mkdir(parents=True, exist_ok=True)
    summary = _summary(curves, budget, metric=metric, opponent=opponent, threshold=parity_threshold, hold=parity_hold)
    summary["surrogate_training_flops"] = _surrogate_flops(q_surrogate)
    if summary["surrogate_training_flops"] is not None:
        # The distillation stage is compute the direct arm never spends, so the
        # surrogate-backed arm's true total is its DQN plus its surrogate.
        arm = summary["arms"][SURROGATE_ARM]
        arm["total_flops_including_distillation"] = arm["total_flops"] + summary["surrogate_training_flops"]

    if attributions:
        studies = _attribution_study(
            surrogate_arm,
            direct_arm,
            out,
            q_surrogate=q_surrogate,
            states=states,
            num_permutations=num_permutations,
        )
        summary["attribution_study"] = studies
        for name, study in studies.items():
            # The coalition-quality side of the iso-FLOP figure: ADR 0015
            # requires reporting it next to the policy metric, because the
            # auxiliary loss can buy one at the other's expense.
            summary["arms"][name]["secondary_value"] = study["deletion_lift"]

    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written = [out / "benchmark_summary.json"]
    if plots:
        from xrl.explainability.benchmark_plots import plot_benchmark

        rendered = plot_benchmark(out)
        if not rendered:
            announce(f"no plots written for {out}; install the monitoring extra")
        written.extend(rendered)

    print(json.dumps({"out_dir": str(out), "artifacts": [str(path) for path in written]}, indent=2))


def _summary(
    curves: list[ArmCurve],
    budget: float,
    *,
    metric: str,
    opponent: str | None,
    threshold: float,
    hold: int,
) -> dict[str, Any]:
    return {
        "env_id": _env_id(curves[0].run_dir),
        "metric": metric,
        "opponent": opponent,
        "flops_source": curves[0].flops_source,
        "iso_flop_budget": budget,
        "parity_threshold": threshold,
        "parity_hold_checkpoints": hold,
        "secondary_metric_name": "faithfulness lift",
        "arms": {
            curve.name: {
                "run_dir": str(curve.run_dir),
                "total_flops": curve.total_flops,
                "quality_at_iso_flop_budget": quality_at_budget(curve, budget),
                "final_value": curve.values[-1],
                "flops_to_parity": flops_to_parity(curve, threshold, hold=hold).flops,
                "checkpoints": len(curve.flops),
            }
            for curve in curves
        },
        "curves": {
            curve.name: {
                "run_dir": str(curve.run_dir),
                "flops": list(curve.flops),
                "values": list(curve.values),
            }
            for curve in curves
        },
    }


def _env_id(run_dir: Path) -> str:
    config = run_dir / "config.json"
    if not config.exists():
        return DEFAULT_ENV_ID
    return str(json.loads(config.read_text(encoding="utf-8")).get("env_id", DEFAULT_ENV_ID))


def _surrogate_flops(q_surrogate: Path | None) -> float | None:
    if q_surrogate is None:
        return None
    path = Path(q_surrogate) / "surrogate_summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    total = payload.get("training_flops")
    return float(total) if total is not None else None


def _attribution_study(
    surrogate_arm: Path,
    direct_arm: Path,
    out: Path,
    *,
    q_surrogate: Path | None,
    states: int,
    num_permutations: int,
) -> dict[str, Any]:
    """Score each arm's attributions against its own policy, in its own study.

    The arms are independently trained networks. A surrogate distilled from the
    surrogate arm approximates *that* Q-function, so it cannot supply coalition
    values for the direct arm's — and ``validate_surrogate_base_run`` rightly
    refuses to let it try. One study per arm is the only correct arrangement,
    matching ADR 0015: each masked model is judged against its own
    full-observation model, never against the other arm's.

    The consequence is that no attribution pair exists *within* a study, so this
    reports no agreement. Cross-policy agreement needs both arms explained on a
    shared probe set with a shared action, which
    ``examples/explainable_q_comparison_walkthrough.py`` does for Tic Tac Toe.
    """

    studies: dict[str, Any] = {}
    if q_surrogate is not None:
        studies[SURROGATE_ARM] = _one_arm_study(
            surrogate_arm,
            out / "attributions-surrogate",
            method="masked-q",
            evaluator="surrogate",
            q_surrogate=q_surrogate,
            states=states,
            num_permutations=num_permutations,
        )
    studies[DIRECT_ARM] = _one_arm_study(
        direct_arm,
        out / "attributions-direct",
        method="masked-q-direct",
        evaluator="direct",
        q_surrogate=None,
        states=states,
        num_permutations=num_permutations,
    )
    return studies


def _one_arm_study(
    run_dir: Path,
    out: Path,
    *,
    method: str,
    evaluator: str,
    q_surrogate: Path | None,
    states: int,
    num_permutations: int,
) -> dict[str, Any]:
    from xrl.explainability.comparison import ComparisonConfig, run_comparison

    config = ComparisonConfig(methods=(method,), states=states, num_permutations=num_permutations)
    report = run_comparison(run_dir, config=config, q_surrogate=q_surrogate)
    report.write(out)
    summary = report.summary().get("faithfulness", [])
    return {
        "run_dir": str(run_dir),
        "out_dir": str(out),
        "method": method,
        "skipped": dict(report.skipped),
        "states": report.state_count,
        # The lift over a random ordering under the arm's own game: bare AOPC
        # would mix explainer quality with how much signal the states carry.
        "deletion_lift": _lift(summary, method, evaluator, "deletion_lift"),
        "insertion_lift": _lift(summary, method, evaluator, "insertion_lift"),
        "marginal_deletion_lift": _lift(summary, method, "marginal", "deletion_lift"),
    }


def _lift(rows: Any, method: str, evaluator: str, field: str) -> float | None:
    for row in rows if isinstance(rows, list) else []:
        if row.get("method") == method and row.get("evaluator") == evaluator:
            value = row.get(field)
            if isinstance(value, dict):
                return value.get("mean")
            return value
    return None
