"""Command-line entrypoints for Q-learning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xrl.training.config import ExperimentConfig
from xrl.training.loop import AECTrainingLoop, summarize_metrics, write_metrics_csv
from xrl.training.q_learning import QLearningAgent
from xrl.training.run_metadata import RunIdentity, execute_training_run
from xrl.training.tracking import DEFAULT_MLFLOW_TRACKING_URI, TrackerConfig, tracker_from_cli_args


def _tabular_progress(config: ExperimentConfig, interval: int):
    """Build the shared progress reporter for a tabular run."""

    from xrl.training.progress import TrainingProgress

    if interval <= 0 and config.eval_interval <= 0:
        return None
    return TrainingProgress(interval=max(0, interval))


def train_q(
    episodes: int = 10_000,
    opponent: str = "random",
    out: Path = Path("runs/q_random"),
    seed: int = 0,
    alpha: float = 0.3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 8_000,
    eval_interval: int = 1_000,
    eval_episodes: int = 200,
    metrics_window: int = 100,
    perturbed_minimax_discount: float = 1.0,
    encoder: str = "identity",
    stats_interval: int = 100,
    seeds: str | None = None,
    no_plots: bool = False,
    mlflow: bool = False,
    mlflow_experiment: str = "xrl",
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict[str, Any]:
    """Train a tabular Q-learning agent.

    Pass ``--seeds 0,1,2`` to run a multi-seed sweep (one ``seed_<k>/`` run per
    seed plus aggregate artifacts; ``--seed`` is ignored in that case).
    """
    config = ExperimentConfig(
        episodes=episodes,
        seed=seed,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        opponent=opponent,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        metrics_window=metrics_window,
        out_dir=out,
        perturbed_minimax_discount=perturbed_minimax_discount,
        encoder=encoder,
    )
    tracker_config = TrackerConfig(
        mlflow=mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    def make_loop(cfg: ExperimentConfig, tracker) -> AECTrainingLoop:
        return AECTrainingLoop(
            cfg,
            progress=_tabular_progress(cfg, stats_interval),
            tracker=tracker,
        )

    if seeds is not None:
        from xrl.training.metrics_io import AEC_SWEEP_EVAL_COLUMNS, AEC_SWEEP_METRIC_COLUMNS
        from xrl.training.seed_sweep import SeedSweepSpec, parse_seeds, run_seed_sweep, trainer_run_seed

        seed_tuple = parse_seeds(seeds)
        spec = SeedSweepSpec(
            seeds=seed_tuple,
            root=config.out_dir,
            run_name=f"tabular_q_seeds{len(seed_tuple)}",
            tags={"workflow": "seed_sweep", "trainer": "tabular_q"},
            base_params=config.to_json_dict(),
            metric_columns=(*AEC_SWEEP_METRIC_COLUMNS, "unique_states_seen", "q_table_rows"),
            eval_columns=AEC_SWEEP_EVAL_COLUMNS,
            eval_group_by="opponent",
            policy_columns=("minimax_action_match_rate",),
        )
        result = run_seed_sweep(
            spec,
            trainer_run_seed(
                config,
                make_loop,
                identity=RunIdentity(
                    environment="tictactoe",
                    algorithm="tabular",
                    eval_opponents=("random", "minimax"),
                ),
            ),
            tracker_config=tracker_config,
            plot=not no_plots,
        )
        return result

    loop = make_loop(config, tracker_from_cli_args(tracker_config))
    summary = execute_training_run(
        config,
        loop,
        RunIdentity(
            environment="tictactoe",
            algorithm="tabular",
            eval_opponents=("random", "minimax"),
        ),
    )
    return {"out_dir": str(config.out_dir), "summary": summary}


def eval_q(
    model: Path,
    opponent: str = "minimax",
    episodes: int = 1_000,
    seed: int = 0,
    out: Path | None = None,
    perturbed_minimax_discount: float = 1.0,
) -> None:
    """Evaluate a saved tabular Q-learning agent."""
    learner = QLearningAgent.load(model, seed=seed)
    config = ExperimentConfig(
        episodes=0,
        seed=seed,
        opponent=opponent,
        out_dir=out or model.parent,
        perturbed_minimax_discount=perturbed_minimax_discount,
    )
    loop = AECTrainingLoop(config)
    metrics = loop.evaluate(learner, episodes=episodes, opponent_name=opponent, seed=seed)
    summary = summarize_metrics(metrics)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_metrics_csv(out / "eval_metrics.csv", metrics)
        (out / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
