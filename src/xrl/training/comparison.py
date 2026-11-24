"""Run and plot comparable Q-learning experiments."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Annotated, Any

import typer

from xrl._optional import MissingNeuralDependenciesError
from xrl.training.config import ExperimentConfig
from xrl.training.loop import AECTrainingLoop
from xrl.training.metrics_io import read_dict_rows, write_dict_rows
from xrl.training.tracking import (
    DEFAULT_MLFLOW_EXPERIMENT,
    DEFAULT_MLFLOW_TRACKING_URI,
    MLflowTracker,
    NoopTracker,
    TrainingTracker,
)
from xrl.training.value_heads import normalize_value_head

METHODS = ("tabular", "tabular-symmetry", "online-q", "dqn-scalar", "dqn-two-hot", "dqn-hl-gauss")
METHOD_MARKERS = {
    "tabular": "o",
    "tabular-symmetry": "^",
    "online-q": "P",
    "dqn-scalar": "s",
    "dqn-two-hot": "D",
    "dqn-hl-gauss": "*",
}
METHOD_LINESTYLES = {
    "tabular": "-",
    "tabular-symmetry": "--",
    "online-q": (0, (5, 1)),
    "dqn-scalar": ":",
    "dqn-two-hot": "-.",
    "dqn-hl-gauss": (0, (3, 1, 1, 1)),
}
DEFAULT_MARKERS = ("P", "X", "<", ">", "h", "8")
DEFAULT_LINESTYLES = ((0, (1, 1)), (0, (5, 1)), (0, (3, 2, 1, 2)))


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Configuration for a comparable Q-learning benchmark run."""

    methods: tuple[str, ...] = ("tabular", "tabular-symmetry")
    seeds: tuple[int, ...] = (0,)
    episodes: int = 1_000
    opponent: str = "random"
    eval_interval: int = 100
    eval_episodes: int = 50
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 800
    out_dir: Path = Path("runs/comparison")
    alpha: float = 0.3
    learning_rate: float = 1e-3
    perturbed_minimax_discount: float = 1.0
    plot: bool = True
    eval_opponents: tuple[str, ...] = ("random", "minimax")
    mlflow: bool = False
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI

    def __post_init__(self) -> None:
        unknown = sorted(set(self.methods) - set(METHODS))
        if unknown:
            raise ValueError(f"unknown comparison methods: {', '.join(unknown)}")
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be non-negative")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")
        unknown_eval_opponents = sorted(set(self.eval_opponents) - {"random", "minimax", "perturbed_minimax"})
        if unknown_eval_opponents:
            raise ValueError(f"unknown eval opponents: {', '.join(unknown_eval_opponents)}")
        if not 0.0 <= self.perturbed_minimax_discount <= 1.0:
            raise ValueError(f"perturbed_minimax_discount must be in [0, 1], got {self.perturbed_minimax_discount}")

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["methods"] = list(self.methods)
        data["seeds"] = list(self.seeds)
        data["eval_opponents"] = list(self.eval_opponents)
        data["out_dir"] = str(self.out_dir)
        return data


@dataclass(frozen=True, slots=True)
class RunResult:
    method: str
    seed: int
    run_dir: Path
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


ProgressReporter = Callable[[str], None]


def run_comparison(config: ComparisonConfig, *, progress_reporter: ProgressReporter | None = None) -> dict[str, Any]:
    """Run configured methods/seeds and write combined comparison artifacts."""

    root = config.out_dir / time.strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison_config.json").write_text(json.dumps(config.to_json_dict(), indent=2), encoding="utf-8")
    _report(progress_reporter, f"comparison output: {root}")

    results: list[RunResult] = []
    total_runs = len(config.methods) * len(config.seeds)
    run_index = 0
    for method in config.methods:
        for seed in config.seeds:
            run_index += 1
            run_dir = root / "runs" / f"{method}_seed_{seed}"
            _report(progress_reporter, f"[{run_index}/{total_runs}] starting {method} seed={seed} -> {run_dir}")
            result = _run_method(method, seed, run_dir, config)
            results.append(result)
            if result.status == "completed":
                _report(
                    progress_reporter,
                    f"[{run_index}/{total_runs}] completed {method} seed={seed} win_rate={result.summary.get('win_rate', 0):.3f} loss_rate={result.summary.get('loss_rate', 0):.3f}",
                )
            else:
                _report(progress_reporter, f"[{run_index}/{total_runs}] skipped {method} seed={seed}: {result.error}")

    _report(progress_reporter, "combining run CSV artifacts")
    metrics_rows = _collect_csv_rows(results, "metrics.csv")
    eval_rows = _collect_csv_rows(results, "eval_metrics.csv")
    policy_rows = _collect_csv_rows(results, "policy_metrics.csv")
    write_dict_rows(root / "comparison_metrics.csv", metrics_rows)
    write_dict_rows(root / "comparison_eval.csv", eval_rows)
    write_dict_rows(root / "comparison_policy.csv", policy_rows)
    write_dict_rows(root / "comparison_summary.csv", [_summary_row(result) for result in results])
    if config.plot:
        _report(progress_reporter, "writing comparison plots")
        plot_paths = plot_comparison(root)
        if plot_paths:
            _report(progress_reporter, f"wrote {len(plot_paths)} plots -> {root / 'plots'}")
        else:
            _report(
                progress_reporter,
                "no plots written; install the monitoring extra or check that CSV metrics are non-empty",
            )

    summary = {
        "out_dir": str(root),
        "runs": [_summary_row(result) for result in results],
        "metrics_rows": len(metrics_rows),
        "eval_rows": len(eval_rows),
        "policy_rows": len(policy_rows),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _report(
        progress_reporter,
        f"done: metrics_rows={len(metrics_rows)} eval_rows={len(eval_rows)} policy_rows={len(policy_rows)} summary={root / 'summary.json'}",
    )
    return summary


def _report(progress_reporter: ProgressReporter | None, message: str) -> None:
    if progress_reporter is not None:
        progress_reporter(message)


def _run_method(method: str, seed: int, run_dir: Path, config: ComparisonConfig) -> RunResult:
    try:
        if method == "tabular":
            summary = _run_tabular(method, seed, run_dir, config, encoder="identity")
        elif method == "tabular-symmetry":
            summary = _run_tabular(method, seed, run_dir, config, encoder="tictactoe_symmetry")
        elif method == "online-q":
            summary = _run_online_q(method, seed, run_dir, config)
        else:
            summary = _run_neural(method, seed, run_dir, config)
        return RunResult(method=method, seed=seed, run_dir=run_dir, status="completed", summary=summary)
    except MissingNeuralDependenciesError as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        error = str(exc)
        (run_dir / "skipped.json").write_text(
            json.dumps({"status": "skipped", "error": error}, indent=2), encoding="utf-8"
        )
        return RunResult(method=method, seed=seed, run_dir=run_dir, status="skipped", error=error)


def _run_tabular(method: str, seed: int, run_dir: Path, config: ComparisonConfig, *, encoder: str) -> dict[str, Any]:
    run_config = ExperimentConfig(
        episodes=config.episodes,
        seed=seed,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_episodes=config.epsilon_decay_episodes,
        opponent=config.opponent,
        eval_interval=config.eval_interval,
        eval_episodes=config.eval_episodes,
        out_dir=run_dir,
        perturbed_minimax_discount=config.perturbed_minimax_discount,
        encoder=encoder,
    )
    loop = AECTrainingLoop(
        run_config,
        tracker=_tracker_for_run(config, method, seed),
        eval_opponents=config.eval_opponents,
    )
    learner, metrics = loop.train()
    summary = loop.write_run(learner, metrics)
    summary["method"] = method
    return summary


def _run_online_q(method: str, seed: int, run_dir: Path, config: ComparisonConfig) -> dict[str, Any]:
    from xrl.training.q_regression import QRegressionConfig, QRegressionTrainer

    run_config = QRegressionConfig(
        episodes=config.episodes,
        seed=seed,
        opponent=config.opponent,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_episodes=config.epsilon_decay_episodes,
        eval_interval=config.eval_interval,
        eval_episodes=config.eval_episodes,
        out_dir=run_dir,
        perturbed_minimax_discount=config.perturbed_minimax_discount,
    )
    trainer = QRegressionTrainer(
        run_config,
        tracker=_tracker_for_run(config, method, seed),
        eval_opponents=config.eval_opponents,
    )
    agent, metrics = trainer.train()
    summary = trainer.write_run(agent, metrics)
    summary["method"] = method
    return summary


def _run_neural(method: str, seed: int, run_dir: Path, config: ComparisonConfig) -> dict[str, Any]:
    from xrl.training.neural_q import NeuralQConfig, NeuralQTrainer

    head = normalize_value_head(method.removeprefix("dqn-"))
    run_config = NeuralQConfig(
        episodes=config.episodes,
        seed=seed,
        value_head=head,
        opponent=config.opponent,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_episodes=config.epsilon_decay_episodes,
        eval_interval=config.eval_interval,
        eval_episodes=config.eval_episodes,
        out_dir=run_dir,
        perturbed_minimax_discount=config.perturbed_minimax_discount,
    )
    trainer = NeuralQTrainer(
        run_config,
        tracker=_tracker_for_run(config, method, seed),
        eval_opponents=config.eval_opponents,
    )
    agent, metrics = trainer.train()
    summary = trainer.write_run(agent, metrics)
    summary["method"] = method
    return summary


def _tracker_for_run(config: ComparisonConfig, method: str, seed: int) -> TrainingTracker:
    if not config.mlflow:
        return NoopTracker()
    return MLflowTracker(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_tracking_uri,
        tags={"workflow": "comparison", "method": method, "seed": str(seed)},
    )


def _collect_csv_rows(results: list[RunResult], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        path = result.run_dir / filename
        if result.status != "completed" or not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                combined = dict(row)
                if "seed" in combined:
                    combined["episode_seed"] = combined["seed"]
                combined["method"] = result.method
                combined["seed"] = result.seed
                rows.append(combined)
    return rows


def _summary_row(result: RunResult) -> dict[str, Any]:
    return {
        "method": result.method,
        "seed": result.seed,
        "status": result.status,
        "run_dir": str(result.run_dir),
        "error": result.error or "",
        **{key: value for key, value in result.summary.items() if isinstance(value, int | float | str | bool)},
    }


def plot_comparison(root: Path) -> list[Path]:
    """Create comparison plots from local CSV artifacts."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    root = Path(root)
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    metrics = read_dict_rows(root / "comparison_metrics.csv")
    eval_rows = read_dict_rows(root / "comparison_eval.csv")
    policy_rows = read_dict_rows(root / "comparison_policy.csv")
    written = [
        _plot_metric(metrics, plots_dir / "learner_return.png", "learner_return", "Learner Return"),
        _plot_metric(metrics, plots_dir / "q_table_rows.png", "q_table_rows", "Q-table Rows"),
        _plot_metric(metrics, plots_dir / "neural_loss.png", "td_error_mean", "TD Error / Neural Loss"),
        _plot_metric(metrics, plots_dir / "elapsed_seconds.png", "elapsed_seconds", "Elapsed Seconds"),
        _plot_metric(
            policy_rows,
            plots_dir / "minimax_action_match_rate.png",
            "minimax_action_match_rate",
            "Minimax Action Match Rate",
        ),
        *_plot_eval_metrics_by_opponent(eval_rows, plots_dir),
    ]
    plt.close("all")
    return [path for path in written if path is not None]


def _plot_metric(rows: list[dict[str, str]], path: Path, column: str, title: str) -> Path | None:
    return _plot_rows(rows, path, column, title, group_suffix=None)


def _plot_eval_metrics_by_opponent(rows: list[dict[str, str]], plots_dir: Path) -> list[Path | None]:
    opponents = sorted({row.get("opponent", "") for row in rows if row.get("opponent")})
    paths: list[Path | None] = []
    for opponent in opponents:
        opponent_rows = [row for row in rows if row.get("opponent") == opponent]
        safe_opponent = opponent.replace(" ", "_").replace("/", "_")
        title_prefix = f"Evaluation vs {opponent}"
        paths.extend(
            [
                _plot_rows(
                    opponent_rows,
                    plots_dir / f"eval_{safe_opponent}_win_rate.png",
                    "win_rate",
                    f"{title_prefix} Win Rate",
                    group_suffix=None,
                ),
                _plot_rows(
                    opponent_rows,
                    plots_dir / f"eval_{safe_opponent}_loss_rate.png",
                    "loss_rate",
                    f"{title_prefix} Loss Rate",
                    group_suffix=None,
                ),
                _plot_rows(
                    opponent_rows,
                    plots_dir / f"eval_{safe_opponent}_draw_rate.png",
                    "draw_rate",
                    f"{title_prefix} Draw Rate",
                    group_suffix=None,
                ),
            ]
        )
    return paths


def _plot_rows(
    rows: list[dict[str, str]], path: Path, column: str, title: str, *, group_suffix: str | None
) -> Path | None:
    if not rows or not any(row.get(column) not in {"", None} for row in rows):
        return None
    import matplotlib.pyplot as plt

    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        try:
            episode = int(float(row["episode"]))
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        group = row["method"]
        if group_suffix is not None and row.get(group_suffix):
            group = f"{group}:{row[group_suffix]}"
        grouped.setdefault(group, {}).setdefault(episode, []).append(value)
    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, (group, by_episode) in enumerate(sorted(grouped.items())):
        marker = marker_for_group(group, index)
        linestyle = linestyle_for_group(group, index)
        xs = sorted(by_episode)
        individual_xs = [episode for episode in xs for _ in by_episode[episode]]
        individual_ys = [value for episode in xs for value in by_episode[episode]]
        if individual_xs:
            ax.scatter(
                individual_xs,
                individual_ys,
                marker=marker,
                facecolors="none",
                edgecolors="black",
                alpha=0.25,
                s=30,
                linewidths=0.7,
            )
        mean_ys = [mean(by_episode[episode]) for episode in xs]
        if len(xs) > 1:
            ax.plot(
                xs,
                mean_ys,
                color="black",
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.65,
            )
        ax.scatter(
            xs,
            mean_ys,
            marker=marker,
            color="black",
            label=group,
            s=72,
            linewidths=0.6,
        )
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def marker_for_group(group: str, index: int) -> str:
    """Return a stable marker for a series, by name where one is reserved."""

    if group in METHOD_MARKERS:
        return METHOD_MARKERS[group]
    return DEFAULT_MARKERS[index % len(DEFAULT_MARKERS)]


def linestyle_for_group(group: str, index: int):
    """Return a stable linestyle for a series, by name where one is reserved."""

    if group in METHOD_LINESTYLES:
        return METHOD_LINESTYLES[group]
    return DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)]


def compare(
    methods: str = "tabular,tabular-symmetry,online-q,dqn-scalar,dqn-two-hot,dqn-hl-gauss",
    seeds: str = "0",
    episodes: int = 1_000,
    opponent: str = "random",
    eval_interval: int = 100,
    eval_episodes: int = 50,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 800,
    alpha: float = 0.3,
    learning_rate: float = 1e-3,
    out: Path = Path("runs/comparison"),
    perturbed_minimax_discount: float = 1.0,
    eval_opponents: str = "random,minimax",
    no_plots: Annotated[bool, typer.Option("--no-plots")] = False,
    mlflow: bool = False,
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> None:
    """Compare tabular, Q-regression, and DQN Q-learning methods."""
    config = ComparisonConfig(
        methods=tuple(item.strip() for item in methods.split(",") if item.strip()),
        seeds=tuple(int(item.strip()) for item in seeds.split(",") if item.strip()),
        episodes=episodes,
        opponent=opponent,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        alpha=alpha,
        learning_rate=learning_rate,
        out_dir=out,
        perturbed_minimax_discount=perturbed_minimax_discount,
        eval_opponents=tuple(item.strip() for item in eval_opponents.split(",") if item.strip()),
        plot=not no_plots,
        mlflow=mlflow,
        mlflow_experiment=mlflow_experiment,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    print(json.dumps(run_comparison(config, progress_reporter=print), indent=2))
