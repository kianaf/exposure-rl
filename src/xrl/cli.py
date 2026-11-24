import typer

from xrl.explainability.benchmark_cli import benchmark
from xrl.explainability.benchmark_plots import benchmark_plot
from xrl.explainability.distributional_q import train_q_surrogate
from xrl.explainability.explain_cli import compare_explanations, explain, train_fastshap
from xrl.remote.cli import remote_runs_app
from xrl.studies.controlled_q import controlled_q_study
from xrl.studies.surrogate_ablation import surrogate_ablation_study
from xrl.training.cli import eval_q
from xrl.training.compare_plots import compare_plot
from xrl.training.compare_runs import compare_runs
from xrl.training.comparison import compare
from xrl.training.minesweeper_dqn import visualize_minesweeper
from xrl.training.symmetry_metrics import minesweeper_q_symmetry
from xrl.training.train_cli import train
from xrl.ui.minesweeper_app import play_minesweeper
from xrl.ui.tictactoe_app import play

app = typer.Typer(
    help="xrl: Tic Tac Toe reinforcement-learning environments and utilities.",
    no_args_is_help=True,
)

app.command(name="play", help="Run a local Tic Tac Toe browser UI.")(play)
app.command(name="play-minesweeper", help="Run a local Minesweeper browser UI.")(play_minesweeper)
app.command(name="train", help="Train an agent: --env tictactoe|minesweeper, --algo tabular|q-regression|dqn.")(train)
app.command(name="visualize-minesweeper", help="Write an ANSI replay for a saved Minesweeper DQN run.")(
    visualize_minesweeper
)
app.command(name="minesweeper-q-symmetry", help="Report board-symmetry Q-value gaps for a saved Minesweeper run.")(
    minesweeper_q_symmetry
)
app.command(name="train-q-surrogate", help="Train a masked distributional surrogate for a categorical DQN.")(
    train_q_surrogate
)
app.command(name="explain", help="Explain a saved run's Q-values with SHAP-style feature attributions.")(explain)
app.command(name="train-fastshap", help="Train an amortized FastSHAP explainer for a saved run.")(train_fastshap)
app.command(
    name="compare-explanations",
    help="Compare explanation methods over many states and write a study directory.",
)(compare_explanations)
app.command(name="eval-q", help="Evaluate a saved tabular Q-learning agent.")(eval_q)
app.command(name="compare", help="Train and compare tabular, online-Q, and DQN methods on Tic Tac Toe.")(compare)
app.command(name="compare-runs", help="Compare finished training runs from disk, without retraining.")(compare_runs)
app.command(name="compare-plot", help="Plot a comparison directory written by xrl compare-runs.")(compare_plot)
app.command(
    name="benchmark",
    help="Compare the surrogate-backed and direct masked-Q arms at matched compute.",
)(benchmark)
app.command(name="benchmark-plot", help="Plot a compute-matched masked-Q benchmark directory.")(benchmark_plot)
app.command(name="mask-ratio-study", help="Run the controlled DQN mask-ratio study.")(controlled_q_study)
app.command(name="surrogate-ablation", help="Ablate surrogate construction against frozen agents.")(
    surrogate_ablation_study
)
app.add_typer(remote_runs_app, name="remote-runs")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
