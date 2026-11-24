"""Unified ``xrl train`` entrypoint across environments and algorithms.

One command trains every supported (environment, algorithm) pair. The
environment-specific trainers stay where they are; this module owns only the
command-line concerns — validating the combination, rejecting flags that do not
apply to it, resolving per-target defaults, and expanding variant sweeps.

Defaults are resolved per target rather than shared, because the sensible
episode budget and evaluation cadence genuinely differ between a tabular
Tic Tac Toe run and a Minesweeper DQN. Every option therefore defaults to
``None`` here and is filled in from :data:`TARGET_DEFAULTS`, which also makes it
possible to tell "the user asked for this" apart from "this is the default" and
so to reject options that do not apply.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from xrl.training.config import parse_hidden_sizes
from xrl.training.progress import announce
from xrl.training.tracking import DEFAULT_MLFLOW_TRACKING_URI
from xrl.training.value_heads import SCALAR, VALUE_HEADS, normalize_value_head

__all__ = [
    "ALGORITHMS",
    "ENVIRONMENTS",
    "TARGET_DEFAULTS",
    "TrainTarget",
    "parse_algorithms",
    "parse_value_heads",
    "resolve_target",
    "resolve_targets",
    "train",
]

ENVIRONMENTS = ("tictactoe", "minesweeper")

# What the agent *is*: how it approximates Q, and how it updates.
#   tabular   Q-table, updated online
#   online-q  MLP, updated online from each transition
#   dqn       MLP, replay buffer + target network
# Orthogonal to the value head, which is what the network emits.
ALGORITHMS = ("tabular", "online-q", "dqn")

# (env, algo) pairs that exist.
#
# `tabular` is Tic Tac Toe only for a real reason: a Q-table over Minesweeper
# boards is not viable. `online-q` is Tic Tac Toe only for an incidental one --
# QRegressionTrainer is built on the two-player AEC episode driver, and nobody
# has written the single-agent Gymnasium equivalent. Its MLP would handle
# Minesweeper fine; the trainer simply does not exist yet.
SUPPORTED_TARGETS = (
    ("tictactoe", "tabular"),
    ("tictactoe", "online-q"),
    ("tictactoe", "dqn"),
    ("minesweeper", "dqn"),
)

# Per-target defaults, preserving the values each dedicated command used.
TARGET_DEFAULTS: dict[tuple[str, str], dict[str, Any]] = {
    ("tictactoe", "tabular"): {
        "episodes": 10_000,
        "epsilon_decay_episodes": 8_000,
        "eval_interval": 1_000,
        "eval_episodes": 200,
        "out": Path("runs/q_random"),
    },
    ("tictactoe", "online-q"): {
        "episodes": 1_000,
        "epsilon_decay_episodes": 800,
        "eval_interval": 0,
        "eval_episodes": 20,
        "out": Path("runs/q_regression"),
    },
    ("tictactoe", "dqn"): {
        "episodes": 1_000,
        "epsilon_decay_episodes": 800,
        "eval_interval": 0,
        "eval_episodes": 20,
        "out": Path("runs/dqn"),
    },
    ("minesweeper", "dqn"): {
        "episodes": 1_000,
        "epsilon_decay_episodes": 800,
        "eval_interval": 50,
        "eval_episodes": 20,
        "out": Path("runs/minesweeper-dqn"),
    },
}

# Options that only apply to some targets, and the human-readable scope used in
# the error message when they are passed anyway.
_TICTACTOE_ONLY = ("opponent", "perturbed_minimax_discount")
_MINESWEEPER_ONLY = ("rows", "columns", "mines")
_TABULAR_ONLY = ("alpha",)
_NEURAL_ONLY = ("learning_rate", "q_probe_states", "hidden_sizes")
_DQN_ONLY = (
    "value_head",
    "exploration",
    "backup",
    "temperature",
    "batch_size",
    "replay_capacity",
    "replay_warmup",
    "target_sync_interval",
    "num_bins",
    "v_min",
    "v_max",
    "hl_gauss_sigma_bins",
    "feature_mask_input",
    "masked_batch_ratio",
    "mask_empty_example_interval",
    "mask_shapley_probability",
    "mask_uniform_probability",
)


@dataclass(frozen=True, slots=True)
class TrainTarget:
    """A validated (environment, algorithm) training target."""

    env: str
    algo: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.env, self.algo)

    @property
    def is_dqn(self) -> bool:
        return self.algo == "dqn"

    @property
    def is_neural(self) -> bool:
        return self.algo in {"online-q", "dqn"}

    def default(self, name: str, fallback: Any = None) -> Any:
        return TARGET_DEFAULTS[self.key].get(name, fallback)


def resolve_target(env: str, algo: str) -> TrainTarget:
    """Validate an environment/algorithm pair, or raise ``typer.BadParameter``."""

    if env not in ENVIRONMENTS:
        raise typer.BadParameter(f"--env must be one of {', '.join(ENVIRONMENTS)}, got {env!r}")
    if algo not in ALGORITHMS:
        raise typer.BadParameter(f"--algo must be one of {', '.join(ALGORITHMS)}, got {algo!r}")
    if (env, algo) not in SUPPORTED_TARGETS:
        available = ", ".join(sorted(pair[1] for pair in SUPPORTED_TARGETS if pair[0] == env))
        raise typer.BadParameter(f"--algo {algo!r} is not available for --env {env!r}; supported: {available}")
    return TrainTarget(env=env, algo=algo)


def parse_algorithms(algos: str) -> tuple[str, ...]:
    """Parse a comma-separated ``--algo`` list, in order, rejecting duplicates."""

    parsed = tuple(name.strip() for name in algos.split(",") if name.strip())
    if not parsed:
        raise typer.BadParameter("--algo must name at least one algorithm")
    if len(set(parsed)) != len(parsed):
        raise typer.BadParameter(f"--algo contains duplicates: {', '.join(parsed)}")
    return parsed


def resolve_targets(env: str, algos: Sequence[str]) -> tuple[TrainTarget, ...]:
    """Validate every selected algorithm against ``env``, reporting all failures.

    Aggregating the errors matters for a sweep: fixing one algorithm at a time
    across three round-trips is worse than being told about all of them once.
    """

    targets: list[TrainTarget] = []
    failures: list[str] = []
    for algo in algos:
        try:
            targets.append(resolve_target(env, algo))
        except typer.BadParameter as error:
            failures.append(str(error))
    if failures:
        raise typer.BadParameter("; ".join(dict.fromkeys(failures)))
    return tuple(targets)


def parse_value_heads(value_heads: str) -> tuple[str, ...]:
    """Parse a comma-separated ``--value-head`` list, in order, rejecting duplicates.

    Legacy spellings (``regression``, ``two_hot``, ``hl_gauss``) are accepted
    and normalized, so a saved run's own recorded name can be pasted back in.
    """

    parsed = [name.strip() for name in value_heads.split(",") if name.strip()]
    if not parsed:
        raise typer.BadParameter("--value-head must name at least one value head")

    canonical: list[str] = []
    for name in parsed:
        try:
            canonical.append(normalize_value_head(name))
        except ValueError as error:
            hint = ""
            if name in ALGORITHMS:
                hint = f" ({name!r} is an --algo, not a value head)"
            raise typer.BadParameter(f"{error}{hint}") from error

    if len(set(canonical)) != len(canonical):
        raise typer.BadParameter(f"--value-head contains duplicates: {', '.join(canonical)}")
    return tuple(canonical)


def train(
    env: Annotated[str, typer.Option(help=f"Environment: {', '.join(ENVIRONMENTS)}.")] = "tictactoe",
    algo: Annotated[
        str,
        typer.Option(help=f"Algorithm, or a comma-separated list to sweep: {', '.join(ALGORITHMS)}."),
    ] = "dqn",
    value_head: Annotated[
        str,
        typer.Option(
            help=f"How the DQN represents a value, or a comma-separated list to sweep: {', '.join(VALUE_HEADS)}."
        ),
    ] = SCALAR,
    episodes: Annotated[int | None, typer.Option(help="Training episodes.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output directory for the saved run.")] = None,
    seed: Annotated[int, typer.Option(help="Base random seed.")] = 0,
    seeds: Annotated[
        str | None, typer.Option(help="Comma-separated seed list for a multi-seed sweep (overrides --seed).")
    ] = None,
    rows: Annotated[int | None, typer.Option(help="Minesweeper board rows.")] = None,
    columns: Annotated[int | None, typer.Option(help="Minesweeper board columns.")] = None,
    mines: Annotated[int | None, typer.Option(help="Minesweeper mine count.")] = None,
    hidden_sizes: Annotated[str | None, typer.Option(help="Comma-separated neural MLP hidden-layer widths.")] = None,
    opponent: Annotated[str | None, typer.Option(help="Tic Tac Toe training opponent.")] = None,
    perturbed_minimax_discount: Annotated[
        float | None, typer.Option(help="Per-episode decay of the perturbed_minimax random-move rate.")
    ] = None,
    alpha: Annotated[float | None, typer.Option(help="Tabular Q-learning update rate.")] = None,
    learning_rate: Annotated[float | None, typer.Option(help="Adam learning rate (neural algorithms).")] = None,
    gamma: Annotated[float, typer.Option(help="Discount factor.")] = 0.95,
    epsilon_start: Annotated[float, typer.Option(help="Initial exploration epsilon.")] = 1.0,
    epsilon_end: Annotated[float, typer.Option(help="Final exploration epsilon.")] = 0.05,
    epsilon_decay_episodes: Annotated[int | None, typer.Option(help="Episodes of linear epsilon decay.")] = None,
    exploration: Annotated[str | None, typer.Option(help="epsilon_greedy or boltzmann (DQN).")] = None,
    backup: Annotated[str | None, typer.Option(help="TD target operator: hard or soft (DQN).")] = None,
    temperature: Annotated[float | None, typer.Option(help="Softmax temperature (DQN).")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Replay minibatch size (DQN).")] = None,
    replay_capacity: Annotated[int | None, typer.Option(help="Replay buffer capacity (DQN).")] = None,
    replay_warmup: Annotated[int | None, typer.Option(help="Replay size before updates begin (DQN).")] = None,
    target_sync_interval: Annotated[int | None, typer.Option(help="Updates between target syncs (DQN).")] = None,
    num_bins: Annotated[int | None, typer.Option(help="Categorical support bins (DQN).")] = None,
    v_min: Annotated[float | None, typer.Option(help="Minimum categorical support center (DQN).")] = None,
    v_max: Annotated[float | None, typer.Option(help="Maximum categorical support center (DQN).")] = None,
    hl_gauss_sigma_bins: Annotated[float | None, typer.Option(help="HL-Gauss sigma in bin widths (DQN).")] = None,
    feature_mask_input: Annotated[
        bool | None,
        typer.Option(
            "--feature-mask-input/--no-feature-mask-input",
            help="Encode DQN inputs as [visible values, known-feature mask].",
        ),
    ] = None,
    masked_batch_ratio: Annotated[
        float | None,
        typer.Option(help="Masked examples appended per full replay example (DQN)."),
    ] = None,
    mask_empty_example_interval: Annotated[
        int | None,
        typer.Option(help="Masked examples per injected empty feature mask (DQN)."),
    ] = None,
    mask_shapley_probability: Annotated[
        float | None,
        typer.Option(help="Relative probability of Shapley-kernel mask-size sampling."),
    ] = None,
    mask_uniform_probability: Annotated[
        float | None,
        typer.Option(help="Relative probability of uniform mask-size sampling."),
    ] = None,
    encoder: Annotated[str, typer.Option(help="State/action encoder.")] = "identity",
    eval_interval: Annotated[int | None, typer.Option(help="Evaluate every N episodes; 0 disables.")] = None,
    eval_episodes: Annotated[int | None, typer.Option(help="Episodes per evaluation checkpoint.")] = None,
    metrics_window: Annotated[int, typer.Option(help="Trailing window for rolling metrics.")] = 100,
    q_probe_states: Annotated[int | None, typer.Option(help="Held-out probe states for Q health.")] = None,
    progress_interval: Annotated[
        int | None,
        typer.Option(help="Print a progress line every N episodes on stderr; 0 disables. Default 1/20th of the run."),
    ] = None,
    no_plots: Annotated[bool, typer.Option("--no-plots", help="Skip sweep plot generation.")] = False,
    mlflow: Annotated[bool, typer.Option(help="Enable MLflow tracking.")] = False,
    mlflow_experiment: Annotated[str, typer.Option(help="MLflow experiment name.")] = "xrl",
    mlflow_tracking_uri: Annotated[str, typer.Option(help="MLflow tracking URI.")] = DEFAULT_MLFLOW_TRACKING_URI,
) -> None:
    """Train an agent. Select the environment with --env and the algorithm with --algo.

    --algo, --value-head, and --seeds all accept comma-separated lists and
    compose: every valid combination is trained into its own subdirectory, with
    a variants_summary.json at the root.
    """

    supplied = _supplied_options(locals())
    targets = resolve_targets(env, parse_algorithms(algo))
    _reject_inapplicable(targets, supplied)

    heads = parse_value_heads(value_head)
    plan = _plan_variants(targets, heads)
    destination = Path(out) if out is not None else targets[0].default("out")

    def arguments_for(target: TrainTarget) -> dict[str, Any]:
        return _shared_arguments(
            target,
            supplied,
            seed=seed,
            seeds=seeds,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            encoder=encoder,
            metrics_window=metrics_window,
            no_plots=no_plots,
            mlflow=mlflow,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

    if len(plan) == 1:
        target, head = plan[0]
        print(json.dumps(_run_one(target, destination, head, arguments_for(target)), indent=2))
        return

    announce(f"{len(plan)} variants: {', '.join(identifier for identifier, _, _ in _labelled(plan))}")
    variants: dict[str, Any] = {}
    for identifier, target, head in _labelled(plan):
        variant_dir = destination / identifier
        announce(f"training {target.env}/{identifier} -> {variant_dir}")
        variants[identifier] = {
            "algo": target.algo,
            "value_head": head or None,
            **_run_one(target, variant_dir, head, arguments_for(target)),
        }

    summary = {
        "out_dir": str(destination),
        "env": env,
        "algos": [target.algo for target in targets],
        "value_heads": [head for head in heads if any(head == used for _, used in plan)],
        "variants": variants,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "variants_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _plan_variants(
    targets: Sequence[TrainTarget],
    heads: Sequence[str],
) -> list[tuple[TrainTarget, str]]:
    """Expand the selected algorithms and value heads into concrete runs.

    The value-head axis collapses for algorithms that have no choice: a tabular
    Q-table and an online-Q MLP always emit a scalar, so ``--algo dqn,online-q
    --value-head scalar,two-hot`` is three runs, not four. That is what makes
    the two axes composable without demanding a rectangular grid.
    """

    plan: list[tuple[TrainTarget, str]] = []
    for target in targets:
        for head in heads if target.is_dqn else ("",):
            plan.append((target, head))
    return plan


def _labelled(plan: Sequence[tuple[TrainTarget, str]]) -> list[tuple[str, TrainTarget, str]]:
    """Attach a filesystem-safe identifier to each variant."""

    return [(f"{target.algo}_{head}" if head else target.algo, target, head) for target, head in plan]


def _run_one(target: TrainTarget, destination: Path, value_head: str, shared: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one (target, value head, out) training run to its environment trainer."""

    arguments = dict(shared, out=destination)
    if target.env == "minesweeper":
        from xrl.training.minesweeper_dqn import train_minesweeper_dqn

        return train_minesweeper_dqn(value_head=value_head, **arguments)
    if target.algo == "tabular":
        from xrl.training.cli import train_q

        return train_q(**arguments)
    if target.algo == "online-q":
        from xrl.training.q_regression import train_q_regression

        return train_q_regression(**arguments)
    from xrl.training.neural_q import train_dqn

    return train_dqn(value_head=value_head, **arguments)


def _shared_arguments(target: TrainTarget, supplied: dict[str, Any], **explicit: Any) -> dict[str, Any]:
    """Build the keyword arguments the selected trainer accepts.

    Options the trainer does not accept are dropped rather than passed as
    ``None``; the trainers own their own defaults for anything not resolved
    here.
    """

    arguments: dict[str, Any] = dict(explicit)
    arguments["episodes"] = _resolve(supplied, target, "episodes")
    arguments["epsilon_decay_episodes"] = _resolve(supplied, target, "epsilon_decay_episodes")
    arguments["eval_interval"] = _resolve(supplied, target, "eval_interval")
    arguments["eval_episodes"] = _resolve(supplied, target, "eval_episodes")
    interval = _progress_interval(supplied, arguments["episodes"])
    # The tabular loop predates the shared reporter and names it stats_interval.
    arguments["stats_interval" if target.algo == "tabular" else "progress_interval"] = interval

    if target.env == "tictactoe":
        _carry(arguments, supplied, _TICTACTOE_ONLY)
    else:
        _carry(arguments, supplied, _MINESWEEPER_ONLY)
    if target.algo == "tabular":
        _carry(arguments, supplied, _TABULAR_ONLY)
        # The tabular loop has no probe-state or learning-rate options.
        arguments.pop("q_probe_states", None)
    if target.is_neural:
        _carry(arguments, supplied, _NEURAL_ONLY)
        if "hidden_sizes" in arguments:
            try:
                arguments["hidden_sizes"] = parse_hidden_sizes(arguments["hidden_sizes"])
            except ValueError as error:
                raise typer.BadParameter(
                    "--hidden-sizes must be a comma-separated list of positive integers"
                ) from error
    if target.is_dqn:
        _carry(arguments, supplied, [name for name in _DQN_ONLY if name != "value_head"])
    return arguments


def _progress_interval(supplied: dict[str, Any], episodes: int) -> int:
    """Resolve --progress-interval, defaulting to roughly twenty lines per run.

    A fixed default would print once per run on a short job and thousands of
    times on a long one; scaling with the episode budget keeps the output
    readable at both ends.
    """

    requested = supplied.get("progress_interval")
    if requested is not None:
        return max(0, int(requested))
    return max(1, int(episodes) // 20)


def _resolve(supplied: dict[str, Any], target: TrainTarget, name: str) -> Any:
    value = supplied.get(name)
    return target.default(name) if value is None else value


def _carry(arguments: dict[str, Any], supplied: dict[str, Any], names: Any) -> None:
    for name in names:
        value = supplied.get(name)
        if value is not None:
            arguments[name] = value


def _supplied_options(scope: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the command's parameters, dropping locals that are not options."""

    return {key: value for key, value in scope.items() if not key.startswith("_") and key not in {"env", "algo"}}


def _reject_inapplicable(targets: Sequence[TrainTarget], supplied: dict[str, Any]) -> None:
    """Fail fast when a *setting* cannot apply to every selected target.

    Silently ignoring ``--opponent`` on Minesweeper would make a run look like
    it honoured a setting it never saw, so a setting must be meaningful for
    every algorithm in the sweep. ``--value-head`` is exempt: it is an *axis*,
    not a setting, and it collapses for algorithms with only one possible head
    (see :func:`_plan_variants`).
    """

    offenders: list[str] = []
    for target in targets:
        scopes: list[tuple[tuple[str, ...], bool, str]] = [
            (_TICTACTOE_ONLY, target.env == "tictactoe", "--env tictactoe"),
            (_MINESWEEPER_ONLY, target.env == "minesweeper", "--env minesweeper"),
            (_TABULAR_ONLY, target.algo == "tabular", "--algo tabular"),
            (_NEURAL_ONLY, target.is_neural, "--algo online-q or --algo dqn"),
            (_DQN_ONLY, target.is_dqn, "--algo dqn"),
        ]
        for names, applies, scope in scopes:
            if applies:
                continue
            for name in names:
                if name == "value_head":
                    continue  # an axis, not a setting
                if supplied.get(name) is not None:
                    offenders.append(f"--{name.replace('_', '-')} (only for {scope})")

    # A non-default value head is only meaningful if some selected algo has one.
    if not any(target.is_dqn for target in targets) and supplied.get("value_head") not in {None, SCALAR}:
        offenders.append("--value-head (only for --algo dqn)")

    if offenders:
        selection = " ".join(f"--algo {target.algo}" for target in targets)
        raise typer.BadParameter(
            f"options not applicable to --env {targets[0].env} {selection}: " + "; ".join(sorted(set(offenders)))
        )
