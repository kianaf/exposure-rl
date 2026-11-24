"""Compare training runs that already finished, without retraining anything.

``xrl compare`` is a trainer: it runs methods from scratch and aggregates the
CSVs it just produced. This module is the other half — it reads run directories
off disk and writes the same kind of cross-seed comparison, so a sweep that
already cost hours of compute can be turned into a table without repeating it.

Nothing here loads a model or imports an environment. A run is
``config.json`` + ``metrics.csv`` + ``eval_metrics.csv``, and that is all this
needs, which is why it works with no optional extras installed and for every
vertical at once.

Two things are *discovered* rather than configured, following the same pattern
as :mod:`xrl.training.progress`:

* **which directories are runs** — the four on-disk shapes are told apart by
  the marker file each one carries (see :data:`DIRECTORY_MARKERS`), so pointing
  at a variant sweep root expands to one method per variant automatically;
* **which columns to compare** — :data:`COMPARISON_METRIC_COLUMNS` and
  :data:`COMPARISON_EVAL_COLUMNS` catalogue every column worth comparing across
  verticals, and a column is used when every selected method actually recorded
  it. Tic Tac Toe contributes a draw rate, Minesweeper a completion ratio, and
  neither has to declare that.

Plotting deliberately lives in :mod:`xrl.training.compare_plots` and reads only
the artifacts written here, so this path never needs matplotlib and its stdout
stays a single JSON object.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from xrl.training.metrics import PROBE_STAT_FIELDS
from xrl.training.metrics_io import DEFAULT_ENV_ID, read_dict_rows, write_dict_rows
from xrl.training.progress import announce
from xrl.training.seed_sweep import aggregate_rows, cross_seed_stats

__all__ = [
    "COMPARISON_EVAL_COLUMNS",
    "COMPARISON_METRIC_COLUMNS",
    "DIRECTORY_MARKERS",
    "MethodSources",
    "RunSource",
    "aggregate_finished_runs",
    "compare_runs",
    "discover_columns",
    "resolve_sources",
]


# Eval rows carry this column only where evaluation is against an opponent.
EVAL_GROUP_COLUMN = "opponent"

# Per-episode columns worth comparing, in display order. Spans verticals: a run
# contributes the ones it recorded. Adding a metric is one entry here.
COMPARISON_METRIC_COLUMNS: tuple[str, ...] = (
    "rolling_win_rate",
    "rolling_draw_rate",
    "rolling_completion_ratio",
    "rolling_learner_return",
    "rolling_episode_length",
    "loss_mean",
    "td_error_mean",
    "steps_per_second",
)

COMPARISON_EVAL_COLUMNS: tuple[str, ...] = (
    "win_rate",
    "draw_rate",
    "loss_rate",
    "mean_completion_ratio",
    "mean_return",
    "mean_learner_return",
    "mean_episode_length",
    "training_flops",
    *PROBE_STAT_FIELDS,
)

# Marker file -> the directory shape it identifies, in resolution order.
DIRECTORY_MARKERS: tuple[tuple[str, str], ...] = (
    ("variants_summary.json", "xrl train variant sweep"),
    ("sweep_config.json", "xrl train --seeds sweep"),
    ("comparison_config.json", "xrl compare output, or runs/<method>_seed_<k>/ beneath it"),
    ("config.json", "single run"),
)


@dataclass(frozen=True, slots=True)
class RunSource:
    """One finished run on disk, belonging to a named method."""

    label: str
    seed: int
    path: Path
    env_id: str
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MethodSources:
    """One comparison series: a label and the runs that make it up."""

    label: str
    runs: tuple[RunSource, ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(run.seed for run in self.runs)


def resolve_sources(specs: Sequence[str]) -> tuple[MethodSources, ...]:
    """Expand ``path`` / ``label=path`` arguments into comparison series.

    A directory is classified by the marker file it carries, so a variant sweep
    root expands to one method per variant and a seed sweep to one method with
    several seeds. Pointing at ``runs/minesweeper-4x4-3-div`` is therefore
    enough to compare three value heads over two seeds each.
    """

    if not specs:
        raise typer.BadParameter("name at least one run directory to compare")

    methods: list[MethodSources] = []
    for spec in specs:
        label, path = _split_spec(spec)
        methods.extend(_resolve_directory(path, label))

    duplicates = sorted({method.label for method in methods if _count(methods, method.label) > 1})
    if duplicates:
        raise typer.BadParameter(
            f"duplicate method labels: {', '.join(duplicates)}; disambiguate with label=path arguments"
        )
    return tuple(methods)


def require_single_environment(methods: Sequence[MethodSources]) -> str:
    """Return the shared env id, or reject a mix of environments.

    A Minesweeper completion ratio and a Tic Tac Toe draw rate are not the same
    axis, and a table that puts them in one column would be read as if they
    were. Same reasoning as ``--env`` not sweeping in ``xrl train``.
    """

    by_env: dict[str, list[str]] = {}
    for method in methods:
        for run in method.runs:
            by_env.setdefault(run.env_id, []).append(method.label)
    if len(by_env) > 1:
        detail = "; ".join(f"{env}: {', '.join(sorted(set(labels)))}" for env, labels in sorted(by_env.items()))
        raise typer.BadParameter(f"cannot compare runs from different environments ({detail})")
    return next(iter(by_env))


def discover_columns(rows_by_method: Mapping[str, list[dict[str, str]]], catalogue: Sequence[str]) -> tuple[str, ...]:
    """Return the catalogued columns every method actually recorded.

    Requiring *every* method to have the column is what keeps a comparison
    honest: a curve present for two of three methods reads as the third having
    scored zero.
    """

    if not rows_by_method:
        return ()
    return tuple(column for column in catalogue if all(_has_numeric(rows, column) for rows in rows_by_method.values()))


def aggregate_finished_runs(
    methods: Sequence[MethodSources],
    *,
    out_dir: Path,
    metric_columns: Sequence[str] | None = None,
    eval_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate finished runs into comparison artifacts under ``out_dir``.

    Writes ``aggregate_metrics.csv``, ``aggregate_eval_metrics.csv``,
    ``final_comparison.csv``, and ``sources.json``; returns the summary that the
    command prints to stdout.
    """

    env_id = require_single_environment(methods)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _tagged_rows(methods, "metrics.csv")
    eval_rows = _tagged_rows(methods, "eval_metrics.csv")
    metrics_used = _columns_for(metric_rows, COMPARISON_METRIC_COLUMNS, metric_columns, "--columns")
    evals_used = _columns_for(eval_rows, COMPARISON_EVAL_COLUMNS, eval_columns, "--eval-columns")

    group_column = _group_column(eval_rows)
    eval_group_keys = ("method", "episode", group_column) if group_column else ("method", "episode")

    aggregate_metrics = aggregate_rows(
        _flatten(metric_rows),
        columns=metrics_used,
        group_keys=("method", "episode"),
    )
    aggregate_evals = aggregate_rows(_flatten(eval_rows), columns=evals_used, group_keys=eval_group_keys)
    final = final_comparison(eval_rows, columns=evals_used, group_column=group_column)

    write_dict_rows(out_dir / "aggregate_metrics.csv", aggregate_metrics)
    write_dict_rows(out_dir / "aggregate_eval_metrics.csv", aggregate_evals)
    write_dict_rows(out_dir / "final_comparison.csv", final)
    sources = _sources_document(methods, env_id)
    (out_dir / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "env_id": env_id,
        "methods": sources["methods"],
        "columns": {"metrics": list(metrics_used), "eval": list(evals_used)},
        "final": final,
    }


def final_comparison(
    rows_by_method: Mapping[str, list[dict[str, str]]],
    *,
    columns: Sequence[str],
    group_column: str | None,
) -> list[dict[str, Any]]:
    """Build the headline table: each method at its own final eval checkpoint.

    Methods trained for different episode budgets keep their own final episode
    rather than being truncated to a shared one, so the recorded ``episode``
    tells you what is actually being compared.
    """

    table: list[dict[str, Any]] = []
    for method, rows in rows_by_method.items():
        episodes = [value for value in (_number(row.get("episode")) for row in rows) if value is not None]
        if not episodes:
            continue
        final_episode = max(episodes)
        at_final = [row for row in rows if _number(row.get("episode")) == final_episode]
        for group, group_rows in _split_by(at_final, group_column).items():
            record: dict[str, Any] = {"method": method}
            if group_column is not None:
                record[group_column] = group
            record["episode"] = int(final_episode)
            record["seeds"] = len(group_rows)
            for column in columns:
                values = [float(row[column]) for row in group_rows if _number(row.get(column)) is not None]
                if not values:
                    continue
                for statistic, value in cross_seed_stats(values).items():
                    record[f"{column}_{statistic}"] = value
            table.append(record)
    return table


def compare_runs(
    sources: Annotated[
        list[str],
        typer.Argument(
            help="Run directories to compare. A sweep root expands to one series per variant. "
            "Use label=path to name a series.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory. Defaults to runs/compare-runs/<timestamp>."),
    ] = None,
    columns: Annotated[
        str | None,
        typer.Option(help="Comma-separated per-episode columns. Default: whichever catalogued ones every run has."),
    ] = None,
    eval_columns: Annotated[
        str | None,
        typer.Option(help="Comma-separated evaluation columns. Default: whichever catalogued ones every run has."),
    ] = None,
) -> None:
    """Compare finished training runs from disk, without retraining them.

    Reads each run's metrics and evaluation CSVs and writes cross-seed
    aggregates plus a final-checkpoint table. Plot the result with
    `xrl compare-plot`.
    """

    methods = resolve_sources(sources)
    destination = Path(out) if out is not None else Path("runs/compare-runs") / time.strftime("%Y%m%d-%H%M%S")
    announce(
        f"{len(methods)} methods, {sum(len(method.runs) for method in methods)} runs: "
        f"{', '.join(method.label for method in methods)}"
    )
    summary = aggregate_finished_runs(
        methods,
        out_dir=destination,
        metric_columns=_parse_columns(columns),
        eval_columns=_parse_columns(eval_columns),
    )
    announce(f"wrote {destination}")
    print(json.dumps(summary, indent=2))


def _split_spec(spec: str) -> tuple[str | None, Path]:
    """Split ``label=path`` into its parts; a bare path has no label."""

    if "=" in spec:
        label, _, path = spec.partition("=")
        if not label.strip() or not path.strip():
            raise typer.BadParameter(f"expected label=path, got {spec!r}")
        return label.strip(), Path(path.strip())
    return None, Path(spec)


def _resolve_directory(path: Path, label: str | None) -> list[MethodSources]:
    if not path.is_dir():
        raise typer.BadParameter(f"not a directory: {path}")
    if (path / "variants_summary.json").exists():
        return _resolve_children(path, label, marker_owner="variant sweep")
    if (path / "sweep_config.json").exists():
        name = label or path.name
        return [MethodSources(name, _seed_runs(path, name))]
    if (path / "comparison_config.json").exists() or _looks_like_comparison_output(path):
        return _resolve_comparison(path, label)
    if (path / "config.json").exists():
        name = label or path.name
        return [MethodSources(name, (_run_source(name, path),))]

    known = ", ".join(f"{marker} ({shape})" for marker, shape in DIRECTORY_MARKERS)
    raise typer.BadParameter(f"{path} is not a run or sweep directory; expected one of: {known}")


def _resolve_children(root: Path, label: str | None, *, marker_owner: str) -> list[MethodSources]:
    """Expand a variant sweep into one method per child directory."""

    methods: list[MethodSources] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        if not any((child / marker).exists() for marker, _ in DIRECTORY_MARKERS):
            continue
        child_label = f"{label}/{child.name}" if label else child.name
        methods.extend(_resolve_directory(child, child_label))
    if not methods:
        raise typer.BadParameter(f"{root} is a {marker_owner} with no run directories beneath it")
    return methods


def _looks_like_comparison_output(path: Path) -> bool:
    """Recognise an ``xrl compare`` output by its ``runs/<method>_seed_<k>/`` layout.

    Detected structurally rather than by ``comparison_config.json`` alone,
    because that file postdates some of the comparison directories on disk.
    """

    runs = path / "runs"
    if not runs.is_dir():
        return False
    return any("_seed_" in child.name and _has_artifacts(child) for child in runs.iterdir() if child.is_dir())


def _resolve_comparison(root: Path, label: str | None) -> list[MethodSources]:
    """Expand an ``xrl compare`` output, whose runs are named ``<method>_seed_<k>``.

    These runs identify themselves by directory name, so comparisons written
    before they saved a ``config.json`` are still readable: the method and the
    seed are both in the name, and ``xrl compare`` is Tic Tac Toe only.
    """

    grouped: dict[str, list[RunSource]] = {}
    for run_dir in sorted((root / "runs").iterdir()) if (root / "runs").is_dir() else []:
        if not run_dir.is_dir() or not _has_artifacts(run_dir):
            continue
        method, separator, seed_text = run_dir.name.rpartition("_seed_")
        name = method if separator else run_dir.name
        full = f"{label}/{name}" if label else name
        grouped.setdefault(full, []).append(_run_source(full, run_dir, fallback_seed=int(_number(seed_text) or 0)))
    if not grouped:
        raise typer.BadParameter(f"{root} is an xrl compare output with no completed runs under runs/")
    return [MethodSources(name, tuple(runs)) for name, runs in grouped.items()]


def _has_artifacts(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "metrics.csv").exists()


def _seed_runs(root: Path, label: str) -> tuple[RunSource, ...]:
    runs: list[RunSource] = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not (seed_dir / "config.json").exists():
            continue
        runs.append(_run_source(label, seed_dir))
    if not runs:
        raise typer.BadParameter(f"{root} is a seed sweep with no seed_<k>/ run directories")
    return tuple(runs)


def _run_source(label: str, path: Path, *, fallback_seed: int = 0) -> RunSource:
    if not (path / "config.json").exists():
        return RunSource(label=label, seed=fallback_seed, path=path, env_id=DEFAULT_ENV_ID, config={})
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise typer.BadParameter(f"unreadable run config in {path}: {error}") from error
    if not isinstance(config, dict):
        raise typer.BadParameter(f"run config must be a JSON object: {path / 'config.json'}")
    return RunSource(
        label=label,
        seed=int(config.get("seed", 0)),
        path=path,
        env_id=str(config.get("env_id", DEFAULT_ENV_ID)),
        config=config,
    )


def _tagged_rows(methods: Sequence[MethodSources], filename: str) -> dict[str, list[dict[str, str]]]:
    """Read one artifact per run, tagged with its method and the run's seed.

    ``run_seed`` rather than ``seed``: Minesweeper's per-episode rows already
    use ``seed`` for the episode's environment seed.
    """

    rows_by_method: dict[str, list[dict[str, str]]] = {}
    for method in methods:
        collected: list[dict[str, str]] = []
        for run in method.runs:
            for row in read_dict_rows(run.path / filename):
                collected.append({**row, "method": method.label, "run_seed": str(run.seed)})
        rows_by_method[method.label] = collected
    return rows_by_method


def _columns_for(
    rows_by_method: Mapping[str, list[dict[str, str]]],
    catalogue: Sequence[str],
    requested: Sequence[str] | None,
    option: str,
) -> tuple[str, ...]:
    if requested is None:
        return discover_columns(rows_by_method, catalogue)
    missing = [
        column for column in requested if not any(_has_numeric(rows, column) for rows in rows_by_method.values())
    ]
    if missing:
        raise typer.BadParameter(f"{option}: no run records {', '.join(missing)}")
    return tuple(requested)


def _group_column(rows_by_method: Mapping[str, list[dict[str, str]]]) -> str | None:
    """Split evaluation rows by opponent where every method names one.

    Tic Tac Toe evaluates against several opponents per checkpoint and must be
    reported per opponent; Minesweeper has no opponent column at all.
    """

    populated = [rows for rows in rows_by_method.values() if rows]
    if not populated:
        return None
    if all(any(row.get(EVAL_GROUP_COLUMN) for row in rows) for rows in populated):
        return EVAL_GROUP_COLUMN
    return None


def _split_by(rows: Sequence[Mapping[str, str]], column: str | None) -> dict[str, list[Mapping[str, str]]]:
    if column is None:
        return {"": list(rows)}
    split: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        split.setdefault(str(row.get(column, "")), []).append(row)
    return dict(sorted(split.items()))


def _sources_document(methods: Sequence[MethodSources], env_id: str) -> dict[str, Any]:
    distinguishing = _distinguishing_config(methods)
    return {
        "env_id": env_id,
        "methods": [
            {
                "label": method.label,
                "seeds": list(method.seeds),
                "runs": [str(run.path) for run in method.runs],
                "config": distinguishing[method.label],
            }
            for method in methods
        ],
    }


def _distinguishing_config(methods: Sequence[MethodSources]) -> dict[str, dict[str, Any]]:
    """Record only the config fields that actually differ between methods.

    A full config dump per method buries the one field that explains the
    difference; ``out_dir`` and ``seed`` always differ and never explain it.
    """

    configs = {method.label: method.runs[0].config for method in methods}
    keys = sorted({key for config in configs.values() for key in config} - {"out_dir", "seed"})
    varying = [
        key
        for key in keys
        if len({json.dumps(config.get(key), sort_keys=True, default=str) for config in configs.values()}) > 1
    ]
    return {label: {key: config.get(key) for key in varying} for label, config in configs.items()}


def _flatten(rows_by_method: Mapping[str, list[dict[str, str]]]) -> Iterable[Mapping[str, Any]]:
    for rows in rows_by_method.values():
        yield from rows


def _has_numeric(rows: Iterable[Mapping[str, Any]], column: str) -> bool:
    return any(_number(row.get(column)) is not None for row in rows)


def _number(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_columns(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(name.strip() for name in value.split(",") if name.strip())
    if not parsed:
        raise typer.BadParameter("column list must name at least one column")
    return parsed


def _count(methods: Sequence[MethodSources], label: str) -> int:
    return sum(1 for method in methods if method.label == label)
