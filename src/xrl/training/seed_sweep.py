"""Game-agnostic multi-seed training sweeps.

``run_seed_sweep`` trains one run per seed through a vertical-supplied
``run_seed`` closure, then aggregates the per-seed CSV artifacts back from
disk — the engine never touches a game's metric dataclasses, so any vertical
(minesweeper, tictactoe, future games) plugs in with one closure and one
``SeedSweepSpec``.

Layout under ``spec.root``::

    sweep_config.json          # base params + the seed list (NOT config.json,
                               # so run discovery skips the sweep root)
    seed_<k>/...               # one full standard run per seed
    aggregate_metrics.csv      # per-episode cross-seed stats (mean/std/min/max)
    aggregate_eval_metrics.csv # per-checkpoint cross-seed stats
    aggregate_summary.json     # per-seed summaries + cross-seed stats (incl. IQM)
    plots/*.png                # per-seed lines + mean ± std band (optional)

MLflow (opt-in): one parent run for the sweep with ``agg/...`` curves and the
aggregate artifacts; one nested child run per seed, opened and closed by the
vertical's own trainer.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from xrl.training.metrics import iqm
from xrl.training.metrics_io import (
    DEFAULT_ENV_ID,
    RunDirectoryConflictError,
    read_dict_rows,
    write_dict_rows,
)
from xrl.training.run_metadata import RunIdentity, execute_training_run
from xrl.training.tracking import MLflowTracker, NoopTracker, TrackerConfig, TrainingTracker

RunSeedFn = Callable[[int, Path, TrainingTracker], dict[str, Any]]
"""``run_seed : ℕ_seed × Path_out × Tracker → Summary`` — one full training run."""


def parse_seeds(seeds: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list (e.g. ``"0,1,2"``) into a tuple."""

    parsed = tuple(int(part.strip()) for part in seeds.split(",") if part.strip())
    if not parsed:
        raise ValueError(f"no seeds parsed from {seeds!r}")
    return parsed


def trainer_run_seed(
    base_config: Any,
    make_trainer: Callable[[Any, TrainingTracker], Any],
    *,
    identity: RunIdentity,
) -> RunSeedFn:
    """Adapt the shared ``Trainer(config, tracker) → train() → write_run()`` shape.

    ``base_config`` must be a frozen dataclass with ``seed`` and ``out_dir``
    fields — true for every vertical's config — which are replaced per seed.
    """

    def run_seed(seed: int, out_dir: Path, tracker: TrainingTracker) -> dict[str, Any]:
        config = replace(base_config, seed=seed, out_dir=out_dir)
        trainer = make_trainer(config, tracker)
        return execute_training_run(config, trainer, identity)

    return run_seed


@dataclass(frozen=True, slots=True)
class SeedSweepSpec:
    """Static description of one multi-seed sweep.

    ``metric_columns``/``eval_columns``/``policy_columns`` name the CSV columns
    to aggregate and plot; columns missing from a vertical's files are skipped
    with a warning. ``eval_group_by`` splits eval aggregation per group value
    (e.g. ``"opponent"`` for tictactoe's per-opponent eval rows).
    """

    seeds: tuple[int, ...]
    root: Path
    run_name: str
    tags: dict[str, str]
    base_params: dict[str, Any]
    metric_columns: tuple[str, ...]
    eval_columns: tuple[str, ...]
    eval_group_by: str | None = None
    policy_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct, got {self.seeds}")


def run_seed_sweep(
    spec: SeedSweepSpec,
    run_seed: RunSeedFn,
    *,
    tracker_config: TrackerConfig | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Run one training run per seed and write cross-seed aggregate artifacts.

    Seeds run sequentially; cross-seed statistics use the population standard
    deviation (``ddof=0``) and the interquartile mean from
    ``xrl.training.metrics.iqm``.
    """

    tracker_config = tracker_config or TrackerConfig()
    root = Path(spec.root)
    root.mkdir(parents=True, exist_ok=True)
    _guard_sweep_root(root, str(spec.base_params.get("env_id", DEFAULT_ENV_ID)))
    (root / "sweep_config.json").write_text(
        json.dumps({**spec.base_params, "seeds": list(spec.seeds)}, indent=2),
        encoding="utf-8",
    )

    parent = _tracker(tracker_config, nested=False, tags=spec.tags)
    parent.start_run(run_name=spec.run_name, tags=spec.tags)
    parent.log_params({**spec.base_params, "seeds": list(spec.seeds), "num_seeds": len(spec.seeds)})
    try:
        summaries: dict[int, dict[str, Any]] = {}
        for seed in spec.seeds:
            child = _tracker(
                tracker_config,
                nested=True,
                tags={"workflow": "seed_sweep", "seed": str(seed), **spec.tags},
            )
            summaries[seed] = run_seed(seed, root / f"seed_{seed}", child)

        metric_rows = _write_aggregate_csv(
            spec,
            filename="metrics.csv",
            out_name="aggregate_metrics.csv",
            columns=spec.metric_columns,
            group_keys=("episode",),
        )
        eval_group_keys = ("episode", spec.eval_group_by) if spec.eval_group_by else ("episode",)
        eval_rows = _write_aggregate_csv(
            spec,
            filename="eval_metrics.csv",
            out_name="aggregate_eval_metrics.csv",
            columns=spec.eval_columns,
            group_keys=eval_group_keys,
        )
        summary = _write_aggregate_summary(spec, summaries, eval_rows)
        _log_parent_curves(parent, spec, metric_rows, eval_rows)

        plots: list[Path] = []
        if plot:
            plots = plot_seed_sweep(
                root,
                metric_columns=spec.metric_columns,
                eval_columns=spec.eval_columns,
                eval_group_by=spec.eval_group_by,
                policy_columns=spec.policy_columns,
            )
        for artifact in (
            root / "sweep_config.json",
            root / "aggregate_metrics.csv",
            root / "aggregate_eval_metrics.csv",
            root / "aggregate_summary.json",
        ):
            parent.log_artifacts(artifact)
        if plots:
            parent.log_artifacts(root / "plots")
    except Exception:
        parent.end_run(status="FAILED")
        raise
    parent.end_run()

    return {
        "out_dir": str(root),
        "seeds": list(spec.seeds),
        "per_seed": {str(seed): summaries[seed] for seed in spec.seeds},
        "aggregate": summary,
        "plots": [str(path) for path in plots],
    }


def plot_seed_sweep(
    root: Path,
    *,
    metric_columns: Sequence[str],
    eval_columns: Sequence[str] = (),
    eval_group_by: str | None = None,
    policy_columns: Sequence[str] = (),
) -> list[Path]:
    """Plot per-seed curves with a mean ± std band from on-disk sweep artifacts.

    Re-runnable against any sweep directory. Returns ``[]`` when matplotlib
    (the optional ``monitoring`` extra) is unavailable.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    root = Path(root)
    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    # Each run writes only the figures its own column set produces, so figures
    # from a previous run with different columns would otherwise survive and be
    # read as this one's.
    for stale in plots_dir.glob("*.png"):
        stale.unlink()
    written: list[Path | None] = []

    metric_rows = _read_seed_rows(root, "metrics.csv")
    for column in metric_columns:
        written.append(_plot_column(metric_rows, column, plots_dir / f"{column}.png", title=column))

    eval_rows = _read_seed_rows(root, "eval_metrics.csv")
    for group_value, group_rows in _split_rows(eval_rows, eval_group_by).items():
        stem = f"eval_{_safe_name(group_value)}" if group_value else "eval"
        for column in eval_columns:
            written.append(
                _plot_column(
                    group_rows,
                    column,
                    plots_dir / f"{stem}_{column}.png",
                    title=f"{stem} {column}".replace("_", " "),
                )
            )

    policy_rows = _read_seed_rows(root, "policy_metrics.csv")
    for column in policy_columns:
        written.append(_plot_column(policy_rows, column, plots_dir / f"policy_{column}.png", title=f"policy {column}"))

    plt.close("all")
    return [path for path in written if path is not None]


def _guard_sweep_root(root: Path, env_id: str) -> None:
    """Refuse a sweep root that already holds a different environment's sweep.

    The per-seed writers guard their own directories, but a sweep root also owns
    ``sweep_config.json`` and the aggregate CSVs, and those are rewritten before
    any seed runs. Catching the conflict here fails before training starts
    rather than after the first seed has already overwritten artifacts.
    """

    existing = root / "sweep_config.json"
    if not existing.exists():
        return
    try:
        previous = json.loads(existing.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    previous_env = str(previous.get("env_id", DEFAULT_ENV_ID))
    if previous_env == env_id:
        return
    raise RunDirectoryConflictError(
        f"{root} already holds a {previous_env} sweep; refusing to write a {env_id} sweep over it. "
        f"Choose a different --out, or delete {root} first."
    )


def _tracker(config: TrackerConfig, *, nested: bool, tags: dict[str, str]) -> TrainingTracker:
    if not config.mlflow:
        return NoopTracker()
    return MLflowTracker(
        experiment_name=config.mlflow_experiment,
        tracking_uri=config.mlflow_tracking_uri,
        nested=nested,
        tags=tags,
    )


def _read_seed_rows(root: Path, filename: str) -> dict[int, list[dict[str, str]]]:
    """Read one artifact file from every ``seed_<k>`` directory under ``root``."""

    rows_by_seed: dict[int, list[dict[str, str]]] = {}
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        rows = read_dict_rows(seed_dir / filename)
        if rows:
            rows_by_seed[seed] = rows
    return rows_by_seed


def aggregate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Cross-run stats per group key (population std, ddof=0).

    Takes a flat row stream: which run each row came from is already encoded in
    its group keys, so a caller can group by episode, by method, or by both.
    """

    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(group_key, "")) for group_key in group_keys)
        groups.setdefault(key, []).append(row)

    missing: set[str] = set()
    aggregated: list[dict[str, Any]] = []
    for key in sorted(groups, key=_group_sort_key):
        record: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        for column in columns:
            values = [float(row[column]) for row in groups[key] if row.get(column) not in {"", None}]
            if not values:
                missing.add(column)
                continue
            data = np.asarray(values, dtype=np.float64)
            record[f"{column}_mean"] = float(data.mean())
            record[f"{column}_std"] = float(data.std())
            record[f"{column}_min"] = float(data.min())
            record[f"{column}_max"] = float(data.max())
        aggregated.append(record)
    for column in sorted(missing):
        warnings.warn(f"aggregate: column {column!r} missing from artifacts; skipped", stacklevel=2)
    return aggregated


def _group_sort_key(key: tuple[str, ...]) -> tuple[Any, ...]:
    """Sort each group component numerically where it parses, lexically otherwise.

    Episodes must order 2 < 10, but a group key may also be a method or opponent
    name, so the choice is made per component rather than by position.
    """

    return tuple(_sortable(component) for component in key)


def _sortable(value: str) -> tuple[int, float, str]:
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def _write_aggregate_csv(
    spec: SeedSweepSpec,
    *,
    filename: str,
    out_name: str,
    columns: Sequence[str],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows_by_seed = _read_seed_rows(Path(spec.root), filename)
    flattened = [row for rows in rows_by_seed.values() for row in rows]
    aggregated = aggregate_rows(flattened, columns=columns, group_keys=group_keys)
    write_dict_rows(Path(spec.root) / out_name, aggregated)
    return aggregated


def cross_seed_stats(values: Sequence[float]) -> dict[str, float]:
    """Mean, population std, min, max, and interquartile mean across runs."""

    data = np.asarray([float(value) for value in values], dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "iqm": iqm([float(value) for value in values]),
    }


def _write_aggregate_summary(
    spec: SeedSweepSpec,
    summaries: Mapping[int, Mapping[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric_per_seed = {
        seed: {
            key: value
            for key, value in summary.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        for seed, summary in summaries.items()
    }
    shared_keys = sorted(set.intersection(*(set(values) for values in numeric_per_seed.values())))
    summary_stats = {key: cross_seed_stats([numeric_per_seed[seed][key] for seed in spec.seeds]) for key in shared_keys}
    # The final checkpoint's cross-seed aggregate row(s) — every run has one
    # thanks to the forced final eval.
    final_eval = [row for row in eval_rows if row.get("episode") == eval_rows[-1].get("episode")] if eval_rows else []
    summary = {
        "seeds": list(spec.seeds),
        "per_seed_summary": {str(seed): numeric_per_seed[seed] for seed in spec.seeds},
        "summary_stats": summary_stats,
        "final_eval": final_eval,
    }
    (Path(spec.root) / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _log_parent_curves(
    parent: TrainingTracker,
    spec: SeedSweepSpec,
    metric_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> None:
    for row in metric_rows:
        payload = {
            f"agg/{column}_{stat}": row[f"{column}_{stat}"]
            for column in spec.metric_columns
            for stat in ("mean", "std")
            if f"{column}_{stat}" in row
        }
        if payload:
            parent.log_metrics(payload, step=int(float(row["episode"])))
    for row in eval_rows:
        group = _safe_name(str(row.get(spec.eval_group_by, ""))) + "_" if spec.eval_group_by else ""
        payload = {
            f"agg/eval_{group}{column}_{stat}": row[f"{column}_{stat}"]
            for column in spec.eval_columns
            for stat in ("mean", "std")
            if f"{column}_{stat}" in row
        }
        if payload:
            parent.log_metrics(payload, step=int(float(row["episode"])))


def _split_rows(
    rows_by_seed: Mapping[int, list[dict[str, str]]],
    group_by: str | None,
) -> dict[str, dict[int, list[dict[str, str]]]]:
    """Split each seed's rows per group value (single empty group when unset)."""

    if group_by is None:
        return {"": dict(rows_by_seed)}
    split: dict[str, dict[int, list[dict[str, str]]]] = {}
    for seed, rows in rows_by_seed.items():
        for row in rows:
            group = str(row.get(group_by, ""))
            split.setdefault(group, {}).setdefault(seed, []).append(row)
    return split


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _plot_column(
    rows_by_seed: Mapping[int, list[dict[str, str]]],
    column: str,
    path: Path,
    *,
    title: str,
) -> Path | None:
    import matplotlib.pyplot as plt

    series: dict[int, tuple[list[int], list[float]]] = {}
    for seed, rows in rows_by_seed.items():
        xs: list[int] = []
        ys: list[float] = []
        for row in rows:
            try:
                episode = int(float(row["episode"]))
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            xs.append(episode)
            ys.append(value)
        if xs:
            series[seed] = (xs, ys)
    if not series:
        return None

    by_episode: dict[int, list[float]] = {}
    for xs, ys in series.values():
        for x, y in zip(xs, ys, strict=True):
            by_episode.setdefault(x, []).append(y)
    episodes = sorted(by_episode)
    means = np.asarray([np.mean(by_episode[episode]) for episode in episodes])
    stds = np.asarray([np.std(by_episode[episode]) for episode in episodes])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for seed, (xs, ys) in sorted(series.items()):
        ax.plot(xs, ys, alpha=0.3, linewidth=1.0, label=f"seed {seed}")
    ax.plot(episodes, means, color="black", linewidth=1.8, label="mean")
    ax.fill_between(episodes, means - stds, means + stds, color="black", alpha=0.15, label="±1 std")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
