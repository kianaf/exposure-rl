"""Shared metrics serialization and summary helpers.

This module is the single home for the CSV writers and metric summaries used by
every trainer. Keeping them here avoids the copy-paste drift that previously had
``loop.py`` and ``neural_q.py`` defining subtly different versions of the same
functions.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Protocol

from xrl.training.episode import EpisodeMetrics
from xrl.training.flops import FLOPS_ROW_FIELDS, learner_training_accounting
from xrl.training.metrics import PROBE_STAT_FIELDS

# Configs written before the field existed are Tic Tac Toe runs.
DEFAULT_ENV_ID = "tictactoe_v0"

# Every filename a trainer may write into its run directory. A run owns this
# whole set: the ones it does not produce are deleted rather than inherited,
# because an orphan reads as if it belonged to the run that outlived it.
# Minesweeper writes no ``policy_metrics.csv``, so a Tic Tac Toe file left in a
# reused directory would otherwise be aggregated as Minesweeper data.
MANAGED_RUN_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "eval_metrics.csv",
    "final_model.msgpack",
    "metrics.csv",
    "model.msgpack",
    "policy_metrics.csv",
    "q_table.json",
    "run_metadata.json",
    "summary.json",
)

EVAL_METRICS_FIELDS: tuple[str, ...] = (
    "episode",
    "opponent",
    "episodes",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "loss_rate",
    "draw_rate",
    "mean_learner_return",
    "mean_return",
    *FLOPS_ROW_FIELDS,
)

POLICY_METRICS_FIELDS: tuple[str, ...] = (
    "episode",
    "states",
    "minimax_matches",
    "minimax_action_match_rate",
    "mean_minimax_regret",
    *PROBE_STAT_FIELDS,
)

# Columns aggregated and plotted by multi-seed sweeps of the AEC verticals.
AEC_SWEEP_METRIC_COLUMNS: tuple[str, ...] = (
    "rolling_win_rate",
    "rolling_draw_rate",
    "rolling_learner_return",
    "rolling_episode_length",
    "td_error_mean",
    "steps_per_second",
)
AEC_SWEEP_EVAL_COLUMNS: tuple[str, ...] = ("win_rate", "loss_rate", "draw_rate", "mean_learner_return")
# Only a learner with a compiled training step can be costed, so tabular and
# online-Q sweeps would warn about a column they can never record.
AEC_DQN_SWEEP_EVAL_COLUMNS: tuple[str, ...] = (*AEC_SWEEP_EVAL_COLUMNS, "training_flops")
AEC_SWEEP_POLICY_COLUMNS: tuple[str, ...] = ("minimax_action_match_rate", *PROBE_STAT_FIELDS)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write arbitrary dict rows, deriving the header from the keys in order.

    Used for derived artifacts (sweep aggregates, comparison tables) whose
    columns depend on which metrics the runs happened to record, rather than on
    a fixed schema. An empty row list writes an empty file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dict rows, treating a missing or empty file as no rows."""

    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_metrics_csv(path: Path, metrics: list[EpisodeMetrics]) -> None:
    """Write per-episode training metrics as a CSV with one row per episode."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(EpisodeMetrics)])
        writer.writeheader()
        for row in metrics:
            writer.writerow(asdict(row))


def write_eval_metrics_csv(path: Path, metrics: list[dict[str, object]]) -> None:
    """Write evaluation rows with a stable column set shared across all learners.

    Uses a fixed header so that tabular, Q-regression, and DQN runs all produce
    comparable ``eval_metrics.csv`` files. Extra keys (e.g. ``bellwether``) are
    dropped; a header is always written even when there are no rows.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVAL_METRICS_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def write_policy_metrics_csv(path: Path, metrics: list[dict[str, object]]) -> None:
    """Write policy-diagnostic rows (minimax match rates) with a stable header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(POLICY_METRICS_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def summarize_learner_metrics(metrics: list[EpisodeMetrics]) -> dict[str, object]:
    """Aggregate win/loss/draw counts and mean return from the learner's view."""
    total = len(metrics)
    wins = sum(1 for metric in metrics if metric.learner_return > 0)
    losses = sum(1 for metric in metrics if metric.learner_return < 0)
    draws = total - wins - losses
    mean_return = sum(metric.learner_return for metric in metrics) / total if total else 0.0
    return {
        "matches": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total if total else 0.0,
        "loss_rate": losses / total if total else 0.0,
        "draw_rate": draws / total if total else 0.0,
        "mean_learner_return": mean_return,
    }


def summarize_metrics(metrics: list[EpisodeMetrics]) -> dict[str, object]:
    """Return the run summary used in ``summary.json`` and evaluation rows."""
    total = len(metrics)
    learner_summary = summarize_learner_metrics(metrics)
    return {
        "episodes": total,
        "matches": total,
        "wins": learner_summary["wins"],
        "losses": learner_summary["losses"],
        "draws": learner_summary["draws"],
        "win_rate": learner_summary["win_rate"],
        "loss_rate": learner_summary["loss_rate"],
        "draw_rate": learner_summary["draw_rate"],
        "mean_learner_return": learner_summary["mean_learner_return"],
    }


class RunDirectoryConflictError(ValueError):
    """A run would overwrite artifacts belonging to a different environment."""


def guard_run_directory(out_dir: Path, env_id: str) -> None:
    """Refuse to write a run into a directory that holds another environment.

    Writing file-by-file leaves behind every artifact the incoming run does not
    itself produce, and those orphans then read as its own. Reusing a directory
    for the same environment still overwrites freely; only a change of
    environment is treated as a mistake worth stopping for.
    """

    existing = out_dir / "config.json"
    if not existing.exists():
        return
    try:
        previous = json.loads(existing.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # An unreadable config carries no identity to conflict with, and
        # refusing here would strand a directory a rerun could simply replace.
        return
    previous_env = str(previous.get("env_id", DEFAULT_ENV_ID))
    if previous_env == env_id:
        return
    raise RunDirectoryConflictError(
        f"{out_dir} already holds a {previous_env} run; refusing to write a {env_id} run over it. "
        f"Choose a different --out, or delete {out_dir} first."
    )


def prune_stale_artifacts(out_dir: Path, written: Iterable[str]) -> list[Path]:
    """Delete managed artifacts this run did not write; return what was removed."""

    keep = set(written)
    removed: list[Path] = []
    for name in MANAGED_RUN_ARTIFACTS:
        if name in keep:
            continue
        path = out_dir / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


class _ArtifactConfig(Protocol):
    """Minimal config surface needed to write run artifacts."""

    @property
    def out_dir(self) -> Path: ...

    def to_json_dict(self) -> dict[str, Any]: ...


class StandardArtifactWriter:
    """Config-driven artifact writer shared by all trainers.

    Writes ``config.json``, the model checkpoint, the three metric CSVs, and
    ``summary.json`` to ``config.out_dir``. The only learner-specific behavior is
    how the model is persisted, supplied via the ``save_model`` callback (e.g.
    ``lambda learner, out_dir: learner.save(out_dir / "model.msgpack")``).
    ``model_filenames`` names what that callback writes, so the stale-artifact
    sweep can tell a checkpoint this run produced from one it inherited.
    """

    def __init__(
        self,
        config: _ArtifactConfig,
        *,
        save_model: Callable[[Any, Path], None],
        model_filenames: Iterable[str] = ("model.msgpack",),
    ):
        self.config = config
        self._save_model = save_model
        self._model_filenames = tuple(model_filenames)

    @property
    def out_dir(self) -> Path:
        return self.config.out_dir

    def write(
        self,
        learner: Any,
        metrics: list[EpisodeMetrics],
        evaluation_metrics: list[dict[str, Any]],
        policy_metrics: list[dict[str, Any]],
    ) -> dict[str, Any]:
        out_dir = self.config.out_dir
        config_json = self.config.to_json_dict()
        out_dir.mkdir(parents=True, exist_ok=True)
        guard_run_directory(out_dir, str(config_json.get("env_id", DEFAULT_ENV_ID)))
        (out_dir / "config.json").write_text(json.dumps(config_json, indent=2), encoding="utf-8")
        self._save_model(learner, out_dir)
        write_metrics_csv(out_dir / "metrics.csv", metrics)
        write_eval_metrics_csv(out_dir / "eval_metrics.csv", evaluation_metrics)
        write_policy_metrics_csv(out_dir / "policy_metrics.csv", policy_metrics)
        summary = summarize_metrics(metrics)
        summary.update(learner_training_accounting(learner))
        if evaluation_metrics:
            summary["evaluations"] = evaluation_metrics
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        prune_stale_artifacts(
            out_dir,
            (
                "config.json",
                "eval_metrics.csv",
                "metrics.csv",
                "policy_metrics.csv",
                "run_metadata.json",
                "summary.json",
                *self._model_filenames,
            ),
        )
        return summary
