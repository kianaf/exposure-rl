"""Optional experiment tracking adapters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Protocol

DEFAULT_MLFLOW_EXPERIMENT = "xrl"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


@dataclass(frozen=True)
class TrackerConfig:
    """Typed tracking options shared by Typer command entrypoints."""

    mlflow: bool = False
    mlflow_experiment: str = DEFAULT_MLFLOW_EXPERIMENT
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI


class TrainingTracker(Protocol):
    """Minimal training tracker interface used by tabular and neural trainers."""

    def start_run(self, *, run_name: str | None = None, tags: dict[str, str] | None = None) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None: ...

    def log_artifacts(self, path: Path) -> None: ...

    def end_run(self, *, status: str = "FINISHED") -> None: ...


class NoopTracker:
    """Tracker implementation used when monitoring is disabled."""

    def start_run(self, *, run_name: str | None = None, tags: dict[str, str] | None = None) -> None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        return None

    def log_artifacts(self, path: Path) -> None:
        return None

    def end_run(self, *, status: str = "FINISHED") -> None:
        return None


class MLflowTracker:
    """MLflow-backed tracker with lazy imports so MLflow stays optional."""

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.nested = nested
        self.tags = tags or {}
        self._active = False

    def _mlflow(self):
        try:
            import mlflow
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra.
            raise RuntimeError(
                "MLflow tracking requires the optional 'monitoring' extra: install xrl[monitoring]"
            ) from exc
        return mlflow

    def start_run(self, *, run_name: str | None = None, tags: dict[str, str] | None = None) -> None:
        mlflow = self._mlflow()
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name is not None:
            mlflow.set_experiment(self.experiment_name)
        merged_tags = {**self.tags, **(tags or {})}
        mlflow.start_run(run_name=run_name, nested=self.nested, tags=merged_tags or None)
        self._active = True

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        flat = flatten_params(params)
        self._mlflow().log_params({key: param_value(value) for key, value in flat.items()})

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self._active:
            return
        converted = {key: value for key, raw in metrics.items() if (value := metric_value(raw)) is not None}
        if converted:
            self._mlflow().log_metrics(converted, step=step)

    def log_artifacts(self, path: Path) -> None:
        if not self._active:
            return
        mlflow = self._mlflow()
        path = Path(path)
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        elif path.exists():
            mlflow.log_artifact(str(path))

    def end_run(self, *, status: str = "FINISHED") -> None:
        if not self._active:
            return
        self._mlflow().end_run(status=status)
        self._active = False


def metric_value(value: Any) -> float | int | None:
    """Convert Python/NumPy/JAX scalar-like values to MLflow-safe metrics."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return value
    try:
        import numpy as np

        if isinstance(value, np.generic):
            item = value.item()
            return item if isinstance(item, Number) else None
        array = np.asarray(value)
        if array.shape == ():
            item = array.item()
            return item if isinstance(item, Number) else None
    except (ImportError, TypeError, ValueError):
        pass
    try:
        import jax

        host_value = jax.device_get(value)
        if hasattr(host_value, "item"):
            item = host_value.item()
            return item if isinstance(item, Number) else None
    except (ImportError, TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def param_value(value: Any) -> Any:
    """Convert config values to MLflow-safe parameter values."""

    if value is None or isinstance(value, (Path, list, tuple, dict)):
        return str(value)
    return value


def flatten_params(params: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested params into MLflow-friendly key/value pairs."""

    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=name))
        else:
            flat[name] = value
    return flat


def add_mlflow_tracking_args(parser: argparse.ArgumentParser) -> None:
    """Add common opt-in MLflow tracking arguments to a training CLI."""

    parser.add_argument("--mlflow", action="store_true", help="Log this training run to MLflow.")
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_MLFLOW_EXPERIMENT,
        help=f"MLflow experiment name. Defaults to {DEFAULT_MLFLOW_EXPERIMENT!r}.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help=f"MLflow tracking URI. Defaults to {DEFAULT_MLFLOW_TRACKING_URI!r}.",
    )


def tracker_from_cli_args(
    args: Any,
    *,
    tags: dict[str, str] | None = None,
) -> TrainingTracker:
    """Return an MLflow tracker when --mlflow is enabled, otherwise a no-op tracker."""

    if not getattr(args, "mlflow", False):
        return NoopTracker()
    return MLflowTracker(
        experiment_name=getattr(args, "mlflow_experiment", DEFAULT_MLFLOW_EXPERIMENT),
        tracking_uri=getattr(args, "mlflow_tracking_uri", DEFAULT_MLFLOW_TRACKING_URI),
        tags=tags,
    )
