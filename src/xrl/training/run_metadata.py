"""Run initialization, provenance, and terminal configuration output."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO

from xrl.training.metrics_io import DEFAULT_ENV_ID, guard_run_directory, prune_stale_artifacts

RUN_METADATA_SCHEMA_VERSION = 1


class RunConfig(Protocol):
    """Configuration surface required by the run lifecycle."""

    @property
    def out_dir(self) -> Path: ...

    def to_json_dict(self) -> dict[str, Any]: ...


class Trainable(Protocol):
    """Trainer surface shared by the command facades."""

    def train(self) -> tuple[Any, list[Any]]: ...

    def write_run(self, agent: Any, metrics: list[Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Human and machine-readable identity not carried by trainer configs."""

    environment: str
    algorithm: str
    value_head: str | None = None
    eval_opponents: tuple[str, ...] = ()


class RunLifecycle(AbstractContextManager["RunLifecycle"]):
    """Publish a run's resolved treatment before training and track its state."""

    def __init__(
        self,
        config: RunConfig,
        identity: RunIdentity,
        *,
        argv: list[str] | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.config = config
        self.identity = identity
        self.argv = list(sys.argv if argv is None else argv)
        self.stream = stream or sys.stderr
        self.out_dir = Path(config.out_dir)
        self.metadata_path = self.out_dir / "run_metadata.json"
        self._started_perf = time.perf_counter()
        self._metadata: dict[str, Any] = {}

    def __enter__(self) -> RunLifecycle:
        config_json = self.config.to_json_dict()
        env_id = str(config_json.get("env_id", DEFAULT_ENV_ID))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        guard_run_directory(self.out_dir, env_id)
        # The new config must never be observable beside an old checkpoint.
        # The guard reads the old config first; only then is the prior managed
        # run replaced. Unmanaged notes and analysis files remain untouched.
        prune_stale_artifacts(self.out_dir, ())
        _write_json(self.out_dir / "config.json", config_json)
        self._metadata = _initial_metadata(self.identity, self.argv)
        _write_json(self.metadata_path, self._metadata)
        _print_resolved_config(config_json, self.identity, self._metadata, stream=self.stream)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        del traceback
        ended = _utc_now()
        self._metadata["completed_at"] = ended
        self._metadata["duration_seconds"] = max(0.0, time.perf_counter() - self._started_perf)
        if exc_type is None:
            self._metadata["status"] = "completed"
        else:
            self._metadata["status"] = "failed"
            self._metadata["failure"] = {"type": exc_type.__name__}
        _write_json(self.metadata_path, self._metadata)
        return False


def execute_training_run(config: RunConfig, trainer: Trainable, identity: RunIdentity) -> dict[str, Any]:
    """Train, write artifacts, and maintain lifecycle metadata around both."""

    with RunLifecycle(config, identity):
        agent, metrics = trainer.train()
        return trainer.write_run(agent, metrics)


def _initial_metadata(identity: RunIdentity, argv: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "identity": {
            "environment": identity.environment,
            "algorithm": identity.algorithm,
            "value_head": identity.value_head,
            "evaluation_opponents": list(identity.eval_opponents),
        },
        "command": _sanitized_argv(argv),
        "code": _code_provenance(),
        "runtime": _runtime_provenance(neural=identity.algorithm in {"online-q", "dqn"}),
    }
    return metadata


def _code_provenance() -> dict[str, Any]:
    try:
        xrl_version = importlib.metadata.version("xrl")
    except importlib.metadata.PackageNotFoundError:
        xrl_version = "unknown"
    commit = _git_output("rev-parse", "HEAD")
    dirty = _git_output("status", "--porcelain")
    return {
        "xrl_version": xrl_version,
        "git_commit": commit,
        "git_dirty": None if dirty is None else bool(dirty),
    }


def _runtime_provenance(*, neural: bool) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if not neural:
        return runtime
    try:
        import jax
    except (ImportError, RuntimeError):
        return runtime
    runtime.update(
        {
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jaxlib_version": _package_version("jaxlib"),
            "jax_backend": jax.default_backend(),
            "jax_devices": sorted({getattr(device, "device_kind", str(device)) for device in jax.devices()}),
        }
    )
    return runtime


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sanitized_argv(argv: list[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if argument == "--mlflow-tracking-uri":
            sanitized.append(argument)
            redact_next = True
            continue
        if argument.startswith("--mlflow-tracking-uri="):
            sanitized.append("--mlflow-tracking-uri=<redacted>")
            continue
        sanitized.append(argument)
    return sanitized


def _print_resolved_config(
    config: dict[str, Any],
    identity: RunIdentity,
    metadata: dict[str, Any],
    *,
    stream: TextIO,
) -> None:
    identity_fields = {
        "environment": identity.environment,
        "algorithm": identity.algorithm,
        "value_head": identity.value_head,
        "seed": config.get("seed"),
        "episodes": config.get("episodes"),
        "out": config.get("out_dir"),
    }
    model_fields = {
        "encoder": config.get("encoder"),
        "hidden_sizes": config.get("hidden_sizes"),
        "board": _board_geometry(config),
        "bins": config.get("num_bins") if identity.value_head not in {None, "scalar"} else None,
        "support": _support(config) if identity.value_head not in {None, "scalar"} else None,
        "feature_mask_input": config.get("feature_mask_input"),
        "masked_batch_ratio": config.get("masked_batch_ratio") if config.get("feature_mask_input") else None,
        "mask_empty_example_interval": (
            config.get("mask_empty_example_interval") if config.get("masked_batch_ratio", 0) > 0 else None
        ),
        "mask_shapley_probability": (
            config.get("mask_shapley_probability") if config.get("masked_batch_ratio", 0) > 0 else None
        ),
        "mask_uniform_probability": (
            config.get("mask_uniform_probability") if config.get("masked_batch_ratio", 0) > 0 else None
        ),
        "hl_gauss_sigma_bins": config.get("hl_gauss_sigma_bins") if identity.value_head == "hl-gauss" else None,
    }
    optimization_fields = {
        "alpha": config.get("alpha"),
        "learning_rate": config.get("learning_rate"),
        "gamma": config.get("gamma"),
        "batch": config.get("batch_size"),
        "replay": config.get("replay_capacity"),
        "warmup": config.get("replay_warmup"),
        "target_sync": config.get("target_sync_interval"),
    }
    behavior_fields = {
        "exploration": config.get("exploration", "epsilon_greedy"),
        "epsilon": _epsilon_schedule(config),
        "backup": config.get("backup"),
        "temperature": config.get("temperature"),
        "opponent": config.get("opponent"),
        "opponent_discount": config.get("perturbed_minimax_discount")
        if config.get("opponent") == "perturbed_minimax"
        else None,
        "learner_role": "seeded-random" if identity.environment == "tictactoe" else None,
    }
    measurement_fields = {
        "eval_interval": config.get("eval_interval"),
        "eval_episodes": config.get("eval_episodes"),
        "eval_opponents": identity.eval_opponents or None,
        "eval_policy": "greedy",
        "metrics_window": config.get("metrics_window"),
        "q_probe_states": config.get("q_probe_states"),
    }
    code = metadata["code"]
    runtime = metadata["runtime"]
    provenance_fields = {
        "git": _short_commit(code.get("git_commit")),
        "dirty": code.get("git_dirty"),
        "backend": runtime.get("jax_backend"),
        "devices": runtime.get("jax_devices"),
    }
    for label, fields in (
        ("identity", identity_fields),
        ("model", model_fields),
        ("optimization", optimization_fields),
        ("behavior", behavior_fields),
        ("measurement", measurement_fields),
        ("provenance", provenance_fields),
    ):
        rendered = "  ".join(f"{key}={_render(value)}" for key, value in fields.items() if value is not None)
        if rendered:
            print(f"=== config/{label}: {rendered}", file=stream, flush=True)


def _board_geometry(config: dict[str, Any]) -> str | None:
    if "rows" not in config or "columns" not in config:
        return None
    return f"{config['rows']}x{config['columns']}/{config.get('mines', 0)}mines"


def _support(config: dict[str, Any]) -> str | None:
    if "v_min" not in config or "v_max" not in config:
        return None
    return f"[{config['v_min']},{config['v_max']}]"


def _epsilon_schedule(config: dict[str, Any]) -> str | None:
    required = ("epsilon_start", "epsilon_end", "epsilon_decay_episodes")
    if not all(key in config for key in required):
        return None
    return f"{config['epsilon_start']}->{config['epsilon_end']}@{config['epsilon_decay_episodes']}ep"


def _short_commit(commit: Any) -> str | None:
    return str(commit)[:12] if commit else None


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
