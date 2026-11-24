"""Provider-neutral workflow graph and artifact publication contracts.

The graph is deliberately data-only.  Provider adapters can serialize a stage,
run it in another process or machine, and return the same :class:`StageResult`.
No Modal, SSH, CUDA, or training modules are imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
StageResource = Literal["cpu", "gpu"]
StageAction = Callable[["StageContext"], Mapping[str, JsonValue] | None]

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_PUBLICATION_FILE = ".xrl-stage.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One independently executable node in a research workflow."""

    key: str
    kind: str
    handler: str
    output: str
    dependencies: tuple[str, ...] = ()
    payload: Mapping[str, JsonValue] = field(default_factory=dict)
    resource: StageResource = "cpu"
    required_outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _KEY_RE.fullmatch(self.key):
            raise ValueError(f"invalid stage key: {self.key!r}")
        if not self.kind or not self.handler:
            raise ValueError("stage kind and handler must be non-empty")
        object.__setattr__(self, "output", _validate_relative_path(self.output, label="stage output"))
        if self.key in self.dependencies or len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"stage {self.key!r} has invalid dependencies")
        for dependency in self.dependencies:
            if not _KEY_RE.fullmatch(dependency):
                raise ValueError(f"invalid dependency key: {dependency!r}")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(
            self,
            "required_outputs",
            tuple(_validate_relative_path(path, label="required output") for path in self.required_outputs),
        )
        # Fail at construction rather than halfway through a remote workflow.
        _canonical_json(self.payload)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "key": self.key,
                "kind": self.kind,
                "handler": self.handler,
                "output": self.output,
                "dependencies": self.dependencies,
                "payload": self.payload,
                "resource": self.resource,
                "required_outputs": self.required_outputs,
            }
        )


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """A validated DAG whose identity is independent of its execution host."""

    name: str
    run_id: str
    source_revision: str
    lock_fingerprint: str
    configuration: Mapping[str, JsonValue]
    stages: tuple[StageSpec, ...]
    device_class: str = "cpu"

    def __post_init__(self) -> None:
        if not _KEY_RE.fullmatch(self.name):
            raise ValueError(f"invalid workflow name: {self.name!r}")
        if not _KEY_RE.fullmatch(self.run_id):
            raise ValueError(f"invalid run id: {self.run_id!r}")
        if not self.source_revision or not self.lock_fingerprint:
            raise ValueError("source revision and lock fingerprint must be non-empty")
        object.__setattr__(self, "stages", tuple(self.stages))
        _canonical_json(self.configuration)
        self._validate_graph()

    def _validate_graph(self) -> None:
        by_key = {stage.key: stage for stage in self.stages}
        if not self.stages or len(by_key) != len(self.stages):
            raise ValueError("workflow stage keys must be non-empty and unique")
        outputs = [stage.output for stage in self.stages]
        if len(outputs) != len(set(outputs)):
            raise ValueError("workflow stage outputs must be unique")
        for stage in self.stages:
            missing = set(stage.dependencies) - by_key.keys()
            if missing:
                raise ValueError(f"stage {stage.key!r} has missing dependencies: {sorted(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError(f"workflow contains a dependency cycle at {key!r}")
            if key in visited:
                return
            visiting.add(key)
            for dependency in by_key[key].dependencies:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in by_key:
            visit(key)

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "name": self.name,
                "source_revision": self.source_revision,
                "lock_fingerprint": self.lock_fingerprint,
                "configuration": self.configuration,
                "device_class": self.device_class,
                "stages": [stage.fingerprint for stage in self.stages],
            }
        )

    def stage(self, key: str) -> StageSpec:
        return next(stage for stage in self.stages if stage.key == key)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "source_revision": self.source_revision,
            "lock_fingerprint": self.lock_fingerprint,
            "configuration": self.configuration,
            "stages": [asdict(stage) for stage in self.stages],
            "device_class": self.device_class,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> WorkflowSpec:
        stages_raw = raw.get("stages")
        if not isinstance(stages_raw, list):
            raise ValueError("workflow stages must be a list")
        stages: list[StageSpec] = []
        for item in stages_raw:
            if not isinstance(item, dict):
                raise ValueError("workflow stage must be an object")
            values = dict(item)
            values["dependencies"] = tuple(values.get("dependencies", ()))
            values["required_outputs"] = tuple(values.get("required_outputs", ()))
            stages.append(StageSpec(**values))
        configuration = raw.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("workflow configuration must be an object")
        return cls(
            name=str(raw["name"]),
            run_id=str(raw["run_id"]),
            source_revision=str(raw["source_revision"]),
            lock_fingerprint=str(raw["lock_fingerprint"]),
            configuration=configuration,
            stages=tuple(stages),
            device_class=str(raw.get("device_class", "cpu")),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Checksummed publication record written last for a successful stage."""

    schema_version: int
    workflow_fingerprint: str
    stage_fingerprint: str
    stage_key: str
    files: Mapping[str, str]
    published_at_unix: float


@dataclass(frozen=True, slots=True)
class StageContext:
    workflow: WorkflowSpec
    stage: StageSpec
    workflow_root: Path
    output_dir: Path
    attempt_dir: Path
    dependency_outputs: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class StageResult:
    key: str
    output_dir: str
    reused: bool
    seconds: float
    manifest: ArtifactManifest
    detail: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    workflow_fingerprint: str
    stages: tuple[StageResult, ...]
    seconds: float


class WorkflowFailed(RuntimeError):
    """A stage failed; no dependent stage was admitted afterwards."""

    def __init__(self, stage_key: str, cause: BaseException, completed: Sequence[StageResult]):
        super().__init__(f"stage {stage_key!r} failed: {cause}")
        self.stage_key = stage_key
        self.cause = cause
        self.completed = tuple(completed)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_artifact_tree(root: Path) -> dict[str, str]:
    """Hash regular files in deterministic order, excluding publication metadata."""

    if not root.is_dir():
        raise FileNotFoundError(f"stage did not produce its output directory: {root}")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact bundles may not contain symlinks: {path}")
        if path.is_file() and path.name != _PUBLICATION_FILE:
            files[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not files:
        raise ValueError(f"stage produced an empty artifact directory: {root}")
    return files


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_publication(output_dir: Path) -> ArtifactManifest | None:
    path = output_dir / _PUBLICATION_FILE
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ArtifactManifest(**raw)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def validate_publication(
    output_dir: Path,
    *,
    workflow_fingerprint: str,
    stage_fingerprint: str,
) -> ArtifactManifest | None:
    manifest = read_publication(output_dir)
    if (
        manifest is None
        or manifest.schema_version != 1
        or manifest.workflow_fingerprint != workflow_fingerprint
        or manifest.stage_fingerprint != stage_fingerprint
    ):
        return None
    # A stage owns the files it published, not caches that a downstream reader
    # may add below the same legacy run directory.  Validate every owned file
    # while deliberately ignoring later, unowned additions.
    try:
        for relative, expected in manifest.files.items():
            path = output_dir / PurePosixPath(relative)
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
                return None
    except OSError:
        return None
    return manifest


def publish_stage(workflow: WorkflowSpec, stage: StageSpec, output_dir: Path) -> ArtifactManifest:
    files = hash_artifact_tree(output_dir)
    manifest = ArtifactManifest(
        schema_version=1,
        workflow_fingerprint=workflow.fingerprint,
        stage_fingerprint=stage.fingerprint,
        stage_key=stage.key,
        files=files,
        published_at_unix=time.time(),
    )
    _write_json_atomic(output_dir / _PUBLICATION_FILE, asdict(manifest))
    return manifest


class WorkflowRunner:
    """Execute ready stages concurrently and publish successful outputs."""

    def __init__(self, *, max_workers: int = 1):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def run(
        self,
        workflow: WorkflowSpec,
        workflow_root: Path,
        action: StageAction,
        *,
        reuse: bool = False,
    ) -> WorkflowResult:
        started = time.perf_counter()
        root = Path(workflow_root)
        root.mkdir(parents=True, exist_ok=True)
        pending = {stage.key: stage for stage in workflow.stages}
        completed: dict[str, StageResult] = {}
        running: dict[Future[StageResult], str] = {}

        executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="xrl-stage")
        failed: tuple[str, BaseException] | None = None
        try:
            while pending or running:
                ready = [
                    stage
                    for stage in pending.values()
                    if all(dependency in completed for dependency in stage.dependencies)
                ]
                for stage in ready[: max(0, self.max_workers - len(running))]:
                    pending.pop(stage.key)
                    future = executor.submit(self._run_stage, workflow, root, stage, action, reuse)
                    running[future] = stage.key

                if not running:
                    raise RuntimeError("workflow scheduler made no progress")

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    key = running.pop(future)
                    try:
                        completed[key] = future.result()
                    except BaseException as error:  # preserve the provider error as the cause
                        failed = (key, error)
                        for other in running:
                            other.cancel()
                        pending.clear()
                        break
                if failed is not None:
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        ordered = tuple(completed[stage.key] for stage in workflow.stages if stage.key in completed)
        if failed is not None:
            raise WorkflowFailed(failed[0], failed[1], ordered) from failed[1]
        return WorkflowResult(workflow.fingerprint, ordered, time.perf_counter() - started)

    def run_stage(
        self,
        workflow: WorkflowSpec,
        workflow_root: Path,
        stage_key: str,
        action: StageAction,
        *,
        reuse: bool = False,
    ) -> StageResult:
        """Run one admitted stage after validating its published dependencies."""

        root = Path(workflow_root)
        stage = workflow.stage(stage_key)
        for dependency_key in stage.dependencies:
            dependency = workflow.stage(dependency_key)
            if (
                validate_publication(
                    root / dependency.output,
                    workflow_fingerprint=workflow.fingerprint,
                    stage_fingerprint=dependency.fingerprint,
                )
                is None
            ):
                raise RuntimeError(f"stage {stage.key!r} has an unpublished dependency {dependency.key!r}")
        return self._run_stage(workflow, root, stage, action, reuse)

    @staticmethod
    def _run_stage(
        workflow: WorkflowSpec,
        root: Path,
        stage: StageSpec,
        action: StageAction,
        reuse: bool,
    ) -> StageResult:
        output_dir = root / stage.output
        valid = validate_publication(
            output_dir,
            workflow_fingerprint=workflow.fingerprint,
            stage_fingerprint=stage.fingerprint,
        )
        if reuse and valid is not None:
            return StageResult(stage.key, str(output_dir), True, 0.0, valid)

        # Existing legacy artifacts can be adopted only when the stage declares
        # its complete minimum set.  A mismatched publication is never adopted.
        if (
            reuse
            and read_publication(output_dir) is None
            and stage.required_outputs
            and all((output_dir / path).is_file() for path in stage.required_outputs)
        ):
            manifest = publish_stage(workflow, stage, output_dir)
            return StageResult(stage.key, str(output_dir), True, 0.0, manifest, {"adopted": True})

        if output_dir.exists():
            raise FileExistsError(f"stage output exists but is not reusable for this fingerprint: {output_dir}")

        attempt_dir = root / ".xrl-attempts" / stage.key / uuid.uuid4().hex
        attempt_dir.mkdir(parents=True)
        status_path = attempt_dir / "status.json"
        _write_json_atomic(status_path, {"state": "running", "stage": stage.key, "started_at_unix": time.time()})
        context = StageContext(
            workflow=workflow,
            stage=stage,
            workflow_root=root,
            output_dir=output_dir,
            attempt_dir=attempt_dir,
            dependency_outputs={key: root / workflow.stage(key).output for key in stage.dependencies},
        )
        started = time.perf_counter()
        try:
            detail = dict(action(context) or {})
            missing = [path for path in stage.required_outputs if not (output_dir / path).is_file()]
            if missing:
                raise FileNotFoundError(f"stage {stage.key!r} omitted required outputs: {missing}")
            manifest = publish_stage(workflow, stage, output_dir)
        except BaseException as error:
            if output_dir.exists():
                os.rename(output_dir, attempt_dir / "failed-output")
            _write_json_atomic(
                status_path,
                {
                    "state": "failed",
                    "stage": stage.key,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at_unix": time.time(),
                },
            )
            raise
        seconds = time.perf_counter() - started
        _write_json_atomic(
            status_path,
            {"state": "succeeded", "stage": stage.key, "seconds": seconds, "finished_at_unix": time.time()},
        )
        return StageResult(stage.key, str(output_dir), False, seconds, manifest, detail)


__all__ = [
    "ArtifactManifest",
    "JsonValue",
    "StageAction",
    "StageContext",
    "StageResult",
    "StageSpec",
    "WorkflowFailed",
    "WorkflowResult",
    "WorkflowRunner",
    "WorkflowSpec",
    "hash_artifact_tree",
    "publish_stage",
    "read_publication",
    "validate_publication",
]
