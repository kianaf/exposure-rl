"""Content-addressed orchestration for reproducible multi-condition studies.

The core knows only conditions, seeds, immutable profile settings, and artifact
trees. Games provide the cell work as a callback, which keeps environment
roles, opponents, board encodings, evaluation, and explanation logic outside
this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STUDY_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "study_manifest.json"
CELL_RECORD_FILENAME = ".study-cell.json"
RESULT_FILENAME = "study_result.json"

_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


class StudyError(RuntimeError):
    """Base class for study orchestration failures."""


class StudyManifestMismatchError(StudyError):
    """The output root already belongs to a different resolved study."""


class StudyCellValidationError(StudyError):
    """A published cell is incomplete, mismatched, or has corrupt artifacts."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"study values must be finite JSON data: {exc}") from exc


def _canonical_mapping(value: Mapping[str, Any] | None) -> str:
    payload = {} if value is None else dict(value)
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("study mapping keys must be strings")
    return _canonical_json(payload)


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_key(value: str, *, field_name: str) -> None:
    if not _KEY_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must match {_KEY_PATTERN.pattern!r}, got {value!r}")


@dataclass(frozen=True, slots=True, init=False)
class StudyCondition:
    """One named treatment with immutable canonical JSON parameters."""

    key: str
    _parameters_json: str = field(repr=False)

    def __init__(self, key: str, parameters: Mapping[str, Any] | None = None) -> None:
        _validate_key(key, field_name="condition key")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "_parameters_json", _canonical_mapping(parameters))

    @property
    def parameters(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the treatment parameters."""

        return json.loads(self._parameters_json)

    def to_json_dict(self) -> dict[str, Any]:
        return {"key": self.key, "parameters": self.parameters}


@dataclass(frozen=True, slots=True, init=False)
class StudyProfile:
    """Named immutable budget/settings bundle shared by every study cell."""

    name: str
    _settings_json: str = field(repr=False)

    def __init__(self, name: str, settings: Mapping[str, Any] | None = None) -> None:
        _validate_key(name, field_name="profile name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_settings_json", _canonical_mapping(settings))

    @property
    def settings(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the profile settings."""

        return json.loads(self._settings_json)

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": self.name, "settings": self.settings}


@dataclass(frozen=True, slots=True)
class StudySpec:
    """Resolved finite product of conditions and seeds for one study root."""

    name: str
    root: Path
    profile: StudyProfile
    conditions: tuple[StudyCondition, ...]
    seeds: tuple[int, ...]
    schema_version: int = STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_key(self.name, field_name="study name")
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.schema_version != STUDY_SCHEMA_VERSION:
            raise ValueError(f"unsupported study schema_version {self.schema_version}; expected {STUDY_SCHEMA_VERSION}")
        if not self.conditions:
            raise ValueError("conditions must be non-empty")
        condition_keys = [condition.key for condition in self.conditions]
        if len(set(condition_keys)) != len(condition_keys):
            raise ValueError(f"condition keys must be distinct, got {condition_keys}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in self.seeds):
            raise ValueError("seeds must be integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"seeds must be distinct, got {self.seeds}")

    def manifest_payload(self) -> dict[str, Any]:
        """Canonical study identity; the relocatable output root is excluded."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "profile": self.profile.to_json_dict(),
            "conditions": [condition.to_json_dict() for condition in self.conditions],
            "seeds": list(self.seeds),
        }

    @property
    def fingerprint(self) -> str:
        return _digest_json(self.manifest_payload())

    def manifest_document(self) -> dict[str, Any]:
        return {**self.manifest_payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class StudyCellContext:
    """Inputs and private staging directory supplied to one cell callback."""

    condition: StudyCondition
    seed: int
    profile: StudyProfile
    out_dir: Path
    publish_dir: Path
    manifest_fingerprint: str

    @property
    def cell_id(self) -> str:
        return f"{self.condition.key}/seed_{self.seed}"


@dataclass(frozen=True, slots=True, init=False)
class StudyCellResult:
    """Validated result of one condition/seed cell."""

    condition_key: str
    seed: int
    path: Path
    artifact_hashes: tuple[tuple[str, str], ...]
    _output_json: str = field(repr=False)
    reused: bool

    def __init__(
        self,
        *,
        condition_key: str,
        seed: int,
        path: Path,
        artifact_hashes: tuple[tuple[str, str], ...],
        output: Mapping[str, Any],
        reused: bool,
    ) -> None:
        object.__setattr__(self, "condition_key", condition_key)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "artifact_hashes", tuple(artifact_hashes))
        object.__setattr__(self, "_output_json", _canonical_mapping(output))
        object.__setattr__(self, "reused", bool(reused))

    @property
    def cell_id(self) -> str:
        return f"{self.condition_key}/seed_{self.seed}"

    @property
    def output(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the cell output."""

        return json.loads(self._output_json)

    def index_row(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "condition_key": self.condition_key,
            "seed": self.seed,
            "path": str(Path("cells") / self.condition_key / f"seed_{self.seed}"),
            "artifact_hashes": dict(self.artifact_hashes),
            "output": self.output,
        }


@dataclass(frozen=True, slots=True)
class StudyResult:
    """One completed traversal of a study's validated cells."""

    root: Path
    manifest_fingerprint: str
    cells: tuple[StudyCellResult, ...]

    @property
    def reused_cells(self) -> int:
        return sum(cell.reused for cell in self.cells)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_fingerprint": self.manifest_fingerprint,
            "cells": [cell.index_row() for cell in self.cells],
            "reused_cells": self.reused_cells,
        }


StudyCellWork = Callable[[StudyCellContext], Mapping[str, Any] | None]


class StudyRunner:
    """Run and validate atomic cells for a resolved :class:`StudySpec`."""

    def __init__(self, spec: StudySpec):
        self.spec = spec

    def run(self, work: StudyCellWork, *, reuse: bool = True, max_workers: int = 1) -> StudyResult:
        """Traverse condition-major/seed-minor cells in deterministic order.

        A callback writes only inside its staging ``context.out_dir``. The
        directory is published to ``cells/...`` by one rename after the output
        and every artifact hash have been recorded. Existing cells are never
        overwritten: they are either validated and reused, or rejected.

        Cells are independent, so ``max_workers`` above one admits that many at
        a time. Published order never follows completion order: results are
        re-indexed by declared order before the result tuple is built. Above one
        worker a failure stops further admission but lets in-flight cells finish,
        so a sibling a serial traversal would have skipped may be published; it
        is a validly content-addressed cell that a resume would produce anyway.
        """

        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._prepare_root()
        cells = tuple((condition, seed) for condition in self.spec.conditions for seed in self.spec.seeds)
        if max_workers == 1:
            results = [self._run_cell(condition, seed, work, reuse=reuse) for condition, seed in cells]
        else:
            results = self._run_cells_concurrently(cells, work, reuse=reuse, max_workers=max_workers)
        result = StudyResult(
            root=self.spec.root,
            manifest_fingerprint=self.spec.fingerprint,
            cells=tuple(results),
        )
        index = {
            "schema_version": self.spec.schema_version,
            "manifest_fingerprint": self.spec.fingerprint,
            "cells": [cell.index_row() for cell in results],
        }
        _write_json_atomic(self.spec.root / RESULT_FILENAME, index)
        return result

    def _run_cells_concurrently(
        self,
        cells: tuple[tuple[StudyCondition, int], ...],
        work: StudyCellWork,
        *,
        reuse: bool,
        max_workers: int,
    ) -> list[StudyCellResult]:
        """Admit at most ``max_workers`` independent cells and stop admitting on failure."""

        pending = list(cells)
        completed: dict[tuple[str, int], StudyCellResult] = {}
        running: dict[Future[StudyCellResult], tuple[str, int]] = {}
        failure: BaseException | None = None
        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xrl-study-cell")
        try:
            while pending or running:
                while pending and failure is None and len(running) < max_workers:
                    condition, seed = pending.pop(0)
                    future = executor.submit(self._run_cell, condition, seed, work, reuse=reuse)
                    running[future] = (condition.key, seed)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    key = running.pop(future)
                    try:
                        completed[key] = future.result()
                    except BaseException as error:  # preserve the cell error exactly as raised
                        if failure is None:
                            failure = error
                        for other in running:
                            other.cancel()
                        pending.clear()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            raise failure
        return [completed[(condition.key, seed)] for condition, seed in cells]

    def validate(self) -> StudyResult:
        """Validate every declared cell without running missing work."""

        self._prepare_root()
        cells = tuple(
            self._load_cell(condition, seed, reused=True)
            for condition in self.spec.conditions
            for seed in self.spec.seeds
        )
        return StudyResult(self.spec.root, self.spec.fingerprint, cells)

    def load_published_cell(self, condition_key: str, seed: int) -> StudyCellResult:
        """Validate and return one already-published cell, read-only.

        A study that reuses another study's cells -- loading a frozen agent, say
        -- must hold them to the same bar the owning runner does: the completion
        record's own fingerprint, its identity against this manifest, and the
        artifact hashes. Nothing here writes, so the borrowed root is untouched.
        """

        for condition in self.spec.conditions:
            if condition.key == condition_key:
                break
        else:
            declared = ", ".join(condition.key for condition in self.spec.conditions)
            raise StudyCellValidationError(
                f"{self.spec.root} declares no condition {condition_key!r}; it has {declared}"
            )
        if seed not in self.spec.seeds:
            raise StudyCellValidationError(f"{self.spec.root} declares no seed {seed}; it has {list(self.spec.seeds)}")
        return self._load_cell(condition, seed, reused=True)

    def _prepare_root(self) -> None:
        self.spec.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.spec.root / MANIFEST_FILENAME
        expected = self.spec.manifest_document()
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise StudyManifestMismatchError(f"cannot validate existing manifest {manifest_path}: {exc}") from exc
            if not isinstance(existing, dict):
                raise StudyManifestMismatchError(f"existing manifest {manifest_path} must be a JSON object")
            if existing != expected:
                raise StudyManifestMismatchError(
                    f"{self.spec.root} belongs to a different study manifest; "
                    f"expected {self.spec.fingerprint}, found {existing.get('fingerprint', 'missing')}"
                )
            return
        if any(self.spec.root.iterdir()):
            raise StudyManifestMismatchError(
                f"{self.spec.root} is non-empty but has no {MANIFEST_FILENAME}; choose a fresh study root"
            )
        _write_json_atomic(manifest_path, expected)

    def _run_cell(
        self,
        condition: StudyCondition,
        seed: int,
        work: StudyCellWork,
        *,
        reuse: bool,
    ) -> StudyCellResult:
        final_dir = self._cell_dir(condition, seed)
        if final_dir.exists():
            if not reuse:
                raise StudyCellValidationError(f"refusing to overwrite existing cell {final_dir}")
            return self._load_cell(condition, seed, reused=True)

        staging_parent = self.spec.root / ".staging"
        staging_parent.mkdir(exist_ok=True)
        staging_dir = staging_parent / f"{condition.key}-seed_{seed}-{uuid.uuid4().hex}"
        staging_dir.mkdir()
        context = StudyCellContext(
            condition=condition,
            seed=seed,
            profile=self.spec.profile,
            out_dir=staging_dir,
            publish_dir=final_dir,
            manifest_fingerprint=self.spec.fingerprint,
        )
        try:
            output_value = work(context)
            if output_value is None:
                output: dict[str, Any] = {}
            elif isinstance(output_value, Mapping):
                output = json.loads(_canonical_json(dict(output_value)))
            else:
                raise TypeError("study cell work must return a mapping or None")
            record_path = staging_dir / CELL_RECORD_FILENAME
            if record_path.exists():
                raise StudyCellValidationError(f"cell work may not create reserved file {CELL_RECORD_FILENAME}")
            artifact_hashes = _artifact_hashes(staging_dir)
            record_body = {
                "schema_version": self.spec.schema_version,
                "manifest_fingerprint": self.spec.fingerprint,
                "cell_id": context.cell_id,
                "condition": condition.to_json_dict(),
                "seed": seed,
                "artifact_hashes": artifact_hashes,
                "output": output,
            }
            _write_json(record_path, {**record_body, "record_fingerprint": _digest_json(record_body)})
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                raise StudyCellValidationError(f"cell appeared while staging; refusing to overwrite {final_dir}")
            os.rename(staging_dir, final_dir)
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise
        return self._load_cell(condition, seed, reused=False)

    def _load_cell(self, condition: StudyCondition, seed: int, *, reused: bool) -> StudyCellResult:
        cell_dir = self._cell_dir(condition, seed)
        record_path = cell_dir / CELL_RECORD_FILENAME
        if not record_path.is_file():
            raise StudyCellValidationError(f"cell {cell_dir} has no completion record")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StudyCellValidationError(f"cannot read completion record {record_path}: {exc}") from exc
        if not isinstance(record, dict):
            raise StudyCellValidationError(f"completion record {record_path} must be a JSON object")
        fingerprint = record.pop("record_fingerprint", None)
        if fingerprint != _digest_json(record):
            raise StudyCellValidationError(f"completion record fingerprint mismatch for {cell_dir}")
        expected_identity = {
            "schema_version": self.spec.schema_version,
            "manifest_fingerprint": self.spec.fingerprint,
            "cell_id": f"{condition.key}/seed_{seed}",
            "condition": condition.to_json_dict(),
            "seed": seed,
        }
        mismatches = [key for key, value in expected_identity.items() if record.get(key) != value]
        if mismatches:
            raise StudyCellValidationError(f"cell identity mismatch for {cell_dir}: {', '.join(mismatches)}")
        recorded_hashes = record.get("artifact_hashes")
        if not isinstance(recorded_hashes, dict) or not all(
            isinstance(path, str) and isinstance(digest, str) for path, digest in recorded_hashes.items()
        ):
            raise StudyCellValidationError(f"invalid artifact hash index for {cell_dir}")
        actual_hashes = _artifact_hashes(cell_dir)
        if actual_hashes != recorded_hashes:
            raise StudyCellValidationError(f"artifact hash mismatch for {cell_dir}")
        output = record.get("output")
        if not isinstance(output, dict):
            raise StudyCellValidationError(f"cell output must be an object for {cell_dir}")
        return StudyCellResult(
            condition_key=condition.key,
            seed=seed,
            path=cell_dir,
            artifact_hashes=tuple(sorted(actual_hashes.items())),
            output=output,
            reused=reused,
        )

    def _cell_dir(self, condition: StudyCondition, seed: int) -> Path:
        return cell_directory(self.spec.root, condition.key, seed)


def digest_manifest_payload(payload: Any) -> str:
    """Fingerprint a manifest payload with the digest studies are keyed by.

    An umbrella study that composes several specs has its own identity, and it
    must hash the same way a spec does or the two fingerprints cannot be
    compared.
    """

    return _digest_json(payload)


def cell_directory(root: Path, condition_key: str, seed: int) -> Path:
    """Resolve one cell's published directory under a study root.

    An isolated worker must derive this with the parent's rule rather than a
    duplicated string, because the value reaches the cell as ``publish_dir``.
    """

    return Path(root) / "cells" / condition_key / f"seed_{seed}"


def load_study_spec(root: Path) -> StudySpec:
    """Reconstruct the resolved spec a study root already published.

    The manifest is the only transport an isolated worker needs, and comparing
    the whole rebuilt document — not just its fingerprint — also rejects extra
    or renamed keys. Because ``manifest_payload`` emits conditions and seeds as
    ordered lists, declared order round-trips by construction.
    """

    root = Path(root)
    document = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise StudyManifestMismatchError(f"{root / MANIFEST_FILENAME} is not a study manifest object")
    spec = StudySpec(
        name=str(document["name"]),
        root=root,
        profile=StudyProfile(str(document["profile"]["name"]), document["profile"]["settings"]),
        conditions=tuple(StudyCondition(str(entry["key"]), entry["parameters"]) for entry in document["conditions"]),
        seeds=tuple(int(seed) for seed in document["seeds"]),
        schema_version=int(document["schema_version"]),
    )
    if spec.manifest_document() != document:
        raise StudyManifestMismatchError(f"{root / MANIFEST_FILENAME} does not round-trip to its own study spec")
    return spec


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path == root / CELL_RECORD_FILENAME:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise StudyCellValidationError(f"study artifacts may not be symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StudyCellValidationError(f"study artifact is not a regular file: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(staging, payload)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
