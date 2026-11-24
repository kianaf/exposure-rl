"""Checksum-aware synchronization of managed remote artifact files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

_RECEIPT_FILE = ".xrl-remote-run.json"


@dataclass(frozen=True, slots=True)
class ArtifactSyncReceipt:
    schema_version: int
    run_id: str
    workflow_fingerprint: str
    files: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact bundles may not contain symlinks: {path}")
        if path.is_file() and path.name != _RECEIPT_FILE:
            result[path.relative_to(root).as_posix()] = _sha256_file(path)
    return result


def _read_receipt(root: Path) -> ArtifactSyncReceipt | None:
    path = root / _RECEIPT_FILE
    if not path.is_file():
        return None
    try:
        return ArtifactSyncReceipt(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_receipt(root: Path, receipt: ArtifactSyncReceipt) -> None:
    target = root / _RECEIPT_FILE
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def synchronize_artifacts(
    source: Path,
    destination: Path,
    *,
    run_id: str,
    workflow_fingerprint: str,
) -> ArtifactSyncReceipt:
    """Synchronize managed files and preserve unrelated destination files.

    A prior receipt is the authority for files this function may replace or
    remove.  On the first pull, an existing path collision is refused rather
    than guessing whether a user-owned file is safe to overwrite.
    """

    source = Path(source)
    destination = Path(destination)
    incoming = _source_files(source)
    prior = _read_receipt(destination)
    managed = dict(prior.files) if prior is not None else {}

    conflicts: list[str] = []
    for relative in incoming:
        target = destination / PurePosixPath(relative)
        if not target.exists():
            continue
        if relative not in managed:
            conflicts.append(relative)
            continue
        if not target.is_file() or _sha256_file(target) != managed[relative]:
            conflicts.append(relative)
    if conflicts:
        raise FileExistsError(f"refusing to overwrite unmanaged or modified files: {sorted(conflicts)}")

    destination.mkdir(parents=True, exist_ok=True)
    for relative, digest in incoming.items():
        target = destination / PurePosixPath(relative)
        if target.is_file() and _sha256_file(target) == digest:
            continue
        _atomic_copy(source / PurePosixPath(relative), target)

    for relative in sorted(set(managed) - set(incoming), reverse=True):
        target = destination / PurePosixPath(relative)
        if target.is_file() and _sha256_file(target) == managed[relative]:
            target.unlink()
            parent = target.parent
            while parent != destination:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    receipt = ArtifactSyncReceipt(1, run_id, workflow_fingerprint, incoming)
    _write_receipt(destination, receipt)
    return receipt


def _relocalize_value(value: object, *, marker: str, local_root: str) -> object:
    if isinstance(value, str):
        index = value.find(marker)
        return f"{local_root}{value[index + len(marker) :]}" if index >= 0 else value
    if isinstance(value, list):
        return [_relocalize_value(item, marker=marker, local_root=local_root) for item in value]
    if isinstance(value, dict):
        return {key: _relocalize_value(item, marker=marker, local_root=local_root) for key, item in value.items()}
    return value


def relocalize_artifacts(root: Path, *, run_id: str, local_root: Path) -> None:
    """Rewrite provider paths in managed JSON before creating a local receipt.

    Provider bundles remain untouched.  Publication records in the downloaded
    copy are re-hashed so local reuse still validates after path relocation.
    """

    root = Path(root)
    marker = f"/runs/{run_id}/artifacts"
    local = str(local_root)
    publication_files: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == ".xrl-stage.json":
            publication_files.append(path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        relocated = _relocalize_value(raw, marker=marker, local_root=local)
        if relocated != raw:
            path.write_text(json.dumps(relocated, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    for publication_path in publication_files:
        raw = json.loads(publication_path.read_text(encoding="utf-8"))
        stage_root = publication_path.parent
        files = raw.get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"invalid stage publication: {publication_path}")
        raw["files"] = {relative: _sha256_file(stage_root / PurePosixPath(relative)) for relative in sorted(files)}
        publication_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["ArtifactSyncReceipt", "relocalize_artifacts", "synchronize_artifacts"]
