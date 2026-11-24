"""Local index of retained provider runs and their synchronization state."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Provider = Literal["ubul", "modal"]
RunStatus = Literal["submitted", "running", "succeeded", "failed", "pruned"]


@dataclass(frozen=True, slots=True)
class RemoteRunRecord:
    schema_version: int
    run_id: str
    provider: Provider
    workflow_name: str
    workflow_fingerprint: str
    source_revision: str
    device_class: str
    status: RunStatus
    remote_path: str
    local_path: str
    created_at_unix: float
    updated_at_unix: float


def registry_root(repo: Path = Path(".")) -> Path:
    return Path(repo).resolve() / "runs" / "remote"


def _record_path(root: Path, provider: Provider, run_id: str) -> Path:
    return root / provider / run_id / "run.json"


def write_run(record: RemoteRunRecord, *, root: Path) -> None:
    path = _record_path(root, record.provider, record.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_run(
    record: RemoteRunRecord, *, root: Path, status: RunStatus, local_path: str | None = None
) -> RemoteRunRecord:
    updated = RemoteRunRecord(
        **{
            **asdict(record),
            "status": status,
            "local_path": record.local_path if local_path is None else local_path,
            "updated_at_unix": time.time(),
        }
    )
    write_run(updated, root=root)
    return updated


def read_run(run_id: str, *, root: Path, provider: Provider | None = None) -> RemoteRunRecord:
    matches = list(root.glob(f"{provider or '*'}/{run_id}/run.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one remote run named {run_id!r}, found {len(matches)}")
    return RemoteRunRecord(**json.loads(matches[0].read_text(encoding="utf-8")))


def list_runs(*, root: Path) -> tuple[RemoteRunRecord, ...]:
    records = [RemoteRunRecord(**json.loads(path.read_text(encoding="utf-8"))) for path in root.glob("*/*/run.json")]
    return tuple(sorted(records, key=lambda record: record.created_at_unix, reverse=True))


__all__ = [
    "Provider",
    "RemoteRunRecord",
    "RunStatus",
    "list_runs",
    "read_run",
    "registry_root",
    "update_run",
    "write_run",
]
