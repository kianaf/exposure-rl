"""Benchmark-backed provider and device qualification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["local", "ubul", "modal", "auto"]


@dataclass(frozen=True, slots=True)
class Qualification:
    provider: Literal["ubul", "modal"]
    workflow_name: str
    resource_class: str
    device_class: str
    representative_execution_seconds: float
    cpu_execution_seconds: float
    projected_end_to_end_seconds: float
    estimated_cost_eur: float
    setup_seconds: float
    queue_seconds: float
    publication_seconds: float
    download_seconds: float
    completed: bool = True
    failure_reason: str = ""

    @property
    def speedup(self) -> float:
        return self.cpu_execution_seconds / self.representative_execution_seconds

    def qualifies(self, *, deadline_seconds: float = 1_200.0, modal_cost_cap_eur: float = 5.0) -> bool:
        if not self.completed:
            return False
        gpu_ok = self.device_class == "cpu" or self.speedup >= 1.2
        cost_ok = self.provider != "modal" or self.estimated_cost_eur < modal_cost_cap_eur
        return gpu_ok and self.projected_end_to_end_seconds < deadline_seconds and cost_ok


def write_qualifications(path: Path, records: list[Qualification]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n", encoding="utf-8")


def read_qualifications(path: Path) -> tuple[Qualification, ...]:
    if not path.is_file():
        return ()
    return tuple(Qualification(**record) for record in json.loads(path.read_text(encoding="utf-8")))


def select_backend(workflow_name: str, records: tuple[Qualification, ...]) -> Qualification:
    candidates = [record for record in records if record.workflow_name == workflow_name and record.qualifies()]
    for provider in ("ubul", "modal"):
        matching = [record for record in candidates if record.provider == provider]
        if matching:
            return min(matching, key=lambda record: record.projected_end_to_end_seconds)
    raise RuntimeError(
        f"no qualified remote provider for {workflow_name!r}; run hardware qualification instead of weakening the gate"
    )


__all__ = ["Backend", "Qualification", "read_qualifications", "select_backend", "write_qualifications"]
