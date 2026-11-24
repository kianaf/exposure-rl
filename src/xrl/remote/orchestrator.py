"""One-provider remote workflow orchestration and automatic selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from xrl.remote.registry import RemoteRunRecord, read_run, registry_root, update_run, write_run
from xrl.remote.selection import Qualification, read_qualifications, select_backend
from xrl.remote.workflow import WorkflowSpec

RemoteProvider = Literal["ubul", "modal"]


@dataclass(frozen=True, slots=True)
class RemoteTarget:
    provider: RemoteProvider
    device_class: str
    qualification: Qualification | None


@dataclass(frozen=True, slots=True)
class RemoteExecutionResult:
    provider: RemoteProvider
    run_id: str
    remote_path: str
    local_path: str


def qualification_path(repo: Path) -> Path:
    return registry_root(repo) / "qualifications.json"


def resolve_target(requested: str, workflow_name: str, *, repo: Path) -> RemoteTarget:
    records = read_qualifications(qualification_path(repo))
    if requested == "auto":
        selected = select_backend(workflow_name, records)
        return RemoteTarget(selected.provider, selected.device_class, selected)
    if requested not in {"ubul", "modal"}:
        raise ValueError(f"not a remote backend: {requested!r}")
    qualified = [record for record in records if record.provider == requested and record.workflow_name == workflow_name]
    selected_record = min(qualified, key=lambda record: record.projected_end_to_end_seconds) if qualified else None
    return RemoteTarget(
        cast(RemoteProvider, requested),
        selected_record.device_class if selected_record and selected_record.qualifies() else "cpu",
        selected_record,
    )


def generated_run_id(workflow: WorkflowSpec) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"{workflow.name}-{stamp}-{workflow.fingerprint[:12]}"


def execute_remote_workflow(
    workflow: WorkflowSpec,
    out_root: Path,
    *,
    provider: RemoteProvider,
    repo: Path,
    reuse: bool,
) -> RemoteExecutionResult:
    root = registry_root(repo)
    if provider == "ubul":
        from xrl.remote.ubul import UbulConfig, launch_ubul, pull_ubul_run, wait_ubul

        remote_path = f"{UbulConfig().cache_root}/runs/{workflow.run_id}"
    else:
        remote_path = f"xrl-research-runs:/runs/{workflow.run_id}"

    now = time.time()
    record = RemoteRunRecord(
        1,
        workflow.run_id,
        provider,
        workflow.name,
        workflow.fingerprint,
        workflow.source_revision,
        workflow.device_class,
        "submitted",
        remote_path,
        str(out_root),
        now,
        now,
    )
    if reuse:
        try:
            existing = read_run(workflow.run_id, root=root, provider=provider)
        except FileNotFoundError:
            existing = None
        if existing is not None and existing.workflow_fingerprint != workflow.fingerprint:
            raise RuntimeError("remote run ID exists with a different workflow fingerprint")
    write_run(record, root=root)
    running = update_run(record, root=root, status="running")
    try:
        if provider == "ubul":
            launch_ubul(workflow, reuse=reuse)
            wait_ubul(workflow)
        else:
            from xrl.remote.modal_backend import run_modal_workflow

            run_modal_workflow(workflow, reuse=reuse)
    except BaseException:
        update_run(running, root=root, status="failed")
        raise
    succeeded = update_run(running, root=root, status="succeeded")
    if provider == "ubul":
        pull_ubul_run(succeeded, out_root)
    else:
        from xrl.remote.modal_backend import pull_modal_run

        pull_modal_run(succeeded, out_root)
    update_run(succeeded, root=root, status="succeeded", local_path=str(out_root))
    return RemoteExecutionResult(provider, workflow.run_id, remote_path, str(out_root))


__all__ = [
    "RemoteExecutionResult",
    "RemoteProvider",
    "RemoteTarget",
    "execute_remote_workflow",
    "generated_run_id",
    "qualification_path",
    "resolve_target",
]
