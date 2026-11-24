"""Lifecycle commands for explicitly retained remote research runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from xrl.remote.registry import list_runs, read_run, registry_root, update_run
from xrl.remote.selection import Qualification, read_qualifications, write_qualifications

remote_runs_app = typer.Typer(help="Inspect, pull, and explicitly prune retained remote research runs.")


def _qualification_file(repo: Path) -> Path:
    return registry_root(repo) / "qualifications.json"


@remote_runs_app.command("list")
def list_remote_runs(
    repo: Annotated[Path, typer.Option(help="Repository whose local remote-run index is inspected.")] = Path("."),
) -> None:
    records = list_runs(root=registry_root(repo))
    typer.echo(json.dumps([asdict(record) for record in records], indent=2))


@remote_runs_app.command("qualifications")
def list_qualifications(
    repo: Annotated[Path, typer.Option(help="Repository whose benchmark records are inspected.")] = Path("."),
) -> None:
    records = read_qualifications(_qualification_file(repo))
    typer.echo(
        json.dumps(
            [{**asdict(record), "speedup": record.speedup, "qualifies": record.qualifies()} for record in records],
            indent=2,
        )
    )


@remote_runs_app.command("record-qualification")
def record_qualification(
    provider: Annotated[str, typer.Option(help="Measured provider: ubul or modal.")],
    resource_class: Annotated[str, typer.Option(help="Measured resource class, such as cpu, RTX-3090, or L4.")],
    device_class: Annotated[str, typer.Option(help="Homogeneous device class recorded by manifests.")],
    execution_seconds: Annotated[float, typer.Option(help="Warm representative-stage execution seconds.")],
    cpu_seconds: Annotated[float, typer.Option(help="Matched CPU representative-stage seconds.")],
    projected_seconds: Annotated[float, typer.Option(help="Cached full-matrix end-to-end projection.")],
    estimated_cost_eur: Annotated[float, typer.Option(help="Estimate from current external provider rates.")],
    setup_seconds: Annotated[float, typer.Option(help="First-time environment/image setup seconds.")] = 0.0,
    queue_seconds: Annotated[float, typer.Option(help="Measured queue seconds.")] = 0.0,
    publication_seconds: Annotated[float, typer.Option(help="Measured publication seconds.")] = 0.0,
    download_seconds: Annotated[float, typer.Option(help="Measured download seconds.")] = 0.0,
    completed: Annotated[
        bool,
        typer.Option("--completed/--failed", help="Whether the representative stage completed."),
    ] = True,
    failure_reason: Annotated[str, typer.Option(help="Reason an incomplete qualification stopped.")] = "",
    workflow_name: Annotated[str, typer.Option(help="Workflow selected by auto.")] = "ttt-matrix",
    repo: Annotated[Path, typer.Option(help="Repository whose benchmark record is updated.")] = Path("."),
) -> None:
    if provider not in {"ubul", "modal"}:
        raise typer.BadParameter("provider must be ubul or modal")
    path = _qualification_file(repo)
    existing = list(read_qualifications(path))
    record = Qualification(
        provider=provider,  # type: ignore[arg-type]
        workflow_name=workflow_name,
        resource_class=resource_class,
        device_class=device_class,
        representative_execution_seconds=execution_seconds,
        cpu_execution_seconds=cpu_seconds,
        projected_end_to_end_seconds=projected_seconds,
        estimated_cost_eur=estimated_cost_eur,
        setup_seconds=setup_seconds,
        queue_seconds=queue_seconds,
        publication_seconds=publication_seconds,
        download_seconds=download_seconds,
        completed=completed,
        failure_reason=failure_reason,
    )
    existing = [
        candidate
        for candidate in existing
        if (candidate.provider, candidate.workflow_name, candidate.resource_class)
        != (record.provider, record.workflow_name, record.resource_class)
    ]
    existing.append(record)
    write_qualifications(path, existing)
    typer.echo(json.dumps({**asdict(record), "speedup": record.speedup, "qualifies": record.qualifies()}, indent=2))


@remote_runs_app.command("pull")
def pull_remote_run(
    run_id: Annotated[str, typer.Argument(help="Explicit retained run ID.")],
    out: Annotated[Path | None, typer.Option(help="Local artifact destination; defaults to the recorded path.")] = None,
    repo: Annotated[Path, typer.Option(help="Repository whose local remote-run index is used.")] = Path("."),
) -> None:
    root = registry_root(repo)
    record = read_run(run_id, root=root)
    destination = out or (Path(record.local_path) if record.local_path else Path(repo) / "runs" / record.workflow_name)
    if record.provider == "ubul":
        from xrl.remote.ubul import pull_ubul_run

        pull_ubul_run(record, destination)
    else:
        from xrl.remote.modal_backend import pull_modal_run

        pull_modal_run(record, destination)
    update_run(record, root=root, status="succeeded", local_path=str(destination))
    typer.echo(str(destination))


@remote_runs_app.command("prune")
def prune_remote_run(
    run_id: Annotated[str, typer.Argument(help="Explicit retained run ID.")],
    yes: Annotated[bool, typer.Option("--yes", help="Confirm deletion after reviewing the preview.")] = False,
    repo: Annotated[Path, typer.Option(help="Repository whose local remote-run index is used.")] = Path("."),
) -> None:
    root = registry_root(repo)
    record = read_run(run_id, root=root)
    preview = {
        "run_id": record.run_id,
        "provider": record.provider,
        "remote_path": record.remote_path,
        "local_artifacts_preserved": record.local_path,
    }
    typer.echo(json.dumps(preview, indent=2))
    if not yes:
        typer.echo("Preview only. Re-run with --yes to delete the retained remote run.")
        return
    if record.provider == "ubul":
        from xrl.remote.ubul import prune_ubul_run

        prune_ubul_run(record)
    else:
        from xrl.remote.modal_backend import prune_modal_run

        prune_modal_run(record)
    update_run(record, root=root, status="pruned")
    typer.echo(f"pruned {record.provider}:{record.run_id}")


__all__ = ["remote_runs_app"]
