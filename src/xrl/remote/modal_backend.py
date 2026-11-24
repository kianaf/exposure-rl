"""Synchronous Modal aggregation over isolated, Volume-backed stage calls."""

from __future__ import annotations

import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any

from xrl.remote.artifacts import relocalize_artifacts, synchronize_artifacts
from xrl.remote.registry import RemoteRunRecord
from xrl.remote.workflow import WorkflowSpec


def _modal():
    try:
        import modal
    except ImportError as error:
        raise RuntimeError("Modal execution requires `uv sync --extra remote`") from error
    return modal


def _upload_inputs(workflow: WorkflowSpec, volume: Any) -> None:
    inputs = [stage for stage in workflow.stages if stage.handler == "ttt-ablation.input"]
    if not inputs:
        return
    with volume.batch_upload(force=True) as batch:
        for stage in inputs:
            source = Path(str(stage.payload["base_run"]))
            remote = f"/runs/{workflow.run_id}/artifacts/{stage.payload['remote_relative']}"
            batch.put_directory(str(source), remote)


def run_modal_workflow(workflow: WorkflowSpec, *, reuse: bool) -> dict[str, dict[str, object]]:
    modal = _modal()
    from xrl.remote.modal_app import app, cpu_stage, l4_stage, volume

    _upload_inputs(workflow, volume)
    pending = {stage.key: stage for stage in workflow.stages}
    completed: set[str] = set()
    results: dict[str, dict[str, object]] = {}
    running: dict[Future[dict[str, object]], str] = {}

    def invoke(stage_key: str) -> dict[str, object]:
        stage = workflow.stage(stage_key)
        function = l4_stage if stage.resource == "gpu" else cpu_stage
        return function.remote(workflow.to_dict(), stage_key, reuse)

    with modal.enable_output(), app.run():
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="xrl-modal")
        try:
            while pending or running:
                ready = [
                    stage
                    for stage in pending.values()
                    if all(dependency in completed for dependency in stage.dependencies)
                ]
                for stage in ready[: max(0, 4 - len(running))]:
                    pending.pop(stage.key)
                    running[executor.submit(invoke, stage.key)] = stage.key
                if not running:
                    raise RuntimeError("Modal scheduler made no progress")
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    key = running.pop(future)
                    results[key] = future.result()
                    completed.add(key)
        except BaseException:
            pending.clear()
            for future in running:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    return results


def pull_modal_run(record: RemoteRunRecord, destination: Path) -> None:
    if record.provider != "modal":
        raise ValueError("not a Modal run")
    _modal()
    from xrl.remote.modal_app import volume

    prefix = f"/runs/{record.run_id}/artifacts"
    with tempfile.TemporaryDirectory(prefix="xrl-modal-pull-") as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        for entry in volume.iterdir(prefix, recursive=True):
            if getattr(entry.type, "name", "") != "FILE":
                continue
            entry_path = PurePosixPath(entry.path.lstrip("/"))
            prefix_path = PurePosixPath(prefix.lstrip("/"))
            relative = entry_path.relative_to(prefix_path)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for chunk in volume.read_file(entry.path):
                    handle.write(chunk)
        relocalize_artifacts(staging, run_id=record.run_id, local_root=destination)
        synchronize_artifacts(
            staging,
            destination,
            run_id=record.run_id,
            workflow_fingerprint=record.workflow_fingerprint,
        )


def prune_modal_run(record: RemoteRunRecord) -> None:
    if record.provider != "modal" or record.remote_path != f"xrl-research-runs:/runs/{record.run_id}":
        raise ValueError("refusing to prune an unexpected Modal path")
    _modal()
    from xrl.remote.modal_app import volume

    volume.remove_file(f"/runs/{record.run_id}", recursive=True)


__all__ = ["prune_modal_run", "pull_modal_run", "run_modal_workflow"]
