"""Detached SSH/tmux executor for the optional ``ubul`` workstation."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from xrl.remote.artifacts import relocalize_artifacts, synchronize_artifacts
from xrl.remote.registry import RemoteRunRecord
from xrl.remote.workflow import WorkflowSpec

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class UbulConfig:
    host: str = "ubul"
    repository: str = "/path/to/xrl"  # set for your remote machine
    cache_root: str = "/path/to/cache/xrl"
    uv: str = "uv"
    min_free_bytes: int = 10 * GIB


@dataclass(frozen=True, slots=True)
class UbulPreflight:
    free_bytes: int
    gpu_count: int
    tmux_version: str


DEFAULT_UBUL_CONFIG = UbulConfig()


def _ssh(config: UbulConfig, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("ssh", config.host, "bash", "-lc", shlex.quote(command)),
        check=check,
        capture_output=True,
        text=True,
    )


def preflight_ubul(config: UbulConfig = DEFAULT_UBUL_CONFIG) -> UbulPreflight:
    tmux = _ssh(config, "command -v tmux >/dev/null && tmux -V", check=False)
    if tmux.returncode != 0:
        raise RuntimeError(
            "ubul requires tmux; install it once interactively with: ssh -t ubul 'sudo apt-get install tmux'"
        )
    free = int(
        _ssh(
            config, f"df -Pk {shlex.quote(str(Path(config.cache_root).parent))} | awk 'NR==2 {{print $4 * 1024}}'"
        ).stdout
    )
    if free < config.min_free_bytes:
        raise RuntimeError(f"ubul admission refused: {free / GIB:.1f} GiB free is below the 10 GiB floor")
    gpu = _ssh(config, "command -v nvidia-smi >/dev/null && nvidia-smi --list-gpus | wc -l", check=False)
    gpu_count = int(gpu.stdout.strip() or "0") if gpu.returncode == 0 else 0
    return UbulPreflight(free, gpu_count, tmux.stdout.strip())


def _remote_paths(workflow: WorkflowSpec, config: UbulConfig) -> dict[str, str]:
    revision = workflow.source_revision
    return {
        "run": f"{config.cache_root}/runs/{workflow.run_id}",
        "worktree": f"{config.cache_root}/worktrees/{revision}",
        "environment": f"{config.cache_root}/envs/{workflow.lock_fingerprint}",
        "lock": f"{config.cache_root}/compute.lock",
    }


def _worker_allocation(workflow: WorkflowSpec) -> tuple[int, str]:
    """Keep admission capacity equal to the physical device pool."""

    if workflow.device_class == "cpu":
        return 4, ""
    return 2, "0,1"


def prepare_ubul(workflow: WorkflowSpec, *, config: UbulConfig = DEFAULT_UBUL_CONFIG) -> dict[str, str]:
    preflight = preflight_ubul(config)
    if workflow.device_class != "cpu" and preflight.gpu_count < 1:
        raise RuntimeError(f"ubul has no GPU available for device class {workflow.device_class!r}")
    paths = _remote_paths(workflow, config)
    extras = "--extra neural --extra explain --extra monitoring"
    if workflow.device_class != "cpu":
        extras += " --extra cuda13"
    script = "\n".join(
        (
            "set -eu",
            f"mkdir -p {shlex.quote(config.cache_root + '/runs')} {shlex.quote(config.cache_root + '/worktrees')} {shlex.quote(config.cache_root + '/envs')}",
            f"git -C {shlex.quote(config.repository)} fetch origin {shlex.quote(workflow.source_revision)}",
            (
                f"test -e {shlex.quote(paths['worktree'] + '/.git')} || "
                f"git -C {shlex.quote(config.repository)} worktree add --detach {shlex.quote(paths['worktree'])} "
                f"{shlex.quote(workflow.source_revision)}"
            ),
            f'test "$(git -C {shlex.quote(paths["worktree"])} rev-parse HEAD)" = {shlex.quote(workflow.source_revision)}',
            f"mkdir -p {shlex.quote(paths['run'])}",
            (
                f"UV_PROJECT_ENVIRONMENT={shlex.quote(paths['environment'])} {shlex.quote(config.uv)} sync "
                f"--project {shlex.quote(paths['worktree'])} --frozen {extras}"
            ),
        )
    )
    _ssh(config, script)
    return paths


def launch_ubul(
    workflow: WorkflowSpec,
    *,
    reuse: bool,
    config: UbulConfig = DEFAULT_UBUL_CONFIG,
) -> str:
    paths = prepare_ubul(workflow, config=config)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", workflow.run_id) is None:
        raise ValueError("ubul run IDs may contain only letters, digits, dots, underscores, and hyphens")
    session = f"xrl-{workflow.run_id}"
    if len(session) > 80:
        raise ValueError("run ID is too long for a readable tmux session")
    with tempfile.TemporaryDirectory(prefix="xrl-ubul-") as temporary:
        spec = Path(temporary) / "workflow.json"
        spec.write_text(json.dumps(workflow.to_dict(), indent=2) + "\n", encoding="utf-8")
        subprocess.run(("rsync", "-a", str(spec), f"{config.host}:{paths['run']}/workflow.json"), check=True)
    for stage in workflow.stages:
        if stage.handler != "ttt-ablation.input":
            continue
        source = Path(str(stage.payload["base_run"]))
        remote_relative = str(stage.payload["remote_relative"])
        _ssh(config, f"mkdir -p {shlex.quote(paths['run'] + '/artifacts/' + remote_relative)}")
        subprocess.run(
            ("rsync", "-a", f"{source}/", f"{config.host}:{paths['run']}/artifacts/{remote_relative}/"),
            check=True,
        )

    max_workers, gpu_devices = _worker_allocation(workflow)
    command = (
        f"cd {shlex.quote(paths['worktree'])} && "
        f"{shlex.quote(paths['worktree'] + '/scripts/remote/run_ubul_workflow.sh')} "
        f"{shlex.quote(paths['run'])} {shlex.quote(paths['lock'])} "
        f"{shlex.quote(paths['environment'] + '/bin/python')} {shlex.quote(paths['run'] + '/workflow.json')} "
        f"{shlex.quote(paths['run'] + '/artifacts')} {max_workers} {shlex.quote(gpu_devices)} 2 "
        f"{'true' if reuse else 'false'} > {shlex.quote(paths['run'] + '/workflow.log')} 2>&1"
    )
    launch = f"tmux has-session -t {shlex.quote(session)} 2>/dev/null || tmux new-session -d -s {shlex.quote(session)} {shlex.quote(command)}"
    _ssh(config, launch)
    return session


def wait_ubul(
    workflow: WorkflowSpec,
    *,
    poll_seconds: float = 5.0,
    config: UbulConfig = DEFAULT_UBUL_CONFIG,
) -> dict[str, object]:
    paths = _remote_paths(workflow, config)
    while True:
        response = _ssh(
            config,
            f"test -f {shlex.quote(paths['run'] + '/status.json')} && cat {shlex.quote(paths['run'] + '/status.json')}",
            check=False,
        )
        if response.returncode == 0 and response.stdout.strip():
            status = json.loads(response.stdout)
            if status.get("state") == "succeeded":
                return status
            if status.get("state") == "failed":
                tail = _ssh(config, f"tail -n 80 {shlex.quote(paths['run'] + '/workflow.log')}", check=False).stdout
                raise RuntimeError(f"ubul workflow failed\n{tail}")
        time.sleep(poll_seconds)


def pull_ubul_run(record: RemoteRunRecord, destination: Path, config: UbulConfig = DEFAULT_UBUL_CONFIG) -> None:
    if record.provider != "ubul":
        raise ValueError("not an ubul run")
    with tempfile.TemporaryDirectory(prefix="xrl-ubul-pull-") as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        subprocess.run(("rsync", "-a", f"{config.host}:{record.remote_path}/artifacts/", f"{staging}/"), check=True)
        relocalize_artifacts(staging, run_id=record.run_id, local_root=destination)
        synchronize_artifacts(
            staging,
            destination,
            run_id=record.run_id,
            workflow_fingerprint=record.workflow_fingerprint,
        )


def prune_ubul_run(record: RemoteRunRecord, config: UbulConfig = DEFAULT_UBUL_CONFIG) -> None:
    expected = f"{config.cache_root}/runs/{record.run_id}"
    if record.provider != "ubul" or record.remote_path != expected:
        raise ValueError("refusing to prune an unexpected ubul path")
    _ssh(config, f"rm -rf -- {shlex.quote(expected)}")


__all__ = [
    "UbulConfig",
    "UbulPreflight",
    "launch_ubul",
    "preflight_ubul",
    "prepare_ubul",
    "prune_ubul_run",
    "pull_ubul_run",
    "wait_ubul",
]
