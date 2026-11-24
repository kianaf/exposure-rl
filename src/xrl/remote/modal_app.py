"""Modal App definitions, imported only when the optional client is installed."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import modal

APP_NAME = "xrl-research-compute"
VOLUME_NAME = "xrl-research-runs"
VOLUME_ROOT = Path("/vol/runs")
CPU_MAX_CONTAINERS = 4
L4_MAX_CONTAINERS = 1
STAGE_TIMEOUT_SECONDS = 20 * 60
IMAGE_PYTHON_VERSION = "3.12"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION)
    .uv_sync(".", extras=["neural", "explain", "monitoring", "remote"], frozen=True)
    .workdir("/root")
    .add_local_python_source("xrl")
    .add_local_dir("examples", "/root/examples")
)
gpu_image = (
    modal.Image.debian_slim(python_version=IMAGE_PYTHON_VERSION)
    .uv_sync(".", extras=["neural", "explain", "monitoring", "remote", "cuda13"], frozen=True)
    .workdir("/root")
    .add_local_python_source("xrl")
    .add_local_dir("examples", "/root/examples")
)


def _execute_stage(workflow_raw: dict[str, object], stage_key: str, reuse: bool) -> dict[str, object]:
    from xrl.remote.stage_worker import _action_for
    from xrl.remote.workflow import WorkflowRunner, WorkflowSpec

    workflow = WorkflowSpec.from_dict(workflow_raw)
    root = VOLUME_ROOT / workflow.run_id / "artifacts"
    volume.reload()
    result = WorkflowRunner().run_stage(
        workflow,
        root,
        stage_key,
        _action_for(workflow.stage(stage_key).handler, Path("/root")),
        reuse=reuse,
    )
    volume.commit()
    return asdict(result)


@app.function(
    image=cpu_image,
    cpu=(2.0, 4.0),
    memory=8192,
    max_containers=CPU_MAX_CONTAINERS,
    min_containers=0,
    buffer_containers=0,
    retries=0,
    timeout=STAGE_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
    env={"XRL_REMOTE_EXECUTION": "1", "MLFLOW_TRACKING_URI": None},
)
def cpu_stage(workflow_raw: dict[str, object], stage_key: str, reuse: bool) -> dict[str, object]:
    return _execute_stage(workflow_raw, stage_key, reuse)


@app.function(
    image=gpu_image,
    gpu="L4",
    memory=16384,
    max_containers=L4_MAX_CONTAINERS,
    min_containers=0,
    buffer_containers=0,
    retries=0,
    timeout=STAGE_TIMEOUT_SECONDS,
    volumes={"/vol": volume},
    env={"XRL_REMOTE_EXECUTION": "1", "MLFLOW_TRACKING_URI": None},
)
def l4_stage(workflow_raw: dict[str, object], stage_key: str, reuse: bool) -> dict[str, object]:
    import jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(f"Modal L4 qualification did not expose a JAX GPU: {jax.devices()}")
    return _execute_stage(workflow_raw, stage_key, reuse)


@app.local_entrypoint()
def smoke() -> None:
    """Build both images and report the persistent Volume identity."""

    print(json.dumps({"app": APP_NAME, "volume": VOLUME_NAME}))
