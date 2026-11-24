"""Run a workflow DAG with one subprocess per stage."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from xrl.remote.workflow import JsonValue, StageContext, WorkflowRunner, WorkflowSpec


class ProcessStageAction:
    """Isolate JAX state and pin GPU stages to one visible device each."""

    def __init__(self, *, spec: Path, python: Path, gpu_devices: tuple[str, ...], cpu_threads: int):
        self.spec = spec
        self.python = python
        self.gpu_devices: queue.Queue[str] = queue.Queue()
        for device in gpu_devices:
            self.gpu_devices.put(device)
        self.cpu_threads = max(1, cpu_threads)

    def __call__(self, context: StageContext) -> dict[str, JsonValue]:
        detail = context.attempt_dir / "detail.json"
        log = context.attempt_dir / "stage.log"
        command = (
            str(self.python),
            "-m",
            "xrl.remote.stage_worker",
            "--spec",
            str(self.spec),
            "--root",
            str(context.workflow_root),
            "--stage",
            context.stage.key,
            "--attempt",
            str(context.attempt_dir),
            "--detail",
            str(detail),
        )
        environment = dict(os.environ)
        # Remote research workflows never inherit the caller's local tracking
        # database. Explicit stage-worker SQLite configuration is still rejected.
        environment.pop("MLFLOW_TRACKING_URI", None)
        device: str | None = None
        if context.stage.resource == "gpu":
            try:
                device = self.gpu_devices.get_nowait()
            except queue.Empty as error:
                raise RuntimeError("no qualified GPU worker is available") from error
            environment["CUDA_VISIBLE_DEVICES"] = device
            environment["JAX_PLATFORMS"] = "cuda"
        else:
            environment["JAX_PLATFORMS"] = "cpu"
            environment["OMP_NUM_THREADS"] = str(self.cpu_threads)
            environment["XLA_FLAGS"] = (
                f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={self.cpu_threads}"
            )
        try:
            with log.open("wb") as output:
                subprocess.run(command, check=True, env=environment, stdout=output, stderr=subprocess.STDOUT)
        finally:
            if device is not None:
                self.gpu_devices.put(device)
        return json.loads(detail.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-devices", default="")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--reuse", action="store_true")
    arguments = parser.parse_args()

    workflow = WorkflowSpec.from_dict(json.loads(arguments.spec.read_text(encoding="utf-8")))
    action = ProcessStageAction(
        spec=arguments.spec,
        python=Path(sys.executable),
        gpu_devices=tuple(device for device in arguments.gpu_devices.split(",") if device),
        cpu_threads=arguments.cpu_threads,
    )
    result = WorkflowRunner(max_workers=arguments.max_workers).run(
        workflow,
        arguments.root,
        action,
        reuse=arguments.reuse,
    )
    arguments.result.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
