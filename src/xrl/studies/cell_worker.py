"""Run one study cell in an isolated interpreter with pinned CPU threads.

Both ends of one contract live here: the parent-side :class:`SubprocessCellWork`
that builds the child environment, and the child entry point that consumes it.
Keeping them in one module keeps the pinning and its consumer visible together.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xrl.studies.core import (
    StudyCellContext,
    StudyCellWork,
    StudyError,
    cell_directory,
    load_study_spec,
)

CONTROLLED_Q_CELL_HANDLER = "controlled-q.policy-cell"
SURROGATE_ABLATION_HANDLER = "surrogate-ablation.cell"

# Fixed by decision, and deliberately not derived from the worker count or the
# host: XLA's CPU intra-op parallelism partitions reductions across threads and
# so changes their association. Floating-point addition is not associative, and
# in a reinforcement-learning loop a last-bit difference is amplified rather
# than damped -- it can flip an argmax, which changes an action, which changes
# the trajectory and the replay contents. Pinning this makes reduction order a
# declared constant of the study instead of a scheduling artifact.
STUDY_CELL_CPU_THREADS = 1


class StudyCellWorkerError(StudyError):
    """A study cell failed inside its isolated worker process."""


def worker_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the fixed, worker-count-independent environment for one cell."""

    environment = dict(base)
    # Study workers never inherit the caller's local tracking database.
    environment.pop("MLFLOW_TRACKING_URI", None)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = str(STUDY_CELL_CPU_THREADS)
    environment["MKL_NUM_THREADS"] = str(STUDY_CELL_CPU_THREADS)
    environment["OPENBLAS_NUM_THREADS"] = str(STUDY_CELL_CPU_THREADS)
    environment["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={STUDY_CELL_CPU_THREADS}"
    )
    environment["PYTHONHASHSEED"] = "0"
    return environment


class SubprocessCellWork:
    """A :data:`StudyCellWork` that runs its handler in a fresh interpreter."""

    def __init__(self, *, root: Path, handler: str, python: Path | None = None) -> None:
        self.root = Path(root)
        self.handler = handler
        self.python = Path(python or sys.executable)

    def __call__(self, context: StudyCellContext) -> Mapping[str, Any]:
        # The output and log are siblings of the staging directory, never inside
        # it: everything under ``context.out_dir`` is hashed as a cell artifact,
        # so a stray log there would make parallel hashes differ from serial.
        scratch = context.out_dir.parent
        output_path = scratch / f"{context.out_dir.name}.output.json"
        log_path = scratch / f"{context.out_dir.name}.log"
        command = (
            str(self.python),
            "-m",
            # A shim nothing imports, so runpy executes the child exactly once.
            "xrl.studies._cell_main",
            "--root",
            str(self.root),
            "--condition",
            context.condition.key,
            "--seed",
            str(context.seed),
            "--staging",
            str(context.out_dir),
            "--handler",
            self.handler,
            "--output",
            str(output_path),
        )
        try:
            with log_path.open("wb") as stream:
                subprocess.run(
                    command,
                    check=True,
                    env=worker_environment(os.environ),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as error:
            raise StudyCellWorkerError(
                f"study cell {context.cell_id} failed in a worker process (exit {error.returncode}); see {log_path}"
            ) from error
        output = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.unlink()
        log_path.unlink()
        return output


def _work_for(handler: str) -> StudyCellWork:
    if handler == SURROGATE_ABLATION_HANDLER:
        from xrl.studies.surrogate_ablation import make_ablation_cell_work

        return make_ablation_cell_work()
    if handler == CONTROLLED_Q_CELL_HANDLER:
        # Deferred: controlled_q imports this module, and this keeps a bare
        # ``import xrl.studies.cell_worker`` free of the neural extra.
        from xrl.studies.controlled_q import train_controlled_q_policy_cell

        return train_controlled_q_policy_cell
    raise ValueError(f"no study cell worker for handler {handler!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--handler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri.startswith("sqlite:"):
        raise StudyCellWorkerError("study cell workers refuse a local SQLite MLflow tracking URI")

    spec = load_study_spec(arguments.root)
    condition = next((entry for entry in spec.conditions if entry.key == arguments.condition), None)
    if condition is None or arguments.seed not in spec.seeds:
        raise StudyCellWorkerError(f"{arguments.condition}/seed_{arguments.seed} is not a cell of {arguments.root}")
    context = StudyCellContext(
        condition=condition,
        seed=arguments.seed,
        profile=spec.profile,
        out_dir=arguments.staging,
        publish_dir=cell_directory(spec.root, condition.key, arguments.seed),
        manifest_fingerprint=spec.fingerprint,
    )
    output = _work_for(arguments.handler)(context)
    payload: dict[str, Any] = {} if output is None else dict(output)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


__all__ = [
    "CONTROLLED_Q_CELL_HANDLER",
    "SURROGATE_ABLATION_HANDLER",
    "STUDY_CELL_CPU_THREADS",
    "StudyCellWorkerError",
    "SubprocessCellWork",
    "worker_environment",
]


if __name__ == "__main__":
    raise SystemExit(main())
