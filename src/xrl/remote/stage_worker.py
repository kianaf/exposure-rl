"""Execute exactly one serialized workflow stage in an isolated process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from xrl.remote.workflow import StageContext, WorkflowSpec


def _load_example(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workflow module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _action_for(handler: str, repo: Path):
    examples = repo / "examples"
    if handler.startswith("ttt-ablation."):
        module = _load_example("ttt_ablation_remote", examples / "ttt_ablation.py")
        return module.execute_ablation_stage
    if handler.startswith("ttt."):
        module = _load_example("ttt_full_pipeline_remote", examples / "ttt_full_pipeline.py")
        return module.execute_pipeline_stage
    raise ValueError(f"no stage-worker dispatcher for handler {handler!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    arguments = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri.startswith("sqlite:"):
        raise RuntimeError("remote workflows refuse a local SQLite MLflow tracking URI")

    workflow = WorkflowSpec.from_dict(json.loads(arguments.spec.read_text(encoding="utf-8")))
    stage = workflow.stage(arguments.stage)
    root = arguments.root
    context = StageContext(
        workflow=workflow,
        stage=stage,
        workflow_root=root,
        output_dir=root / stage.output,
        attempt_dir=arguments.attempt,
        dependency_outputs={key: root / workflow.stage(key).output for key in stage.dependencies},
    )
    action = _action_for(stage.handler, Path.cwd())
    detail = dict(action(context) or {})
    arguments.detail.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
