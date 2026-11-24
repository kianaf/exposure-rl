"""Optional offline research execution without provider dependencies."""

from xrl.remote.artifacts import ArtifactSyncReceipt, synchronize_artifacts
from xrl.remote.workflow import (
    ArtifactManifest,
    StageContext,
    StageResult,
    StageSpec,
    WorkflowFailed,
    WorkflowResult,
    WorkflowRunner,
    WorkflowSpec,
)

__all__ = [
    "ArtifactManifest",
    "ArtifactSyncReceipt",
    "StageContext",
    "StageResult",
    "StageSpec",
    "WorkflowFailed",
    "WorkflowResult",
    "WorkflowRunner",
    "WorkflowSpec",
    "synchronize_artifacts",
]
