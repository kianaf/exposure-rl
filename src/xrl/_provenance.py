"""Tie a derived explanation artifact to the training run it was built from.

A surrogate or FastSHAP artifact records which base run it was distilled from.
Readers must decide whether that record names the run the user just selected.
Comparing the recorded *path* cannot answer this: the string is written under
the trainer's working directory and read under the caller's, so the comparison
silently changes meaning between the two. See ADR 0023.

The witness used instead is the base run's ``started_at``, written into
``run_metadata.json`` by every run (ADR 0019). It depends on the run's contents
rather than its location, so a run copied out of a container still matches, and
it is rewritten whenever an output directory is reused, so a retrained base run
no longer matches artifacts distilled from the parameters it replaced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_METADATA_FILE = "run_metadata.json"
STARTED_AT_KEY = "base_run_started_at"
BASE_RUN_DIR_KEY = "base_run_dir"

# Mirrors xrl.training.run_metadata.RUN_METADATA_SCHEMA_VERSION. Duplicated
# rather than imported so this module stays standard-library only and the
# browser UI can import it without pulling in the training stack.
SUPPORTED_METADATA_SCHEMA = 1


def run_started_at(run_dir: str | Path) -> str | None:
    """Return a run's start instant, or ``None`` when it records no metadata.

    ``None`` means "this run carries no identity", not "identity mismatch";
    callers decide what to do about that.
    """

    path = Path(run_dir) / RUN_METADATA_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != SUPPORTED_METADATA_SCHEMA:
        return None
    started_at = payload.get("started_at")
    return started_at if isinstance(started_at, str) and started_at else None


@dataclass(frozen=True, slots=True)
class RunInstanceRef:
    """One execution of a run, as opposed to the run's descriptive identity.

    ``xrl.training.run_metadata.RunIdentity`` names *what kind of run* this is
    (environment, algorithm, value head). This names *which execution* it was.
    """

    path: Path
    started_at: str | None

    @classmethod
    def of(cls, run_dir: str | Path) -> RunInstanceRef:
        path = Path(run_dir)
        return cls(path=path.resolve(), started_at=run_started_at(path))


@dataclass(frozen=True, slots=True)
class ProvenanceMatch:
    """Whether an artifact belongs to a run, and on what evidence.

    ``recorded`` carries the base run the artifact names, so a caller reporting
    a mismatch does not have to read the config a second time -- a second read
    could disagree with the one the decision was made on, and would escape the
    ``ValueError`` contract if the file had become unreadable in between.
    """

    matched: bool
    basis: str
    detail: str = ""
    recorded: str = ""

    def __bool__(self) -> bool:
        return self.matched


def provenance_payload(base_run_dir: str | Path) -> dict[str, str]:
    """Return the provenance fields an artifact writer merges into its config."""

    ref = RunInstanceRef.of(base_run_dir)
    payload = {BASE_RUN_DIR_KEY: str(ref.path)}
    if ref.started_at is not None:
        payload[STARTED_AT_KEY] = ref.started_at
    return payload


def _path_candidates(recorded: Path, artifact_dir: Path) -> list[Path]:
    # A relative record was written from the trainer's working directory, which
    # is not recoverable here. Try it as given, then anchored at the artifact's
    # parent and grandparent -- the two conventional `runs/<run>/<artifact>`
    # and `runs/<artifact>` layouts. The set is deliberately not widened
    # further: each extra anchor lets one artifact match more runs, and a
    # relation that matches several runs cannot decide which one is meant.
    if recorded.is_absolute():
        return [recorded]
    return [recorded, artifact_dir.parent / recorded, artifact_dir.parent.parent / recorded]


def payload_matches_run(
    payload: Mapping[str, Any],
    *,
    artifact_dir: str | Path,
    ref: RunInstanceRef,
) -> ProvenanceMatch:
    """Decide a parsed artifact config against a run, without touching disk.

    The *artifact* selects the rule. One that records a start instant is matched
    on that alone; one written before this existed falls back to paths.
    """

    recorded_value = payload.get(BASE_RUN_DIR_KEY) if isinstance(payload, Mapping) else None
    if not isinstance(recorded_value, str) or not recorded_value:
        raise ValueError(f"artifact config is missing {BASE_RUN_DIR_KEY}")

    recorded_started_at = payload.get(STARTED_AT_KEY)
    if isinstance(recorded_started_at, str) and recorded_started_at:
        if ref.started_at is None:
            # Never fall back here. The weaker path rule would then be
            # reachable by deleting one file, which turns the check into an
            # opt-out precisely when it would have refused.
            raise ValueError(
                f"artifact records a base run started at {recorded_started_at}, but "
                f"{Path(ref.path) / RUN_METADATA_FILE} is missing or unreadable, so the "
                "selected run cannot be identified"
            )
        matched = recorded_started_at == ref.started_at
        detail = "" if matched else f" (recorded start {recorded_started_at}, selected run started {ref.started_at})"
        return ProvenanceMatch(matched=matched, basis="started_at", detail=detail, recorded=recorded_value)

    artifact_path = Path(artifact_dir)
    candidates = _path_candidates(Path(recorded_value), artifact_path)
    matched = any(candidate.resolve() == ref.path for candidate in candidates)
    return ProvenanceMatch(matched=matched, basis="path", recorded=recorded_value)


def artifact_matches_run(
    artifact_dir: str | Path,
    base_run_dir: str | Path,
    *,
    config_name: str,
    label: str,
    ref: RunInstanceRef | None = None,
) -> ProvenanceMatch:
    """Read an artifact's config and decide it against the selected run."""

    artifact_path = Path(artifact_dir)
    config_path = artifact_path / config_name
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} provenance from {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} config is not an object: {config_path}")
    try:
        return payload_matches_run(
            payload,
            artifact_dir=artifact_path,
            ref=ref if ref is not None else RunInstanceRef.of(base_run_dir),
        )
    except ValueError as error:
        raise ValueError(f"{label} config at {config_path}: {error}") from error


def require_artifact_matches_run(
    artifact_dir: str | Path,
    base_run_dir: str | Path,
    *,
    config_name: str,
    label: str,
    ref: RunInstanceRef | None = None,
) -> None:
    """Raise unless the artifact belongs to the selected run."""

    artifact_path = Path(artifact_dir)
    match = artifact_matches_run(
        artifact_path,
        base_run_dir,
        config_name=config_name,
        label=label,
        ref=ref,
    )
    if match:
        return
    raise ValueError(
        f"{label} {artifact_path} was trained for base run {match.recorded!r}, "
        f"not selected run {base_run_dir}{match.detail}"
    )
