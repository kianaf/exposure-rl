"""Canonical explanation documents persisted under ``<run>/explain/results/``.

Every computed explanation — CLI or browser UI — is serialized into one JSON
document with a stable schema, so results can be listed and re-rendered later
(including the board snapshot they were computed on) without recomputing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xrl.explainability.core import ExplanationResult

RESULTS_DIRNAME = "results"

_ID_FORBIDDEN = ("/", "\\", "..")


def json_safe(value: Any) -> Any:
    """Coerce a value into something both Python and browsers can parse.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens, which are valid for
    Python's own loader but rejected by ``JSON.parse`` — a saved document (or
    an HTTP response built from one) containing them is unreadable in the
    browser. Non-finite floats become ``null``; numpy scalars and arrays become
    plain Python values.
    """

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def results_dir(run_dir: str | Path) -> Path:
    """Return the saved-explanations directory for a run."""

    return Path(run_dir) / "explain" / RESULTS_DIRNAME


def explanation_document(
    *,
    method: str,
    env_id: str,
    run_dir: str | Path,
    model_type: str,
    state_source: str,
    state_seed: int,
    observation: np.ndarray,
    action_mask: np.ndarray,
    result: ExplanationResult,
    extras: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON document for one explanation result."""

    created = created_at or datetime.now(UTC)
    document: dict[str, Any] = {
        "created_at": created.isoformat(),
        "duration_seconds": None if duration_seconds is None else round(float(duration_seconds), 6),
        "method": method,
        "env_id": env_id,
        "run": str(run_dir),
        "model_type": model_type,
        "state": {
            "source": state_source,
            "seed": int(state_seed),
            "observation": [float(value) for value in np.asarray(observation).reshape(-1)],
            "action_mask": [int(value) for value in np.asarray(action_mask).reshape(-1)],
        },
        "action": result.action,
        "target": result.target,
        "baseline_value": result.baseline_value,
        "full_value": result.full_value,
        "attributions": result.attributions,
        "characteristic_assumption": result.characteristic_assumption,
        "semantics": asdict(result.semantics),
    }
    if extras:
        collisions = set(extras) & set(document)
        if collisions:
            raise ValueError(f"extras cannot replace canonical explanation fields: {sorted(collisions)}")
        document.update(dict(extras))
    return json_safe(document)


def save_explanation(run_dir: str | Path, document: Mapping[str, Any]) -> Path:
    """Write one document under the run's results dir; returns the file path.

    The filename (and thus the document ``id``) encodes creation time, method,
    and explained action; collisions get a ``-{n}`` suffix.
    """

    directory = results_dir(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stored = json_safe(dict(document))
    created = datetime.fromisoformat(str(stored["created_at"]))
    stem = f"{created.strftime('%Y%m%dT%H%M%S%f')}Z_{stored['method']}_a{stored['action']}"
    path = directory / f"{stem}.json"
    suffix = 0
    while path.exists():
        suffix += 1
        path = directory / f"{stem}-{suffix}.json"
    stored["id"] = path.stem
    path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return path


def load_explanation(run_dir: str | Path, explanation_id: str) -> dict[str, Any]:
    """Load one saved document by id (the filename stem)."""

    if not explanation_id or any(token in explanation_id for token in _ID_FORBIDDEN):
        raise ValueError(f"invalid explanation id {explanation_id!r}")
    path = results_dir(run_dir) / f"{explanation_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved explanation {explanation_id!r} in {results_dir(run_dir)}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"saved explanation {explanation_id!r} is not a JSON object")
    return document


def list_explanations(run_dir: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Return saved documents newest-first; unreadable files are skipped."""

    directory = results_dir(run_dir)
    if not directory.exists():
        return []
    documents: list[dict[str, Any]] = []
    # Zero-padded UTC timestamps make the filename stem sort chronologically.
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stem, reverse=True):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        document.setdefault("id", path.stem)
        documents.append(document)
        if limit is not None and len(documents) >= limit:
            break
    return documents
