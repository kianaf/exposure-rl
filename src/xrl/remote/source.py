"""Clean, GitHub-reachable source admission for remote research runs."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    revision: str
    origin_url: str
    origin_refs: tuple[str, ...]
    lock_fingerprint: str


class SourceAdmissionError(RuntimeError):
    pass


def _git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        )
        raise SourceAdmissionError(f"git {' '.join(arguments)} failed: {detail}") from error


def admit_remote_source(repo: Path = Path(".")) -> SourceIdentity:
    """Require a clean HEAD contained by at least one fetched ``origin/*`` ref."""

    repo = Path(repo).resolve()
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        preview = ", ".join(line[3:] for line in dirty.splitlines()[:5])
        raise SourceAdmissionError(f"remote execution requires a clean worktree; changed files: {preview}")
    revision = _git(repo, "rev-parse", "HEAD")
    origin_url = _git(repo, "remote", "get-url", "origin")
    refs = tuple(
        line
        for line in _git(
            repo,
            "for-each-ref",
            "--format=%(refname:short)",
            "--contains",
            revision,
            "refs/remotes/origin",
        ).splitlines()
        if line.startswith("origin/") and line != "origin/HEAD"
    )
    if not refs:
        raise SourceAdmissionError(
            f"commit {revision[:12]} is not contained by a fetched GitHub origin ref; push it before remote execution"
        )
    lock = repo / "uv.lock"
    if not lock.is_file():
        raise SourceAdmissionError("remote execution requires uv.lock")
    return SourceIdentity(revision, origin_url, refs, hashlib.sha256(lock.read_bytes()).hexdigest())


__all__ = ["SourceAdmissionError", "SourceIdentity", "admit_remote_source"]
