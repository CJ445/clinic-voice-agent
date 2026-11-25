"""Git provenance for every results JSON: commit hash, dirty flag, and the clean-tree rule.

Rule 3 says every number is tied to a commit. A number from a dirty tree is tied to code that no
commit contains, so bench and eval scripts refuse to run on a dirty tree unless --allow-dirty is
passed, and the record says which happened (F-002 follow-up). Checked at the start of a run, so
nothing touches the GPU first, and again at the end, so edits made during the run are caught.
"""

import subprocess
from pathlib import Path


class DirtyTreeError(RuntimeError):
    """The working tree has uncommitted or untracked changes and --allow-dirty was not passed."""


def is_dirty(porcelain: str) -> bool:
    """True if `git status --porcelain` printed anything (modified, staged or untracked files)."""
    return bool(porcelain.strip())


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    """(short commit hash, dirty) for the repo at cwd (default: the current directory)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout

    return git("rev-parse", "--short", "HEAD").strip(), is_dirty(git("status", "--porcelain"))


def require_clean(*, dirty: bool, allow_dirty: bool) -> None:
    if dirty and not allow_dirty:
        raise DirtyTreeError(
            "working tree is dirty: commit or stash first, or pass --allow-dirty "
            "(recorded in the result as allow_dirty: true)"
        )


def provenance(*, allow_dirty: bool, cwd: Path | None = None) -> dict[str, object]:
    """Check the tree and return the record's provenance fields. Call at the start of a run."""
    commit, dirty = git_state(cwd)
    require_clean(dirty=dirty, allow_dirty=allow_dirty)
    return {"commit": commit, "dirty": dirty, "allow_dirty": allow_dirty}


def finish(start: dict[str, object], cwd: Path | None = None) -> dict[str, object]:
    """Provenance at the end of a run: dirty if the tree was dirty at the start *or* is now."""
    commit, dirty_now = git_state(cwd)
    if commit != start["commit"]:
        raise DirtyTreeError(f"HEAD moved during the run ({start['commit']} -> {commit})")
    return {**start, "dirty": bool(start["dirty"]) or dirty_now}
