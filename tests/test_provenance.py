"""Unit tests for bench/provenance.py: the commit + dirty flag in every results JSON, and the
rule that bench/eval scripts refuse a dirty tree unless --allow-dirty is passed (F-002 follow-up).

git_state is tested against a real throwaway git repo in tmp_path, not a mock of git's output.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from bench import min_util, vram
from bench.provenance import (
    DirtyTreeError,
    finish,
    git_state,
    is_dirty,
    provenance,
    require_clean,
)


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", *identity, "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    git(tmp_path, "add", "a.txt")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


@pytest.mark.parametrize(
    ("porcelain", "dirty"),
    [("", False), ("\n", False), (" M bench/vram.py\n", True), ("?? new.txt\n", True)],
)
def test_is_dirty(porcelain: str, dirty: bool) -> None:
    assert is_dirty(porcelain) is dirty


def test_clean_repo(repo: Path) -> None:
    commit, dirty = git_state(repo)
    assert len(commit) >= 7
    assert dirty is False


def test_modified_file_is_dirty(repo: Path) -> None:
    (repo / "a.txt").write_text("two\n")
    assert git_state(repo)[1] is True


def test_untracked_file_is_dirty(repo: Path) -> None:
    (repo / "b.txt").write_text("new\n")
    assert git_state(repo)[1] is True


def test_require_clean_refuses_a_dirty_tree() -> None:
    with pytest.raises(DirtyTreeError, match="--allow-dirty"):
        require_clean(dirty=True, allow_dirty=False)


def test_require_clean_allows_dirty_when_asked_and_clean_always() -> None:
    require_clean(dirty=True, allow_dirty=True)
    require_clean(dirty=False, allow_dirty=False)


def test_provenance_records_both_flags(repo: Path) -> None:
    (repo / "b.txt").write_text("new\n")
    record = provenance(allow_dirty=True, cwd=repo)
    assert record["dirty"] is True
    assert record["allow_dirty"] is True
    assert record["commit"] == git_state(repo)[0]


def test_provenance_refuses_dirty_without_flag(repo: Path) -> None:
    (repo / "b.txt").write_text("new\n")
    with pytest.raises(DirtyTreeError):
        provenance(allow_dirty=False, cwd=repo)


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (vram, ["bench.vram", "--beam-size", "5", "--llm", "x/y@rev:0.5"]),
        (min_util, ["bench.min_util", "--llm", "x/y@rev", "--lo", "0.2", "--hi", "0.4"]),
    ],
)
def test_scripts_refuse_a_dirty_tree_before_touching_the_gpu(
    module: object, argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_gpu(*cmd: str) -> str:
        raise AssertionError(f"ran {cmd} before the dirty-tree check")

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("bench.provenance.git_state", lambda cwd=None: ("abc1234", True))
    monkeypatch.setattr(module, "sh", no_gpu)
    with pytest.raises(DirtyTreeError):
        module.main()  # type: ignore[attr-defined]


def test_finish_marks_dirty_if_tree_changed_during_the_run(repo: Path) -> None:
    start = provenance(allow_dirty=False, cwd=repo)
    (repo / "a.txt").write_text("edited mid-run\n")
    assert finish(start, cwd=repo)["dirty"] is True


def test_finish_keeps_clean_when_nothing_changed(repo: Path) -> None:
    start = provenance(allow_dirty=False, cwd=repo)
    assert finish(start, cwd=repo) == start


def test_finish_refuses_if_head_moved_during_the_run(repo: Path) -> None:
    start = provenance(allow_dirty=False, cwd=repo)
    git(repo, "commit", "-q", "--allow-empty", "-m", "moved")
    with pytest.raises(DirtyTreeError, match="HEAD moved"):
        finish(start, cwd=repo)
