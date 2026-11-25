"""Unit tests for agent/serve.py: which --gpu-memory-utilization the serve script uses (PLAN M3.1).

Cyril's rules: the util comes from the published min_util record for the exact model, plus a
safety margin; an unmeasured model is refused; a full SHA may match a recorded short revision
(>= 10 characters) only if exactly one record matches; --util is an explicit, printed override.
"""

import json
from pathlib import Path

import pytest

from agent import serve

FULL = "b37d1ccb77e27a70fe4222d995fda27940eead5a"
REPO = "cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit"


def write_record(folder: Path, name: str, repo: str, revision: str, best: float | None) -> Path:
    path = folder / f"min_util_{name}.json"
    record = {"llm": {"repo": repo, "revision": revision}, "min_gpu_memory_utilization": best}
    path.write_text(json.dumps(record))
    return path


def test_full_sha_matches_a_recorded_short_revision(tmp_path: Path) -> None:
    path = write_record(tmp_path, "a", REPO, FULL[:10], 0.59)
    chosen = serve.choose_util(REPO, FULL, override=None, records=tmp_path)
    assert chosen == serve.ServedUtil(
        util=round(0.59 + serve.UTIL_MARGIN, 2), measured=0.59, source=str(path), override=False
    )


def test_full_sha_matches_a_recorded_full_sha(tmp_path: Path) -> None:
    write_record(tmp_path, "a", REPO, FULL, 0.59)
    assert serve.choose_util(REPO, FULL, override=None, records=tmp_path).measured == 0.59


@pytest.mark.parametrize(
    ("repo", "recorded"),
    [
        (REPO, "c" * 10),  # another commit
        (REPO, FULL[:9]),  # too short to trust as a prefix
        ("Qwen/Qwen2.5-3B-Instruct-AWQ", FULL[:10]),  # same revision string, other repo
    ],
)
def test_unmeasured_model_is_refused(tmp_path: Path, repo: str, recorded: str) -> None:
    write_record(tmp_path, "a", repo, recorded, 0.59)
    with pytest.raises(LookupError, match="no measured minimum"):
        serve.choose_util(REPO, FULL, override=None, records=tmp_path)


def test_two_matching_records_are_refused(tmp_path: Path) -> None:
    write_record(tmp_path, "a", REPO, FULL[:10], 0.59)
    write_record(tmp_path, "b", REPO, FULL[:12], 0.60)
    with pytest.raises(LookupError, match="2 records"):
        serve.choose_util(REPO, FULL, override=None, records=tmp_path)


def test_a_search_that_found_no_minimum_is_refused(tmp_path: Path) -> None:
    write_record(tmp_path, "a", REPO, FULL[:10], None)
    with pytest.raises(LookupError, match="found no workable"):
        serve.choose_util(REPO, FULL, override=None, records=tmp_path)


def test_short_revision_is_refused(tmp_path: Path) -> None:
    write_record(tmp_path, "a", REPO, FULL[:10], 0.59)
    with pytest.raises(ValueError, match="40"):
        serve.choose_util(REPO, FULL[:10], override=None, records=tmp_path)


def test_override_is_used_and_marked(tmp_path: Path) -> None:
    chosen = serve.choose_util(REPO, FULL, override=0.7, records=tmp_path)
    assert chosen == serve.ServedUtil(util=0.7, measured=None, source="--util", override=True)


def test_published_qwen3_4b_record_is_found() -> None:
    # The real published file, so a rename or format change of results/published breaks this.
    path = Path("results/published/min_util_Qwen3-4B-Instruct-2507-AWQ-4bit.json")
    measured = json.loads(path.read_text())["min_gpu_memory_utilization"]
    chosen = serve.choose_util(REPO, FULL, override=None, records=path.parent)
    assert (chosen.measured, chosen.source) == (measured, str(path))


def test_budget_is_labelled_computed_not_measured() -> None:
    text = serve.budget(serve.ServedUtil(0.5, 0.48, "x.json", False), total_mib=8000)
    assert "4000 MiB" in text and "4000 MiB left" in text
    assert "computed, not measured" in text


def test_budget_names_an_override() -> None:
    text = serve.budget(serve.ServedUtil(0.7, None, "--util", True), total_mib=8000)
    assert "override" in text
