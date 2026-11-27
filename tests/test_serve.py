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
    # Live run, 2026-11-25: vLLM used more than util x total (CUDA context, graphs), so the
    # remainder is an upper bound and must say so.
    assert "at most" in text and "outside its share" in text


def test_budget_names_an_override() -> None:
    text = serve.budget(serve.ServedUtil(0.7, None, "--util", True), total_mib=8000)
    assert "override" in text


def test_main_refuses_an_unmeasured_model_before_starting_vllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_start(*args: object, **kwargs: object) -> None:
        raise AssertionError("started vLLM for an unmeasured model")

    monkeypatch.setattr(serve, "RECORDS", tmp_path)
    monkeypatch.setattr(serve.vllm_server, "start", must_not_start)
    monkeypatch.setattr("sys.argv", ["agent.serve", "--llm", f"{REPO}@{FULL}"])
    with pytest.raises(LookupError, match="no measured minimum"):
        serve.main()


def test_stop_cannot_be_interrupted_by_a_second_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    # Review #4: an impatient second Ctrl-C during stop()'s 60 s grace raised KeyboardInterrupt
    # inside it and left vLLM's process group holding VRAM.
    import signal

    seen: list[object] = []
    monkeypatch.setattr(
        serve.vllm_server,
        "stop",
        lambda proc: seen.extend(
            [signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)]
        ),
    )
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    serve.stop_uninterruptibly(object())  # type: ignore[arg-type]
    assert seen == [signal.SIG_IGN, signal.SIG_IGN]
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


@pytest.mark.parametrize("util", [0.0, -0.1, 0.96, 5.0])
def test_util_out_of_range_is_refused(tmp_path: Path, util: float) -> None:
    # Review #8, Cyril: 0 < util <= 0.95, so `--util 5` fails here, not deep in vLLM's startup.
    with pytest.raises(ValueError, match="0.95"):
        serve.choose_util(REPO, FULL, override=util, records=tmp_path)


def test_util_at_the_cap_is_accepted(tmp_path: Path) -> None:
    assert serve.choose_util(REPO, FULL, override=serve.MAX_UTIL, records=tmp_path).util == 0.95


def test_measured_plus_margin_is_also_capped(tmp_path: Path) -> None:
    write_record(tmp_path, "a", REPO, FULL[:10], 0.94)
    with pytest.raises(ValueError, match="0.95"):
        serve.choose_util(REPO, FULL, override=None, records=tmp_path)


def test_published_served_record_is_what_serve_would_use() -> None:
    # Review (ADR-0003 PR) #1: if bench.min_util's minimum moves, agent.serve's util moves with it
    # and the measured served record goes stale. This makes that drift fail CI.
    path = Path("results/published/vram_served_Qwen3-4B-Instruct-2507-AWQ-4bit.json")
    llm_run = json.loads(path.read_text())["runs"][0]["llm"]
    chosen = serve.choose_util(
        llm_run["repo"], llm_run["revision"], override=None, records=serve.RECORDS
    )
    assert llm_run["gpu_memory_utilization"] == chosen.util
    assert llm_run["tool_calling"] is True
    assert "--tool-call-parser" in llm_run["argv"]


# --- measured headroom (Cyril, 2026-11-27): only on an exact match, else "computed" -------------


GPU = serve.Gpu("NVIDIA GeForce RTX 4060 Laptop GPU", 8188)


def lookup(
    records: Path, gpu: serve.Gpu = GPU, vllm: str | None = "0.19.1"
) -> serve.Measured | None:
    return serve.measured_headroom(REPO, FULL, 0.61, records=records, gpu=gpu, vllm_version=vllm)


def write_vram(
    folder: Path,
    name: str = "served",
    *,
    repo: str = REPO,
    revision: str = FULL,
    util: float = 0.61,
    tools: bool = True,
    dirty: bool = False,
    headroom: int | None = 550,
    fits: bool = True,
    commit: str = "27d99b6",
    gpu: str = GPU.name,
    total: int = GPU.total_mib,
    vllm: str | None = None,
) -> Path:
    from agent import vllm_server

    argv = vllm_server.server_command(repo, revision, util, tools=tools)[1:]
    llm_run = {"repo": repo, "revision": revision, "gpu_memory_utilization": util, "argv": argv}
    run = {"llm": llm_run, "mib": {"headroom": headroom}, "fits": fits}
    path = folder / f"vram_{name}.json"
    record = {"commit": commit, "dirty": dirty, "gpu": gpu, "total_mib": total, "runs": [run]}
    if vllm is not None:
        record["vllm_version"] = vllm
    path.write_text(json.dumps(record))
    return path


def test_exact_match_gives_the_measured_headroom(tmp_path: Path) -> None:
    path = write_vram(tmp_path)
    assert lookup(tmp_path) == serve.Measured(550, str(path), "27d99b6", vllm_checked=False)


@pytest.mark.parametrize(
    "mismatch",
    [
        {"revision": "c" * 40},  # another commit of the model
        {"repo": "Qwen/Qwen2.5-3B-Instruct-AWQ"},
        {"util": 0.59},  # measured at another utilization
        {"tools": False},  # measured without the served tool-calling flags
        {"dirty": True},  # measured on a dirty tree
        {"headroom": None, "fits": False},  # a stage failed, so there is no headroom
        {"headroom": 300, "fits": False},  # measured, but under the 500 MiB floor
        {"gpu": "NVIDIA GeForce RTX 4070 Laptop GPU"},  # measured on another GPU
        {"total": 12282},  # same name, different memory size
    ],
)
def test_any_mismatch_falls_back_to_computed(tmp_path: Path, mismatch: dict[str, object]) -> None:
    write_vram(tmp_path, **mismatch)  # type: ignore[arg-type]
    assert lookup(tmp_path) is None


def test_a_record_without_argv_falls_back(tmp_path: Path) -> None:
    # Records before 27d99b6 don't say what vLLM was launched with, so they can't match.
    path = write_vram(tmp_path)
    record = json.loads(path.read_text())
    del record["runs"][0]["llm"]["argv"]
    path.write_text(json.dumps(record))
    assert lookup(tmp_path) is None


@pytest.mark.parametrize(
    "drop",
    [
        ("dirty",),
        ("commit",),
        ("gpu",),
        ("total_mib",),
        ("runs",),
        ("runs", 0, "mib"),
        ("runs", 0, "fits"),
        ("runs", 0, "llm"),
    ],
)
def test_an_older_shaped_record_falls_back_instead_of_crashing(
    tmp_path: Path, drop: tuple[str | int, ...]
) -> None:
    # Review should-fix #1: a print-only feature must never stop `make serve` from starting.
    path = write_vram(tmp_path)
    record = json.loads(path.read_text())
    parent = record
    for key in drop[:-1]:
        parent = parent[key]
    del parent[drop[-1]]
    path.write_text(json.dumps(record))
    # A record without its commit has no provenance, so it isn't trusted either.
    assert lookup(tmp_path) is None


def test_two_matches_use_the_lower_headroom(tmp_path: Path) -> None:
    write_vram(tmp_path, "a", headroom=550)
    low = write_vram(tmp_path, "b", headroom=520, commit="abc1234")
    found = lookup(tmp_path)
    assert found == serve.Measured(520, str(low), "abc1234", vllm_checked=False)


def test_budget_prints_measured_headroom_when_given() -> None:
    chosen = serve.ServedUtil(0.61, 0.59, "x.json", False)
    measured = serve.Measured(550, "v.json", "27d99b6", vllm_checked=True)
    text = serve.budget(chosen, total_mib=8188, measured=measured)
    assert "measured headroom 550 MiB" in text and "v.json" in text and "27d99b6" in text
    assert "computed, not measured" not in text and "vLLM version not checked" not in text


def test_same_vllm_version_is_checked(tmp_path: Path) -> None:
    write_vram(tmp_path, vllm="0.19.1")
    found = lookup(tmp_path, vllm="0.19.1")
    assert found is not None and found.vllm_checked is True


def test_other_vllm_version_falls_back(tmp_path: Path) -> None:
    write_vram(tmp_path, vllm="0.18.0")
    assert lookup(tmp_path, vllm="0.19.1") is None


def test_record_without_vllm_version_says_it_was_not_checked(tmp_path: Path) -> None:
    # Cyril: records from before e899a69 still count, but the line must say what wasn't checked.
    write_vram(tmp_path)
    found = lookup(tmp_path)
    assert found is not None and found.vllm_checked is False
    chosen = serve.ServedUtil(0.61, 0.59, "x.json", False)
    assert "vLLM version not checked" in serve.budget(chosen, total_mib=8188, measured=found)


def test_published_served_record_matches_what_serve_runs() -> None:
    chosen = serve.choose_util(REPO, FULL, override=None, records=serve.RECORDS)
    path = serve.RECORDS / "vram_served_Qwen3-4B-Instruct-2507-AWQ-4bit.json"
    record = json.loads(path.read_text())
    gpu = serve.Gpu(record["gpu"], record["total_mib"])  # CI has no GPU: take the record's own
    found = serve.measured_headroom(
        REPO, FULL, chosen.util, records=serve.RECORDS, gpu=gpu, vllm_version="0.19.1"
    )
    assert found is not None
    assert found.source == "results/published/vram_served_Qwen3-4B-Instruct-2507-AWQ-4bit.json"
