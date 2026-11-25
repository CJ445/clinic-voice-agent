"""Unit tests for bench/gpu_mem.py: nvidia-smi and /proc/meminfo parsing, headroom, record checks.

These run on CPU in CI. Fixtures are copied from real output on the RTX 4060 laptop
(driver 570.207), including its broken `[N/A]` / 590 W power fields.
"""

import pytest

from bench.gpu_mem import (
    MIN_HEADROOM_MIB,
    GpuInfo,
    check_record,
    fits,
    headroom_mib,
    parse_compute_apps,
    parse_gpu_query,
    parse_meminfo,
    parse_mib,
)

# nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits
GPU_QUERY = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 11, 570.207\n"

# /proc/meminfo, the two lines we use (values in kB)
MEMINFO = "MemTotal:       15640988 kB\nMemFree:  999 kB\nMemAvailable:   10176736 kB\n"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" 8188 ", 8188), ("0", 0), ("[N/A]", None), (" [N/A] ", None)],
)
def test_parse_mib(raw: str, expected: int | None) -> None:
    assert parse_mib(raw) == expected


def test_parse_mib_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_mib("8188 MiB")


def test_parse_gpu_query() -> None:
    assert parse_gpu_query(GPU_QUERY) == GpuInfo(
        name="NVIDIA GeForce RTX 4060 Laptop GPU", total_mib=8188, used_mib=11, driver="570.207"
    )


def test_parse_gpu_query_fails_loudly_when_no_gpu() -> None:
    # Earlier today nvidia-smi printed nothing usable; a smoke run must stop, not record 0 MiB.
    with pytest.raises(ValueError):
        parse_gpu_query("")


def test_parse_gpu_query_rejects_unknown_total() -> None:
    with pytest.raises(ValueError):
        parse_gpu_query("NVIDIA GeForce RTX 4060 Laptop GPU, [N/A], 11, 570.207\n")


def test_parse_compute_apps() -> None:
    text = "4242, 5100\n4343, 1500\n"
    assert parse_compute_apps(text) == {4242: 5100, 4343: 1500}


def test_parse_compute_apps_empty_when_nothing_runs() -> None:
    # Real output with only Xorg (a graphics, not compute, process) is an empty string.
    assert parse_compute_apps("") == {}


def test_parse_compute_apps_keeps_unknown_memory_as_none() -> None:
    # Some laptop drivers report per-process memory as [N/A]; callers must then use totals.
    assert parse_compute_apps("4242, [N/A]\n") == {4242: None}


def test_parse_meminfo() -> None:
    # used = MemTotal - MemAvailable, both converted kB -> MiB (floor).
    assert parse_meminfo(MEMINFO) == (15274, 5336)


@pytest.mark.parametrize(("peak_used", "expected"), [(7689, 499), (7688, 500), (7687, 501)])
def test_headroom_mib(peak_used: int, expected: int) -> None:
    assert headroom_mib(total_mib=8188, peak_used_mib=peak_used) == expected


@pytest.mark.parametrize(("headroom", "ok"), [(499, False), (500, True), (501, True)])
def test_fits_at_the_500_mib_boundary(headroom: int, ok: bool) -> None:
    assert MIN_HEADROOM_MIB == 500
    assert fits(headroom) is ok


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "commit": "2a85ab5",
        "command": "make vram",
        "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "driver": "570.207",
    }
    record.update(overrides)
    return record


def test_check_record_accepts_complete_record() -> None:
    check_record(_record(), home="/home/tester")


@pytest.mark.parametrize("key", ["commit", "command", "gpu", "driver"])
def test_check_record_requires_provenance(key: str) -> None:
    # Gate G4: every published number carries command + commit hash + hardware.
    with pytest.raises(ValueError, match=key):
        check_record(_record(**{key: ""}), home="/home/tester")


def test_check_record_rejects_home_paths() -> None:
    # results/published/vram.json is committed; it must not leak the username or home dir.
    bad = _record(notes={"model_dir": "/home/tester/.cache/huggingface/hub"})
    with pytest.raises(ValueError, match="home"):
        check_record(bad, home="/home/tester")
