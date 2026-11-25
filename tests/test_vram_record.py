"""Unit tests for the CPU-testable parts of bench/vram.py: the peak sampler and the run record.

Review of M0.3 (#1, #6): a failed nvidia-smi sample must never turn into a pass. The record must
fail closed: any sampler error fails the run, an unmeasured peak is None (never 0), and stages
that were never reached are None. Error messages must not carry the home directory.
"""

import threading
from collections.abc import Callable

from bench.vram import Peaks, combine_errors, describe_error, summarize

TOTAL = 8188
FULL = {"baseline": 11, "after_llm": 3653, "after_stt": 5252, "after_tts": 5252}


def scripted_read(values: list[tuple[int, int] | Exception]) -> tuple[Callable, threading.Event]:
    """Fake sampler read: returns or raises values in order; sets `done` on the last one.

    The tests wait on `done` instead of sleeping, so they don't depend on thread scheduling
    (second review #3). After the last value it keeps returning it.
    """
    done = threading.Event()
    calls = 0

    def read() -> tuple[int, int]:
        nonlocal calls
        value = values[min(calls, len(values) - 1)]
        calls += 1
        if calls >= len(values):
            done.set()
        if isinstance(value, Exception):
            raise value
        return value

    return read, done


def test_sampler_failure_on_first_sample_leaves_peak_unmeasured() -> None:
    read, done = scripted_read([RuntimeError("nvidia-smi: Unknown Error")])
    with Peaks(read, interval_s=0.001) as peaks:
        assert done.wait(timeout=5)
    assert peaks.gpu is None
    assert peaks.ram is None
    assert peaks.error is not None and "Unknown Error" in peaks.error


def test_sampler_failure_after_good_samples_is_still_an_error() -> None:
    read, done = scripted_read([(5000, 8000), (5400, 8100), RuntimeError("failed sample")])
    with Peaks(read, interval_s=0.001) as peaks:
        assert done.wait(timeout=5)
    assert peaks.gpu == 5400
    assert peaks.error is not None and "failed sample" in peaks.error


def test_sampler_keeps_the_maximum() -> None:
    read, done = scripted_read([(5000, 8000), (5430, 7000), (5100, 8118)])
    with Peaks(read, interval_s=0.001) as peaks:
        assert done.wait(timeout=5)
    assert (peaks.gpu, peaks.ram, peaks.error) == (5430, 8118, None)


def test_full_run_fits() -> None:
    mib, ok = summarize(FULL, peak_gpu=5430, total=TOTAL, error=None)
    assert (mib["llm"], mib["stt"], mib["tts_measured_delta"]) == (3642, 1599, 0)
    assert mib["headroom"] == TOTAL - 5430
    assert ok is True


def test_simulated_nvidia_smi_failure_does_not_fit() -> None:
    # The sampler died before it measured anything: the old code read this as peak 0, headroom 8188.
    mib, ok = summarize(FULL, peak_gpu=None, total=TOTAL, error="sampler: RuntimeError: boom")
    assert ok is False
    assert mib["peak_used"] is None


def test_unmeasured_peak_without_error_still_does_not_fit() -> None:
    _, ok = summarize(FULL, peak_gpu=None, total=TOTAL, error=None)
    assert ok is False


def test_missing_stages_are_none_and_do_not_fit() -> None:
    at = {"baseline": 11, "after_llm": 7139}  # STT ran out of memory
    mib, ok = summarize(at, peak_gpu=7234, total=TOTAL, error="RuntimeError: out of memory")
    assert (mib["stt"], mib["tts_measured_delta"]) == (None, None)
    assert mib["llm"] == 7128
    assert ok is False


def test_error_message_has_home_dir_stripped() -> None:
    exc = FileNotFoundError("/home/someone/.cache/huggingface/hub/x not found")
    message = describe_error(exc, home="/home/someone")
    assert "/home/someone" not in message
    assert message.startswith("FileNotFoundError: ~/.cache")


def test_error_message_is_bounded() -> None:
    assert len(describe_error(RuntimeError("x" * 1000), home="/home/someone")) <= 300


def test_incomplete_set_has_no_headroom() -> None:
    # Review #5: the 7B record said headroom 954 beside fits false; 954 was measured with no STT
    # loaded, so anything checking headroom >= 500 alone would have passed it.
    at = {"baseline": 11, "after_llm": 7139}
    mib, ok = summarize(at, peak_gpu=7234, total=TOTAL, error="RuntimeError: out of memory")
    assert mib["headroom"] is None
    assert mib["room_left_after_llm"] == TOTAL - 7139
    assert ok is False


def test_all_stages_but_an_error_later_has_no_headroom() -> None:
    # Loaded everything, then the round trip failed: the peak never saw inference load.
    mib, ok = summarize(FULL, peak_gpu=5300, total=TOTAL, error="URLError: refused")
    assert mib["headroom"] is None
    assert ok is False


def test_sampler_error_alone_fails_the_run() -> None:
    # Second review #2: the sampler died after a few good samples, so the peak is set but too
    # low; with no stage error, the run must still fail.
    mib, ok = summarize(FULL, peak_gpu=5300, total=TOTAL, error=None, sampler_error="sampler: x")
    assert ok is False
    assert mib["headroom"] is None


def test_both_errors_are_kept() -> None:
    assert combine_errors("OOM", "sampler: x") == "OOM; sampler: x"
    assert combine_errors(None, "sampler: x") == "sampler: x"
    assert combine_errors("OOM", None) == "OOM"
    assert combine_errors(None, None) is None
