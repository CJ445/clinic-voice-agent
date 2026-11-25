"""Unit tests for the CPU-testable part of agent/vllm_server.py: reading vLLM's startup log.

The minimum-utilization search (PLAN M0.3) accepts a --gpu-memory-utilization only if vLLM's log
shows at least 1x concurrency at max-model-len 4096. The fixture lines are copied verbatim from a
real vLLM 0.19.1 start of Qwen2.5-3B-Instruct-AWQ at 0.40 on the RTX 4060 (2026-11-25).
"""

import os
import socket
import subprocess
from pathlib import Path

import pytest

from agent import vllm_server
from agent.vllm_server import parse_max_concurrency, port_in_use

REAL_LOG = (
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [gpu_worker.py:436] "
    "Available KV cache memory: 0.43 GiB\n"
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [kv_cache_utils.py:1319] "
    "GPU KV cache size: 12,624 tokens\n"
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [kv_cache_utils.py:1324] "
    "Maximum concurrency for 4,096 tokens per request: 3.08x\n"
)


def test_reads_real_vllm_line_with_thousands_separator() -> None:
    assert parse_max_concurrency(REAL_LOG, max_model_len=4096) == pytest.approx(3.08)


def test_none_when_vllm_died_before_sizing_kv_cache() -> None:
    # F-001: the engine can exit before this line is ever printed.
    assert parse_max_concurrency("EngineCore failed to start.\n", max_model_len=4096) is None


def test_ignores_line_for_a_different_max_model_len() -> None:
    # A stale log from a run with another --max-model-len must not satisfy the check.
    line = "Maximum concurrency for 8,192 tokens per request: 1.54x\n"
    assert parse_max_concurrency(line, max_model_len=4096) is None


def test_below_one_is_reported_not_rounded() -> None:
    line = "Maximum concurrency for 4,096 tokens per request: 0.97x\n"
    assert parse_max_concurrency(line, max_model_len=4096) == pytest.approx(0.97)


def test_last_matching_line_wins() -> None:
    # One log file per run is the rule, but if two starts were appended, trust the latest.
    log = REAL_LOG + "Maximum concurrency for 4,096 tokens per request: 1.20x\n"
    assert parse_max_concurrency(log, max_model_len=4096) == pytest.approx(1.20)


def test_port_in_use_detects_a_listener() -> None:
    # Review #9: a stale server on vLLM's port would answer /health and pass wait_ready.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert port_in_use(port) is True
    assert port_in_use(port) is False


def test_start_refuses_when_port_is_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        monkeypatch.setattr(vllm_server, "PORT", port)
        with pytest.raises(RuntimeError, match=str(port)):
            vllm_server.start("some/model", "rev", 0.5, tmp_path / "vllm.log")
    assert not (tmp_path / "vllm.log").exists()  # refused before launching anything


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_kills_a_surviving_child_after_the_leader_exited() -> None:
    # Review #3: if the API server dies but its engine child (which holds the VRAM) survives,
    # stop() used to return early and never kill the group.
    proc = subprocess.Popen(["sh", "-c", "sleep 60 & exit 0"], start_new_session=True)
    proc.wait()  # the leader is gone; `sleep` is still running in its process group
    assert _group_alive(proc.pid)
    vllm_server.stop(proc)
    assert not _group_alive(proc.pid)


def test_stop_kills_a_running_leader() -> None:
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    vllm_server.stop(proc)
    assert proc.returncode is not None
    assert not _group_alive(proc.pid)


def test_stop_is_safe_when_everything_already_exited() -> None:
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    vllm_server.stop(proc)  # must not raise ProcessLookupError


def test_stop_raises_when_sigkill_leaves_the_group_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Review M0.3 #1: a process stuck in a CUDA call can survive SIGKILL for a while; stop() used to
    # ignore that and return (or hang) while the VRAM was still held.
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    real_killpg = os.killpg

    def signals_do_nothing(pgid: int, sig: int) -> None:
        if sig == 0:
            real_killpg(pgid, 0)  # existence checks are real, so the group looks alive

    monkeypatch.setattr(vllm_server.os, "killpg", signals_do_nothing)
    monkeypatch.setattr(vllm_server, "TERM_GRACE_S", 0.3)
    monkeypatch.setattr(vllm_server, "KILL_GRACE_S", 0.3)
    try:
        with pytest.raises(RuntimeError, match=f"{proc.pid} survived SIGKILL"):
            vllm_server.stop(proc)
    finally:
        real_killpg(proc.pid, 9)
        proc.wait()


def test_stop_final_wait_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    timeouts: list[float | None] = []

    def never_reaped(timeout: float | None = None) -> int:
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired("vllm", timeout or 0)

    monkeypatch.setattr(proc, "wait", never_reaped)
    with pytest.raises(RuntimeError, match=f"{proc.pid} was not reaped"):
        vllm_server.stop(proc)
    assert timeouts == [vllm_server.WAIT_S]


def test_server_binds_to_localhost_only() -> None:
    # Second review: without --host, the unauthenticated bench server listened on every
    # interface while it ran.
    cmd = vllm_server.server_command("some/model", "rev", 0.5)
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--port") + 1] == str(vllm_server.PORT)


def test_full_sha_is_accepted() -> None:
    sha = "b37d1ccb77e27a70fe4222d995fda27940eead5a"
    assert vllm_server.require_full_sha(sha) == sha


@pytest.mark.parametrize(
    "revision",
    [
        "b37d1ccb77",
        "main",
        "B37D1CCB77E27A70FE4222D995FDA27940EEAD5A",
        "b37d1ccb77e27a70fe4222d9" + "z" * 16,
    ],
)
def test_short_or_named_revisions_are_refused(revision: str) -> None:
    # Cyril, M3.1: a 10-character revision only resolved offline because the local HF cache had a
    # refs/<short> file, so records must name the full commit.
    with pytest.raises(ValueError, match="40"):
        vllm_server.require_full_sha(revision)


def test_tool_calling_flags_only_when_asked() -> None:
    # The bench's measured command line must not change; the serve script asks for tools=True.
    # hermes reads the <tool_call> JSON blocks the Qwen2.5 and Qwen3-2507 templates emit.
    plain = vllm_server.server_command("some/model", "rev", 0.5)
    served = vllm_server.server_command("some/model", "rev", 0.5, tools=True)
    assert "--enable-auto-tool-choice" not in plain and "--tool-call-parser" not in plain
    assert served[: len(plain)] == plain
    assert served[len(plain) :] == ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
