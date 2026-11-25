"""Start, health-check, query and stop a local `vllm serve` process.

Used by bench/vram.py now and by the M3.1 serve script later. vLLM pre-allocates VRAM when it
starts, so callers must start it before loading any other GPU model (gpu-budget skill).
Stdlib only, so it type-checks in CI where the gpu dependency group is not installed.
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 8011
MAX_MODEL_LEN = 4096
BASE_URL = f"http://127.0.0.1:{PORT}"
# stop(): SIGTERM grace, then SIGKILL grace, then the bound on reaping the server process.
TERM_GRACE_S = 60
KILL_GRACE_S = 10
WAIT_S = 5


# vLLM 0.19.1 prints e.g. "Maximum concurrency for 4,096 tokens per request: 3.08x".
_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def parse_max_concurrency(log_text: str, *, max_model_len: int) -> float | None:
    """Return the last reported concurrency for max_model_len, or None if vLLM never printed it."""
    found = None
    for tokens, value in _CONCURRENCY.findall(log_text):
        if int(tokens.replace(",", "")) == max_model_len:
            found = float(value)
    return found


_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def require_full_sha(revision: str) -> str:
    """Return revision if it is a full 40-character commit SHA; raise ValueError otherwise.

    A short revision resolves offline only if the local HF cache happens to hold a refs/<short>
    file, so a record that names one can't be reproduced elsewhere (Cyril, M3.1).
    """
    if not _FULL_SHA.fullmatch(revision):
        raise ValueError(f"revision {revision!r} is not a full 40-character commit SHA")
    return revision


def port_in_use(port: int) -> bool:
    """True if something already accepts connections on 127.0.0.1:port."""
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def server_command(repo: str, revision: str, util: float) -> list[str]:
    """The `vllm serve` command line. No --enforce-eager, so CUDA graphs stay on.

    Bound to 127.0.0.1: the bench server has no authentication, so it must not listen on
    every network interface while it runs.
    """
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        repo,
        "--revision",
        revision,
        "--served-model-name",
        "llm",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]


def start(repo: str, revision: str, util: float, log: Path) -> subprocess.Popen[bytes]:
    """Launch the OpenAI-compatible server in its own session.

    Refuses if the port is already taken: a stale server there would answer /health and
    wait_ready would accept it as ours (review #9). The check is best-effort (another process
    could take the port between the check and the launch); fine for a single-user bench.
    """
    if port_in_use(PORT):
        raise RuntimeError(f"port {PORT} is already in use; stop the server holding it first")
    # New session so stop() can kill vLLM's engine child process too, not just the server.
    return subprocess.Popen(
        server_command(repo, revision, util),
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )


def wait_ready(proc: subprocess.Popen[bytes], timeout_s: float = 900) -> None:
    """Poll /health until it answers. Raise if vLLM exits first or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode} before becoming ready")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"vLLM not ready after {timeout_s:.0f}s")


def _signal_and_wait(proc: subprocess.Popen[bytes], sig: int, grace_s: float) -> bool:
    """Send sig to the process group; return True once nothing is left in it."""
    deadline = time.monotonic() + grace_s
    try:
        os.killpg(proc.pid, sig)
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader so it doesn't linger as a zombie in the group
            os.killpg(proc.pid, 0)  # raises ProcessLookupError once the group is empty
            time.sleep(0.2)
    except ProcessLookupError:
        return True
    return False


def stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM vLLM's whole process group, then SIGKILL whatever is left after TERM_GRACE_S.

    Always signals the group, even if the server process already exited: its engine child
    holds the VRAM and can outlive it (review #3). Raises RuntimeError rather than returning
    while the group may still hold VRAM, and never waits without a limit (review M0.3 #1).
    """
    gone = _signal_and_wait(proc, signal.SIGTERM, grace_s=TERM_GRACE_S)
    if not gone:
        gone = _signal_and_wait(proc, signal.SIGKILL, grace_s=KILL_GRACE_S)
    if not gone:
        raise RuntimeError(f"vLLM process group {proc.pid} survived SIGKILL; its VRAM may be held")
    try:
        proc.wait(timeout=WAIT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"vLLM process {proc.pid} was not reaped after {WAIT_S} s") from exc


def chat(system: str, user: str, max_tokens: int = 64) -> str:
    """One greedy chat completion against the running server."""
    body = {
        "model": "llm",
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"].strip()
