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


# vLLM 0.19.1 prints e.g. "Maximum concurrency for 4,096 tokens per request: 3.08x".
_CONCURRENCY = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


def parse_max_concurrency(log_text: str, *, max_model_len: int) -> float | None:
    """Return the last reported concurrency for max_model_len, or None if vLLM never printed it."""
    found = None
    for tokens, value in _CONCURRENCY.findall(log_text):
        if int(tokens.replace(",", "")) == max_model_len:
            found = float(value)
    return found


def port_in_use(port: int) -> bool:
    """True if something already accepts connections on 127.0.0.1:port."""
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def start(repo: str, revision: str, util: float, log: Path) -> subprocess.Popen[bytes]:
    """Launch the OpenAI-compatible server. No --enforce-eager, so CUDA graphs stay on.

    Refuses if the port is already taken: a stale server there would answer /health and
    wait_ready would accept it as ours (review #9).
    """
    if port_in_use(PORT):
        raise RuntimeError(f"port {PORT} is already in use; stop the server holding it first")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        repo,
        "--revision",
        revision,
        "--served-model-name",
        "llm",
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(util),
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]
    # New session so stop() can kill vLLM's engine child process too, not just the server.
    return subprocess.Popen(
        cmd,
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


def stop(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the whole process group; SIGKILL it if it has not exited after 60 s."""
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


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
