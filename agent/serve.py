"""Serve the dialogue LLM with vLLM, at the GPU share measured for that exact model (PLAN M3.1).

The model is an argument because the LLM choice is still open (ADR-0003). The
--gpu-memory-utilization is not typed by whoever runs it: it is the minimum that
bench.min_util published for the same REPO@REVISION, plus UTIL_MARGIN. An unmeasured model is
refused, so nothing can be served at a share nobody checked. `--util` overrides the lookup and is
printed as an override.

Usage: python -m agent.serve --llm REPO@FULL_40_CHAR_SHA [--util X]   (or `make serve LLM=...`)
Prints the VRAM budget, starts vLLM with tool calling, waits for /health, sends one warm-up
completion, then serves until Ctrl-C or SIGTERM and stops vLLM's whole process group.
"""

import argparse
import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from agent import vllm_server

RECORDS = Path("results/published")
LOG = Path("results/vllm_serve.log")
# Policy (Cyril, M3.1; to be recorded in ADR-0003): serve 0.02 above the measured minimum, so a
# run-to-run wobble in vLLM's startup memory doesn't leave the KV cache short of one request.
UTIL_MARGIN = 0.02
# Policy (Cyril, review #8): the highest utilization the serve script accepts.
MAX_UTIL = 0.95
# A recorded revision shorter than this is too ambiguous to match a full SHA by prefix.
MIN_PREFIX = 10


@dataclass(frozen=True)
class ServedUtil:
    """The utilization to serve at and where it came from."""

    util: float
    measured: float | None  # the published minimum; None for an override
    source: str  # the record's path, or "--util"
    override: bool


def _matches(recorded: str, revision: str) -> bool:
    return recorded == revision or (len(recorded) >= MIN_PREFIX and revision.startswith(recorded))


def choose_util(repo: str, revision: str, *, override: float | None, records: Path) -> ServedUtil:
    """Pick the utilization for repo@revision (a full SHA). Raise LookupError if unmeasured."""
    vllm_server.require_full_sha(revision)
    if override is not None:
        return _in_range(ServedUtil(util=override, measured=None, source="--util", override=True))
    found = []
    for path in sorted(records.glob("min_util_*.json")):
        record = json.loads(path.read_text())
        if record["llm"]["repo"] == repo and _matches(record["llm"]["revision"], revision):
            found.append((path, record["min_gpu_memory_utilization"]))
    if not found:
        raise LookupError(f"no measured minimum for {repo}@{revision}; run bench.min_util first")
    if len(found) > 1:
        names = ", ".join(str(path) for path, _ in found)
        raise LookupError(f"{len(found)} records match {repo}@{revision}: {names}")
    path, measured = found[0]
    if measured is None:
        raise LookupError(f"{path}: the search found no workable utilization")
    return _in_range(ServedUtil(round(measured + UTIL_MARGIN, 2), measured, str(path), False))


def _in_range(chosen: ServedUtil) -> ServedUtil:
    if not 0 < chosen.util <= MAX_UTIL:
        raise ValueError(f"gpu-memory-utilization {chosen.util} is outside 0 < u <= {MAX_UTIL}")
    return chosen


def budget(chosen: ServedUtil, *, total_mib: int) -> str:
    """Say what share of VRAM vLLM gets and the most that can be left for STT/TTS and headroom."""
    reserved = round(chosen.util * total_mib)
    if chosen.override:
        origin = "--util override"
    else:
        origin = f"measured {chosen.measured} + margin {UTIL_MARGIN} ({chosen.source})"
    return (
        f"gpu-memory-utilization {chosen.util}: {origin}\n"
        f"vLLM's share is {reserved} MiB of {total_mib} MiB, so at most {total_mib - reserved} MiB "
        "left for STT, TTS and headroom (computed, not measured: the vLLM process also uses memory "
        "outside its share, so the real remainder is smaller; make vram measures it)"
    )


def total_mib() -> int:
    """Total VRAM of GPU 0, from nvidia-smi."""
    query = ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    return int(subprocess.run(query, check=True, capture_output=True, text=True).stdout.split()[0])


def _sigterm_as_interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def stop_uninterruptibly(proc: subprocess.Popen[bytes]) -> None:
    """Stop vLLM with Ctrl-C and SIGTERM ignored, so a second one can't abandon a group that
    still holds VRAM (review #4). The previous handlers come back afterwards."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        vllm_server.stop(proc)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the dialogue LLM with vLLM (PLAN M3.1)")
    parser.add_argument("--llm", required=True, help="REPO@FULL_40_CHAR_SHA")
    parser.add_argument("--util", type=float, help="override the published minimum + margin")
    args = parser.parse_args()
    repo, _, revision = args.llm.partition("@")
    chosen = choose_util(repo, revision, override=args.util, records=RECORDS)
    print(budget(chosen, total_mib=total_mib()), flush=True)
    proc = vllm_server.start(repo, revision, chosen.util, LOG, tools=True)
    signal.signal(signal.SIGTERM, _sigterm_as_interrupt)
    try:
        vllm_server.wait_ready(proc)
        # Warm-up: the first request pays one-off costs that a caller's first turn should not.
        vllm_server.chat("Reply with one word.", "Hello", max_tokens=1)
        print(f"ready: {vllm_server.BASE_URL}/v1 model 'llm' = {repo}@{revision} (log {LOG})")
        sys.stdout.flush()
        code = proc.wait()
        sys.exit(f"vLLM exited on its own with code {code}; see {LOG}")
    except KeyboardInterrupt:
        print("stopping vLLM", flush=True)
    finally:
        stop_uninterruptibly(proc)


if __name__ == "__main__":
    main()
