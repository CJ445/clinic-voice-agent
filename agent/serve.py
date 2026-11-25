"""Serve the dialogue LLM with vLLM, at the GPU share measured for that exact model (PLAN M3.1).

The model is an argument because the LLM choice is still open (ADR-0003). The
--gpu-memory-utilization is not typed by whoever runs it: it is the minimum that
bench.min_util published for the same REPO@REVISION, plus UTIL_MARGIN. An unmeasured model is
refused, so nothing can be served at a share nobody checked. `--util` overrides the lookup and is
printed as an override.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from agent import vllm_server

RECORDS = Path("results/published")
# Policy (Cyril, M3.1; to be recorded in ADR-0003): serve 0.02 above the measured minimum, so a
# run-to-run wobble in vLLM's startup memory doesn't leave the KV cache short of one request.
UTIL_MARGIN = 0.02
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
        return ServedUtil(util=override, measured=None, source="--util", override=True)
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
    return ServedUtil(round(measured + UTIL_MARGIN, 2), measured, str(path), override=False)


def budget(chosen: ServedUtil, *, total_mib: int) -> str:
    """Say how much VRAM vLLM will reserve and what is left for STT/TTS and headroom."""
    reserved = round(chosen.util * total_mib)
    if chosen.override:
        origin = "--util override"
    else:
        origin = f"measured {chosen.measured} + margin {UTIL_MARGIN} ({chosen.source})"
    return (
        f"gpu-memory-utilization {chosen.util}: {origin}\n"
        f"vLLM reserves {reserved} MiB of {total_mib} MiB; {total_mib - reserved} MiB left for "
        "STT, TTS and headroom (computed, not measured)"
    )
