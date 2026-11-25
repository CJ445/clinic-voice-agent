"""Find the lowest --gpu-memory-utilization at which vLLM starts with >= 1x concurrency (PLAN M0.3).

Cyril's rule: for each LLM, the utilization to record is the lowest one where vLLM starts and its
log reports at least 1x concurrency at max-model-len 4096. A real probe is a full vLLM start
(~50 s), so the search bisects on a 0.01 grid between a `lo` and a `hi`, and records every trial.
It assumes more memory never makes vLLM fail. Writes results/min_util_<model>.json.

Usage: python -m bench.min_util --llm REPO@REVISION --lo 0.25 --hi 0.40
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from bench import vllm_server
from bench.gpu_mem import check_record, parse_compute_apps, parse_gpu_query

STEP = 0.01
Probe = Callable[[float], float | None]


def find_min_util(probe: Probe, *, lo: float, hi: float) -> tuple[float | None, list[dict]]:
    """Return (lowest passing utilization or None, trials). Passing means probe(u) >= 1.0."""
    trials: list[dict] = []

    def passes(grid: int) -> bool:
        util = round(grid * STEP, 2)
        concurrency = probe(util)
        ok = concurrency is not None and concurrency >= 1.0
        trials.append({"util": util, "concurrency": concurrency, "ok": ok})
        return ok

    lo_i, hi_i = round(lo / STEP), round(hi / STEP)
    if not passes(hi_i):
        return None, trials
    if passes(lo_i):
        return round(lo_i * STEP, 2), trials
    while hi_i - lo_i > 1:  # invariant: lo_i fails, hi_i passes
        mid = (lo_i + hi_i) // 2
        if passes(mid):
            hi_i = mid
        else:
            lo_i = mid
    return round(hi_i * STEP, 2), trials


def _wait_gpu_idle(timeout_s: float = 120) -> None:
    query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    deadline = time.monotonic() + timeout_s
    while parse_compute_apps(subprocess.run(query, capture_output=True, text=True).stdout):
        if time.monotonic() > deadline:
            raise RuntimeError("GPU still busy after the last vLLM probe was stopped")
        time.sleep(2)


def vllm_probe(repo: str, revision: str, log_dir: Path) -> Probe:
    """Real probe: start vLLM at `util`, read its concurrency line, always stop it."""

    def probe(util: float) -> float | None:
        _wait_gpu_idle()
        log = log_dir / f"vllm_{repo.split('/')[-1]}_u{util:.2f}.log"
        proc = vllm_server.start(repo, revision, util, log)
        try:
            vllm_server.wait_ready(proc)
        except (RuntimeError, TimeoutError):
            return None
        finally:
            vllm_server.stop(proc)
        text = log.read_text(errors="replace")
        return vllm_server.parse_max_concurrency(text, max_model_len=vllm_server.MAX_MODEL_LEN)

    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Lowest workable vLLM utilization (PLAN M0.3)")
    parser.add_argument("--llm", required=True, help="REPO@REVISION")
    parser.add_argument("--lo", type=float, required=True, help="a utilization expected to fail")
    parser.add_argument("--hi", type=float, required=True, help="a utilization expected to pass")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    repo, _, revision = args.llm.partition("@")

    def sh(*cmd: str) -> str:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

    gpu_query = "--query-gpu=name,memory.total,memory.used,driver_version"
    gpu = parse_gpu_query(sh("nvidia-smi", gpu_query, "--format=csv,noheader,nounits"))
    best, trials = find_min_util(vllm_probe(repo, revision, args.out_dir), lo=args.lo, hi=args.hi)
    record = {
        "commit": sh("git", "rev-parse", "--short", "HEAD").strip(),
        "dirty": bool(sh("git", "status", "--porcelain").strip()),
        "command": "python -m bench.min_util " + " ".join(sys.argv[1:]),
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": gpu.name,
        "driver": gpu.driver,
        "llm": {"repo": repo, "revision": revision, "cuda_graphs": True},
        "max_model_len": vllm_server.MAX_MODEL_LEN,
        "criterion": "vLLM ready and 'Maximum concurrency' >= 1.0x at max_model_len",
        "min_gpu_memory_utilization": best,
        "trials": trials,
    }
    check_record(record, home=str(Path.home()))
    out = args.out_dir / f"min_util_{repo.split('/')[-1]}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"min_gpu_memory_utilization": best, "trials": trials}, indent=2))


if __name__ == "__main__":
    main()
