"""GPU co-residency smoke test (PLAN M0.3): vLLM + STT + TTS loaded together, one round trip.

Starts vLLM first (it pre-allocates VRAM), then loads faster-whisper on the GPU and the TTS on
CPU, runs TTS -> STT -> LLM -> TTS once, and samples GPU memory and system RAM throughout.
Writes results/vram.json. Every number comes from nvidia-smi or /proc/meminfo via bench/gpu_mem.

Per-model MiB is the change in *total* GPU memory, because vLLM's memory belongs to its engine
child process, not the server PID. vLLM's figure is what it reserved, not what its weights need.

Usage: python -m bench.vram --beam-size N --llm REPO@REVISION:UTIL [--llm ...]
       [--allow-dirty] [--tools]
Refuses a dirty tree unless --allow-dirty (bench/provenance.py).
GPU-only imports live inside functions: CI does not install the gpu dependency group.
"""

import argparse
import datetime as dt
import importlib.metadata
import json
import multiprocessing
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from agent import vllm_server
from bench import provenance
from bench.gpu_mem import (
    MIN_HEADROOM_MIB,
    check_record,
    fits,
    headroom_mib,
    parse_compute_apps,
    parse_gpu_query,
    parse_meminfo,
)

STT_REPO = "Systran/faster-whisper-large-v3"
STT_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
STT_COMPUTE = "int8_float16"
TTS_REPO = "Indus-Labs/indus-pocket-tts"
TTS_REVISION = "39899d3c1dcda47a7ef167cd222419feb86abeb9"
TTS_RATE = 24000
PROMPT_HI = "नमस्ते, मुझे कल शाम डॉक्टर शर्मा के साथ अपॉइंटमेंट चाहिए।"
SYSTEM = "You are a clinic receptionist. Reply in one short Hindi sentence, in Devanagari script."
GPU_QUERY = "--query-gpu=name,memory.total,memory.used,driver_version"
APPS_QUERY = "--query-compute-apps=pid,used_memory"
CSV = "--format=csv,noheader,nounits"
# Review #2: spikes shorter than one interval can be missed, so the peak is a lower bound and
# headroom an upper bound. Stated in every record.
SAMPLE_INTERVAL_S = 1.0


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def gpu_used_mib() -> int:
    return parse_gpu_query(sh("nvidia-smi", GPU_QUERY, CSV)).used_mib


def ram_mib() -> tuple[int, int]:
    return parse_meminfo(Path("/proc/meminfo").read_text())


class Peaks:
    """Samples (GPU used MiB, RAM used MiB) every interval_s and keeps the maximum of each.

    Fails closed: if any sample raises, sampling stops and `error` is set, so the run fails.
    A peak that was never measured stays None, never 0 (0 would read as 8,188 MiB headroom).
    """

    def __init__(self, read: Callable[[], tuple[int, int]], interval_s: float = 1.0) -> None:
        self.gpu: int | None = None
        self.ram: int | None = None
        self.error: str | None = None
        self._read = read
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                gpu, ram = self._read()
            except Exception as exc:
                self.error = "sampler: " + describe_error(exc, home=str(Path.home()))
                return
            self.gpu = gpu if self.gpu is None else max(self.gpu, gpu)
            self.ram = ram if self.ram is None else max(self.ram, ram)
            self._stop.wait(self._interval_s)

    def __enter__(self) -> "Peaks":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def describe_error(exc: BaseException, *, home: str) -> str:
    """One-line error for the record, home directory replaced by ~ and capped at 300 chars."""
    return f"{type(exc).__name__}: {exc}".replace(home, "~")[:300]


def combine_errors(stage_error: str | None, sampler_error: str | None) -> str | None:
    """Keep both errors for the record; None only if neither happened."""
    return "; ".join(e for e in (stage_error, sampler_error) if e) or None


def summarize(
    at: dict[str, int],
    *,
    peak_gpu: int | None,
    total: int,
    error: str | None,
    sampler_error: str | None = None,
) -> tuple[dict[str, int | None], bool]:
    """Turn per-stage GPU readings into the record's MiB block and the fits verdict.

    `at` holds total GPU MiB read after each stage that completed. Stages never reached are None.
    Fits only for the full set with no stage or sampler error, a measured peak, and headroom >=
    MIN_HEADROOM_MIB. A sampler error alone fails the run: the peak may be too low (review #1).
    """
    error = combine_errors(error, sampler_error)

    def change(after: str, before: str) -> int | None:
        return at[after] - at[before] if after in at and before in at else None

    mib: dict[str, int | None] = {**at}
    mib["llm"] = change("after_llm", "baseline")
    mib["stt"] = change("after_stt", "after_llm")
    mib["tts_measured_delta"] = change("after_tts", "after_stt")
    mib["peak_used"] = peak_gpu
    if "after_llm" in at:
        mib["room_left_after_llm"] = headroom_mib(total_mib=total, peak_used_mib=at["after_llm"])
    # Headroom only means something for the full set under load: every stage loaded, the round
    # trip finished, and the peak was measured. Otherwise null; room_left_after_llm still shows
    # what the LLM alone left (review #5).
    headroom = None
    if error is None and peak_gpu is not None and "after_tts" in at:
        headroom = headroom_mib(total_mib=total, peak_used_mib=peak_gpu)
    mib["headroom"] = headroom
    return mib, headroom is not None and fits(headroom)


def load_stt() -> Any:
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    path = snapshot_download(STT_REPO, revision=STT_REVISION, local_files_only=True)
    return WhisperModel(path, device="cuda", compute_type=STT_COMPUTE)


def load_tts() -> Any:
    """Return speak(text) -> float32 numpy audio at 24 kHz, via the repo's speak.py.

    speak.py and normalizer/ were read before first use (2026-11-25): stdlib + torch +
    pocket_tts only, safetensors weights, no network or process calls. load() sets
    torch.set_num_threads(1) process-wide; stream() yields torch tensors; sampling uses
    temperature 0.3, so the seed makes the round trip repeatable.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    sys.path.insert(0, snapshot_download(TTS_REPO, revision=TTS_REVISION, local_files_only=True))
    from speak import load, stream  # pyright: ignore[reportMissingImports]

    torch.manual_seed(0)
    model, voice = load(int8=True)
    return lambda text: torch.cat([c.reshape(-1) for c in stream(model, voice, text)]).numpy()


def to_16k(audio: Any) -> Any:
    """Linear resample 24 kHz -> 16 kHz. Enough for a smoke test; STT quality is M4.3's job."""
    import numpy as np  # pyright: ignore[reportMissingImports]

    n = len(audio) * 16000 // TTS_RATE
    positions = np.linspace(0, len(audio), n, endpoint=False)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def vllm_version() -> str | None:
    """The installed vLLM version, recorded so agent.serve can tell a stale measurement."""
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def llm_record(
    repo: str, revision: str, util: float, argv: list[str], *, log: str
) -> dict[str, object]:
    """The LLM part of a run record, from the command vLLM was actually started with."""
    return {
        "repo": repo,
        "revision": revision,
        "gpu_memory_utilization": util,
        "max_model_len": vllm_server.MAX_MODEL_LEN,
        "cuda_graphs": "--enforce-eager" not in argv,
        "tool_calling": "--enable-auto-tool-choice" in argv,  # --tools: as agent.serve starts it
        "argv": argv,
        "log": log,  # file name only (no home path), so a -2/-3 log maps to its run
    }


def log_path(log_dir: Path, repo: str, commit: str, util: float, tools: bool, dirty: bool) -> Path:
    """Where this run's vLLM log goes: named by model, commit, utilization and tool flags, and
    never an existing file (Cyril, 2026-11-27), so ADR evidence logs are not overwritten. A
    dirty-tree run gets `_dirty`: its code is not that commit's (review #2)."""
    stem = f"vllm_{repo.split('/')[-1]}_{commit}_u{util}" + ("_tools" if tools else "")
    stem += "_dirty" if dirty else ""
    path, n = log_dir / f"{stem}.log", 1
    while path.exists():
        n += 1
        path = log_dir / f"{stem}-{n}.log"
    return path


def run_one(
    llm: str, beam: int, total: int, log_dir: Path, tools: bool, commit: str, dirty: bool
) -> dict[str, object]:
    repo, _, rest = llm.partition("@")
    revision, _, util = rest.partition(":")
    if parse_compute_apps(sh("nvidia-smi", APPS_QUERY, CSV)):
        raise RuntimeError("another compute process is using the GPU; stop it first")
    at = {"baseline": gpu_used_mib()}  # total GPU MiB used, read after each stage
    log = log_path(log_dir, repo, commit, float(util), tools, dirty)
    transcript, reply, reply_samples, error = "", "", 0, None
    proc = vllm_server.start(repo, revision, float(util), log, tools=tools)
    if not isinstance(proc.args, list):
        raise TypeError(f"expected vLLM's argv as a list, got {type(proc.args).__name__}")
    argv = [str(arg) for arg in proc.args][1:]  # what was launched, minus the interpreter path
    try:
        with Peaks(lambda: (gpu_used_mib(), ram_mib()[1]), SAMPLE_INTERVAL_S) as peaks:
            # A model set that does not fit (e.g. STT out of memory) is a result, not a crash:
            # record how far it got and the error, so ADR-0003 sees every option.
            try:
                vllm_server.wait_ready(proc)
                at["after_llm"] = gpu_used_mib()
                stt = load_stt()
                at["after_stt"] = gpu_used_mib()
                speak = load_tts()
                at["after_tts"] = gpu_used_mib()
                audio = to_16k(speak(PROMPT_HI))
                segments, _ = stt.transcribe(audio, language="hi", beam_size=beam)
                transcript = " ".join(s.text.strip() for s in segments)
                reply = vllm_server.chat(SYSTEM, transcript)
                reply_samples = len(speak(reply))
            except Exception as exc:
                error = describe_error(exc, home=str(Path.home()))
    finally:
        vllm_server.stop(proc)

    mib, ok = summarize(at, peak_gpu=peaks.gpu, total=total, error=error, sampler_error=peaks.error)
    error = combine_errors(error, peaks.error)
    return {
        "llm": llm_record(repo, revision, float(util), argv, log=log.name),
        "stt": {
            "repo": STT_REPO,
            "revision": STT_REVISION,
            "device": "cuda",
            "compute_type": STT_COMPUTE,
            "beam_size": beam,
        },
        # vram_mib 0 is the expectation for a CPU model, not a measurement (review #7);
        # the measured change is mib.tts_measured_delta.
        "tts": {
            "repo": TTS_REPO,
            "revision": TTS_REVISION,
            "device": "cpu",
            "vram_mib": 0,
            "measured": False,
        },
        "mib": mib,
        "fits": ok,
        "error": error,
        "peak_ram_used_mib": peaks.ram,
        "round_trip": {
            "transcript": transcript,
            "reply": reply,
            "reply_tts_samples": reply_samples,
            "ok": error is None and bool(transcript and reply and reply_samples),
        },
    }


def run_isolated(
    llm: str, beam: int, total: int, log_dir: Path, tools: bool, commit: str, dirty: bool
) -> dict[str, object]:
    """Run one model set in a fresh process, so its STT model and CUDA context are freed on exit
    and the next run starts from an idle GPU (the baseline check would otherwise see us)."""
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as pool:
        return pool.submit(run_one, llm, beam, total, log_dir, tools, commit, dirty).result()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU co-residency smoke test (PLAN M0.3)")
    parser.add_argument("--llm", action="append", required=True, help="REPO@REVISION:UTIL")
    parser.add_argument("--beam-size", type=int, required=True, help="production STT beam size")
    parser.add_argument("--out", type=Path, default=Path("results/vram.json"))
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree (recorded)")
    parser.add_argument("--tools", action="store_true", help="start vLLM as agent.serve does")
    args = parser.parse_args()
    start = provenance.provenance(allow_dirty=args.allow_dirty)  # before touching the GPU
    commit, dirty = str(start["commit"]), bool(start["dirty"])  # name the vLLM logs

    import torch  # pyright: ignore[reportMissingImports]  # only for torch's CUDA build version

    gpu = parse_gpu_query(sh("nvidia-smi", GPU_QUERY, CSV))
    runs = [
        run_isolated(llm, args.beam_size, gpu.total_mib, args.out.parent, args.tools, commit, dirty)
        for llm in args.llm
    ]
    record = {
        **provenance.finish(start),
        "command": "python -m bench.vram " + " ".join(sys.argv[1:]),
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": gpu.name,
        "driver": gpu.driver,
        "torch_cuda": torch.version.cuda,
        "vllm_version": vllm_version(),
        "total_mib": gpu.total_mib,
        "ram_total_mib": ram_mib()[0],
        "min_headroom_mib": MIN_HEADROOM_MIB,
        "sampling": {
            "interval_s": SAMPLE_INTERVAL_S,
            "note": "peak_used is sampled, so spikes shorter than interval_s can be missed: "
            "headroom is an upper bound",
        },
        "runs": runs,
    }
    check_record(record, home=str(Path.home()))
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
