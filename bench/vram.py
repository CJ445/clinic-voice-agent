"""GPU co-residency smoke test (PLAN M0.3): vLLM + STT + TTS loaded together, one round trip.

Starts vLLM first (it pre-allocates VRAM), then loads faster-whisper on the GPU and the TTS on
CPU, runs TTS -> STT -> LLM -> TTS once, and samples GPU memory and system RAM throughout.
Writes results/vram.json. Every number comes from nvidia-smi or /proc/meminfo via bench/gpu_mem.

Per-model MiB is the change in *total* GPU memory, because vLLM's memory belongs to its engine
child process, not the server PID. vLLM's figure is what it reserved, not what its weights need.

Usage: python -m bench.vram --beam-size N --llm REPO@REVISION:UTIL [--llm ...]
GPU-only imports live inside functions: CI does not install the gpu dependency group.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from bench import vllm_server
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
STT_REVISION = "edaa852ec7"
STT_COMPUTE = "int8_float16"
TTS_REPO = "Indus-Labs/indus-pocket-tts"
TTS_REVISION = "39899d3c1d"
TTS_RATE = 24000
PROMPT_HI = "नमस्ते, मुझे कल शाम डॉक्टर शर्मा के साथ अपॉइंटमेंट चाहिए।"
SYSTEM = "You are a clinic receptionist. Reply in one short Hindi sentence, in Devanagari script."
GPU_QUERY = "--query-gpu=name,memory.total,memory.used,driver_version"
APPS_QUERY = "--query-compute-apps=pid,used_memory"
CSV = "--format=csv,noheader,nounits"


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def gpu_used_mib() -> int:
    return parse_gpu_query(sh("nvidia-smi", GPU_QUERY, CSV)).used_mib


def ram_mib() -> tuple[int, int]:
    return parse_meminfo(Path("/proc/meminfo").read_text())


class Peaks:
    """Samples GPU used MiB and system RAM used MiB once a second; keeps the maximum of each."""

    def __init__(self) -> None:
        self.gpu = 0
        self.ram = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.gpu = max(self.gpu, gpu_used_mib())
            self.ram = max(self.ram, ram_mib()[1])
            self._stop.wait(1.0)

    def __enter__(self) -> "Peaks":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


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


def run_one(llm: str, beam: int, total: int, log_dir: Path) -> dict[str, object]:
    repo, _, rest = llm.partition("@")
    revision, _, util = rest.partition(":")
    if parse_compute_apps(sh("nvidia-smi", APPS_QUERY, CSV)):
        raise RuntimeError("another compute process is using the GPU; stop it first")
    mib = {"baseline": gpu_used_mib()}
    log = log_dir / f"vllm_{repo.split('/')[-1]}.log"
    proc = vllm_server.start(repo, revision, float(util), log)
    try:
        with Peaks() as peaks:
            vllm_server.wait_ready(proc)
            mib["after_llm"] = gpu_used_mib()
            stt = load_stt()
            mib["after_stt"] = gpu_used_mib()
            speak = load_tts()
            mib["after_tts"] = gpu_used_mib()
            audio = to_16k(speak(PROMPT_HI))
            segments, _ = stt.transcribe(audio, language="hi", beam_size=beam)
            transcript = " ".join(s.text.strip() for s in segments)
            reply = vllm_server.chat(SYSTEM, transcript)
            reply_samples = len(speak(reply))
    finally:
        vllm_server.stop(proc)
    mib["llm"] = mib["after_llm"] - mib["baseline"]
    mib["stt"] = mib["after_stt"] - mib["after_llm"]
    mib["tts_measured_delta"] = mib["after_tts"] - mib["after_stt"]
    mib["peak_used"] = peaks.gpu
    mib["room_left_after_llm"] = headroom_mib(total_mib=total, peak_used_mib=mib["after_llm"])
    mib["headroom"] = headroom_mib(total_mib=total, peak_used_mib=peaks.gpu)
    return {
        "llm": {
            "repo": repo,
            "revision": revision,
            "gpu_memory_utilization": float(util),
            "max_model_len": vllm_server.MAX_MODEL_LEN,
            "cuda_graphs": True,
        },
        "stt": {
            "repo": STT_REPO,
            "revision": STT_REVISION,
            "device": "cuda",
            "compute_type": STT_COMPUTE,
            "beam_size": beam,
        },
        "tts": {"repo": TTS_REPO, "revision": TTS_REVISION, "device": "cpu", "vram_mib": 0},
        "mib": mib,
        "fits": fits(mib["headroom"]),
        "peak_ram_used_mib": peaks.ram,
        "round_trip": {
            "transcript": transcript,
            "reply": reply,
            "reply_tts_samples": reply_samples,
            "ok": bool(transcript and reply and reply_samples),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU co-residency smoke test (PLAN M0.3)")
    parser.add_argument("--llm", action="append", required=True, help="REPO@REVISION:UTIL")
    parser.add_argument("--beam-size", type=int, required=True, help="production STT beam size")
    parser.add_argument("--out", type=Path, default=Path("results/vram.json"))
    args = parser.parse_args()

    import torch  # pyright: ignore[reportMissingImports]  # only for torch's CUDA build version

    gpu = parse_gpu_query(sh("nvidia-smi", GPU_QUERY, CSV))
    runs = [run_one(llm, args.beam_size, gpu.total_mib, args.out.parent) for llm in args.llm]
    record = {
        "commit": sh("git", "rev-parse", "--short", "HEAD").strip(),
        "dirty": bool(sh("git", "status", "--porcelain").strip()),
        "command": "python -m bench.vram " + " ".join(sys.argv[1:]),
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "gpu": gpu.name,
        "driver": gpu.driver,
        "torch_cuda": torch.version.cuda,
        "total_mib": gpu.total_mib,
        "ram_total_mib": ram_mib()[0],
        "min_headroom_mib": MIN_HEADROOM_MIB,
        "runs": runs,
    }
    check_record(record, home=str(Path.home()))
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(runs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
