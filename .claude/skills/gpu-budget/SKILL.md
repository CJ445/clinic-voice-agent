---
name: gpu-budget
description: VRAM budgeting rules for the single 8 GB RTX 4060. Use before adding, swapping, or reconfiguring any model that runs on the GPU.
---
# GPU budget (8 GB)

Before any GPU change, state the expected VRAM delta and check it against `results/vram.json`.

- vLLM pre-allocates: always pass `--gpu-memory-utilization` explicitly (start 0.55) and
  `--max-model-len 4096`. Start vLLM FIRST, then load STT/TTS, or they will OOM.
- Keep ≥ 500 MiB headroom; the desktop compositor also uses VRAM on a laptop.
- Prefer CPU for VAD, turn detection, and TTS unless a benchmark proves GPU is needed.
- Fallback order if it doesn't fit: shrink `max-model-len` → smaller LLM (4B) → STT to CPU int8.
  Record whichever you choose as an ADR amendment.
- Measure with `nvidia-smi --query-gpu=memory.used --format=csv -l 1` during a full call,
  not at idle.
