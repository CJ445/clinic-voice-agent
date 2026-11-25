---
name: gpu-budget
description: VRAM budgeting rules for the single 8 GB RTX 4060. Use before adding, swapping, or reconfiguring any model that runs on the GPU.
---
# GPU budget (8 GB)

Before any GPU change, state the expected VRAM delta and check it against `results/vram.json`.

- vLLM pre-allocates: always pass `--gpu-memory-utilization` explicitly, at the model's
  measured minimum, and `--max-model-len 4096`. Measured: **0.36** for Qwen2.5-3B-Instruct-AWQ,
  **0.80** for Qwen2.5-7B-Instruct-AWQ (searched by `python -m bench.min_util`:
  `results/published/min_util_*.json`, commit `0a764e5`; used in the co-residency run
  `results/published/vram.json`, commit `1c71cb2`). A new model gets its own `min_util` run
  first. Start vLLM FIRST, then load STT/TTS, or they will OOM.
- Keep ≥ 500 MiB headroom; the desktop compositor also uses VRAM on a laptop.
- Prefer CPU for VAD, turn detection, and TTS unless a benchmark proves GPU is needed.
- Fallback order if it doesn't fit: shrink `max-model-len` → smaller LLM (4B) → STT to CPU int8.
  Record whichever you choose as an ADR amendment.
- Measure with `nvidia-smi --query-gpu=memory.used --format=csv -l 1` during a full call,
  not at idle.
