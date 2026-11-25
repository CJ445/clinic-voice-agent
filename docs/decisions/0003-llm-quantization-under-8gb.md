# ADR-0003: LLM + quantization under 8 GB

- **Status:** proposed
- **Date:** 2026-11-25
- **Decider:** Cyril

## Context

The LLM (vLLM), STT and TTS must run together on one RTX 4060 Laptop GPU (8,188 MiB), with
at least 500 MiB headroom (PLAN M0.3). vLLM must start first because it pre-allocates memory.
TTS (Indus Pocket TTS) runs on CPU and measured +0 MiB, so the real contest is LLM vs STT.

All numbers below come from `results/published/vram.json`: commit `1c71cb2`, clean tree; the
exact command (full revision SHAs) is in the file. The utilizations are each model's minimum
from `bench.min_util` (lowest value where vLLM starts with ≥ 1x concurrency at max-model-len
4096; `results/published/min_util_*.json`, commit `0a764e5`, clean). CUDA graphs on. STT is
faster-whisper large-v3, `int8_float16`, beam 5 (provisional).

**Peaks are sampled at 1 Hz, so every headroom figure is an upper bound.** The same 3B set
peaked at 5,430 MiB at `4c1d4cc` and 5,336 MiB at `1c71cb2` (every pre-inference reading was
identical); the derivation below uses the larger.

| | Qwen2.5-3B-Instruct-AWQ @ 0.36 | Qwen2.5-7B-Instruct-AWQ @ 0.80 |
|---|---|---|
| GPU MiB after vLLM | 3,653 | 7,139 |
| Room left for STT | 4,535 | 1,049 |
| Whisper large-v3 load | +1,599 | **out of memory** (`RuntimeError: CUDA failed with error out of memory`) |
| Peak GPU MiB / headroom | 5,336 / **2,758 — fits** (conservative: lowest across two runs; this run 2,852) | 7,234 / `null` (no STT loaded) — **does not fit** |
| Peak system RAM (of 15,274 MiB) | 8,955 | 8,600 (STT never loaded) |
| Round trip | completed | not reached |
| Licence | Qwen Research License — research/evaluation only, **non-commercial** | Apache-2.0 |

vLLM's own log lines at those utilizations (`results/vllm_*_u0.36.log`, `_u0.80.log`; these logs
are **local only**, not committed, so the figures below can't be checked against a commit):

| | 3B | 7B |
|---|---|---|
| Weights | 1.95 GiB | 5.2 GiB |
| CUDA graph memory (estimated) | 0.62 GiB | 0.61 GiB |
| KV cache | 0.19 GiB, 5,408 tokens | 0.29 GiB, 5,440 tokens |

Other forces:

- **Hindi quality.** The 3B's one smoke-test reply to a Devanagari prompt was romanised and
  near-nonsense: `namaste_m_m_sharma_sham_ka_sharam_aapni_sharam_aapni_chahi_sey`. That is one
  sample at temperature 0 with a throwaway prompt, not an evaluation. **The 7B's Hindi is
  unverified**: it never reached the round trip.
- **vLLM uses more than its setting.** The 3B at 0.36 is ≈ 2,948 MiB by the setting but measured
  +3,642 MiB (CUDA context and non-torch memory). Budgets must use measured MiB.
- **Overlap with ADR-0002.** Which STT model to use is ADR-0002, decided after the M4.3 WER
  benchmark. Any option here that changes the STT model is provisional until then.

### What the 7B is short by (derived from the numbers above, not measured)

To fit, after-LLM MiB + STT load + inference peak + 500 must be ≤ 8,188. With the 3B's
larger measured inference peak (5,430 − 5,252 = +178 MiB, at `4c1d4cc`) as a stand-in: 7,139 + 1,599 + 178 + 500 =
9,416, so the 7B set is **≈ 1,230 MiB over**. The 7B's own inference peak is unverified.

## Options

1. **7B on GPU, Whisper large-v3 on CPU (int8).**
   - VRAM: STT moves off the GPU; the 7B alone peaked at 7,234 MiB, leaving 954 MiB with
     no STT on the GPU (derived from the record; not a headroom measurement of this option).
   - Latency: **unverified.** CPU large-v3 with beam 5 on the Ryzen 9 8945HS (16 threads) is
     likely far slower than GPU. It must be measured against the M7 latency budget before
     this can be accepted.
   - RAM: **unverified.** System RAM already peaked at 7.4 GiB of 15 GiB without STT. The
     pinned large-v3 snapshot is 2.88 GiB on disk (float16, converted to int8 at load); its
     working set on CPU is not measured.
   - Licence: Apache-2.0 (7B) + MIT (Whisper). Cost: about one `make vram` variant plus a CPU
     STT latency run.

2. **A 3–4B LLM instead of the 7B (the Qwen 3B, or another model of that size).**
   - VRAM: measured for the Qwen 3B: fits with 2,758 MiB headroom (conservative upper bound), leaving room for a larger
     KV cache or a second STT pass.
   - Quality: the Qwen 3B's one Hindi sample was unusable (see Context). Other 3–4B instruct
     models with AWQ builds and Hindi ability are **unverified**: none has been tried here.
   - Licence: the Qwen 3B is **non-commercial**. Fine for a portfolio/research build; blocks any
     real clinic deployment. Another model's licence would need checking.
   - Cost: for another model, a download (~2–3 GB), a `bench.min_util` search and a
     `make vram` run; a fair Hindi comparison needs the M3/M5 eval, which doesn't exist yet.

3. **Keep the 7B and Whisper on GPU, shrink the footprint.** Levers, with what is known:
   - `--enforce-eager` (CUDA graphs off): vLLM estimates 0.61 GiB (≈ 625 MiB) of graph memory.
     Whether that lowers the minimum utilization by the same amount is **unverified**; it needs
     a new `bench.min_util` search. Costs decode speed (unmeasured on this GPU).
   - Shorter `--max-model-len` (e.g. 2048): KV is ≈ 56 KiB/token for the 7B (0.29 GiB / 5,440
     tokens), so halving one request's need saves only ≈ 110 MiB (derived). Also limits how long
     a call's conversation can get before truncation.
   - Smaller Whisper (medium, or a distilled/turbo variant): VRAM and Hindi WER on this setup are
     **unverified**; overlaps ADR-0002.
   - From the derived ≈ 1,230 MiB shortfall, eager mode plus a 2,048 context (≈ 735 MiB) does not
     close the gap on its own; a smaller Whisper would also be needed. Each combination needs its
     own `make vram` run to count.
   - Cost: one to three extra `min_util` + `make vram` runs (~10–15 min each), no new downloads
     except the smaller Whisper.

## Decision

TODO(Cyril)

## Consequences

TODO(Cyril) — depends on the option. Whatever is chosen, these must now be tested:
- the chosen set's `make vram` result with ≥ 500 MiB headroom (re-run on any version bump);
- Hindi reply quality of the chosen LLM (M3/M5 eval), not just "non-empty";
- the chosen LLM's licence recorded against the intended use.

## How we'd know we were wrong

TODO(Cyril)

## Evidence

- `results/published/vram.json` — commit `1c71cb2`, both model sets, per-stage MiB, peak RAM.
  The earlier run at `4c1d4cc` is in git history (`git show f5e252d:results/published/vram.json`).
- `results/published/min_util_Qwen2.5-3B-Instruct-AWQ.json`, `results/published/min_util_Qwen2.5-7B-Instruct-AWQ.json`
  — commit `0a764e5`, clean tree, every trial.
- vLLM logs `results/vllm_Qwen2.5-*-AWQ_u*.log` (gitignored; local only).
- Licences: model cards at the pinned revisions (`3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd`,
  `b25037543e9394b818fdfca67ab2a00ecc7dd641`), checked 2026-11-25.
- Qwen3-4B-Instruct-2507 AWQ (`cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` at
  `b37d1ccb77e27a70fe4222d995fda27940eead5a`; Apache-2.0 per its model card, base model's
  LICENSE file Apache 2.0, checked 2026-11-25):
  - `results/published/min_util_Qwen3-4B-Instruct-2507-AWQ-4bit.json`: commit `c9d35d8`, clean
    tree, every trial (its record names the 10-character revision; written before full SHAs
    were required).
  - `results/published/vram_Qwen3-4B-Instruct-2507-AWQ-4bit.json`: commit `c9d35d8`, clean, at
    the measured minimum, without tool calling.
  - `results/published/vram_served_Qwen3-4B-Instruct-2507-AWQ-4bit.json`: commit `674fa44`, clean,
    the configuration `agent.serve` runs: `llm.tool_calling` true (the same vLLM command line)
    and the utilization from `agent.serve.choose_util` (measured minimum + `UTIL_MARGIN`).
- Serve-script policies to weigh in the Decision (`agent/serve.py`, Cyril, M3.1): `UTIL_MARGIN`
  (served utilization = measured minimum + 0.02) and `MAX_UTIL` (0.95 cap).
