# Related open-source work — notes (2026-11-25)

What an external research report turned up that bears on open decisions, after checking.
**Verified** means Cyril checked it against the source named (outside the Claude session,
2026-11-25). **Unverified** means it is only claimed somewhere and must not be used as a number
in our docs. Nothing here is measured on our hardware; our own numbers live in `results/`.

## LLM — input to ADR-0003 (option 2)

**Qwen3-4B-Instruct-2507** — source: huggingface.co/Qwen/Qwen3-4B-Instruct-2507

| Claim | Status |
|---|---|
| Licence Apache-2.0 (base model) | verified |
| AWQ build is community-made: `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` (W4A16) | verified |
| `Qwen/Qwen3-4B-AWQ` is the thinking-mode model, **not** the instruct model; don't use it | verified |
| Quality of the community AWQ quant (vs the base model) | **unverified** |
| Licence of the community quant repo | **unverified** — check before download |
| Fits beside Whisper large-v3 on our GPU; Hindi quality | **unmeasured** — PLAN task under ADR-0003 |

Why it matters: the Qwen2.5-3B fits (2,758 MiB headroom) but is non-commercial and gave a bad
Hindi reply; a 4B Apache-2.0 instruct model is the obvious next candidate.

## STT — input to ADR-0002 and M4.3

**Tara** (Trelis) — source: huggingface.co/Trelis/tara

| Claim | Status |
|---|---|
| Whisper-large-v3 architecture (fine-tune) | verified |
| Vistaar benchmark mean WER 12.06 | verified as published, but **self-measured** by the authors |
| Licence | **unverified** |
| 21% WER on GramVaani (telephone speech) | **unverified** |
| Converts to CTranslate2 (would slot into faster-whisper at our measured ~1,600 MiB) | **unverified** |

**Named entities matter more than WER for booking.** The report cited a Hindi/Telugu
named-entity benchmark where overall WER improved but names and dates were still lost
(unverified, but our own smoke test already turned "कल" into "कर"). PLAN M4.3 now also scores
entity errors.

## Evaluation — input to ADR-0007 (build vs adopt)

**Pipecat Evals** — source: docs.pipecat.ai/pipecat/evals/overview

| Claim | Status |
|---|---|
| Two modes: scripted scenarios and simulated (LLM-played) callers | verified |
| Tool-call assertions | verified |
| Text and audio modes | verified |
| Simulated callers don't script interruption timing | verified |
| Minimum Pipecat version that has Evals | **unverified** — check before pinning Pipecat |

Cyril's leaning (not yet an ADR): adopt the runner; build ourselves the DB-state scoring,
scripted interruption timing, the Hinglish personas, P95/CI statistics and the gate.

## Patterns worth borrowing (no dependency)

- **Slot-ID booking** (from LiveKit Agents' front-desk example, per the report): `book` takes a
  slot ID returned by `check_slots`, never a free-text date/time from the LLM, and rejects an
  unknown ID with a tool error. Adopted for PLAN M2.2.
- **Interrupted turns never commit** — the report's invariant matches CLAUDE.md rule 7 and M4.4.

## Not adopted (conflicts with decided stack; would need an ADR and a measured reason)

LiveKit Agents as the runtime, PostgreSQL + Redis, Kokoro TTS, llama.cpp instead of vLLM.
The report's VRAM estimates (e.g. 7–8B AWQ "5–7 GB") are contradicted by our measurement
(Qwen2.5-7B-AWQ +7,128 MiB at its minimum utilization); use `results/published/vram.json`.
