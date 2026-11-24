---
name: latency-bench
description: Protocol for measuring voice-agent latency on this repo. Use whenever measuring, reporting, or optimizing P50/P95 response time or per-stage timings.
---
# Latency benchmark protocol

## Metric (ADR-0005)
`t_end` = timestamp of the last caller audio frame containing speech (from VAD, not from the
turn detector). `t_start` = timestamp the first agent audio frame is written to the transport.
Headline = `t_start - t_end`. Always also report the turn-detection wait separately.

## Stage events (must all be logged per turn)
`user_speech_end`, `turn_detected`, `stt_final`, `llm_first_token`, `tool_call_start`,
`tool_call_end`, `tts_first_audio`, `agent_audio_out`. Use `time.perf_counter_ns()` in one
process; never mix clocks across processes without an offset.

## Procedure
1. Close other GPU apps. Record `nvidia-smi --query-gpu=name,driver_version,clocks.sm --format=csv`.
2. Warm up: 5 turns discarded.
3. Run the fixed bench set: 30 scenarios × 3 repeats, audio mode, 8 kHz input, fixed seed.
4. Report P50, P95, and n. With n < 100 turns, P95 is noisy — show the bootstrap 95% CI.
5. Write `results/latency_<git-short-hash>.json` with hardware, versions (`uv pip freeze`
   hash), command, and raw per-turn data.

## Rules
- A change "helps" only if the P95 CI of after is below the CI of before.
- Never compare runs from different hardware states (laptop on battery vs plugged in!).
  Check the power profile and record it.
- Report VRAM peak (`torch.cuda.max_memory_allocated` + `nvidia-smi` for vLLM).
