# Design — clinic-voice-agent

> Write this yourself, in your own words, before M2 code. Claude may ask you questions
> about it, but should not write the prose. Keep it to ~2 pages. Update it when an ADR changes it.

## 1. Problem

Who calls, what they want, what "done" looks like for one call. (3–5 sentences.)

## 2. Constraints

- Hardware: one RTX 4060 (8 GB VRAM), Ubuntu, no paid GPU.
- Languages: Hindi, English, and code-mixed Hinglish.
- Phone-quality audio (8 kHz).
- Latency target: P95 end-of-user-speech → first agent audio < ___ ms (pick a number and justify it).
- Correctness: never double-book; never confirm a slot the tools did not return.

## 3. Architecture

Diagram (ASCII or Mermaid) of: browser client ⇄ WebRTC ⇄ Pipecat pipeline
(VAD → turn detector → STT → LangGraph agent ⇄ tools ⇄ SQLite → TTS) and the simulator
driving the same interface.

For each component: what it does, which model, where it runs (GPU/CPU), VRAM cost.

## 4. Key choices and why

Link to ADRs. One line each:
- Pipecat for transport/pipeline — ADR-0001
- STT model — ADR-0002
- LLM + quantization — ADR-0003
- LangGraph scope — ADR-0004
- Latency metric — ADR-0005
- Eval scoring — ADR-0006

## 5. What I rejected

| Option | Why not |
|---|---|
| | |

## 6. Evaluation design

How a call is scored, why each metric exists, how noise is handled, what blocks a merge.

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Models don't co-reside in 8 GB | | |
| Barge-in commits a booking | | |
| LLM judge disagrees with humans | | |
| Flaky CI from LLM nondeterminism | | |
