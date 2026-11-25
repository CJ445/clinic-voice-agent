# PLAN — milestones, tasks, quality gates

~19 working days. One branch + one PR per task (yes, even solo: the PR description is your
review record). Tag each finished milestone: `git tag m3-done`.

## Quality gates

| Gate | When | What must be true | Enforced by |
|---|---|---|---|
| **G1 Code** | every commit | ruff, pyright, unit tests pass | PostToolUse hook, `make check`, CI |
| **G2 Task** | every PR | reviewer subagent run; you can explain the diff; docs updated | `/review`, `/explain`, PR checklist |
| **G3 Milestone** | end of milestone | exit criteria below met, with evidence linked | you, + `m*-done` tag |
| **G4 Claim** | any number in README/RESULTS | produced by a script, has command + commit hash + hardware | `make bench` / `make eval-*` write JSON |
| **G5 Regression** | every PR from M6 on | eval scores within noise band of baseline | `make gate` in CI on self-hosted GPU runner |

---

## M0 — Foundations (day 1)

- [x] M0.1 `uv init`, pyproject with ruff/pyright/pytest, Makefile targets wired, `.gitignore`
- [x] M0.2 GitHub Actions: `make check` on ubuntu-latest (CPU only)
- [x] M0.3 **GPU co-residency smoke test**: vLLM + STT + TTS loaded together, one round trip.
      Script `bench/vram.py` writes `results/vram.json` (per-model MiB, headroom).
- [ ] M0.4 Fill `docs/DESIGN.md` §1–3 yourself (problem, constraints, architecture sketch)

**Exit (G3):** CI green on an empty-but-real package; `results/vram.json` shows all three
models fit with ≥ 500 MiB headroom, or an ADR records the fallback (smaller LLM / CPU STT).

## M1 — Decisions (day 1–2) — *you write these, Claude can only draft*

- [ ] ADR-0001 Pipecat vs hand-rolled asyncio pipeline
- [ ] ADR-0002 STT: faster-whisper vs IndicConformer (decide **after** M4.3 WER bench; record "provisional")
- [ ] ADR-0003 LLM model + quantization under 8 GB
  - [ ] Qwen3-4B-Instruct-2507 AWQ (`cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`, community quant):
        `bench.min_util` search + `make vram` run → input to ADR-0003 option 2
        (sources: `docs/research/2026-11-25-related-repos.md`)
- [ ] ADR-0004 LangGraph scope: logic only, hybrid single-LLM-node + guard nodes
- [ ] ADR-0005 Latency metric definition (what "start" and "end" mean, exactly)
- [ ] ADR-0006 Eval scoring: deterministic DB-state checks + LLM judge only for hallucination
- [ ] ADR-0007 Build vs adopt Pipecat Evals for the simulator (Cyril's leaning: adopt the runner;
      build DB-state scoring, scripted interruption timing, Hinglish personas, P95/CI stats, the gate)

**Exit:** each ADR has Context / Options / Decision / Consequences / "How we'd know we were wrong".

## M2 — Clinic backend + tools (day 2–3)

- [ ] M2.1 SQLite schema (doctors, slots, appointments, patients), deterministic `seed.py`
- [x] M2.2 Tools `check_slots`, `book`, `cancel` with Pydantic I/O schemas; `book` takes a slot ID
      returned by `check_slots` (never a free-text date/time) and rejects unknown IDs
- [ ] M2.3 Edge-case tests: double booking, past slot, cancel nonexistent, IST midnight boundary,
      idempotent retry of `book`
- [ ] M2.4 Date resolution helper ("next Friday", "parso", "kal shaam") + table-driven tests

**Exit:** tools 100% branch-covered; date helper passes ≥ 40 table cases incl. Hinglish.

## M3 — Agent, text mode (day 3–5)

- [ ] M3.1 vLLM serve script + health check; OpenAI client wrapper with timeouts
- [ ] M3.2 LangGraph: one LLM node with tools + guard nodes (confirm-before-mutate, slot-grounding)
- [ ] M3.3 Text REPL `python -m agent.cli` for manual calls
- [ ] M3.4 20 hand-written golden conversations as pytest (text mode)

**Exit:** 20/20 golden conversations pass at temperature 0, 3 runs each.

## M4 — Voice pipeline + honest baseline (day 5–7)

- [ ] M4.1 Pipecat pipeline: Silero VAD → STT → LangGraph → TTS, browser WebRTC client
- [ ] M4.2 Per-stage timestamps (events per ADR-0005) → `results/latency_*.json`
- [ ] M4.3 `bench/stt_wer.py`: Whisper vs IndicConformer on your own 50 recorded clinic utterances
      (8 kHz resampled); score WER **and** entity errors (dates, times, names, phone numbers)
      → finalize ADR-0002
- [ ] M4.4 Barge-in: cancel in-flight LLM/TTS; **test: no DB mutation from a cancelled turn**
- [ ] M4.5 Record baseline P50/P95 (reasonable v1, not sandbagged) → `make bench`

**Exit:** a real Hindi call books an appointment end-to-end; baseline latency in RESULTS with hash.

## M5 — Caller simulator + scoring (day 8–11)

- [ ] M5.1 Persona schema (YAML): goal, behaviours, language mix, date ambiguity, interrupt plan,
      expected DB end-state, expected tool calls
- [ ] M5.2 ~40 templates × variations → 200 scenarios (generated, then **you spot-check 30**)
- [ ] M5.3 Caller loop, text mode
- [ ] M5.4 Scorers: task success (DB diff), tool-call accuracy, latency, hallucination judge
- [ ] M5.5 **Validate the judge**: hand-label 30 transcripts; report judge-vs-you agreement
- [ ] M5.6 Audio mode (persona TTS → agent → STT back), 30-scenario nightly subset
- [ ] M5.7 Scorer unit tests on known-good and known-bad transcripts

**Exit:** `make eval-text` runs 200 scenarios < 15 min; judge agreement ≥ 85% (or limitation logged).

## M6 — CI regression gate (day 11–12)

- [ ] M6.1 Self-hosted GitHub Actions runner on the 4060 box (label `gpu`)
- [ ] M6.2 `ci/gate.py`: compares metrics vs `ci/baseline.json`, noise bands from 3 repeated runs
- [ ] M6.3 Workflows: PR → text eval + gate; nightly → audio subset
- [ ] M6.4 **Prove it works**: open a PR with a deliberately worse prompt; it must fail. Link it in README.

**Exit:** the sabotage PR is red, a normal PR is green.

## M7 — Latency work (day 12–16) — one optimization per PR

Each PR: hypothesis → change → `make bench` before/after → `make eval-text` → RESULTS row.
- [ ] M7.1 LLM token streaming + sentence-chunked TTS
- [ ] M7.2 Streaming/partial STT
- [ ] M7.3 vLLM prefix caching for system prompt + tool schemas
- [ ] M7.4 int8 STT / AWQ LLM (VRAM + latency + quality all reported)
- [ ] M7.5 Warm-up, CUDA graphs, turn-detection timeout tuning with Smart Turn
- [ ] M7.6 Measure LangGraph overhead on its own

**Exit:** P95 improvement table with every row reproducible; no eval metric regressed beyond its band.

## M8 — Ship (day 17–19)

- [ ] M8.1 README: problem, demo GIF, architecture diagram, results table, reproduce section,
      known limitations, "How I used AI"
- [ ] M8.2 Demo video (≤ 2 min): Hindi call, interruption, unclear date, CI blocking a bad PR
- [ ] M8.3 GitHub Pages site from `docs/site/`
- [ ] M8.4 `/explain` every module; rewrite anything you couldn't explain; log rewrites in AI_USAGE
- [ ] M8.5 Final FAILURES + LIMITATIONS pass

## Out of scope (taste in scope — say no to these)

Real telephony, more than two languages, multi-clinic tenancy, auth, a UI beyond the demo client,
fine-tuning models. Note them under "Future work" in the README.
