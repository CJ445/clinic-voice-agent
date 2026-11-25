# HANDOVER

Newest entry on top. Written at the end of every Claude Code session with `/handover`,
then checked by Cyril. The next session starts by reading the top entry.

---

## 2026-11-25 — session 2 (M0.3)

- **Milestone / task:** M0.3 GPU co-residency smoke test (branch `m0/gpu-smoke`, not pushed)
- **Done:**
  - Commits `c97ca54` gpu dependency group (vLLM 0.19.1, faster-whisper) → `270dd1b`
    nvidia-smi/headroom helpers + tests → `e96bfb6` vLLM server wrapper → `0fa62b1`
    pocket-tts pinned git commit → `ac4319e` `make vram` → `7a5bb24` uv-managed Python (F-001)
    + `tests/test_env.py` guard → `c99518d` concurrency-line parser → `0a764e5`
    min-utilization search → `4c1d4cc` record a set that doesn't fit instead of crashing.
  - **First `/review`: 0 must-fix.** Fixed on Cyril's call: `37f61c3` fail closed on a sampler
    error (#1, #6) → `a4ab698` + `f6a5edf` headroom `null` unless the full set ran (#5) →
    `3866b39` port-free check (#9) → `0d5ae11` kill vLLM's process group (#3) → `65fe9cf` TTS
    `"measured": false` (#7) → `34d7dc7` 1 Hz / upper-bound note (#2) → `cea09af` full
    revision SHAs → `1c71cb2` publish `min_util` results (#8). F-002 logs two gate slips.
  - **Second `/review` (`f5e252d..HEAD`): 0 must-fix.** Fixed on Cyril's call: `e0c9f16` sampler
    error merged inside `summarize` (tested) → `ba975eb` sampler tests wait on a condition →
    `6c98636` bench vLLM bound to `127.0.0.1` (checked with one real start).
  - **Minimum utilization** (`python -m bench.min_util`, commit `0a764e5`, clean;
    `results/published/min_util_*.json`): Qwen2.5-3B-Instruct-AWQ **0.36**, Qwen2.5-7B-
    Instruct-AWQ **0.80**. Rule: vLLM ready and ≥ 1x concurrency at max-model-len 4096, CUDA
    graphs on. Every trial is in the JSON.
  - **Co-residency** (`results/published/vram.json`, commit `1c71cb2`, clean for the whole run;
    command in the file): 3B + Whisper large-v3 `int8_float16` beam 5 + Indus Pocket TTS (CPU,
    +0 MiB) **fits, 2,758 MiB headroom** (conservative: the lower of two runs; this run measured
    2,852; both are upper bounds because of 1 Hz sampling), round trip completed.
    7B: Whisper **out of memory** after vLLM left 1,049 MiB; `"fits": false`, `headroom: null`.
    The earlier run at `4c1d4cc` matched every pre-inference reading; only the sampled peak
    differed (5,430 vs 5,336 MiB). Checked: no home path or username in it.
  - **ADR-0003 drafted** (`docs/decisions/0003-llm-quantization-under-8gb.md`, status
    proposed): three options with measured vs unverified marked; Decision is `TODO(Cyril)`.
  - **NeMo dry-run** (not added to `uv.lock`, per Cyril): `nemo-toolkit` 3.0.0 resolves
    alongside vLLM 0.19.1; torch stays 2.10.0, 0 cu13 packages. It would change transformers
    5.24.0→5.24.1, fsspec 2026.9.0→2025.12.0, packaging 26.4→24.2, and add 66 packages.
    No conflict, so no ADR-0002 notes needed. **Unverified:** whether IndicConformer needs
    AI4Bharat's NeMo fork or an ONNX build instead of stock NeMo.
  - **TTS:** Indus Pocket TTS works on CPU (`torch.cuda.is_initialized()` False); Whisper
    transcribed its Hindi output close to the prompt, so Piper fallback not needed.
- **In progress (exact state):** nothing uncommitted on `m0/gpu-smoke`. The M1 PLAN additions
  and research notes are in `git stash` ("m1 plan+research"), for branch `m1/plan-research`
  after this PR merges.
- **Next step (first command to run):** second `/review` (new commits only); Cyril pushes and
  opens the PR. After merge: `chore/precommit-gate` (F-002 guard), then `m1/plan-research`.
- **Carry-forward from the first `/review` (not fixed in this PR, Cyril's call):**
  - #4: `vram.py` doesn't wait for the GPU to go idle between runs, a child-process crash
    still loses every run's numbers, and a hard crash can leave vLLM running. Fix: reuse
    `min_util._wait_gpu_idle` and turn per-run exceptions in `main` into error records.
  - Nits: `vllm_server.start` never closes the log file handle in the parent (one leaked fd per
    probe); `find_min_util` doesn't check `lo < hi`; one bisection is noisy near the threshold
    (repeat the chosen value once); `# noqa: ANN201` in `tests/test_min_util.py` suppresses a
    rule that isn't enabled.
  - `load_tts` puts the HF snapshot on `sys.path` and imports its `speak.py`: code from a model
    repo runs in our process. Mitigated by the full-SHA pin and the read-before-use note;
    logged in `docs/LIMITATIONS.md`.
- **Carry-forward from the second `/review` (Cyril's call):**
  - #1 `vllm_server.stop()` ignores a failed SIGKILL and then waits with no time limit: a
    process stuck in a CUDA call can hang the run, or `stop()` returns while VRAM is still
    held. **Do in M3.1** (the serve script): raise if SIGKILL fails, bound the final wait.
  - Headroom uses only the sampled peak; it should use `max(peak, stage readings)`, since the
    1 Hz sampler can miss the moment a stage reading captured.
  - Nits: `results/published/min_util_*.json` still record 10-character revisions (written
    before `cea09af`); no tests for `"measured": false`, the `sampling` block, or full 40-char
    SHAs; `Peaks`'s `interval_s` default duplicates `SAMPLE_INTERVAL_S` (make it required);
    `wait_ready` still trusts any `/health` responder; `test_port_in_use_detects_a_listener`
    can flake if another process takes the freed port; one over-long line in this file.
- **Open questions for Cyril:**
  - ADR-0003 Decision and "how we'd know we were wrong". M0.3's exit criterion ("or an ADR
    records the fallback") is only met once that's written.
  - `.claude/skills/gpu-budget/SKILL.md` still says "start 0.55"; measured minimums are 0.36
    (3B) and 0.80 (7B). Your file; update it?
  - Still open from session 1: where to keep the blocked-merge screenshot.
- **Licences and terms to carry:**
  - Qwen2.5-3B-Instruct-AWQ: Qwen Research License, **non-commercial** (research/evaluation).
  - Qwen2.5-7B-Instruct-AWQ: Apache-2.0. faster-whisper large-v3: MIT.
  - Indus Pocket TTS: CC-BY-4.0, derived from kyutai/pocket-tts: attribution required, Kyutai's
    prohibited-use terms apply, and callers must be told they are talking to an automated system.
- **Not proven by these runs:**
  - The 7B's Hindi (it never reached the round trip). The 3B's Hindi looked bad in one
    sample: reply `namaste_m_m_sharma_sham_ka_sharam_aapni_sharam_aapni_chahi_sey`
    (romanised, so TTS reads it as English). One prompt at temperature 0; an M3 prompt issue
    as much as a model issue.
  - STT accuracy: Whisper heard synthetic speech resampled 24→16 kHz by linear interpolation
    ("कल"→"कर", "अपॉइंटमेंट"→"अपॉइंडमिन"). Not a WER number.
  - Latency of anything; CPU STT speed and RAM (ADR-0003 option 1).
- **Things worth knowing:**
  - vLLM uses ~700 MiB more than its setting (3B at 0.36: setting ≈ 2,948, measured +3,642).
  - Concurrency at the same utilization varies between starts: 3B at 0.40 read 3.08x and then
    3.49x. Hence the ≥ 1x rule rather than an exact target.
  - nvidia-smi power still reads 590 W with limit `[N/A]`; memory readings are consistent,
    so only memory numbers are used.
  - The laptop slept partway through the 3B search; it was unaffected (one vLLM start per trial).
- **Decisions made this session (Cyril; no ADR unless noted):** pin `vllm==0.19.1` for CUDA
  12.8 builds (driver 570.207 can't run cu13 torch); uv-managed CPython (F-001); STT beam 5
  **provisional**, revisit with M7 latency numbers; 3B first then 7B, each at its minimum;
  Indus Pocket TTS on CPU; re-download models into the default HF cache.
- **Things that broke:** F-001 (vLLM couldn't find `Python.h`; logged and fixed in
  `7a5bb24`). F-002 (G1 bypassed twice by Claude; fixed forward in `f6a5edf`, hook guard is a follow-up PR). Not logged in FAILURES: torch 2.14 resolved to cu13 (fixed by the pin); uv and
  Hugging Face downloads failing on the flaky network (fixed with one download at a time,
  long timeouts, `HF_HUB_DISABLE_XET=1` and retries); model weights lost when the folder
  they'd been moved to disappeared (re-downloaded).
- **Gate status:** `make check` ✅ (ruff clean, pyright 0 errors, 63 passed); CI: not run
  (not pushed); eval vs baseline: n/a

## 2026-11-24 — session 1, part 2 (M0.2)

- **Milestone / task:** M0.2 (branch `m0/ci-check`, PR #3)
- **Done:**
  - PR #1 (M0.1) merged as `a4eae64`.
  - `main` ruleset `27975112` active: PR required (0 approvals) + `check` status, not strict;
    no bypass actors, so direct pushes to `main` are rejected.
  - `2a85ab5` ci: `permissions: contents: read`, `concurrency` (cancel stale PR runs),
    `timeout-minutes: 10`.
  - **Green on a PR:** run 45373444158 (PR #3, hardened workflow, CPython 3.11.18, 5 passed):
    https://github.com/CJ445/clinic-voice-agent/actions/runs/45373444158
    (also run 45367970564 on PR #1, pre-hardening).
  - **Red on purpose:** run 45372931560 (PR #2, `throwaway/ci-red-test` @ `46de256`): ruff and
    pyright passed, pytest `1 failed, 5 passed`, job failed at the test step:
    https://github.com/CJ445/clinic-voice-agent/actions/runs/45372931560
    Cyril took a screenshot of the merge being blocked after marking it ready for review.
    PR #2 was closed without merging and its branch was deleted.
- **In progress (exact state):** PR #3 is a draft; still needs `/review`, then Cyril merges.
- **Next step (first command to run):** `/task M0.3`
- **Open questions for Cyril:** where to keep the blocked-merge screenshot (e.g. `docs/evidence/`)
  so it can be linked from here and the README.
- **Not proven by these runs:** `concurrency` cancelling and `timeout-minutes` never fired; the
  red run used `main`'s pre-hardening workflow (it proves the gate, not the hardening).
- **M6 prerequisite (self-hosted GPU runner, public repo):** the `gpu` runner must never run
  fork PRs. Trigger GPU jobs only on own-branch PRs, push to `main`, or `workflow_dispatch`;
  require approval for workflows from outside contributors; pin every action to a commit SHA
  at that point (until then Cyril chose to keep version tags). When enabling the commented-out
  `eval-gate` job in `ci.yml`, change its `uv sync` to `uv sync --locked` to match `check`.
- **Decisions made this session:** ruleset settings above; keep action version tags until M6
  (Cyril; no ADR).
- **Things that broke:** none.
- **Gate status:** `make check` ✅ (5 passed); CI `check` ✅ on PR #3; eval vs baseline: n/a

## 2026-11-24 — session 1

- **Milestone / task:** M0.1 (done; branch `m0/project-tooling`, not pushed)
- **Done:**
  - `fa0fbd5` chore(env): pin Python 3.11 (`.python-version`, `requires-python = ">=3.11,<3.12"`);
    `uv sync --locked` in CI and `make setup` (CI part pulled forward from M0.2).
  - `c52c586` test(pkg): five package roots with docstrings; parametrized
    `tests/test_packages.py` replaces the `assert True` smoke test.
  - `7761895` chore: removed `claude-config.tar.gz` (duplicate of tracked `.claude/` files).
  - Confirmed `.claude/settings.local.json` is gitignored (`.gitignore:14`).
  - `/review`: 0 must-fix; should-fix 1–3 addressed, nits left as is. `/explain`: verified,
    no rewrite (row in `docs/AI_USAGE.md`).
- **In progress (exact state):** nothing half-done. This entry, the PLAN tick and the
  AI_USAGE row are commit 4. Branch needs pushing + PR by Cyril.
- **Next step (first command to run):** `/task M0.2`
- **Open questions for Cyril:**
  - Add ruff `D100`/`D104` so every module needs a docstring? (Today only package roots are
    checked.) Would be its own small task.
- **Carry-forward for M0.3:**
  - Confirm vLLM, Pipecat, faster-whisper and NeMo (IndicConformer) install on Python 3.11
    *with CUDA builds* (check models are really on the GPU, not a CPU-only fallback).
    If any does not, write an ADR before changing the pin.
  - `nvidia-smi` failed mid-session (`Unable to determine the device handle for GPU0:
    0000:01:00.0: Unknown Error`); fixed by a reboot. After reboot: RTX 4060, 8188 MiB,
    driver 570.207, CUDA 12.8, kernel 6.8.0-149. Power read 590W / 55W cap, which can't be
    right; check it isn't still glitching before trusting any GPU numbers in M0.3.
- **Decisions made this session:** Python 3.11 pin and archive removal (Cyril; no ADR — the
  pin matches the existing stack decision).
- **Things that broke:** GPU not visible to `nvidia-smi`; a reboot fixed it (above). Not logged in FAILURES.
- **Gate status:** `make check` ✅ (ruff clean, pyright 0 errors, 5 passed), eval vs baseline: n/a
