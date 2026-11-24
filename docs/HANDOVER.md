# HANDOVER

Newest entry on top. Written at the end of every Claude Code session with `/handover`,
then checked by Cyril. The next session starts by reading the top entry.

---

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
