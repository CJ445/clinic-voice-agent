# HANDOVER

Newest entry on top. Written at the end of every Claude Code session with `/handover`,
then checked by Cyril. The next session starts by reading the top entry.

---

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
