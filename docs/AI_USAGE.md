# How I used AI on this project

Built with Claude Code as a junior engineer. I designed the system, made every decision
in `docs/decisions/`, reviewed every diff, and own every line.

## Summary (fill in at M8)

- **I designed:** …
- **Claude wrote first drafts of:** …
- **I verified by:** tests, evals, benchmarks, line-by-line review
- **I rewrote:** … (and why — usually clarity, a bug, or a wrong assumption)
- **Where Claude was wrong, and how I caught it:** (link FAILURES entries)

## Ledger

| Module / PR | Drafted by | Verified how | Rewritten? | Notes |
|---|---|---|---|---|
| M0.1 (`m0/project-tooling`) — Python 3.11 pin, `uv sync --locked`, package roots, `tests/test_packages.py` | Claude | Cyril: plan approved with edits; reviewer subagent (0 must-fix); `/explain` 5/5 answered | No | **Verified.** Weak spots in `/explain`: test is a *guard*, not what creates the paths; `pythonpath = ["."]` + `package = false` is why imports resolve; `requires-python` controls lock resolution, not just the interpreter. Known gap: docstring test covers package roots only (ruff `D100`/`D104` would cover all modules — undecided). |
