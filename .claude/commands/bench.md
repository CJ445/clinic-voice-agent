---
description: Before/after benchmark for one optimization (M7 workflow)
argument-hint: <hypothesis, e.g. "prefix caching cuts LLM first token">
---
Hypothesis: $ARGUMENTS
Follow the `latency-bench` skill exactly:
1. On `main`, run `make bench` → save as `results/before.json`.
2. On this branch, run `make bench` → `results/after.json`.
3. Run `make eval-text` and `make gate` to confirm quality held.
4. Show a table of P50/P95 per stage, before/after/Δ, plus VRAM, plus eval deltas.
5. If the improvement is inside run-to-run noise, say so plainly. Do not round up.
