---
description: Run quality gate G1 (and G5 if evals exist) and report honestly
---
1. Run `make check`. If it fails, show the first failure and propose the smallest fix.
   Do not modify tests to make them pass.
2. If `ci/baseline.json` exists and the diff touches `agent/`, `prompts/`, or model config,
   run `make eval-text` then `make gate` and show the metric table vs baseline.
3. Report: ✅/❌ per gate, plus anything you skipped and why.
