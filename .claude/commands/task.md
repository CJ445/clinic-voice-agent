---
description: Plan one PLAN.md task before writing code
argument-hint: <task id, e.g. M2.2>
---
Task: $ARGUMENTS

1. Find task $ARGUMENTS in `docs/PLAN.md` and any ADRs it depends on.
2. Propose a plan (no code yet):
   - files to create/change, each with a one-line purpose
   - the tests you will write first, and what each proves
   - the commit sequence: each commit one concern, < 200 lines
   - any decision NOT covered by an ADR → list as an open question with 2–3 options
   - VRAM / latency impact, if any
3. Suggest a branch name: `m<milestone>/<short-slug>`.
4. Stop and wait for approval. If Cyril edits the plan, follow the edited plan.
