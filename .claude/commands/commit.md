---
description: Split staged/unstaged work into small conventional commits and propose them
---
1. Run `git status` and `git diff`.
2. Group changes into commits that each do ONE thing (< 200 lines). If one file mixes
   concerns, say so and propose `git add -p` hunks.
3. For each commit propose: files, message in Conventional Commits format, a body that
   explains *why*, `Refs: <PLAN task>`, and the Co-authored-by trailer if Claude wrote most of it.
4. Run `make check` before each commit. Then ask Cyril to approve each `git commit`.
   Never push.
