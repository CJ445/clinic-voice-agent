---
description: Independent review of the current diff by the reviewer subagent
---
Use the `reviewer` subagent on the diff `git diff main...HEAD` (plus unstaged changes).
It has not seen how the code was written — do not summarise intent for it beyond the
PLAN task id. Then list its findings as: must-fix / should-fix / nit, and propose fixes
for must-fix items only. Cyril decides on the rest.
