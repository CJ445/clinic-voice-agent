---
description: Quiz Cyril on a file so he can defend every line in an interview
argument-hint: <path>
---
File: $ARGUMENTS

Act as a sharp interviewer at a voice-AI startup. Do NOT explain the file first.
1. Ask 5 questions about $ARGUMENTS, one at a time, waiting for each answer:
   what it does, why this design, a failure mode, a latency/VRAM implication, and
   "what would you change at 100 concurrent calls?"
2. After each answer, say what was right, what was missing, and point to the exact lines.
3. At the end, list lines or functions Cyril could not explain → recommend rewrite or
   simplification, and add a row to `docs/AI_USAGE.md` (verified / needs rewrite).
