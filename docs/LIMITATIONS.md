# Known limitations

Honest list, updated as found. Each item: what, impact, why not fixed, what would fix it.

- Browser WebRTC only; no real telephony (out of scope, see PLAN).
- Simulator personas are synthetic; real callers are messier. Mitigation: 50 real recorded
  utterances in the STT bench.
- Single-GPU, single-call; no concurrency numbers.
