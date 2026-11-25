# Known limitations

Honest list, updated as found. Each item: what, impact, why not fixed, what would fix it.

- Browser WebRTC only; no real telephony (out of scope, see PLAN).
- Simulator personas are synthetic; real callers are messier. Mitigation: 50 real recorded
  utterances in the STT bench.
- Single-GPU, single-call; no concurrency numbers.
- TTS loads by importing `speak.py` from the Indus Pocket TTS model repo (`sys.path.insert` in
  `bench/vram.py`), so code from a model repo runs in our process. Impact: a changed repo
  could run anything. Why not fixed: it's the model's only documented loader. Mitigation: full
  commit SHA pin, and `speak.py` was read before first use. Fix: vendor the loader, or load the
  weights through `pocket_tts` directly.
