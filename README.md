# clinic-voice-agent

A Hindi/English phone agent that books clinic appointments, and a caller simulator
(200 scripted personas) that scores every change and blocks regressions in CI.

> 🚧 Work in progress. Numbers below appear only once measured — see `docs/RESULTS.md`.

## Results

| | Baseline | Now |
|---|---|---|
| P95 end-of-speech → first audio | — | — |
| Task success (200 scenarios) | — | — |
| Tool-call accuracy | — | — |
| Hallucination rate | — | — |

Hardware: single RTX 4060 Laptop (8 GB). Reproduce: see [Reproducing](#reproducing).

## Architecture

See [`docs/DESIGN.md`](docs/DESIGN.md) and the decision records in [`docs/decisions/`](docs/decisions/).

## Reproducing

```bash
make setup && make check
make bench        # latency
make eval-text    # 200-scenario simulator
```

## Known limitations

[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) · What broke and how I fixed it: [`docs/FAILURES.md`](docs/FAILURES.md)

## How I used AI

Built with Claude Code acting as a junior engineer. I wrote the design and every decision
record, reviewed every diff, and verified behaviour with tests, evals and benchmarks.
Details, including what I rewrote: [`docs/AI_USAGE.md`](docs/AI_USAGE.md).
