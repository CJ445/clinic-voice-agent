.PHONY: setup lint format types test check eval-text eval-audio bench gate

setup:
	uv sync --locked
	@command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no GPU visible"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

types:
	uv run pyright

test:
	uv run pytest -q -m "not gpu"

# Gate G1 — must pass before every commit
check: lint types test

# Targets below are implemented in later milestones; they fail loudly until then.
eval-text:
	@test -f sim/run.py || (echo "eval-text: not implemented yet (PLAN M5.3)"; exit 1)
	uv run python -m sim.run --mode text --out results/eval_text.json

eval-audio:
	@test -f sim/run.py || (echo "eval-audio: not implemented yet (PLAN M5.6)"; exit 1)
	uv run python -m sim.run --mode audio --subset nightly --out results/eval_audio.json

bench:
	@test -f bench/latency.py || (echo "bench: not implemented yet (PLAN M4.2)"; exit 1)
	uv run python -m bench.latency --out results/latency_$$(git rev-parse --short HEAD).json

gate:
	@test -f ci/gate.py || (echo "gate: not implemented yet (PLAN M6.2)"; exit 1)
	uv run python -m ci.gate --current results/eval_text.json --baseline ci/baseline.json
