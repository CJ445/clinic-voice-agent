.PHONY: setup lint format types test cov check eval-text eval-audio bench vram gate

setup:
	uv sync --locked --group gpu  # plain `uv sync` would uninstall the gpu group
	git config core.hooksPath .githooks  # pre-commit runs make check (F-002)
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
# M2 exit criterion: clinic/tools.py 100% branch-covered. Fails if it drops.
cov:
	uv run pytest -q -m "not gpu" --cov=clinic.tools --cov-branch \
		--cov-report=term-missing --cov-fail-under=100

check: lint types test

# Targets below are implemented in later milestones; they fail loudly until then.
eval-text:
	@test -f sim/run.py || (echo "eval-text: not implemented yet (PLAN M5.3)"; exit 1)
	uv run python -m sim.run --mode text --out results/eval_text.json

eval-audio:
	@test -f sim/run.py || (echo "eval-audio: not implemented yet (PLAN M5.6)"; exit 1)
	uv run python -m sim.run --mode audio --subset nightly --out results/eval_audio.json

# GPU co-residency smoke test (PLAN M0.3). LLMS = space-separated REPO@REVISION:UTIL specs,
# with UTIL from the minimum-utilization search. BEAM = 5 is provisional (Cyril, 2026-11-25),
# to be revisited with latency numbers in M7.
BEAM ?= 5
vram:
	@test -n "$(LLMS)" || (echo "vram: set LLMS='REPO@REVISION:UTIL ...'"; exit 1)
	uv run python -m bench.vram --beam-size $(BEAM) $(foreach spec,$(LLMS),--llm $(spec)) $(if $(ALLOW_DIRTY),--allow-dirty)

bench:
	@test -f bench/latency.py || (echo "bench: not implemented yet (PLAN M4.2)"; exit 1)
	uv run python -m bench.latency --out results/latency_$$(git rev-parse --short HEAD).json

gate:
	@test -f ci/gate.py || (echo "gate: not implemented yet (PLAN M6.2)"; exit 1)
	uv run python -m ci.gate --current results/eval_text.json --baseline ci/baseline.json
