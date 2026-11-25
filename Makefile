.PHONY: setup lint format types test check eval-text eval-audio bench vram serve gate

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

# One pass: unit tests plus coverage: clinic/tools.py (M2 exit) and clinic/dates.py (Cyril,
# M2 close-out) must stay 100% branch-covered.
test:
	uv run pytest -q -m "not gpu" --cov=clinic.tools --cov=clinic.dates --cov-branch \
		--cov-report=term-missing:skip-covered --cov-fail-under=100

# Gate G1 — must pass before every commit
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

# Serve the dialogue LLM (PLAN M3.1). LLM = REPO@FULL_40_CHAR_SHA; the utilization is the
# published minimum + margin (agent/serve.py) unless UTIL overrides it.
serve:
	@test -n "$(LLM)" || (echo "serve: set LLM='REPO@FULL_40_CHAR_SHA'"; exit 1)
	uv run python -m agent.serve --llm $(LLM) $(if $(UTIL),--util $(UTIL))

bench:
	@test -f bench/latency.py || (echo "bench: not implemented yet (PLAN M4.2)"; exit 1)
	uv run python -m bench.latency --out results/latency_$$(git rev-parse --short HEAD).json

gate:
	@test -f ci/gate.py || (echo "gate: not implemented yet (PLAN M6.2)"; exit 1)
	uv run python -m ci.gate --current results/eval_text.json --baseline ci/baseline.json
