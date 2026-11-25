# What broke and how I fixed it

Interview gold. One entry per real problem. Be specific: symptoms, numbers, wrong guesses.

---

## F-001 — vLLM would not start: Triton could not find `Python.h`

- **Date / milestone:** 2026-11-25, M0.3 (first `vllm serve` of Qwen2.5-3B-Instruct-AWQ)
- **Symptom:** vLLM 0.19.1 loaded the weights (`Model loading took 1.95 GiB`), then the engine
  died while sizing the KV cache: `/tmp/.../cuda_utils.c:7:10: fatal error: Python.h: No such
  file or directory` → `EngineCore failed to start`, exit code 1. GPU memory was released cleanly.
- **Wrong hypotheses:** none about the error itself, but an earlier check was too weak. The M0.3
  install check (Claude) reported vLLM as working because `import vllm` succeeded and torch saw
  the GPU. Importing never starts the engine, so it could not catch a failure in Triton's JIT
  compile step. Separately, during the fix Claude first built the new venv as `.venv-managed` and
  renamed it to `.venv`; that broke every entry-point script (`pyright`, `pytest`, `hf`), because
  their shebang lines hard-code the venv path. `make check` failed with
  `Failed to spawn: pyright`. Rebuilt in place instead.
- **Root cause:** the venv was created on the system interpreter `/usr/bin/python3.11` (3.11.17),
  and Ubuntu's `python3.11-dev` (which provides `/usr/include/python3.11/Python.h`) was not
  installed. Triton compiles a small C helper on first use and needs those headers. CI never
  saw it: it uses uv-managed CPython (3.11.18), which ships the headers.
- **Fix:** (Cyril chose option B over `apt install python3.11-dev`) `python-preference =
  "only-managed"` under `[tool.uv]`, `uv python install 3.11` (installed 3.11.15), venv rebuilt
  in place with `uv sync --locked --group gpu --offline` from uv's cache (no re-download).
  Commit: `7a5bb24`.
- **Guard added:** (1) `tests/test_env.py::test_python_dev_headers_present` asserts `Python.h`
  exists for the running interpreter. It is in `make check`, so G1 catches it locally and in CI
  (a `make setup` check was considered, but CI never runs `make setup`). Checked: it reports
  `exists: False` for `/usr/bin/python3.11`. (2) Rule: "vLLM works" means
  `bench/vllm_server.start` + `wait_ready` succeeded (the M0.3 smoke run), never `import vllm`.
- **Lesson:** an import is not a start. Check the thing at the level it will actually be used.
  And never rename a venv; rebuild it where it will live.
