"""Unit tests for the CPU-testable part of bench/vllm_server.py: reading vLLM's startup log.

The minimum-utilization search (PLAN M0.3) accepts a --gpu-memory-utilization only if vLLM's log
shows at least 1x concurrency at max-model-len 4096. The fixture lines are copied verbatim from a
real vLLM 0.19.1 start of Qwen2.5-3B-Instruct-AWQ at 0.40 on the RTX 4060 (2026-11-25).
"""

import pytest

from bench.vllm_server import parse_max_concurrency

REAL_LOG = (
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [gpu_worker.py:436] "
    "Available KV cache memory: 0.43 GiB\n"
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [kv_cache_utils.py:1319] "
    "GPU KV cache size: 12,624 tokens\n"
    "(EngineCore pid=21675) INFO 11-25 02:30:09 [kv_cache_utils.py:1324] "
    "Maximum concurrency for 4,096 tokens per request: 3.08x\n"
)


def test_reads_real_vllm_line_with_thousands_separator() -> None:
    assert parse_max_concurrency(REAL_LOG, max_model_len=4096) == pytest.approx(3.08)


def test_none_when_vllm_died_before_sizing_kv_cache() -> None:
    # F-001: the engine can exit before this line is ever printed.
    assert parse_max_concurrency("EngineCore failed to start.\n", max_model_len=4096) is None


def test_ignores_line_for_a_different_max_model_len() -> None:
    # A stale log from a run with another --max-model-len must not satisfy the check.
    line = "Maximum concurrency for 8,192 tokens per request: 1.54x\n"
    assert parse_max_concurrency(line, max_model_len=4096) is None


def test_below_one_is_reported_not_rounded() -> None:
    line = "Maximum concurrency for 4,096 tokens per request: 0.97x\n"
    assert parse_max_concurrency(line, max_model_len=4096) == pytest.approx(0.97)


def test_last_matching_line_wins() -> None:
    # One log file per run is the rule, but if two starts were appended, trust the latest.
    log = REAL_LOG + "Maximum concurrency for 4,096 tokens per request: 1.20x\n"
    assert parse_max_concurrency(log, max_model_len=4096) == pytest.approx(1.20)
