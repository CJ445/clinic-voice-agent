"""Unit tests for bench/min_util.find_min_util: the search for the lowest workable utilization.

A probe returns vLLM's reported concurrency at max-model-len 4096, or None if vLLM did not start.
A utilization passes only when the probe returns >= 1.0 (Cyril, PLAN M0.3). Each real probe is a
full vLLM start (~50 s), so the search bisects instead of stepping, and must record every trial.
"""

import pytest

from bench.min_util import find_min_util


def threshold_probe(threshold: float, calls: list[float]):  # noqa: ANN201
    """Fake vLLM: starts with 1.5x concurrency at or above `threshold`, fails below it."""

    def probe(util: float) -> float | None:
        calls.append(util)
        return 1.5 if util >= threshold - 1e-9 else None

    return probe


def test_finds_the_threshold_on_the_grid() -> None:
    calls: list[float] = []
    best, trials = find_min_util(threshold_probe(0.37, calls), lo=0.25, hi=0.40)
    assert best == 0.37
    assert [t["util"] for t in trials] == calls


def test_bisects_rather_than_steps() -> None:
    calls: list[float] = []
    find_min_util(threshold_probe(0.37, calls), lo=0.25, hi=0.40)
    # hi + lo + ceil(log2(15)) = 6 probes at most, versus up to 16 when stepping by 0.01.
    assert len(calls) <= 6


def test_none_when_even_hi_does_not_work() -> None:
    best, trials = find_min_util(lambda util: None, lo=0.80, hi=0.90)
    assert best is None
    assert trials == [{"util": 0.90, "concurrency": None, "ok": False}]


def test_lo_is_returned_when_it_already_works() -> None:
    best, _ = find_min_util(lambda util: 2.0, lo=0.25, hi=0.40)
    assert best == 0.25


@pytest.mark.parametrize("concurrency", [None, 0.0, 0.99])
def test_below_one_x_or_no_start_is_a_failure(concurrency: float | None) -> None:
    def probe(util: float) -> float | None:
        return 1.0 if util >= 0.30 else concurrency

    best, _ = find_min_util(probe, lo=0.25, hi=0.40)
    assert best == 0.30


def test_exactly_one_x_passes() -> None:
    best, trials = find_min_util(lambda util: 1.0, lo=0.25, hi=0.40)
    assert best == 0.25
    assert all(t["ok"] for t in trials)


def test_utilizations_are_clean_two_decimal_values() -> None:
    # Values go straight onto vLLM's command line and into the published JSON.
    calls: list[float] = []
    find_min_util(threshold_probe(0.37, calls), lo=0.25, hi=0.40)
    assert all(u == round(u, 2) for u in calls)
