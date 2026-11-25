"""Unit tests for clinic/clock.py and the DB time round-trip: everything stays aware, in IST."""

from datetime import datetime

import pytest

from clinic.clock import IST, FrozenClock, SystemClock
from clinic.db import from_db_time, to_db_time


def test_frozen_clock_rejects_naive() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FrozenClock(datetime(2026, 11, 30, 9, 0))  # noqa: DTZ001 (naive on purpose)


def test_frozen_clock_is_frozen_and_in_ist() -> None:
    clock = FrozenClock(datetime.fromisoformat("2026-11-30T03:30:00+00:00"))
    assert clock.now() == clock.now()
    assert clock.now().isoformat() == "2026-11-30T09:00:00+05:30"


def test_system_clock_is_aware_ist() -> None:
    now = SystemClock().now()
    assert now.tzinfo is IST
    assert now.isoformat().endswith("+05:30")


def test_db_time_round_trip() -> None:
    moment = datetime(2026, 11, 30, 23, 30, tzinfo=IST)
    assert from_db_time(to_db_time(moment)) == moment


def test_from_db_time_rejects_naive() -> None:
    # Review: a naive string was read as the server's local time and silently shifted.
    with pytest.raises(ValueError, match="timezone-aware"):
        from_db_time("2026-11-30T09:00:00")
