"""Unit tests for clinic/seed.py: deterministic seed data from a frozen clock (PLAN M2.1).

"Identical" is checked on the full SQL dump, so any difference in any row or column fails.
"""

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from clinic.clock import IST, FrozenClock
from clinic.db import from_db_time
from clinic.seed import SEED_NOW, build


def dump(path: Path) -> list[str]:
    return list(sqlite3.connect(path).iterdump())


def test_seed_is_identical_across_runs(tmp_path: Path) -> None:
    build(tmp_path / "a.db")
    build(tmp_path / "b.db")
    assert dump(tmp_path / "a.db") == dump(tmp_path / "b.db")


def test_seed_cli_matches_in_process_build(tmp_path: Path) -> None:
    build(tmp_path / "a.db")
    subprocess.run([sys.executable, "-m", "clinic.seed", str(tmp_path / "cli.db")], check=True)
    assert dump(tmp_path / "a.db") == dump(tmp_path / "cli.db")


def test_seed_follows_the_clock(tmp_path: Path) -> None:
    # Checks slot data itself: created_at alone would make the dumps differ (review #3).
    later = SEED_NOW + timedelta(days=7)
    build(tmp_path / "later.db", clock=FrozenClock(later))
    starts = [
        from_db_time(s)
        for (s,) in sqlite3.connect(tmp_path / "later.db").execute("SELECT start FROM slots")
    ]
    assert min(starts).date() == later.date()
    last_open_day = max(
        later.date() + timedelta(days=d)
        for d in range(14)
        if (later.date() + timedelta(days=d)).weekday() != 6
    )
    assert max(starts).date() == last_open_day  # day 14 is a Sunday here: closed


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 11, 30, 10, 15, tzinfo=IST),  # Monday mid-morning session
        datetime(2026, 11, 30, 9, 0, tzinfo=IST),  # exactly on a slot start
        datetime(2026, 12, 6, 11, 0, tzinfo=IST),  # Sunday: closed today
        datetime(2026, 12, 5, 21, 0, tzinfo=IST),  # Saturday after the last session
    ],
)
def test_any_clock_gives_only_future_slots_and_valid_bookings(
    tmp_path: Path, now: datetime
) -> None:
    # Review #2: the "skip slots at or before now" branch, and the position-based picks that
    # depend on it, were never exercised by SEED_NOW (08:00, before every slot).
    build(tmp_path / "c.db", clock=FrozenClock(now))
    conn = sqlite3.connect(tmp_path / "c.db")
    starts = [from_db_time(s) for (s,) in conn.execute("SELECT start FROM slots")]
    assert min(starts) > now
    assert not any(s.date() == now.date() for s in starts if now.weekday() == 6)
    rows = conn.execute(
        "SELECT s.status, s.start FROM appointments a JOIN slots s ON s.id = a.slot_id"
    ).fetchall()
    assert len(rows) == 5
    assert all(status == "open" and from_db_time(start) > now for status, start in rows)


def test_build_refuses_to_overwrite(tmp_path: Path) -> None:
    build(tmp_path / "a.db")
    with pytest.raises(FileExistsError):
        build(tmp_path / "a.db")


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    build(tmp_path / "clinic.db")
    return sqlite3.connect(tmp_path / "clinic.db")


def test_three_doctors(db: sqlite3.Connection) -> None:
    assert db.execute("SELECT COUNT(*) FROM doctors").fetchone() == (3,)


def test_two_weeks_of_future_slots_no_sundays(db: sqlite3.Connection) -> None:
    starts = [from_db_time(s) for (s,) in db.execute("SELECT start FROM slots")]
    assert starts
    assert min(starts) > SEED_NOW
    assert max(starts).date() < SEED_NOW.date() + timedelta(days=14)
    assert {s.date() for s in starts} == {
        SEED_NOW.date() + timedelta(days=d)
        for d in range(14)
        if (SEED_NOW.date() + timedelta(days=d)).weekday() != 6
    }
    raw = [s for (s,) in db.execute("SELECT start FROM slots")]
    assert all(s.endswith("+05:30") for s in raw)  # the stored data, not the helper


def test_each_doctor_keeps_their_hours(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT doctor_id, start FROM slots").fetchall()
    hours = {doc: {from_db_time(s).hour for d, s in rows if d == doc} for doc in (1, 2, 3)}
    assert hours[2] <= set(range(9, 13))  # mornings only
    assert hours[3] <= set(range(17, 20))  # evenings only
    assert hours[1] & set(range(9, 13)) and hours[1] & set(range(17, 20))  # both


def test_some_slots_are_blocked(db: sqlite3.Connection) -> None:
    assert db.execute("SELECT COUNT(*) FROM slots WHERE status = 'blocked'").fetchone()[0] > 0


def test_pre_booked_appointments_sit_on_open_future_slots(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT a.status, s.status, s.start FROM appointments a JOIN slots s ON s.id = a.slot_id"
    ).fetchall()
    statuses = sorted(a for a, _, _ in rows)
    assert statuses == ["booked"] * 4 + ["cancelled"]
    assert all(slot_status == "open" for _, slot_status, _ in rows)
    assert all(from_db_time(start) > SEED_NOW for _, _, start in rows)


def test_seed_now_is_a_monday_morning_in_ist() -> None:
    assert SEED_NOW.tzinfo is IST and SEED_NOW.weekday() == 0 and SEED_NOW.hour < 9
    assert datetime(2026, 11, 30, 8, 0, tzinfo=IST) == SEED_NOW
