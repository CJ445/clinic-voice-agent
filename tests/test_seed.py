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
    build(tmp_path / "a.db")
    build(tmp_path / "later.db", clock=FrozenClock(SEED_NOW + timedelta(days=7)))
    assert dump(tmp_path / "a.db") != dump(tmp_path / "later.db")


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
    assert min(starts) >= SEED_NOW
    assert max(starts).date() < SEED_NOW.date() + timedelta(days=14)
    assert {s.date() for s in starts} == {
        SEED_NOW.date() + timedelta(days=d)
        for d in range(14)
        if (SEED_NOW.date() + timedelta(days=d)).weekday() != 6
    }
    assert all(
        s.tzinfo is not None and s.utcoffset() == timedelta(hours=5, minutes=30) for s in starts
    )


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
    assert statuses.count("booked") >= 3 and statuses.count("cancelled") >= 1
    assert all(slot_status == "open" for _, slot_status, _ in rows)
    assert all(from_db_time(start) > SEED_NOW for _, _, start in rows)


def test_seed_now_is_a_monday_morning_in_ist() -> None:
    assert SEED_NOW.tzinfo is IST and SEED_NOW.weekday() == 0 and SEED_NOW.hour < 9
    assert datetime(2026, 11, 30, 8, 0, tzinfo=IST) == SEED_NOW
