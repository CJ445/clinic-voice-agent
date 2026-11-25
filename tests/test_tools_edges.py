"""Edge cases for the booking tools (PLAN M2.3): concurrency, time boundaries, retries.

Concurrency tests use one DB file and a separate connection per writer, first as threads, then
as real OS processes (SQLite file locking, not Python's GIL, is what must keep it to one booking).
Time-boundary tests pin the exact instant a slot stops being bookable and the IST midnight edge,
where a UTC-based date filter would put a 00:00 IST slot on the previous day.
"""

import json
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from clinic.clock import IST, FrozenClock
from clinic.db import connect
from clinic.seed import SEED_NOW, build
from clinic.tools import (
    BookIn,
    BookOut,
    CancelIn,
    CancelOut,
    CheckSlotsIn,
    CheckSlotsOut,
    ToolError,
    book,
    cancel,
    check_slots,
    run,
)

FREE = "d1-20261130-0930"  # open Dr. Sharma slot in the seed (see conftest)
REPO = Path(__file__).resolve().parents[1]


def args(i: int, key: str | None = None, slot_id: str = FREE) -> dict[str, Any]:
    return {
        "slot_id": slot_id,
        "patient_name": f"Caller {i}",
        "phone": f"98765{i:05d}",
        "idempotency_key": key or f"k-{i}",
    }


def booked_on(path: Path, slot_id: str) -> int:
    with sqlite3.connect(path) as conn:
        query = "SELECT COUNT(*) FROM appointments WHERE slot_id = ? AND status = 'booked'"
        return conn.execute(query, (slot_id,)).fetchone()[0]


def race_threads(path: Path, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run each book call on its own thread and connection, all released at once."""
    barrier = threading.Barrier(len(calls))
    results: list[dict[str, Any]] = [{} for _ in calls]

    def worker(i: int) -> None:
        conn = connect(path)
        barrier.wait()
        results[i] = run(conn, FrozenClock(SEED_NOW), "book", calls[i])
        conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(calls))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# A separate Python process per writer: imports, waits for a shared start time, books once.
WORKER = """
import json, sys, time
from clinic.clock import FrozenClock
from clinic.db import connect
from clinic.seed import SEED_NOW
from clinic.tools import run
path, start_at, call = sys.argv[1], float(sys.argv[2]), json.loads(sys.argv[3])
conn = connect(path)
while time.time() < start_at:
    pass
print(json.dumps(run(conn, FrozenClock(SEED_NOW), "book", call)))
"""


def race_processes(path: Path, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start_at = str(time.time() + 3)  # time for every process to import before the race
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", WORKER, str(path), start_at, json.dumps(call)],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for call in calls
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        results.append(json.loads(out))
    return results


@pytest.mark.parametrize("race", [race_threads, race_processes])
def test_concurrent_double_booking_books_exactly_once(tmp_path: Path, race: Any) -> None:
    build(tmp_path / "c.db")
    results = race(tmp_path / "c.db", [args(i) for i in range(6)])
    winners = [r for r in results if r["ok"]]
    assert len(winners) == 1, results
    assert {r["code"] for r in results if not r["ok"]} == {"slot_taken"}
    assert booked_on(tmp_path / "c.db", FREE) == 1


@pytest.mark.parametrize("race", [race_threads, race_processes])
def test_concurrent_retries_with_one_key_make_one_appointment(tmp_path: Path, race: Any) -> None:
    # The same request delivered several times at once (e.g. a retried network call).
    build(tmp_path / "c.db")
    results = race(tmp_path / "c.db", [args(1, key="same") for _ in range(4)])
    assert all(r["ok"] for r in results), results
    assert len({r["appointment_id"] for r in results}) == 1
    assert sorted(r["replayed"] for r in results) == [False, True, True, True]
    assert booked_on(tmp_path / "c.db", FREE) == 1


def test_retry_after_reconnect_returns_the_same_appointment(tmp_path: Path) -> None:
    # The agent process restarts between the first call and its retry.
    build(tmp_path / "c.db")
    clock = FrozenClock(SEED_NOW)
    req = BookIn.model_validate(args(1))
    first_conn = connect(tmp_path / "c.db")
    first = book(first_conn, clock, req)
    first_conn.close()
    again_conn = connect(tmp_path / "c.db")
    again = book(again_conn, clock, req)
    again_conn.close()
    assert isinstance(first, BookOut) and isinstance(again, BookOut)
    assert (again.appointment_id, again.replayed) == (first.appointment_id, True)
    assert booked_on(tmp_path / "c.db", FREE) == 1


# --- past-slot boundary: a slot stops being bookable at the instant it starts ---


def at(hour: int, minute: int, second: int = 0) -> FrozenClock:
    return FrozenClock(datetime(2026, 11, 30, hour, minute, second, tzinfo=IST))


def test_slot_is_bookable_one_second_before_it_starts(conn: sqlite3.Connection) -> None:
    out = book(conn, at(9, 29, 59), BookIn.model_validate(args(1)))
    assert isinstance(out, BookOut)


def test_slot_is_past_at_the_instant_it_starts(conn: sqlite3.Connection) -> None:
    out = book(conn, at(9, 30), BookIn.model_validate(args(1)))
    assert isinstance(out, ToolError) and out.code == "past_slot"
    monday = CheckSlotsIn(doctor="Sharma", date_from=date(2026, 11, 30), date_to=date(2026, 11, 30))
    offered = check_slots(conn, at(9, 30), monday)
    assert isinstance(offered, CheckSlotsOut) and offered.slots[0].slot_id == "d1-20261130-1000"


def test_cancel_is_refused_at_the_instant_the_appointment_starts(conn: sqlite3.Connection) -> None:
    asha = CancelIn(appointment_id=1, phone="9800000001")  # seed: 09:00
    assert isinstance(cancel(conn, at(9, 0), asha), ToolError)
    assert isinstance(cancel(conn, at(8, 59, 59), asha), CancelOut)


# --- IST midnight: slot dates are IST calendar days, whatever zone the clock reports in ---

LATE, MIDNIGHT = "d1-20261201-2330", "d1-20261202-0000"


@pytest.fixture
def night_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    with conn:
        conn.executemany(
            "INSERT INTO slots (id, doctor_id, start, duration_min, status)"
            " VALUES (?, 1, ?, 30, 'open')",
            [(LATE, "2026-12-01T23:30:00+05:30"), (MIDNIGHT, "2026-12-02T00:00:00+05:30")],
        )
    return conn


def utc_clock(hour: int, minute: int) -> FrozenClock:
    """A clock that reports UTC on 1 Dec (IST is UTC+05:30)."""
    return FrozenClock(datetime(2026, 12, 1, hour, minute, tzinfo=UTC))


def slot_ids(conn: sqlite3.Connection, clock: FrozenClock, day: date) -> list[str]:
    out = check_slots(
        conn, clock, CheckSlotsIn(doctor="Sharma", date_from=day, date_to=day, max_results=50)
    )
    assert isinstance(out, CheckSlotsOut)
    return [s.slot_id for s in out.slots]


def test_midnight_ist_slot_belongs_to_the_next_ist_day(night_conn: sqlite3.Connection) -> None:
    # 00:00 IST on 2 Dec is 18:30 UTC on 1 Dec; a UTC date filter would list it under 1 Dec.
    clock = FrozenClock(SEED_NOW)
    dec1, dec2 = (
        slot_ids(night_conn, clock, date(2026, 12, 1)),
        slot_ids(night_conn, clock, date(2026, 12, 2)),
    )
    assert dec1[-1] == LATE and MIDNIGHT not in dec1
    assert dec2[0] == MIDNIGHT and LATE not in dec2


def test_clock_in_utc_is_read_as_ist(night_conn: sqlite3.Connection) -> None:
    # 18:15 UTC = 23:45 IST on 1 Dec: the 23:30 slot has started, the 00:00 slot hasn't.
    clock = utc_clock(18, 15)
    assert LATE not in slot_ids(night_conn, clock, date(2026, 12, 1))
    assert slot_ids(night_conn, clock, date(2026, 12, 2))[0] == MIDNIGHT
    late = book(night_conn, clock, BookIn.model_validate(args(1, slot_id=LATE)))
    assert isinstance(late, ToolError) and late.code == "past_slot"
    midnight = book(night_conn, clock, BookIn.model_validate(args(2, slot_id=MIDNIGHT)))
    assert isinstance(midnight, BookOut)
    assert midnight.start.isoformat() == "2026-12-02T00:00:00+05:30"


def test_midnight_slot_is_past_from_1830_utc(night_conn: sqlite3.Connection) -> None:
    out = book(night_conn, utc_clock(18, 30), BookIn.model_validate(args(1, slot_id=MIDNIGHT)))
    assert isinstance(out, ToolError) and out.code == "past_slot"


# --- cancel: nonexistent, twice, and a cancel/rebook cycle on one slot ---


def test_cancel_nonexistent_and_twice_by_phone_and_slot(conn: sqlite3.Connection) -> None:
    clock = FrozenClock(SEED_NOW)
    ravi = CancelIn(phone="9800000002", slot_id="d1-20261130-1730")
    nobody = CancelIn(phone="9800000002", slot_id=FREE)
    assert (
        isinstance(missing := cancel(conn, clock, nobody), ToolError)
        and missing.code == "not_found"
    )
    assert isinstance(cancel(conn, clock, ravi), CancelOut)
    again = cancel(conn, clock, ravi)
    assert isinstance(again, ToolError) and again.code == "already_cancelled"


def test_cancel_rebook_cancel_on_one_slot(conn: sqlite3.Connection) -> None:
    # After a cancel and a rebook, phone + slot matches two rows; only the active one is cancelled.
    clock = FrozenClock(SEED_NOW)
    ravi = CancelIn(phone="9800000002", slot_id="d1-20261130-1730")
    assert isinstance(cancel(conn, clock, ravi), CancelOut)
    rebook = BookIn(
        slot_id="d1-20261130-1730",
        patient_name="Ravi Kumar",
        phone="9800000002",
        idempotency_key="r2",
    )
    rebooked = book(conn, clock, rebook)
    assert isinstance(rebooked, BookOut) and rebooked.appointment_id != 2
    second = cancel(conn, clock, ravi)
    assert isinstance(second, CancelOut) and second.appointment_id == rebooked.appointment_id
    third = cancel(conn, clock, ravi)
    assert isinstance(third, ToolError) and third.code == "already_cancelled"
