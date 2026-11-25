"""Unit tests for clinic/db.py: the schema enforces the booking invariants itself (PLAN M2.1).

Most tests write raw SQL on purpose: the database must refuse a double booking or a naive
timestamp even when no Python code is checking. Truly concurrent writers (SQLite's busy/locked
path) are tested in M2.3, not here.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from clinic.clock import IST
from clinic.db import connect, create_schema, to_db_time

START = to_db_time(datetime(2026, 11, 30, 9, 0, tzinfo=IST))
INSERT_SLOT = (
    "INSERT INTO slots (id, doctor_id, start, duration_min, status) VALUES (?, 1, ?, 30, 'open')"
)
INSERT_APPT = (
    "INSERT INTO appointments (id, slot_id, patient_id, status, idempotency_key, created_at)"
    " VALUES (?, ?, ?, 'booked', ?, ?)"
)


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "clinic.db")
    create_schema(conn)
    conn.execute("INSERT INTO doctors (id, name, speciality) VALUES (1, 'Dr. Sharma', 'General')")
    conn.execute("INSERT INTO patients (id, name, phone) VALUES (1, 'Asha', '+919800000001')")
    conn.execute("INSERT INTO patients (id, name, phone) VALUES (2, 'Ravi', '+919800000002')")
    conn.execute(INSERT_SLOT, ("s1", START))
    return conn


def book(conn: sqlite3.Connection, appt_id: int, patient_id: int, key: str) -> None:
    conn.execute(INSERT_APPT, (appt_id, "s1", patient_id, key, START))


def test_db_rejects_a_second_active_appointment_on_one_slot(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "k1")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        book(db, 2, 2, "k2")


def test_slot_can_be_rebooked_after_cancellation(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "k1")
    db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = 1")
    book(db, 2, 2, "k2")  # allowed: only *active* appointments are unique per slot


def test_idempotency_key_is_unique(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "same")
    db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = 1")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        book(db, 2, 2, "same")


def test_constraint_holds_for_a_second_connection_after_commit(tmp_path: Path) -> None:
    # Sequential, not a race: a committed booking is enforced for every other connection.
    path = tmp_path / "clinic.db"
    first = connect(path)
    create_schema(first)
    first.execute("INSERT INTO doctors (id, name, speciality) VALUES (1, 'Dr. Sharma', 'General')")
    first.execute("INSERT INTO patients (id, name, phone) VALUES (1, 'Asha', '+919800000001')")
    first.execute(INSERT_SLOT, ("s1", START))
    first.commit()
    book(first, 1, 1, "k1")
    first.commit()
    second = connect(path)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        book(second, 2, 1, "k2")


@pytest.mark.parametrize(
    "bad_start",
    ["2026-11-30T09:00:00", "2026-11-30T03:30:00+00:00", "2026-11-30 09:00", "tomorrow 9am"],
)
def test_db_rejects_a_start_that_is_not_ist(db: sqlite3.Connection, bad_start: str) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.execute(
            "INSERT INTO slots (id, doctor_id, start, duration_min, status)"
            " VALUES ('s2', 1, ?, 30, 'open')",
            (bad_start,),
        )


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [("slots", "status", "booked"), ("appointments", "status", "pending")],
)
def test_status_values_are_checked(
    db: sqlite3.Connection, table: str, column: str, value: str
) -> None:
    book(db, 1, 1, "k1")
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.execute(f"UPDATE {table} SET {column} = ?", (value,))


def test_foreign_keys_are_enforced(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        db.execute(INSERT_APPT, (9, "no-such-slot", 1, "k9", START))


def test_to_db_time_rejects_naive_and_converts_to_ist() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        to_db_time(datetime(2026, 11, 30, 9, 0))  # noqa: DTZ001 (naive on purpose)
    utc = datetime.fromisoformat("2026-11-30T03:30:00+00:00")
    assert to_db_time(utc) == "2026-11-30T09:00:00+05:30"


def test_db_rejects_booking_a_blocked_slot(db: sqlite3.Connection) -> None:
    # Cyril's option 3 (M2.1 review): the DB is the safety net; book() also checks first.
    db.execute("UPDATE slots SET status = 'blocked' WHERE id = 's1'")
    with pytest.raises(sqlite3.IntegrityError, match="slot is blocked"):
        book(db, 1, 1, "k1")


def test_db_rejects_reactivating_an_appointment_on_a_blocked_slot(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "k1")
    db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = 1")
    db.execute("UPDATE slots SET status = 'blocked' WHERE id = 's1'")
    with pytest.raises(sqlite3.IntegrityError, match="slot is blocked"):
        db.execute("UPDATE appointments SET status = 'booked' WHERE id = 1")


def test_db_rejects_blocking_a_slot_that_is_booked(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "k1")
    with pytest.raises(sqlite3.IntegrityError, match="slot has an active appointment"):
        db.execute("UPDATE slots SET status = 'blocked' WHERE id = 's1'")


def test_blocking_is_allowed_once_the_appointment_is_cancelled(db: sqlite3.Connection) -> None:
    book(db, 1, 1, "k1")
    db.execute("UPDATE appointments SET status = 'cancelled' WHERE id = 1")
    db.execute("UPDATE slots SET status = 'blocked' WHERE id = 's1'")
