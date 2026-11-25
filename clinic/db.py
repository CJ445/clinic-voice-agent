"""SQLite schema for the clinic: doctors, slots, patients, appointments (PLAN M2.1).

The booking invariants live in the database, not only in Python, so they hold for every
connection, whatever code is writing (concurrent writers are tested in M2.3):
- one slot holds at most one *active* appointment: a partial UNIQUE index on
  appointments(slot_id) WHERE status = 'booked'. A cancelled appointment frees the slot.
- whether a slot is booked is derived from appointments only. slots.status is just 'open' or
  'blocked' (doctor unavailable), so the two tables can never disagree.
- times are ISO 8601 strings in IST with an explicit +05:30 offset; a CHECK (a format check,
  not full date validation) rejects anything else, including naive times. One fixed offset
  also makes string order equal time order.
- a blocked slot never holds an active appointment: triggers refuse booking (or re-booking)
  a blocked slot, and refuse blocking a slot that is booked. book() checks first so callers get
  a typed error; the triggers are the safety net for any other writer (Cyril, option 3).
- every idempotency_key is unique, so a retried book() can find its first appointment.
- the phone identifies a patient, but the name is stored per appointment: families share one
  phone, so Sunil can book on Asha's number and be confirmed as Sunil (Cyril, option A).
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from clinic.clock import IST

IST_TIME = "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+05:30'"

SCHEMA = f"""
CREATE TABLE doctors (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    speciality  TEXT NOT NULL
);
CREATE TABLE patients (
    id     INTEGER PRIMARY KEY,
    name   TEXT NOT NULL,
    phone  TEXT NOT NULL UNIQUE
);
CREATE TABLE slots (
    id            TEXT PRIMARY KEY,
    doctor_id     INTEGER NOT NULL REFERENCES doctors(id),
    start         TEXT NOT NULL CHECK (start GLOB {IST_TIME}),
    duration_min  INTEGER NOT NULL CHECK (duration_min > 0),
    status        TEXT NOT NULL CHECK (status IN ('open', 'blocked')),
    UNIQUE (doctor_id, start)
);
CREATE TABLE appointments (
    id               INTEGER PRIMARY KEY,
    slot_id          TEXT NOT NULL REFERENCES slots(id),
    patient_id       INTEGER NOT NULL REFERENCES patients(id),
    patient_name     TEXT NOT NULL CHECK (length(trim(patient_name)) > 0),
    status           TEXT NOT NULL CHECK (status IN ('booked', 'cancelled')),
    idempotency_key  TEXT NOT NULL UNIQUE,
    created_at       TEXT NOT NULL CHECK (created_at GLOB {IST_TIME})
);
CREATE UNIQUE INDEX one_active_appointment_per_slot
    ON appointments (slot_id) WHERE status = 'booked';
CREATE TRIGGER no_booking_on_blocked_slot
    BEFORE INSERT ON appointments
    WHEN NEW.status = 'booked'
     AND (SELECT status FROM slots WHERE id = NEW.slot_id) = 'blocked'
BEGIN SELECT RAISE(ABORT, 'slot is blocked'); END;
CREATE TRIGGER no_rebooking_on_blocked_slot
    BEFORE UPDATE OF status, slot_id ON appointments
    WHEN NEW.status = 'booked'
     AND (SELECT status FROM slots WHERE id = NEW.slot_id) = 'blocked'
BEGIN SELECT RAISE(ABORT, 'slot is blocked'); END;
CREATE TRIGGER no_blocking_a_booked_slot
    BEFORE UPDATE OF status ON slots
    WHEN NEW.status = 'blocked'
     AND EXISTS (SELECT 1 FROM appointments WHERE slot_id = NEW.id AND status = 'booked')
BEGIN SELECT RAISE(ABORT, 'slot has an active appointment'); END;
"""


def connect(path: Path | str, timeout: float = 5) -> sqlite3.Connection:
    """Open the clinic DB with foreign keys on (SQLite leaves them off by default).

    timeout: seconds to wait for another connection's write lock before "database is locked".
    """
    conn = sqlite3.connect(path, timeout=timeout)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def to_db_time(moment: datetime) -> str:
    """Timezone-aware datetime -> the stored form, e.g. '2026-11-30T09:00:00+05:30'."""
    if moment.tzinfo is None:
        raise ValueError("refusing a naive datetime: slot logic must be timezone-aware")
    return moment.astimezone(IST).replace(microsecond=0).isoformat()


def from_db_time(text: str) -> datetime:
    """Stored form -> aware datetime in IST. Refuses naive text: astimezone() would read it as
    the server's local time and silently shift it (review of M2.1)."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"refusing naive time {text!r}: slot logic must be timezone-aware")
    return moment.astimezone(IST)
