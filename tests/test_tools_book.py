"""Unit tests for clinic.tools.book: slot-ID booking, idempotency, typed errors (PLAN M2.2)."""

import sqlite3
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from clinic.clock import IST, FrozenClock
from clinic.tools import BookIn, BookOut, CheckSlotsIn, CheckSlotsOut, ToolError, book, check_slots

FREE = "d1-20261130-0930"  # Dr. Sharma, open (conftest docstring lists the seed facts)


def req(slot_id: str = FREE, key: str = "k-1", phone: str = "9876543210") -> BookIn:
    return BookIn(slot_id=slot_id, patient_name="Sunil Rao", phone=phone, idempotency_key=key)


def count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_books_a_free_slot(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = book(conn, clock, req())
    assert isinstance(out, BookOut) and out.ok is True and out.replayed is False
    assert (out.slot_id, out.doctor, out.patient_name) == (FREE, "Dr. Sharma", "Sunil Rao")
    assert out.start.isoformat() == "2026-11-30T09:30:00+05:30"
    row = conn.execute("SELECT status FROM appointments WHERE id = ?", (out.appointment_id,))
    assert row.fetchone() == ("booked",)
    free = check_slots(
        conn,
        clock,
        CheckSlotsIn(date_from=date(2026, 11, 30), date_to=date(2026, 11, 30), max_results=50),
    )
    assert isinstance(free, CheckSlotsOut) and FREE not in [s.slot_id for s in free.slots]


def test_existing_patient_is_reused_by_phone(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    before = count(conn, "patients")
    assert isinstance(book(conn, clock, req(phone="+91 98000 00001")), BookOut)  # Asha's number
    assert count(conn, "patients") == before


@pytest.mark.parametrize(
    ("slot_id", "code"),
    [
        ("kal 5 baje", "unknown_slot"),  # a free-form time is never a slot id
        ("d9-20261130-0930", "unknown_slot"),
        ("d1-20261130-0900", "slot_taken"),
        ("d2-20261201-0900", "slot_blocked"),
    ],
)
def test_typed_errors(
    conn: sqlite3.Connection, clock: FrozenClock, slot_id: str, code: str
) -> None:
    before = count(conn, "appointments")
    out = book(conn, clock, req(slot_id=slot_id))
    assert isinstance(out, ToolError) and out.code == code and out.ok is False
    assert count(conn, "appointments") == before  # rolled back
    assert not conn.in_transaction


def test_past_slot(conn: sqlite3.Connection) -> None:
    later = FrozenClock(datetime(2026, 11, 30, 10, 15, tzinfo=IST))
    out = book(conn, later, req())
    assert isinstance(out, ToolError) and out.code == "past_slot"


def test_retry_with_same_key_returns_the_same_appointment(
    conn: sqlite3.Connection, clock: FrozenClock
) -> None:
    first = book(conn, clock, req())
    again = book(conn, clock, req())
    assert isinstance(first, BookOut) and isinstance(again, BookOut)
    assert again.appointment_id == first.appointment_id and again.replayed is True
    assert conn.execute(
        "SELECT COUNT(*) FROM appointments WHERE slot_id = ?", (FREE,)
    ).fetchone() == (1,)


@pytest.mark.parametrize(
    ("other", "why"),
    [
        (lambda: req(slot_id="d1-20261130-1000"), "same key, different slot"),
        (lambda: req(phone="9000000009"), "same key, different patient"),
    ],
)
def test_reused_key_with_different_details_conflicts(
    conn: sqlite3.Connection, clock: FrozenClock, other, why: str
) -> None:  # noqa: ANN001
    assert isinstance(book(conn, clock, req()), BookOut)
    out = book(conn, clock, other())
    assert isinstance(out, ToolError) and out.code == "idempotency_conflict", why


def test_key_of_a_cancelled_appointment_conflicts(
    conn: sqlite3.Connection, clock: FrozenClock
) -> None:
    # seed-5 belongs to a cancelled appointment on d3-20261130-1700 (Asha).
    out = book(conn, clock, req(slot_id="d3-20261130-1700", key="seed-5", phone="+919800000001"))
    assert isinstance(out, ToolError) and out.code == "idempotency_conflict"


@pytest.mark.parametrize("phone", ["9876543210", "+919876543210", "91 98765 43210", "098765-43210"])
def test_phone_is_normalized(phone: str) -> None:
    assert req(phone=phone).phone == "+919876543210"


@pytest.mark.parametrize(
    "phone",
    ["12345", "5876543210", "+1 415 555 0100", "abcdefghij", "98765x43210", "98+76543210", ""],
)
def test_bad_phone_is_rejected_by_the_model(phone: str) -> None:
    with pytest.raises(ValidationError):
        req(phone=phone)


def test_refuses_to_run_inside_a_caller_transaction(
    conn: sqlite3.Connection, clock: FrozenClock
) -> None:
    conn.execute("UPDATE patients SET name = name")  # opens an implicit transaction
    with pytest.raises(RuntimeError, match="transaction"):
        book(conn, clock, req())
