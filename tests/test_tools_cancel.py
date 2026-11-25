"""Unit tests for clinic.tools.cancel: phone + appointment id, or phone + slot (PLAN M2.2).

The phone is always required, so a guessed appointment id can't cancel someone else's booking.
"""

import sqlite3
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from clinic.clock import IST, FrozenClock
from clinic.tools import (
    CancelIn,
    CancelOut,
    CheckSlotsIn,
    CheckSlotsOut,
    ToolError,
    cancel,
    check_slots,
)

# Seed: appointment 1 = Asha (+919800000001) on d1-20261130-0900; 2 = Ravi (+919800000002) on
# d1-20261130-1730; 3 = Meera (+919800000003) on d2-20261130-1030; 5 = Asha, already cancelled.
ASHA, RAVI, MEERA = "9800000001", "9800000002", "9800000003"


def status(conn: sqlite3.Connection, appointment_id: int) -> str:
    return conn.execute(
        "SELECT status FROM appointments WHERE id = ?", (appointment_id,)
    ).fetchone()[0]


def test_cancel_by_id_frees_the_slot(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = cancel(conn, clock, CancelIn(appointment_id=1, phone=ASHA))
    assert isinstance(out, CancelOut) and out.ok is True
    assert (out.appointment_id, out.slot_id, out.status) == (1, "d1-20261130-0900", "cancelled")
    assert status(conn, 1) == "cancelled"
    monday = CheckSlotsIn(doctor="Sharma", date_from=date(2026, 11, 30), date_to=date(2026, 11, 30))
    free = check_slots(conn, clock, monday)
    assert isinstance(free, CheckSlotsOut) and free.slots[0].slot_id == "d1-20261130-0900"


def test_cancel_by_phone_and_slot(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = cancel(conn, clock, CancelIn(phone="98000 00002", slot_id="d1-20261130-1730"))
    assert isinstance(out, CancelOut) and out.appointment_id == 2
    assert status(conn, 2) == "cancelled"


@pytest.mark.parametrize(
    ("make", "code"),
    [
        (lambda: CancelIn(appointment_id=999, phone=ASHA), "not_found"),
        (lambda: CancelIn(appointment_id=3, phone=ASHA), "not_found"),  # Meera's, wrong phone
        (lambda: CancelIn(slot_id="d2-20261130-1030", phone=RAVI), "not_found"),  # Meera's slot
        (
            lambda: CancelIn(phone="9800000001", slot_id="d1-20261130-1730"),
            "not_found",
        ),  # not Asha's
        (lambda: CancelIn(appointment_id=5, phone=ASHA), "already_cancelled"),
        (lambda: CancelIn(phone="9800000001", slot_id="d3-20261130-1700"), "already_cancelled"),
    ],
)
def test_typed_errors(conn: sqlite3.Connection, clock: FrozenClock, make, code: str) -> None:  # noqa: ANN001
    before = conn.execute("SELECT id, status FROM appointments ORDER BY id").fetchall()
    out = cancel(conn, clock, make())
    assert isinstance(out, ToolError) and out.code == code
    assert conn.execute("SELECT id, status FROM appointments ORDER BY id").fetchall() == before
    assert not conn.in_transaction


def test_cancelling_twice(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    assert isinstance(cancel(conn, clock, CancelIn(appointment_id=1, phone=ASHA)), CancelOut)
    again = cancel(conn, clock, CancelIn(appointment_id=1, phone=ASHA))
    assert isinstance(again, ToolError) and again.code == "already_cancelled"


def test_cannot_cancel_an_appointment_that_has_started(conn: sqlite3.Connection) -> None:
    later = FrozenClock(datetime(2026, 11, 30, 9, 10, tzinfo=IST))
    out = cancel(conn, later, CancelIn(appointment_id=1, phone=ASHA))
    assert isinstance(out, ToolError) and out.code == "past_slot"
    assert status(conn, 1) == "booked"


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"phone": "9800000002"},
        {"slot_id": "d1-20261130-1730"},
        {"appointment_id": 2},  # an id alone is not enough: the phone proves whose it is
        {"appointment_id": 2, "slot_id": "d1-20261130-1730"},
        {"appointment_id": 2, "phone": "9800000002", "slot_id": "d1-20261130-1730"},
        {"phone": "12345", "slot_id": "d1-20261130-1730"},
    ],
)
def test_input_must_be_id_or_phone_plus_slot(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CancelIn.model_validate(fields)


def test_exception_mid_write_rolls_back(
    conn: sqlite3.Connection, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import clinic.tools

    def fail_after_update(**_: object) -> None:
        raise RuntimeError("crash after the UPDATE")

    monkeypatch.setattr(clinic.tools, "CancelOut", fail_after_update)
    with pytest.raises(RuntimeError, match="after the UPDATE"):
        cancel(conn, clock, CancelIn(appointment_id=1, phone=ASHA))
    assert status(conn, 1) == "booked"
    assert not conn.in_transaction


def test_refuses_to_run_inside_a_caller_transaction(
    conn: sqlite3.Connection, clock: FrozenClock
) -> None:
    conn.execute("UPDATE patients SET name = name")
    with pytest.raises(RuntimeError, match="transaction"):
        cancel(conn, clock, CancelIn(appointment_id=1, phone=ASHA))


@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_appointment_id_must_be_a_real_integer(value: object) -> None:
    # Review: {"appointment_id": true} was coerced to 1 and cancelled someone's appointment.
    with pytest.raises(ValidationError):
        CancelIn.model_validate({"appointment_id": value, "phone": ASHA})


def test_meera_cannot_be_cancelled_from_another_phone(
    conn: sqlite3.Connection, clock: FrozenClock
) -> None:
    # Review must-fix #1: the bare id 3 used to cancel Meera's appointment with no phone at all.
    for other in (ASHA, RAVI, "9876543210"):
        out = cancel(conn, clock, CancelIn(appointment_id=3, phone=other))
        assert isinstance(out, ToolError) and out.code == "not_found"
        assert "3" not in out.message  # don't confirm the id exists
    assert status(conn, 3) == "booked"
    assert isinstance(cancel(conn, clock, CancelIn(appointment_id=3, phone=MEERA)), CancelOut)
