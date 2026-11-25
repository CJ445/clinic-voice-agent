"""Unit tests for clinic.tools.run and tool_schemas: the LLM-facing entry point (PLAN M2.2).

run() takes the raw JSON arguments an LLM produced. Whatever they are, it must answer with a
JSON-serialisable dict, and turn bad input into a typed invalid_input error, never an exception.
"""

import json
import sqlite3

import pytest

from clinic.clock import FrozenClock
from clinic.tools import run, tool_schemas


def test_check_slots_from_raw_json_args(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = run(
        conn,
        clock,
        "check_slots",
        {"doctor": "Khan", "date_from": "2026-11-30", "date_to": "2026-11-30"},
    )
    assert out["ok"] is True and out["slots"][0]["slot_id"] == "d3-20261130-1700"
    assert out["slots"][0]["start"] == "2026-11-30T17:00:00+05:30"
    json.dumps(out)


def test_book_then_replay_through_run(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    args = {
        "slot_id": "d1-20261130-0930",
        "patient_name": "Sunil",
        "phone": "9876543210",
        "idempotency_key": "k",
    }
    first = run(conn, clock, "book", args)
    again = run(conn, clock, "book", args)
    assert first["ok"] is True and again["appointment_id"] == first["appointment_id"]
    assert again["replayed"] is True


def test_cancel_through_run(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    assert run(conn, clock, "cancel", {"appointment_id": 1})["status"] == "cancelled"


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        (
            "book",
            {
                "slot_id": "d1-20261130-0930",
                "patient_name": "S",
                "phone": "123",
                "idempotency_key": "k",
            },
            "phone",
        ),
        ("book", {"slot_id": "d1-20261130-0930"}, "patient_name"),
        ("check_slots", {"date_from": "kal", "date_to": "2026-11-30"}, "date_from"),
        (
            "check_slots",
            {"date_from": "2026-11-30", "date_to": "2026-11-30", "when": "shaam"},
            "when",
        ),
        ("cancel", {"phone": "9800000002"}, "slot_id"),
    ],
)
def test_bad_args_are_a_typed_error(
    conn: sqlite3.Connection, clock: FrozenClock, name: str, args: dict[str, object], field: str
) -> None:
    out = run(conn, clock, name, args)
    assert out["ok"] is False and out["code"] == "invalid_input"
    assert field in out["message"]
    json.dumps(out)


def test_unknown_tool(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = run(conn, clock, "reschedule", {})
    assert out == {"ok": False, "code": "unknown_tool", "message": out["message"]}
    assert "check_slots" in out["message"]


def test_every_tool_has_a_json_schema() -> None:
    schemas = tool_schemas()
    assert set(schemas) == {"check_slots", "book", "cancel"}
    for name, schema in schemas.items():
        assert schema["name"] == name and schema["description"]
        assert schema["parameters"]["type"] == "object"
        json.dumps(schema)
    book = schemas["book"]["parameters"]
    assert set(book["required"]) == {"slot_id", "patient_name", "phone", "idempotency_key"}
    assert "check_slots" in schemas["book"]["description"]  # tells the LLM where slot_ids come from
    assert book["additionalProperties"] is False


def test_db_contention_is_a_typed_busy_error(tmp_path, clock: FrozenClock) -> None:  # noqa: ANN001
    # Review must-fix #2: two lines booking at once can outlast the busy timeout; the turn must
    # get a typed error it can recover from, not a crash.
    from clinic.db import connect
    from clinic.seed import build

    build(tmp_path / "c.db")
    holder = connect(tmp_path / "c.db")
    holder.execute("BEGIN IMMEDIATE")  # another phone line holds the write lock
    waiter = connect(tmp_path / "c.db", timeout=0.05)
    for name, args in [
        ("cancel", {"phone": "9800000002", "slot_id": "d1-20261130-1730"}),
        (
            "book",
            {
                "slot_id": "d1-20261130-0930",
                "patient_name": "S",
                "phone": "9876543210",
                "idempotency_key": "k",
            },
        ),
    ]:
        out = run(waiter, clock, name, args)
        assert (out["ok"], out["code"]) == (False, "busy")
        assert not waiter.in_transaction
    holder.rollback()


@pytest.mark.parametrize(
    ("db_error", "code"),
    [
        ("slot is blocked", "slot_blocked"),
        ("UNIQUE constraint failed: appointments.slot_id", "slot_taken"),
    ],
)
def test_db_constraint_raised_mid_write_is_a_typed_error(
    conn: sqlite3.Connection,
    clock: FrozenClock,
    monkeypatch: pytest.MonkeyPatch,
    db_error: str,
    code: str,
) -> None:
    # The DB constraints (unique index, option-3 triggers) are the safety net behind book()'s own
    # checks; if one fires, the LLM still gets the matching typed error.
    import clinic.tools

    def raise_integrity(*_: object) -> None:
        raise sqlite3.IntegrityError(db_error)

    monkeypatch.setattr(clinic.tools, "_book", raise_integrity)
    args = {
        "slot_id": "d1-20261130-0930",
        "patient_name": "S",
        "phone": "9876543210",
        "idempotency_key": "k",
    }
    out = run(conn, clock, "book", args)
    assert (out["ok"], out["code"]) == (False, code)
    assert not conn.in_transaction


def test_programming_errors_still_raise(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    # A caller holding an open transaction is a bug in our code, not bad LLM input: fail loudly.
    conn.execute("UPDATE patients SET name = name")
    with pytest.raises(RuntimeError, match="transaction"):
        run(conn, clock, "cancel", {"appointment_id": 1})


@pytest.mark.parametrize(
    "exc",
    [
        sqlite3.IntegrityError("NOT NULL constraint failed: x"),
        sqlite3.OperationalError("no such table: x"),
    ],
)
def test_other_db_errors_are_not_disguised(
    conn: sqlite3.Connection, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    # Only contention and the known safety-net constraints are mapped; anything else is a bug.
    import clinic.tools

    def boom(*_: object) -> None:
        raise exc

    monkeypatch.setattr(clinic.tools, "_cancel", boom)
    with pytest.raises(type(exc)):
        run(conn, clock, "cancel", {"appointment_id": 1})
