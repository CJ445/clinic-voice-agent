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
