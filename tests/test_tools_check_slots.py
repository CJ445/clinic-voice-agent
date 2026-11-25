"""Unit tests for clinic.tools.check_slots: only bookable slots, typed errors (PLAN M2.2)."""

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from clinic.clock import IST, FrozenClock
from clinic.tools import CheckSlotsIn, CheckSlotsOut, ToolError, check_slots

MON, TUE, SUN = date(2026, 11, 30), date(2026, 12, 1), date(2026, 12, 6)


def ids(out: CheckSlotsOut | ToolError) -> list[str]:
    assert isinstance(out, CheckSlotsOut), out
    return [s.slot_id for s in out.slots]


def ask(conn: sqlite3.Connection, clock: FrozenClock, **kw: object) -> CheckSlotsOut | ToolError:
    return check_slots(conn, clock, CheckSlotsIn.model_validate(kw))


def test_booked_slots_are_not_offered(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    got = ids(ask(conn, clock, doctor="Dr. Sharma", date_from=MON, date_to=MON, max_results=50))
    assert "d1-20261130-0900" not in got and "d1-20261130-1730" not in got
    assert "d1-20261130-0930" in got
    assert len(got) == 14 - 2  # 8 morning + 6 evening slots, two booked


def test_cancelled_appointment_frees_the_slot(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    got = ids(ask(conn, clock, doctor="Khan", date_from=MON, date_to=MON, part_of_day="evening"))
    assert got == [
        "d3-20261130-1700",
        "d3-20261130-1730",
        "d3-20261130-1800",
        "d3-20261130-1830",
        "d3-20261130-1930",
    ]


def test_blocked_slots_are_not_offered(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    got = ids(ask(conn, clock, doctor="Iyer", date_from=TUE, date_to=TUE))
    assert "d2-20261201-0900" not in got and "d2-20261201-0930" not in got
    assert got[0] == "d2-20261201-1000"


@pytest.mark.parametrize(
    ("part", "hours"), [("morning", {9, 10, 11}), ("afternoon", {12}), ("evening", {17, 18, 19})]
)
def test_part_of_day(
    conn: sqlite3.Connection, clock: FrozenClock, part: str, hours: set[int]
) -> None:
    out = ask(conn, clock, doctor="Sharma", date_from=MON, date_to=MON, part_of_day=part)
    assert isinstance(out, CheckSlotsOut) and out.slots
    assert {s.start.hour for s in out.slots} == hours


@pytest.mark.parametrize("name", ["Sharma", "dr. sharma", "Dr. Sharma", "  DR SHARMA "])
def test_doctor_name_forms(conn: sqlite3.Connection, clock: FrozenClock, name: str) -> None:
    out = ask(conn, clock, doctor=name, date_from=MON, date_to=MON)
    assert isinstance(out, CheckSlotsOut) and {s.doctor for s in out.slots} == {"Dr. Sharma"}


def test_unknown_doctor_is_a_typed_error(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = ask(conn, clock, doctor="Dr. Who", date_from=MON, date_to=MON)
    assert isinstance(out, ToolError) and out.code == "unknown_doctor" and out.ok is False


def test_reversed_range_is_a_typed_error(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = ask(conn, clock, date_from=TUE, date_to=MON)
    assert isinstance(out, ToolError) and out.code == "invalid_range"


def test_any_doctor_and_sunday(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = ask(conn, clock, date_from=MON, date_to=MON, max_results=50)
    assert isinstance(out, CheckSlotsOut) and len({s.doctor for s in out.slots}) == 3
    assert ids(ask(conn, clock, date_from=SUN, date_to=SUN)) == []


def test_past_slots_are_not_offered(conn: sqlite3.Connection) -> None:
    later = FrozenClock(datetime(2026, 11, 30, 10, 15, tzinfo=IST))
    out = ask(conn, later, doctor="Sharma", date_from=MON, date_to=MON)
    assert isinstance(out, CheckSlotsOut) and min(s.start for s in out.slots) > later.now()


def test_output_is_aware_ist_with_end(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = ask(conn, clock, doctor="Sharma", date_from=MON, date_to=MON)
    assert isinstance(out, CheckSlotsOut) and out.ok is True
    first = out.slots[0]
    assert first.start.isoformat() == "2026-11-30T09:30:00+05:30"
    assert first.end - first.start == timedelta(minutes=30)


def test_results_are_capped_and_say_so(conn: sqlite3.Connection, clock: FrozenClock) -> None:
    out = ask(conn, clock, date_from=MON, date_to=TUE, max_results=3)
    assert isinstance(out, CheckSlotsOut) and len(out.slots) == 3 and out.more is True
    assert [s.start for s in out.slots] == sorted(s.start for s in out.slots)
    few = ask(conn, clock, doctor="Khan", date_from=MON, date_to=MON, max_results=50)
    assert isinstance(few, CheckSlotsOut) and few.more is False
