"""The agent's only way to read or change appointments: check_slots, book, cancel (PLAN M2.2).

Why tools and not free text: the LLM never writes a date into the DB. It asks check_slots for
real, bookable slots and books one by the slot_id it got back (slot-ID booking), so a
hallucinated time can't become an appointment. Every failure is a typed ToolError with a code
the agent can act on, never an exception string. Inputs and outputs are pydantic models, so each
tool also has a JSON schema for the LLM.

Scope: book() accepts any slot_id that check_slots *could* return (exists, open, future, free).
Checking it was returned in *this* conversation needs session state: the M3.2 slot-grounding guard.
"""

import sqlite3
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from clinic.clock import Clock
from clinic.db import from_db_time, to_db_time

PartOfDay = Literal["morning", "afternoon", "evening"]
# Hours [start, end) on the clinic's IST clock. A 12:30 slot is "afternoon".
PART_OF_DAY_HOURS: dict[str, tuple[int, int]] = {
    "morning": (5, 12),
    "afternoon": (12, 17),
    "evening": (17, 22),
}
ErrorCode = Literal[
    "invalid_input",
    "invalid_range",
    "unknown_doctor",
    "unknown_slot",
    "past_slot",
    "slot_blocked",
    "slot_taken",
    "idempotency_conflict",
    "not_found",
    "already_cancelled",
    "unknown_tool",
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolError(Model):
    ok: Literal[False] = False
    code: ErrorCode
    message: str


class CheckSlotsIn(Model):
    """Find bookable slots. Dates are calendar days in IST, inclusive."""

    doctor: str | None = Field(None, description="Doctor's name, e.g. 'Dr. Sharma'; omit for any")
    date_from: date
    date_to: date
    part_of_day: PartOfDay | None = None
    max_results: int = Field(10, ge=1, le=50)


class Slot(Model):
    slot_id: str
    doctor: str
    start: datetime
    end: datetime


class CheckSlotsOut(Model):
    ok: Literal[True] = True
    slots: list[Slot]
    more: bool = Field(description="True if more slots matched than max_results")


def _doctor_key(name: str) -> str:
    """'  DR. Sharma ' -> 'sharma': case, spaces and the Dr prefix don't matter."""
    key = " ".join(name.lower().replace(".", " ").split())
    return key.removeprefix("dr ").strip()


def _find_doctor(conn: sqlite3.Connection, name: str) -> tuple[int, str] | None:
    for doctor_id, full_name in conn.execute("SELECT id, name FROM doctors"):
        if _doctor_key(full_name) == _doctor_key(name):
            return doctor_id, full_name
    return None


def check_slots(
    conn: sqlite3.Connection, clock: Clock, req: CheckSlotsIn
) -> CheckSlotsOut | ToolError:
    if req.date_from > req.date_to:
        return ToolError(code="invalid_range", message="date_from is after date_to")
    doctor_id = None
    if req.doctor is not None:
        found = _find_doctor(conn, req.doctor)
        if found is None:
            return ToolError(code="unknown_doctor", message=f"no doctor named {req.doctor!r}")
        doctor_id = found[0]
    # Stored times share one +05:30 offset, so string comparison is time comparison.
    rows = conn.execute(
        """
        SELECT s.id, d.name, s.start, s.duration_min
        FROM slots s JOIN doctors d ON d.id = s.doctor_id
        WHERE s.status = 'open' AND s.start > ? AND substr(s.start, 1, 10) BETWEEN ? AND ?
          AND (? IS NULL OR s.doctor_id = ?)
          AND NOT EXISTS (
              SELECT 1 FROM appointments a WHERE a.slot_id = s.id AND a.status = 'booked'
          )
        ORDER BY s.start, d.id
        """,
        (
            to_db_time(clock.now()),
            req.date_from.isoformat(),
            req.date_to.isoformat(),
            doctor_id,
            doctor_id,
        ),
    ).fetchall()
    slots = []
    for slot_id, doctor, start_text, duration in rows:
        start = from_db_time(start_text)
        if req.part_of_day is not None:
            first, last = PART_OF_DAY_HOURS[req.part_of_day]
            if not first <= start.hour < last:
                continue
        slots.append(
            Slot(
                slot_id=slot_id, doctor=doctor, start=start, end=start + timedelta(minutes=duration)
            )
        )
    return CheckSlotsOut(slots=slots[: req.max_results], more=len(slots) > req.max_results)
