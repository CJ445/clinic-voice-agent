"""The agent's only way to read or change appointments: check_slots, book, cancel (PLAN M2.2).

Why tools and not free text: the LLM never writes a date into the DB. It asks check_slots for
real, bookable slots and books one by the slot_id it got back (slot-ID booking), so a
hallucinated time can't become an appointment. Every failure is a typed ToolError with a code
the agent can act on, never an exception string. Inputs and outputs are pydantic models, so each
tool also has a JSON schema for the LLM.

Scope: book() accepts any slot_id that check_slots *could* return (exists, open, future, free).
Checking it was returned in *this* conversation needs session state: the M3.2 slot-grounding guard.
"""

import inspect
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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
    "busy",
]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolError(Model):
    ok: Literal[False] = False
    code: ErrorCode
    message: str


class CheckSlotsIn(Model):
    """Find bookable slots. Dates are calendar days in IST, inclusive."""

    doctor: str | None = Field(
        default=None, description="Doctor's name, e.g. 'Dr. Sharma'; omit for any"
    )
    date_from: date
    date_to: date
    part_of_day: PartOfDay | None = None
    max_results: int = Field(default=10, ge=1, le=50)


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


def normalize_phone(raw: str) -> str:
    """Indian mobile number in a common spelling -> '+91XXXXXXXXXX'. Raises ValueError otherwise.

    Accepts digits, spaces and dashes, an optional leading '+', and a 91 or 0 prefix.
    """
    text = raw.strip()
    if text.startswith("+"):
        text = text[1:]
    if not text or any(ch not in "0123456789 -" for ch in text):
        raise ValueError("phone may contain only digits, spaces, dashes and a leading +")
    digits = text.replace(" ", "").replace("-", "")
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) != 10 or digits[0] not in "6789":
        raise ValueError("expected a 10-digit Indian mobile number, e.g. 98765 43210")
    return "+91" + digits


class BookIn(Model):
    """Book one slot. slot_id must come from check_slots; never a free-form date or time."""

    slot_id: str = Field(min_length=1, max_length=40)
    patient_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(description="Indian mobile number; normalized to +91XXXXXXXXXX")
    idempotency_key: str = Field(
        min_length=1, max_length=100, description="Same key = same booking"
    )

    @field_validator("patient_name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("patient_name is blank")
        return " ".join(value.split())

    @field_validator("phone")
    @classmethod
    def _phone(cls, value: str) -> str:
        return normalize_phone(value)


class BookOut(Model):
    ok: Literal[True] = True
    appointment_id: int
    slot_id: str
    doctor: str
    start: datetime
    end: datetime
    patient_name: str
    replayed: bool = Field(description="True if this idempotency_key had already booked it")


def book(conn: sqlite3.Connection, clock: Clock, req: BookIn) -> BookOut | ToolError:
    """Book req.slot_id, all checks and the insert under one write lock (BEGIN IMMEDIATE).

    A ToolError rolls the transaction back, so a failed call never leaves a partial write.
    """
    if conn.in_transaction:
        raise RuntimeError("book() must start its own transaction; commit or roll back first")
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = _book(conn, clock, req)
    except BaseException:
        conn.rollback()
        raise
    if isinstance(result, ToolError):
        conn.rollback()
    else:
        conn.commit()
    return result


def _booked(conn: sqlite3.Connection, appointment_id: int, replayed: bool) -> BookOut:
    row = conn.execute(
        """
        SELECT a.slot_id, d.name, s.start, s.duration_min, p.name
        FROM appointments a JOIN slots s ON s.id = a.slot_id
        JOIN doctors d ON d.id = s.doctor_id JOIN patients p ON p.id = a.patient_id
        WHERE a.id = ?
        """,
        (appointment_id,),
    ).fetchone()
    slot_id, doctor, start_text, duration, patient = row
    start = from_db_time(start_text)
    return BookOut(
        appointment_id=appointment_id,
        slot_id=slot_id,
        doctor=doctor,
        start=start,
        end=start + timedelta(minutes=duration),
        patient_name=patient,
        replayed=replayed,
    )


def _book(conn: sqlite3.Connection, clock: Clock, req: BookIn) -> BookOut | ToolError:
    prior = conn.execute(
        """
        SELECT a.id, a.slot_id, p.phone, a.status FROM appointments a
        JOIN patients p ON p.id = a.patient_id WHERE a.idempotency_key = ?
        """,
        (req.idempotency_key,),
    ).fetchone()
    if prior is not None:
        appointment_id, slot_id, phone, status = prior
        if (slot_id, phone, status) == (req.slot_id, req.phone, "booked"):
            return _booked(conn, appointment_id, replayed=True)
        return ToolError(
            code="idempotency_conflict",
            message="this idempotency_key was already used for a different or cancelled booking",
        )
    slot = conn.execute("SELECT status, start FROM slots WHERE id = ?", (req.slot_id,)).fetchone()
    if slot is None:
        return ToolError(code="unknown_slot", message="no such slot_id; use one from check_slots")
    status, start_text = slot
    if from_db_time(start_text) <= clock.now():
        return ToolError(code="past_slot", message="that slot has already started")
    if status == "blocked":
        return ToolError(code="slot_blocked", message="the doctor is unavailable for that slot")
    taken = "SELECT 1 FROM appointments WHERE slot_id = ? AND status = 'booked'"
    if conn.execute(taken, (req.slot_id,)).fetchone() is not None:
        return ToolError(code="slot_taken", message="that slot is already booked")
    patient = conn.execute("SELECT id FROM patients WHERE phone = ?", (req.phone,)).fetchone()
    if patient is None:
        cursor = conn.execute(
            "INSERT INTO patients (name, phone) VALUES (?, ?)", (req.patient_name, req.phone)
        )
        patient_id = cursor.lastrowid
    else:
        patient_id = patient[0]
    cursor = conn.execute(
        """
        INSERT INTO appointments (slot_id, patient_id, status, idempotency_key, created_at)
        VALUES (?, ?, 'booked', ?, ?)
        """,
        (req.slot_id, patient_id, req.idempotency_key, to_db_time(clock.now())),
    )
    assert cursor.lastrowid is not None
    return _booked(conn, cursor.lastrowid, replayed=False)


class CancelIn(Model):
    """Cancel by appointment_id, or by the patient's phone plus the slot_id. Exactly one way."""

    appointment_id: int | None = Field(default=None, ge=1)
    phone: str | None = None
    slot_id: str | None = Field(default=None, min_length=1, max_length=40)

    @field_validator("phone")
    @classmethod
    def _phone(cls, value: str | None) -> str | None:
        return None if value is None else normalize_phone(value)

    @model_validator(mode="after")
    def _one_way(self) -> "CancelIn":
        by_id = self.appointment_id is not None
        by_patient = self.phone is not None and self.slot_id is not None
        partial = (self.phone is None) != (self.slot_id is None)
        if by_id == by_patient or partial:
            raise ValueError("give appointment_id, or phone and slot_id (not both, not neither)")
        return self


class CancelOut(Model):
    ok: Literal[True] = True
    appointment_id: int
    slot_id: str
    status: Literal["cancelled"] = "cancelled"


def cancel(conn: sqlite3.Connection, clock: Clock, req: CancelIn) -> CancelOut | ToolError:
    """Cancel one appointment under a write lock; a ToolError rolls back (same pattern as book;
    factor it out on a third use, CLAUDE.md rule 5)."""
    if conn.in_transaction:
        raise RuntimeError("cancel() must start its own transaction; commit or roll back first")
    conn.execute("BEGIN IMMEDIATE")
    try:
        result = _cancel(conn, clock, req)
    except BaseException:
        conn.rollback()
        raise
    if isinstance(result, ToolError):
        conn.rollback()
    else:
        conn.commit()
    return result


def _cancel(conn: sqlite3.Connection, clock: Clock, req: CancelIn) -> CancelOut | ToolError:
    query = """
        SELECT a.id, a.slot_id, a.status, s.start FROM appointments a
        JOIN slots s ON s.id = a.slot_id JOIN patients p ON p.id = a.patient_id
    """
    if req.appointment_id is not None:
        rows = conn.execute(query + " WHERE a.id = ?", (req.appointment_id,)).fetchall()
    else:
        rows = conn.execute(
            query + " WHERE p.phone = ? AND a.slot_id = ?", (req.phone, req.slot_id)
        ).fetchall()
    if not rows:
        return ToolError(code="not_found", message="no matching appointment")
    active = [row for row in rows if row[2] == "booked"]  # at most one, by the unique index
    if not active:
        return ToolError(code="already_cancelled", message="that appointment is already cancelled")
    appointment_id, slot_id, _, start_text = active[0]
    if from_db_time(start_text) <= clock.now():
        return ToolError(code="past_slot", message="that appointment has already started")
    conn.execute("UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,))
    return CancelOut(appointment_id=appointment_id, slot_id=slot_id)


# name -> (input model, function). The only tools the agent has.
TOOLS: dict[str, tuple[type[Model], Any]] = {
    "check_slots": (CheckSlotsIn, check_slots),
    "book": (BookIn, book),
    "cancel": (CancelIn, cancel),
}


def run(conn: sqlite3.Connection, clock: Clock, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call a tool with the LLM's raw JSON arguments and return a JSON-ready dict.

    Bad arguments become ToolError(invalid_input) naming the field, so the agent can ask the
    caller again instead of crashing the turn. DB contention (another line holding the write lock
    past the busy timeout) becomes ToolError(busy), and a DB constraint firing mid-write becomes
    the matching slot error. Programming errors (e.g. calling inside an open transaction) still
    raise: they are bugs in our code, not caller input.
    """
    if name not in TOOLS:
        known = ", ".join(TOOLS)
        return ToolError(
            code="unknown_tool", message=f"no tool {name!r}; tools: {known}"
        ).model_dump(mode="json")
    model, tool = TOOLS[name]
    try:
        req = model.model_validate(args)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or 'input'}: {err['msg']}"
            for err in exc.errors()
        )
        return ToolError(code="invalid_input", message=problems).model_dump(mode="json")
    try:
        result = tool(conn, clock, req)
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc) and "busy" not in str(exc):
            raise
        result = ToolError(code="busy", message="the booking system is busy; try again shortly")
    except sqlite3.IntegrityError as exc:
        result = _constraint_error(exc)
    return result.model_dump(mode="json")


def _constraint_error(exc: sqlite3.IntegrityError) -> ToolError:
    """Map a DB safety-net constraint to the typed error book()/cancel() would have returned."""
    text = str(exc)
    if "slot is blocked" in text:
        return ToolError(code="slot_blocked", message="the doctor is unavailable for that slot")
    if "appointments.slot_id" in text:
        return ToolError(code="slot_taken", message="that slot is already booked")
    raise exc


def tool_schemas() -> dict[str, dict[str, Any]]:
    """One function schema per tool (name, description, JSON-schema parameters) for the LLM."""
    return {
        name: {
            "name": name,
            "description": inspect.cleandoc(model.__doc__ or name),
            "parameters": model.model_json_schema(),
        }
        for name, (model, _) in TOOLS.items()
    }
