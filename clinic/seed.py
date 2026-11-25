"""Deterministic seed data for the clinic DB (PLAN M2.1).

Everything is derived from one frozen clock, with no randomness, so the same clock always gives
the same database, row for row. Tests, the simulator and the CI gate can then check exact DB
state. Three doctors with different hours (so "subah" and "shaam" queries have different
answers), two weeks of 30-minute slots from the clock's day, closed on Sundays, a few blocked
slots, and a few pre-booked and cancelled appointments for cancel and conflict cases.

Usage: python -m clinic.seed PATH   (refuses to overwrite an existing file)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from clinic.clock import IST, Clock, FrozenClock
from clinic.db import connect, create_schema, to_db_time

SEED_NOW = datetime(2026, 11, 30, 8, 0, tzinfo=IST)  # a Monday, before the first slot
DAYS = 14
SLOT_MIN = 30
MORNING = (9, 13)  # 09:00-13:00
EVENING = (17, 20)  # 17:00-20:00

# (id, name, speciality, sessions)
DOCTORS = [
    (1, "Dr. Sharma", "General Physician", [MORNING, EVENING]),
    (2, "Dr. Iyer", "Paediatrics", [MORNING]),
    (3, "Dr. Khan", "Dermatology", [EVENING]),
]
PATIENTS = [
    (1, "Asha Verma", "+919800000001"),
    (2, "Ravi Kumar", "+919800000002"),
    (3, "Meera Nair", "+919800000003"),
    (4, "Imran Shaikh", "+919800000004"),
]
# (doctor_id, n-th slot of that doctor counted from the clock) -> blocked (doctor unavailable)
BLOCKED = [(2, 8), (2, 9), (1, 20)]
# (doctor_id, n-th slot, patient_id, status). Picked by position, so any clock works.
APPOINTMENTS = [
    (1, 0, 1, "booked"),
    (1, 9, 2, "booked"),
    (2, 3, 3, "booked"),
    (3, 4, 4, "booked"),
    (3, 0, 1, "cancelled"),
]


def slot_starts(sessions: list[tuple[int, int]], now: datetime) -> list[datetime]:
    """Every slot start for one doctor, in order, over DAYS days from now's date, after now."""
    starts = []
    for day in range(DAYS):
        date = now.date() + timedelta(days=day)
        if date.weekday() == 6:  # Sunday: closed
            continue
        for first, last in sessions:
            t = datetime(date.year, date.month, date.day, first, 0, tzinfo=IST)
            while t.hour < last:
                if t > now:
                    starts.append(t)
                t += timedelta(minutes=SLOT_MIN)
    return starts


def slot_id(doctor_id: int, start: datetime) -> str:
    """Stable, readable id, e.g. 'd1-20261130-0900'. Same doctor and time -> same id, always."""
    return f"d{doctor_id}-{start:%Y%m%d-%H%M}"


def build(path: Path | str, clock: Clock | None = None) -> None:
    """Create a new clinic DB at path, seeded from clock (default: FrozenClock(SEED_NOW))."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; seed only builds fresh databases")
    now = (clock or FrozenClock(SEED_NOW)).now()
    conn = connect(path)
    with conn:
        create_schema(conn)
        conn.executemany("INSERT INTO patients VALUES (?, ?, ?)", PATIENTS)
        ids: dict[int, list[str]] = {}
        for doctor_id, name, speciality, sessions in DOCTORS:
            conn.execute("INSERT INTO doctors VALUES (?, ?, ?)", (doctor_id, name, speciality))
            starts = slot_starts(sessions, now)
            ids[doctor_id] = [slot_id(doctor_id, s) for s in starts]
            conn.executemany(
                "INSERT INTO slots VALUES (?, ?, ?, ?, 'open')",
                [(slot_id(doctor_id, s), doctor_id, to_db_time(s), SLOT_MIN) for s in starts],
            )
        for doctor_id, n in BLOCKED:
            conn.execute("UPDATE slots SET status = 'blocked' WHERE id = ?", (ids[doctor_id][n],))
        names = {patient_id: name for patient_id, name, _ in PATIENTS}
        for i, (doctor_id, n, patient_id, status) in enumerate(APPOINTMENTS, start=1):
            conn.execute(
                "INSERT INTO appointments (id, slot_id, patient_id, patient_name, status,"
                " idempotency_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    i,
                    ids[doctor_id][n],
                    patient_id,
                    names[patient_id],
                    status,
                    f"seed-{i}",
                    to_db_time(now),
                ),
            )
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m clinic.seed PATH")
    build(sys.argv[1])
