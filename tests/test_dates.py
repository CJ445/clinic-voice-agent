"""Table tests for clinic.dates.resolve (PLAN M2.4): English, Devanagari Hindi and Hinglish.

Each row is (what the caller said, what resolve must return). "2026-12-01 evening" means one
day plus a part of day; "ambiguous: <word>" means an Ambiguous whose reason contains <word>.
Unless a row says otherwise, now is MON = Monday 30 Nov 2026, 08:00 IST.
"""

from datetime import UTC, datetime

import pytest

from clinic.clock import IST
from clinic.dates import Ambiguous, Resolved, resolve

MON = datetime(2026, 11, 30, 8, 0, tzinfo=IST)


def show(result: Resolved | Ambiguous) -> str:
    if isinstance(result, Ambiguous):
        return f"ambiguous: {result.reason}"
    assert result.date_from == result.date_to  # single days only, so far
    day = result.date_from.isoformat()
    return f"{day} {result.part_of_day}" if result.part_of_day else day


RELATIVE_DAYS = [
    ("aaj", "2026-11-30"),
    ("आज शाम", "2026-11-30 evening"),
    ("today morning", "2026-11-30 morning"),
    ("kal", "2026-12-01"),
    ("kal shaam", "2026-12-01 evening"),
    ("कल सुबह", "2026-12-01 morning"),
    ("kal subah aana hai", "2026-12-01 morning"),
    ("Kal Shaam!", "2026-12-01 evening"),
    ("tomorrow afternoon", "2026-12-01 afternoon"),
    ("kal dopahar ko", "2026-12-01 afternoon"),
    ("कल दोपहर", "2026-12-01 afternoon"),
    ("parso", "2026-12-02"),
    ("परसों शाम को", "2026-12-02 evening"),
    ("day after tomorrow", "2026-12-02"),
    ("the day after tomorrow in the morning", "2026-12-02 morning"),
    ("आज शाम बुक कर दो", "2026-11-30 evening"),  # कर here is "do", a day word is present
    ("book kar do kal subah", "2026-12-01 morning"),
]

AMBIGUOUS_RELATIVE = [
    ("कर", "ambiguous: कल"),  # STT error for कल: never guess tomorrow from it
    ("कर शाम को", "ambiguous: कल"),
    ("kal gaya tha", "ambiguous: past"),  # kal with past tense = yesterday, not bookable
    ("कल आया था", "ambiguous: past"),
    ("yesterday", "ambiguous: past"),
    ("aaj ya kal", "ambiguous: more than one day"),
    ("kal parso", "ambiguous: more than one day"),  # idiom: "in a day or two"
    ("shaam ko", "ambiguous: no day"),
    ("", "ambiguous: no day"),
    ("kal subah ya shaam", "ambiguous: more than one part of the day"),
]


def check(text: str, expected: str, now: datetime = MON) -> None:
    got = show(resolve(text, now))
    if expected.startswith("ambiguous: "):
        assert got.startswith("ambiguous: "), got
        assert expected.removeprefix("ambiguous: ") in got, got
    else:
        assert got == expected


@pytest.mark.parametrize(("text", "expected"), RELATIVE_DAYS + AMBIGUOUS_RELATIVE)
def test_relative_days(text: str, expected: str) -> None:
    check(text, expected)


def test_now_is_read_in_ist() -> None:
    # 19:00 UTC on 30 Nov is 00:30 IST on 1 Dec, so "aaj" is 1 Dec.
    late = datetime(2026, 11, 30, 19, 0, tzinfo=UTC)
    assert show(resolve("aaj", late)) == "2026-12-01"
    assert show(resolve("kal", late)) == "2026-12-02"


def test_naive_now_is_refused() -> None:
    with pytest.raises(ValueError, match="aware"):
        resolve("aaj", datetime(2026, 11, 30, 8, 0))  # noqa: DTZ001
