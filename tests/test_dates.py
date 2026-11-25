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


FRI = datetime(2026, 12, 4, 10, 0, tzinfo=IST)
SAT = datetime(2026, 12, 5, 10, 0, tzinfo=IST)

WEEKDAYS = [
    ("this Friday", MON, "2026-12-04"),
    ("is shukravar", MON, "2026-12-04"),
    ("इस शुक्रवार शाम", MON, "2026-12-04 evening"),
    ("Friday", MON, "2026-12-04"),
    ("fri morning", MON, "2026-12-04 morning"),
    ("shukravar ko subah", MON, "2026-12-04 morning"),
    ("mangalwar", MON, "2026-12-01"),
    ("budhvar dopahar", MON, "2026-12-02 afternoon"),
    ("guruvar", MON, "2026-12-03"),
    ("brihaspativar", MON, "2026-12-03"),
    ("गुरुवार", MON, "2026-12-03"),
    ("shaniwar", MON, "2026-12-05"),
    ("itwar", MON, "2026-12-06"),
    ("रविवार", MON, "2026-12-06"),
    ("this Monday", MON, "2026-11-30"),  # "this" + today's weekday = today
    ("next Monday", MON, "2026-12-07"),  # today's weekday: only next week's reading
    ("agle somvar", MON, "2026-12-07"),
    ("next Monday", SAT, "2026-12-07"),  # Monday has passed this week: one reading
    ("next Friday", FRI, "2026-12-11"),
    ("agle budhvar", FRI, "2026-12-09"),
    ("kal mangalvar", MON, "2026-12-01"),  # two day words, same day: fine
    ("next Friday", MON, "ambiguous: this week's"),  # Cyril: in-week "next" -> ask
    ("agle shukravar", MON, "ambiguous: this week's"),
    ("अगले शुक्रवार", MON, "ambiguous: this week's"),
    ("next Saturday", FRI, "ambiguous: this week's"),
    ("Monday", MON, "ambiguous: today"),  # bare weekday said on that weekday
    ("kal friday", MON, "ambiguous: more than one day"),
    ("agle hafte", MON, "ambiguous: no day"),  # "next week" names no day
    ("sun lijiye, kal shaam", MON, "2026-12-01 evening"),  # "sun" = listen, not Sunday
    ("kal sat baje", MON, "2026-12-01"),  # "sat" = seven (saat), not Saturday
]


@pytest.mark.parametrize(("text", "now", "expected"), WEEKDAYS)
def test_weekdays(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


def test_in_week_next_offers_both_readings() -> None:
    result = resolve("next Friday", MON)
    assert isinstance(result, Ambiguous)
    assert [d.isoformat() for d in result.candidates] == ["2026-12-04", "2026-12-11"]


DEC15 = datetime(2026, 12, 15, 10, 0, tzinfo=IST)
JAN31 = datetime(2027, 1, 31, 10, 0, tzinfo=IST)

# Cyril: the next such date on or after today; a month without that day is ambiguous.
DAYS_OF_MONTH = [
    ("the 5th", MON, "2026-12-05"),  # the 5th of November has passed
    ("5th", MON, "2026-12-05"),
    ("5 tarikh", MON, "2026-12-05"),
    ("5 tareekh ko shaam", MON, "2026-12-05 evening"),
    ("5 तारीख", MON, "2026-12-05"),
    ("५ तारीख", MON, "2026-12-05"),  # Devanagari digits
    ("the 30th", MON, "2026-11-30"),  # today
    ("1st", MON, "2026-12-01"),
    ("the 2nd", MON, "2026-12-02"),
    ("3rd morning", MON, "2026-12-03 morning"),
    ("the 10th", DEC15, "2027-01-10"),  # into next year
    ("friday the 4th", MON, "2026-12-04"),  # weekday and date agree
    ("the 1st, kal", MON, "2026-12-01"),
    ("kal 5 baje", MON, "2026-12-01"),  # a bare number is a time, not a date
    ("the 31st", MON, "ambiguous: no day 31"),  # November has no 31st
    ("the 30th", JAN31, "ambiguous: no day 30"),  # February has no 30th
    ("32nd", MON, "ambiguous: not a day of the month"),
    ("0 tarikh", MON, "ambiguous: not a day of the month"),
    ("friday the 5th", MON, "ambiguous: more than one day"),
    ("5 baje", MON, "ambiguous: no day"),
]


@pytest.mark.parametrize(("text", "now", "expected"), DAYS_OF_MONTH)
def test_days_of_month(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


def test_the_table_has_at_least_40_cases() -> None:
    # PLAN M2.4 exit: >= 40 table cases incl. Hinglish and ambiguous ones.
    table = RELATIVE_DAYS + AMBIGUOUS_RELATIVE + WEEKDAYS + DAYS_OF_MONTH
    ambiguous = [row for row in table if row[-1].startswith("ambiguous")]
    assert len(table) >= 40 and len(ambiguous) >= 10


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
