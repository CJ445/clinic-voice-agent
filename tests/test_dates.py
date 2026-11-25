"""Table tests for clinic.dates.resolve (PLAN M2.4): English, Devanagari Hindi and Hinglish.

Each row is (what the caller said, what resolve must return). "2026-12-01 evening" means one
day plus a part of day; "ambiguous: <code>" means an Ambiguous with exactly that code.
Unless a row says otherwise, now is MON = Monday 30 Nov 2026, 08:00 IST.
"""

from datetime import UTC, datetime

import pytest

from clinic.clock import IST
from clinic.dates import Ambiguous, Resolved, resolve

MON = datetime(2026, 11, 30, 8, 0, tzinfo=IST)


def show(result: Resolved | Ambiguous) -> str:
    if isinstance(result, Ambiguous):
        assert result.reason, "every Ambiguous carries a reason the agent can use"
        return f"ambiguous: {result.code}"
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
    ("कर", "ambiguous: heard_kar"),  # STT error for कल: never guess tomorrow from it
    ("कर शाम को", "ambiguous: heard_kar"),
    ("kal gaya tha", "ambiguous: past"),  # kal with past tense = yesterday, not bookable
    ("कल आया था", "ambiguous: past"),
    ("yesterday", "ambiguous: past"),
    ("aaj ya kal", "ambiguous: two_days"),
    ("kal parso", "ambiguous: two_days"),  # idiom: "in a day or two"
    ("shaam ko", "ambiguous: no_day"),
    ("", "ambiguous: no_day"),
    ("kal subah ya shaam", "ambiguous: two_parts"),
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
    ("next Friday", MON, "ambiguous: which_week"),  # Cyril: in-week "next" -> ask
    ("agle shukravar", MON, "ambiguous: which_week"),
    ("अगले शुक्रवार", MON, "ambiguous: which_week"),
    ("next Saturday", FRI, "ambiguous: which_week"),
    ("Monday", MON, "ambiguous: weekday_is_today"),  # bare weekday said on that weekday
    ("kal friday", MON, "ambiguous: two_days"),
    ("agle hafte", MON, "ambiguous: no_day"),  # "next week" names no day
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
    ("the 31st", MON, "ambiguous: no_such_day"),  # November has no 31st
    ("the 30th", JAN31, "ambiguous: no_such_day"),  # February has no 30th
    ("32nd", MON, "ambiguous: not_a_day"),
    ("0 tarikh", MON, "ambiguous: not_a_day"),
    ("friday the 5th", MON, "ambiguous: two_days"),
    ("5 baje", MON, "ambiguous: no_day"),
]


@pytest.mark.parametrize(("text", "now", "expected"), DAYS_OF_MONTH)
def test_days_of_month(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


NOV2 = datetime(2026, 11, 2, 10, 0, tzinfo=IST)

# Review must-fix 2-5, #11, #12: each of these used to come back as a confident wrong date.
NOT_A_PLAIN_DAY = [
    ("कल आया था।", MON, "ambiguous: past"),  # danda glued to था hid the past tense
    ("कल शाम।", MON, "2026-12-01 evening"),
    ("आज शाम।", MON, "2026-11-30 evening"),
    ("कल शा\u200cम", MON, "2026-12-01 evening"),  # zero-width non-joiner inside a word
    ("kal nahi", MON, "ambiguous: negated"),
    ("not tomorrow", MON, "ambiguous: negated"),
    ("कल नहीं, परसों", MON, "ambiguous: negated"),
    ("Friday ko gaya tha", MON, "ambiguous: past"),
    ("aaj subah call kiya tha", MON, "ambiguous: past"),
    ("last Friday", MON, "ambiguous: past"),
    ("pichhle shukravar", MON, "ambiguous: past"),
    ("पिछले शुक्रवार", MON, "ambiguous: past"),
    ("kal raat", MON, "ambiguous: night"),  # often "last night"; and the clinic is shut
    ("आज रात", MON, "ambiguous: night"),
    ("tonight", MON, "ambiguous: night"),
    ("the 5th of January", MON, "ambiguous: month_named"),
    ("10th Jan", MON, "ambiguous: month_named"),
    ("5 tarikh agle mahine", NOV2, "ambiguous: month_named"),
    ("दिसंबर 5 तारीख", MON, "ambiguous: month_named"),
    ("may I come kal", MON, "2026-12-01"),  # "may" is not read as a month
    ("kiya kal slot hai", MON, "ambiguous: past"),  # STT "kiya" for kya: safe, costs a question
]


@pytest.mark.parametrize(("text", "now", "expected"), NOT_A_PLAIN_DAY)
def test_not_a_plain_day(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


SUN = datetime(2026, 12, 6, 10, 0, tzinfo=IST)

# Review #1 and #7 (Cyril: "next week X" = X in the following week; "next X" said on a Sunday is
# ambiguous, since Monday- and Sunday-start weeks disagree only then), #6 and #10.
WEEKS_AND_POSITIONS = [
    ("agle hafte friday", MON, "2026-12-11"),
    ("next week Friday", MON, "2026-12-11"),
    ("Friday next week", MON, "2026-12-11"),
    ("agle hafte ke somvar ko subah", MON, "2026-12-07 morning"),
    ("अगले हफ्ते शुक्रवार", MON, "2026-12-11"),
    ("next week Monday", SAT, "2026-12-07"),
    ("pichhle hafte friday", MON, "ambiguous: past"),
    ("next Saturday", SUN, "ambiguous: which_week"),
    ("next Monday", SUN, "ambiguous: which_week"),
    ("agle hafte friday", SUN, "ambiguous: which_week"),
    ("Friday", SUN, "2026-12-11"),  # bare weekday: still the coming one
    ("Is Monday ok?", MON, "ambiguous: weekday_is_today"),  # English "is", not Hindi "this"
    ("is somvar", MON, "2026-11-30"),  # Hindi "is" = this
    ("mera 2nd appointment hai shaam ko", MON, "ambiguous: no_day"),  # not a date position
    ("2nd opinion chahiye kal", MON, "2026-12-01"),
    ("3rd floor pe aana hai parso", MON, "2026-12-02"),
]


@pytest.mark.parametrize(("text", "now", "expected"), WEEKS_AND_POSITIONS)
def test_weeks_and_positions(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


AT_0030 = datetime(2026, 12, 1, 0, 30, tzinfo=IST)
AT_0359 = datetime(2026, 12, 1, 3, 59, tzinfo=IST)
AT_0400 = datetime(2026, 12, 1, 4, 0, tzinfo=IST)

# Review #8 (Cyril): from 00:00 to 04:00 IST, aaj/kal/tomorrow/parso are ambiguous: at 00:30, "kal
# subah" probably means the morning a few hours away, not the calendar day after.
AFTER_MIDNIGHT = [
    ("kal subah", AT_0030, "ambiguous: after_midnight"),
    ("कल", AT_0030, "ambiguous: after_midnight"),
    ("tomorrow", AT_0359, "ambiguous: after_midnight"),
    ("aaj shaam", AT_0030, "ambiguous: after_midnight"),
    ("kal subah", AT_0400, "2026-12-02 morning"),
    ("aaj shaam", AT_0400, "2026-12-01 evening"),
    ("parso", AT_0030, "ambiguous: after_midnight"),  # Cyril at M2 close-out: same as kal
    ("परसों शाम", AT_0030, "ambiguous: after_midnight"),
    ("day after tomorrow", AT_0359, "ambiguous: after_midnight"),
    ("parso", AT_0400, "2026-12-03"),
    ("Friday", AT_0030, "2026-12-04"),
]


@pytest.mark.parametrize(("text", "now", "expected"), AFTER_MIDNIGHT)
def test_after_midnight(text: str, now: datetime, expected: str) -> None:
    check(text, expected, now)


def test_after_midnight_kal_offers_both_days() -> None:
    result = resolve("kal subah", AT_0030)
    assert isinstance(result, Ambiguous)
    assert [d.isoformat() for d in result.candidates] == ["2026-12-01", "2026-12-02"]


def test_after_midnight_parso_offers_both_days() -> None:
    # At 00:30 on 1 Dec, "parso" may mean 2 Dec (counting from the coming morning) or 3 Dec.
    result = resolve("parso", AT_0030)
    assert isinstance(result, Ambiguous)
    assert [d.isoformat() for d in result.candidates] == ["2026-12-02", "2026-12-03"]


@pytest.mark.parametrize(
    ("text", "candidates"),
    [
        ("Friday ya parso", ["2026-12-02", "2026-12-03", "2026-12-04"]),  # Friday is kept
        ("parso ya kal", ["2026-12-01", "2026-12-02", "2026-12-03"]),
        ("kal ya parso", ["2026-12-01", "2026-12-02", "2026-12-03"]),  # word order doesn't matter
    ],
)
def test_after_midnight_keeps_every_day_named(text: str, candidates: list[str]) -> None:
    # Review (follow-up) #1: the early return used to drop the other days the caller named.
    result = resolve(text, AT_0030)
    assert isinstance(result, Ambiguous) and result.code == "after_midnight"
    assert [d.isoformat() for d in result.candidates] == candidates


@pytest.mark.parametrize("text", ["parso gaya tha", "kal aaya tha", "aaj call kiya tha"])
def test_past_tense_wins_after_midnight(text: str) -> None:
    # Review (follow-up) #2: past tense is the better reason to give the caller.
    check(text, "ambiguous: past", AT_0030)


def test_the_table_has_at_least_40_cases() -> None:
    # PLAN M2.4 exit: >= 40 table cases incl. Hinglish and ambiguous ones.
    table = RELATIVE_DAYS + AMBIGUOUS_RELATIVE + WEEKDAYS + DAYS_OF_MONTH
    table += NOT_A_PLAIN_DAY + WEEKS_AND_POSITIONS + AFTER_MIDNIGHT
    ambiguous = [row for row in table if row[-1].startswith("ambiguous")]
    assert len(table) >= 40 and len(ambiguous) >= 10


def check(text: str, expected: str, now: datetime = MON) -> None:
    assert show(resolve(text, now)) == expected


@pytest.mark.parametrize(("text", "expected"), RELATIVE_DAYS + AMBIGUOUS_RELATIVE)
def test_relative_days(text: str, expected: str) -> None:
    check(text, expected)


def test_now_is_read_in_ist() -> None:
    # 23:00 UTC on 30 Nov is 04:30 IST on 1 Dec, so "aaj" is 1 Dec. (Before 04:00 IST aaj/kal
    # are ambiguous; see test_after_midnight.)
    late = datetime(2026, 11, 30, 23, 0, tzinfo=UTC)
    assert show(resolve("aaj", late)) == "2026-12-01"
    assert show(resolve("kal", late)) == "2026-12-02"


def test_naive_now_is_refused() -> None:
    with pytest.raises(ValueError, match="aware"):
        resolve("aaj", datetime(2026, 11, 30, 8, 0))  # noqa: DTZ001


def test_every_word_list_is_nfc() -> None:
    # Input is NFC-normalised; a list word typed another way (e.g. precomposed फ़) never matches.
    import unicodedata

    from clinic import dates

    lists = [value for value in vars(dates).values() if isinstance(value, set)]
    lists += [set(dates.WEEKDAYS), *dates.PARTS.values()]
    assert len(lists) > 10
    for word in set().union(*lists):
        assert unicodedata.normalize("NFC", word) == word, word
