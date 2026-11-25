"""Turn what a caller said about *when* into IST dates, or say why it can't (PLAN M2.4).

resolve("kal shaam", now) -> Resolved(1 Dec, 1 Dec, "evening"); resolve("कर", now) -> Ambiguous.
It understands English, Devanagari Hindi and romanised Hinglish.

Why deterministic code and not the LLM: a wrong date is the one mistake a booking agent must
never make silently. LLMs are bad at calendar arithmetic ("next Friday" from a Monday), differ
run to run, and can't be unit-tested exhaustively; this function is pure, so every phrase has a
table test and the same input always gives the same date. When the words are unclear it doesn't
guess: it returns Ambiguous with a reason, and the agent asks the caller. The LLM's job is only
to pass the caller's words here and say the answer back.

Rules (Cyril's rulings in M2.4 are marked *):
- kal/कल/tomorrow is the next calendar day*, aaj is today, parso is the day after tomorrow.
  From 00:00 to 04:00 IST, aaj/kal/tomorrow/parso are ambiguous*: at 00:30 "kal subah" probably
  means the coming morning, and parso is off by one the same way.
- A weekday is the coming one; said on that weekday, it is ambiguous. "this/is X" is the coming X.
  "next X" is ambiguous while this week (Monday start) still has an X*, and always on a Sunday*.
  "agle hafte X" / "next week X" is X in the following week*.
- "the 5th" / "5 tarikh" is the next 5th on or after today*; a month without that day is
  ambiguous. An ordinal counts only in a date position ("2nd appointment" is not a date), and a
  bare number never does ("5 baje" is a time).
- Always ambiguous: a past-tense marker (tha, gaya, था...), last/pichhle, no/nahi, raat (night),
  any month word (not parsed yet), two different days, two parts of the day, no day at all.
- कर, a common STT error for कल, is ambiguous unless another day word is present (then it is
  just "do", as in "book kar do").
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from clinic.clock import IST
from clinic.tools import PartOfDay


@dataclass(frozen=True)
class Resolved:
    date_from: date
    date_to: date
    part_of_day: PartOfDay | None = None


AmbiguousCode = Literal[
    "no_day",  # no day word at all
    "two_days",  # two day expressions naming different days
    "two_parts",  # e.g. subah ya shaam
    "past",  # yesterday, or kal in the past tense
    "heard_kar",  # कर: maybe an STT error for कल
    "which_week",  # "next Friday" while this week still has a Friday
    "weekday_is_today",  # a bare weekday said on that weekday
    "not_a_day",  # "32nd"
    "no_such_day",  # "the 31st" in a 30-day month
    "negated",  # kal nahi: a day the caller just turned down
    "night",  # raat: "kal raat" is often last night, and the clinic is shut
    "month_named",  # a month word: not parsed yet, so never guess the month
    "after_midnight",  # aaj/kal/parso between 00:00 and 04:00 IST: which day is meant?
]


@dataclass(frozen=True)
class Ambiguous:
    code: AmbiguousCode  # for code and tests
    reason: str  # said to the agent, which asks the caller
    candidates: tuple[date, ...] = ()  # the readings, when there are a few


TODAY = {"aaj", "aj", "today", "आज"}
TOMORROW = {"tomorrow"}
KAL = {"kal", "कल"}  # tomorrow, or yesterday in a past-tense sentence
DAY_AFTER = {"parso", "parson", "parsoon", "परसों", "परसो"}
YESTERDAY = {"yesterday"}
STT_KAL = {"कर"}  # what STT made of कल in our M0 smoke test
PAST_MARKERS = {
    "tha",
    "thi",
    "gaya",
    "gayi",
    "gaye",
    "kiya",
    "kiye",
    "hua",
    "hui",
    "was",
    "were",
    "था",
    "थी",
    "थे",
    "गया",
    "गई",
    "गए",
    "किया",
    "हुआ",
    "हुई",
}  # romanised "the" (they were) is left out: it collides with English "the"
SMALL_HOURS_END = 4  # IST hour; before it, aaj/kal/tomorrow/parso are ambiguous
LAST = {"last", "pichhle", "pichla", "pichli", "पिछले", "पिछला", "पिछली"}  # before a weekday
NEGATION = {"nahi", "nahin", "nai", "not", "no", "नहीं", "नही"}  # not "na": a common filler
NIGHT = {"raat", "rat", "night", "tonight", "रात"}
# Month words aren't parsed yet; any of them makes the answer Ambiguous. No "may": English verb.
MONTH_WORDS = set(
    [
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "month",
        "mahina",
        "mahine",
        "जनवरी",
        "फरवरी",
        "फ़रवरी",
        "मार्च",
        "अप्रैल",
        "मई",
        "जून",
        "जुलाई",
        "अगस्त",
        "सितंबर",
        "सितम्बर",
        "अक्टूबर",
        "अक्तूबर",
        "नवंबर",
        "नवम्बर",
        "दिसंबर",
        "दिसम्बर",
        "महीना",
        "महीने",
    ]
)
PARTS: dict[PartOfDay, set[str]] = {
    "morning": {"subah", "subha", "morning", "सुबह"},
    "afternoon": {"dopahar", "dopehar", "afternoon", "दोपहर"},
    "evening": {"shaam", "sham", "evening", "शाम"},
}
# Monday = 0, as date.weekday(). Romanised spellings as callers and STT write them. No "sat" or
# "sun": in Hinglish they are "seven" (saat) and "listen" (sun lo).
WEEKDAYS: dict[str, int] = {}
for number, names in enumerate(
    [
        "monday mon somvar somwar सोमवार",
        "tuesday tue tues mangalvar mangalwar मंगलवार",
        "wednesday wed budhvar budhwar बुधवार",
        "thursday thu thur thurs guruvar guruwar brihaspativar veervar गुरुवार बृहस्पतिवार",
        "friday fri shukravar shukrawar शुक्रवार",
        "saturday shanivar shaniwar शनिवार",
        "sunday ravivar raviwar itvaar itwar रविवार इतवार",
    ]
):
    WEEKDAYS.update(dict.fromkeys(names.split(), number))
NEXT = {"next", "agle", "agla", "agli", "अगले", "अगला", "अगली"}
THIS = {"this", "is", "iss", "coming", "इस"}  # "is"/"iss" only before a Hindi weekday name
ENGLISH_DAY_NAMES = {
    "monday",
    "mon",
    "tuesday",
    "tue",
    "tues",
    "wednesday",
    "wed",
    "thursday",
    "thu",
    "thur",
    "thurs",
    "friday",
    "fri",
    "saturday",
    "sunday",
}  # "Is Monday ok?" is English: "is" there is not Hindi "this"
WEEK = {"week", "hafte", "hafta", "हफ्ते", "हफ़्ते", "हफ्ता", "सप्ताह"}  # after next/agle or last
TARIKH = {"tarikh", "tareekh", "tarik", "तारीख", "तारिख"}  # "date", after a number
ORDINAL = re.compile(r"(\d{1,2})(?:st|nd|rd|th)")
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
# Latin, digits and Devanagari, except the danda । and double danda ॥ (U+0964/5), which are
# punctuation: STT puts them right after a word ("था।").
WORD = re.compile(r"[0-9a-z\u0900-\u0963\u0966-\u097f]+")
ZERO_WIDTH = str.maketrans("", "", "\u200c\u200d")  # (non-)joiners can sit inside a word


def words(text: str) -> list[str]:
    """Lower-cased words, punctuation dropped. Devanagari is NFC-normalised so that the same
    word typed or transcribed two ways compares equal."""
    text = unicodedata.normalize("NFC", text).lower().translate(DEVANAGARI_DIGITS)
    text = text.translate(ZERO_WIDTH)
    text = re.sub(r"\bday after tomorrow\b", "parso", text)
    return WORD.findall(text)


def resolve(text: str, now: datetime) -> Resolved | Ambiguous:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(IST).date()
    small_hours = now.astimezone(IST).hour < SMALL_HOURS_END
    said = words(text)
    if NEGATION.intersection(said):
        return Ambiguous("negated", "the caller said no/nahi; ask which day they do want")
    if NIGHT.intersection(said):
        return Ambiguous("night", "night (raat) may mean last night; the clinic is shut at night")
    if MONTH_WORDS.intersection(said):
        return Ambiguous("month_named", "a month was named; confirm the full date")

    parts: list[PartOfDay] = [p for p, names in PARTS.items() if names.intersection(said)]
    if len(parts) > 1:
        return Ambiguous("two_parts", f"more than one part of the day: {' or '.join(parts)}")
    part = parts[0] if parts else None

    days: list[date] = []  # one entry per day expression the caller used
    # From 00:00 to 04:00, aaj/kal/parso may be counted from the coming morning (Cyril): note it
    # and the earlier reading here, and decide after the loop, so other days named are kept.
    after_midnight = False
    earlier: list[date] = []
    for i, word in enumerate(said):
        number = day_number(said, i)
        if word in WEEKDAYS:
            day = weekday(said, i, today)
            if isinstance(day, Ambiguous):
                return day
            days.append(day)
        elif number is not None:
            day = day_of_month(number, today)
            if isinstance(day, Ambiguous):
                return day
            days.append(day)
        elif word in TODAY:
            days.append(today)
            after_midnight = after_midnight or small_hours
        elif word in TOMORROW or word in KAL:
            days.append(today + timedelta(days=1))
            if small_hours:  # at 00:30, "kal subah" probably means this coming morning
                after_midnight = True
                earlier.append(today)
        elif word in DAY_AFTER:
            days.append(today + timedelta(days=2))
            if small_hours:  # parso is off by one the same way (Cyril, M2 close-out)
                after_midnight = True
                earlier.append(today + timedelta(days=1))
        elif word in YESTERDAY:
            return Ambiguous("past", "yesterday is in the past; ask which day")

    if not days:
        if STT_KAL.intersection(said):
            return Ambiguous(
                "heard_kar", "heard 'कर', which may be 'कल' (tomorrow); ask to confirm the day"
            )
        return Ambiguous("no_day", "no day mentioned; ask which day")
    if PAST_MARKERS.intersection(said):  # "kal gaya tha" = yesterday; "Friday ko gaya tha"
        return Ambiguous("past", "the sentence is in the past tense; ask which day to book")
    if after_midnight:
        readings = tuple(sorted(set(days + earlier)))
        return Ambiguous("after_midnight", "it is just after midnight; confirm the date", readings)
    distinct = sorted(set(days))
    if len(distinct) > 1:
        return Ambiguous("two_days", "more than one day mentioned; ask which one", tuple(distinct))
    return Resolved(distinct[0], distinct[0], part)


def follows(said: list[str], first: set[str], second: set[str]) -> bool:
    """True if a word from `first` comes right before a word from `second` anywhere."""
    return any(a in first and b in second for a, b in zip(said, said[1:], strict=False))


def weekday(said: list[str], i: int, today: date) -> date | Ambiguous:
    """The date the weekday name said[i] means, given the words around it.

    Weeks start on Monday, except that "next ..." said on a Sunday is ambiguous: Monday- and
    Sunday-start weeks disagree only then (Cyril).
    """
    word, before = said[i], said[i - 1] if i else ""
    number = WEEKDAYS[word]
    ahead = (number - today.weekday()) % 7  # 0 when it is that weekday today
    sunday = today.weekday() == 6
    if before in LAST or follows(said, LAST, WEEK):
        return Ambiguous("past", "'last' names a past day; ask which day to book")
    if follows(said, NEXT, WEEK):  # "agle hafte friday": that weekday in the following week
        day = today + timedelta(days=7 - today.weekday() + number)
        if sunday:
            return Ambiguous(
                "which_week",
                "'next week' said on a Sunday; ask which",
                (day, day + timedelta(days=7)),
            )
        return day
    if before in THIS and not (before in {"is", "iss"} and word in ENGLISH_DAY_NAMES):
        return today + timedelta(days=ahead)
    coming = today + timedelta(days=ahead or 7)
    if before in NEXT:
        if sunday or number > today.weekday():  # this week's X may still be ahead
            later = coming + timedelta(days=7)
            return Ambiguous(
                "which_week",
                "'next' can mean this week's or next week's; ask which",
                (coming, later),
            )
        return coming
    if ahead == 0:
        return Ambiguous(
            "weekday_is_today",
            "that weekday is today; ask if they mean today or next week",
            (today, coming),
        )
    return coming


def day_number(said: list[str], i: int) -> int | None:
    """The day number if said[i] names a day of the month ("5th", or "5" before tarikh).

    An ordinal counts only in a date position: after "the" or a weekday, before tarikh/of/ko or
    a part of the day, or last. "2nd appointment" and "3rd floor" are not dates.
    """
    ordinal = ORDINAL.fullmatch(said[i])
    if ordinal:
        before = said[i - 1] if i else ""
        after = said[i + 1] if i + 1 < len(said) else ""
        parts = set().union(*PARTS.values())
        dated = before == "the" or before in WEEKDAYS or not after
        if dated or after in TARIKH | {"of", "ko"} | parts:
            return int(ordinal.group(1))
        return None
    if said[i].isdigit() and i + 1 < len(said) and said[i + 1] in TARIKH:
        return int(said[i])
    return None


def day_of_month(number: int, today: date) -> date | Ambiguous:
    """The next date on or after today that falls on day `number` of its month."""
    if not 1 <= number <= 31:
        return Ambiguous("not_a_day", f"{number} is not a day of the month; ask which date")
    year, month = today.year, today.month
    if number < today.day:  # passed this month: next month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    try:
        return date(year, month, number)
    except ValueError:
        return Ambiguous("no_such_day", f"that month has no day {number}; ask which date")
