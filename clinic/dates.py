"""Turn what a caller said about *when* into IST dates, or say why it can't (PLAN M2.4).

resolve("kal shaam", now) -> Resolved(1 Dec, 1 Dec, "evening"); resolve("कर", now) -> Ambiguous.
It understands English, Devanagari Hindi and romanised Hinglish.

Why deterministic code and not the LLM: a wrong date is the one mistake a booking agent must
never make silently. LLMs are bad at calendar arithmetic ("next Friday" from a Monday), differ
run to run, and can't be unit-tested exhaustively; this function is pure, so every phrase has a
table test and the same input always gives the same date. When the words are unclear it doesn't
guess: it returns Ambiguous with a reason, and the agent asks the caller. The LLM's job is only
to pass the caller's words here and say the answer back.

Ambiguity rules (Cyril, M2.4): a bare kal/कल is tomorrow (a booking call is future context);
kal with a past-tense marker (tha, gaya, था...) is ambiguous; कर, a common STT error for कल, is
ambiguous unless another day word is present (then it is just "do", as in "book kar do").
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from clinic.clock import IST
from clinic.tools import PartOfDay


@dataclass(frozen=True)
class Resolved:
    date_from: date
    date_to: date
    part_of_day: PartOfDay | None = None


@dataclass(frozen=True)
class Ambiguous:
    reason: str  # said to the agent, which asks the caller
    candidates: tuple[date, ...] = ()  # the readings, when there are a few


TODAY = {"aaj", "aj", "today", "आज"}
TOMORROW = {"tomorrow"}
KAL = {"kal", "कल"}  # tomorrow, or yesterday in a past-tense sentence
DAY_AFTER = {"parso", "parson", "parsoon", "परसों", "परसो"}
YESTERDAY = {"yesterday"}
STT_KAL = {"कर"}  # what STT made of कल in our M0 smoke test
PAST_MARKERS = {
    *("tha", "thi", "gaya", "gayi", "gaye", "kiya", "kiye", "hua", "hui", "was", "were"),
    *("था", "थी", "थे", "गया", "गई", "गए", "किया", "हुआ", "हुई"),
}  # romanised "the" (they were) is left out: it collides with English "the"
PARTS: dict[PartOfDay, set[str]] = {
    "morning": {"subah", "subha", "morning", "सुबह"},
    "afternoon": {"dopahar", "dopehar", "afternoon", "दोपहर"},
    "evening": {"shaam", "sham", "evening", "शाम"},
}
WORD = re.compile(r"[0-9a-zऀ-ॿ]+")  # Latin, digits, and the whole Devanagari block


def words(text: str) -> list[str]:
    """Lower-cased words, punctuation dropped. Devanagari is NFC-normalised so that the same
    word typed or transcribed two ways compares equal."""
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\bday after tomorrow\b", "parso", text)
    return WORD.findall(text)


def resolve(text: str, now: datetime) -> Resolved | Ambiguous:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = now.astimezone(IST).date()
    said = words(text)

    parts: list[PartOfDay] = [p for p, names in PARTS.items() if names.intersection(said)]
    if len(parts) > 1:
        return Ambiguous(f"more than one part of the day: {' or '.join(parts)}")
    part = parts[0] if parts else None

    days: list[date] = []  # one entry per day expression the caller used
    for word in said:
        if word in TODAY:
            days.append(today)
        elif word in TOMORROW:
            days.append(today + timedelta(days=1))
        elif word in KAL:
            if PAST_MARKERS.intersection(said):
                return Ambiguous(
                    "kal in the past tense means yesterday, which is past; ask which day"
                )
            days.append(today + timedelta(days=1))
        elif word in DAY_AFTER:
            days.append(today + timedelta(days=2))
        elif word in YESTERDAY:
            return Ambiguous("yesterday is in the past; ask which day")

    if not days:
        if STT_KAL.intersection(said):
            return Ambiguous("heard 'कर', which may be 'कल' (tomorrow); ask to confirm the day")
        return Ambiguous("no day mentioned; ask which day")
    distinct = sorted(set(days))
    if len(distinct) > 1:
        return Ambiguous("more than one day mentioned; ask which one", tuple(distinct))
    return Resolved(distinct[0], distinct[0], part)
