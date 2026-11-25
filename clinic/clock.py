"""The clinic's clock: one place that says what "now" is, in Asia/Kolkata (CLAUDE.md rule 6).

Everything that needs the current time takes a Clock instead of calling datetime.now(), so seed
data, tests and simulator runs can freeze time and get identical results.
"""

from datetime import datetime
from typing import Protocol
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, timezone-aware, in IST."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(IST)


class FrozenClock:
    """Always returns the same instant. Rejects naive datetimes so no test can smuggle one in."""

    def __init__(self, at: datetime) -> None:
        if at.tzinfo is None:
            raise ValueError("FrozenClock needs a timezone-aware datetime")
        self._at = at.astimezone(IST)

    def now(self) -> datetime:
        return self._at
