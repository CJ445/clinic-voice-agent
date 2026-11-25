"""Shared fixtures: a freshly seeded clinic DB and the frozen seed clock (used by the tool tests).

Seed facts the tool tests rely on (clinic/seed.py, SEED_NOW = Mon 2026-11-30 08:00 IST):
- Dr. Sharma (1): d1-20261130-0900 booked (Asha), d1-20261130-1730 booked (Ravi),
  d1-20261201-1200 blocked.
- Dr. Iyer (2), mornings: d2-20261130-1030 booked (Meera), d2-20261201-0900/0930 blocked.
- Dr. Khan (3), evenings: d3-20261130-1700 had a cancelled appointment (free),
  d3-20261130-1900 booked (Imran).
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from clinic.clock import FrozenClock
from clinic.db import connect
from clinic.seed import SEED_NOW, build


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(SEED_NOW)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    build(tmp_path / "clinic.db")
    connection = connect(tmp_path / "clinic.db")
    yield connection
    connection.close()
