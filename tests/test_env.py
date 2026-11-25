"""Environment guards: facts about the interpreter that the GPU stack depends on.

F-001 (docs/FAILURES.md): vLLM died at startup because Triton could not compile its C helper
without Python.h. The venv was on the system python3.11, which has no dev headers; uv-managed
CPython ships them. This runs in `make check`, so G1 catches it locally and in CI.
"""

import sysconfig
from pathlib import Path


def test_python_dev_headers_present() -> None:
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    assert header.exists(), (
        f"{header} is missing: Triton (and so vLLM) cannot start. Use uv-managed Python "
        "(pyproject: python-preference = 'only-managed') — see docs/FAILURES.md F-001."
    )
