"""Every top-level package imports cleanly and carries a docstring.

Why: pyright's `include` list points at these packages, and CLAUDE.md rule 5 requires a
top docstring on every module. If a package goes missing or loses its docstring, this fails.
"""

import importlib

import pytest

PACKAGES = ["agent", "clinic", "sim", "bench", "ci"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports_and_has_docstring(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip(), f"{name}/__init__.py needs a docstring"
