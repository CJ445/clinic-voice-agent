"""Deliberately failing test: proves CI goes red and the main ruleset blocks the merge (M0.2).

Never merge. Close the PR and delete the branch after the evidence is captured.
"""


def test_ci_must_fail() -> None:
    raise AssertionError("deliberate failure for M0.2 red-CI proof")
