"""A pytest plugin that writes a machine-readable report for the web UI.

Parsing pytest's human-readable output with regexes works until it doesn't:
the format is meant for people, and it changes between versions. pytest
exposes hooks precisely so that tools do not have to scrape it.

This plugin implements one hook, `pytest_runtest_logreport`, which pytest
calls for every phase (setup / call / teardown) of every test. We record the
"call" phase -- the test body itself -- and write one JSON object per line to
the file named by the PLAYGROUND_REPORT environment variable.

To use it from the command line and see what the UI sees:

    PLAYGROUND_REPORT=report.jsonl pytest tests/test_lab01.py -p app.pytest_report_plugin

Writing a plugin is a good way to understand that pytest is a framework you
can extend, not a black box you run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Matches an ANSI escape sequence, e.g. "\x1b[31m" (red) or "\x1b[0m" (reset).
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def pytest_runtest_logreport(report) -> None:
    """Called by pytest after each phase of each test."""
    destination = os.environ.get("PLAYGROUND_REPORT")
    if not destination:
        return

    # setup/teardown also produce reports. We want the test body, except when
    # setup itself failed (a broken fixture), which we do want to surface --
    # otherwise a fixture error would show as no result at all.
    if report.when != "call" and not (report.when == "setup" and report.failed):
        return

    # `markers` is how the UI tells a worked example from an exercise. Reading
    # it from the report is exact; guessing from the test's name is not.
    #
    # report.keywords is a dict whose KEYS include every marker name that
    # applies to the test -- its values are bookkeeping integers, not marker
    # objects, so membership is what to test. It also picks up markers
    # inherited from an enclosing class, which is what makes the grouped
    # TestRegionQueryValidation tests in lab 07 classify correctly.
    markers = [name for name in ("demo", "exercise") if name in report.keywords]

    record = {
        "nodeid": report.nodeid,
        "name": report.nodeid.split("::", 1)[-1],
        "outcome": report.outcome,          # "passed" | "failed" | "skipped"
        "duration": round(report.duration, 4),
        "markers": markers,
        "message": _first_error_line(report),
    }

    with Path(destination).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _first_error_line(report) -> str | None:
    """Pull the assertion message out of a failure, if there is one."""
    if not report.failed or report.longrepr is None:
        return None

    # --color=no controls what pytest prints to the terminal, but the
    # longrepr object builds its own string with ANSI escapes embedded. Sent
    # to a browser those render as literal "[1m[31m" noise in front of every
    # message, so strip them here rather than in the UI -- the report should
    # be clean data, and it is the producer's job to make it so.
    text = _ANSI.sub("", str(report.longrepr))

    # pytest prefixes the explanation lines of a failure with "E   ".
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E "):
            return stripped[1:].strip()

    lines = text.splitlines()
    return lines[-1].strip() if lines else None
