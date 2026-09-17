"""Run a lab's pytest suite in a subprocess and return a structured result.

WHY A SUBPROCESS, not pytest.main() in this process
---------------------------------------------------
Python caches imported modules in sys.modules. If the web server imported
labs.lab01_loading to run your tests, then you edited that file and clicked
"Run" again, you would be testing the OLD code still sitting in memory. You
would fix a bug, watch the test fail anyway, and lose an hour to it.

importlib.reload() can paper over that, but it gets unreliable as soon as
modules import each other -- reloading lab04 does not reload the reference to
it that lab07 already holds.

A fresh subprocess has no cache at all. It costs a few hundred milliseconds
and it is always right. "Start a clean process rather than try to un-import
things" is a pattern worth recognising.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_lab_tests(test_file: str, timeout: int = 120) -> dict:
    """Run one lab's tests and return a structured summary.

    Args:
        test_file: Path to the test file, relative to the repo root.
        timeout: Seconds to wait before giving up.

    Returns:
        {
          "ok": bool,          # every test passed
          "passed": int, "failed": int, "total": int,
          "demo_passed": int, "demo_total": int,
          "exercise_passed": int, "exercise_total": int,
          "tests": [{"name", "outcome", "kind", "message", "duration"}],
          "first_error": str | None,
          "raw": str,          # full pytest output, shown in the UI's log pane
        }
    """
    target = REPO_ROOT / test_file
    if not target.exists():
        return _error_result(f"Test file not found: {test_file}")

    # A temp file for the plugin's JSONL report. delete=False because Windows
    # will not let the subprocess open a NamedTemporaryFile that this process
    # still holds -- a real cross-platform gotcha worth knowing about.
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    handle.close()
    report_path = Path(handle.name)

    env = os.environ.copy()
    env["PLAYGROUND_REPORT"] = str(report_path)

    try:
        completed = subprocess.run(
            [
                sys.executable, "-m", "pytest", str(target),
                "-v",
                "--tb=short",
                "-p", "app.pytest_report_plugin",  # our JSONL reporter
                "-p", "no:cacheprovider",          # do not litter .pytest_cache
                "--color=no",                      # ANSI codes would reach the UI
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = completed.stdout + completed.stderr
        records = _read_report(report_path)
    except subprocess.TimeoutExpired:
        return _error_result(
            f"Tests did not finish within {timeout}s. An infinite loop in your "
            f"implementation is the usual cause."
        )
    finally:
        report_path.unlink(missing_ok=True)

    return _summarise(records, output)


def _read_report(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial final line can happen if pytest was killed. Skip
                # it rather than failing the whole report.
                continue
    return records


def _summarise(records: list[dict], output: str) -> dict:
    tests = []
    for record in records:
        markers = record.get("markers", [])
        kind = "demo" if "demo" in markers else "exercise" if "exercise" in markers else "other"
        tests.append(
            {
                "name": record.get("name", record.get("nodeid", "?")),
                "outcome": record.get("outcome", "unknown"),
                "kind": kind,
                "message": record.get("message"),
                "duration": record.get("duration", 0.0),
            }
        )

    def count(kind: str, outcome: str | None = None) -> int:
        return sum(
            1 for t in tests
            if t["kind"] == kind and (outcome is None or t["outcome"] == outcome)
        )

    passed = sum(1 for t in tests if t["outcome"] == "passed")
    failed = sum(1 for t in tests if t["outcome"] == "failed")

    first_error = next(
        (t["message"] for t in tests if t["outcome"] == "failed" and t["message"]), None
    )

    return {
        "ok": failed == 0 and passed > 0,
        "passed": passed,
        "failed": failed,
        "total": len(tests),
        "demo_passed": count("demo", "passed"),
        "demo_total": count("demo"),
        "exercise_passed": count("exercise", "passed"),
        "exercise_total": count("exercise"),
        "tests": tests,
        "first_error": first_error,
        "raw": output,
    }


def _error_result(message: str) -> dict:
    return {
        "ok": False,
        "passed": 0, "failed": 0, "total": 0,
        "demo_passed": 0, "demo_total": 0,
        "exercise_passed": 0, "exercise_total": 0,
        "tests": [],
        "first_error": message,
        "raw": message,
    }
