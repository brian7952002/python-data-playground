"""Tests for Lab 06 - consuming an API.

Every test here runs offline in microseconds, because the lab functions take
their I/O as arguments. Read the fakes below: simulating "the server returned
503 three times and then succeeded" is three lines here, and would be nearly
impossible to arrange reliably against a live API.
"""

from __future__ import annotations

import pytest

from labs.lab06_api_consume import (
    ApiError,
    ApiResponse,
    collect_paginated,
    fetch_with_retry,
    is_transient,
)


class FakeEndpoint:
    """Returns a scripted sequence of responses, and records what happened."""

    def __init__(self, *statuses: int, payload: dict | None = None) -> None:
        self.statuses = list(statuses)
        self.payload = payload if payload is not None else {"ok": True}
        self.calls = 0

    def __call__(self) -> ApiResponse:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return ApiResponse(status_code=status, payload=self.payload)


class SleepRecorder:
    """Stands in for time.sleep and remembers the delays it was asked for."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_walks_every_page() -> None:
    pages = {
        1: {"results": [{"id": 1}, {"id": 2}], "next_page": 2},
        2: {"results": [{"id": 3}], "next_page": 3},
        3: {"results": [{"id": 4}], "next_page": None},
    }
    assert collect_paginated(pages.__getitem__) == [
        {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}
    ]


@pytest.mark.demo
def test_single_page_response() -> None:
    assert collect_paginated(lambda p: {"results": [{"id": 1}], "next_page": None}) == [
        {"id": 1}
    ]


@pytest.mark.demo
def test_tolerates_a_page_with_no_results_key() -> None:
    """Be forgiving in what you accept from someone else's server."""
    assert collect_paginated(lambda p: {"next_page": None}) == []


@pytest.mark.demo
def test_endless_pagination_is_stopped() -> None:
    with pytest.raises(ApiError, match="Stopped after"):
        collect_paginated(lambda p: {"results": [], "next_page": p + 1}, max_pages=5)


@pytest.mark.demo
def test_transient_classification() -> None:
    assert is_transient(429) and is_transient(500) and is_transient(503)
    assert not is_transient(200)
    assert not is_transient(404)
    assert not is_transient(401)


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_success_on_the_first_try_does_not_sleep() -> None:
    endpoint = FakeEndpoint(200, payload={"temperature": 12.5})
    sleeper = SleepRecorder()

    assert fetch_with_retry(endpoint, sleep=sleeper) == {"temperature": 12.5}
    assert endpoint.calls == 1
    assert sleeper.delays == [], "Nothing failed, so nothing should have waited."


@pytest.mark.exercise
def test_retries_a_transient_failure_then_succeeds() -> None:
    endpoint = FakeEndpoint(503, 200, payload={"ok": True})
    sleeper = SleepRecorder()

    assert fetch_with_retry(endpoint, sleep=sleeper) == {"ok": True}
    assert endpoint.calls == 2
    assert sleeper.delays == [0.5]


@pytest.mark.exercise
def test_backoff_doubles_each_time() -> None:
    endpoint = FakeEndpoint(503, 429, 500, 200)
    sleeper = SleepRecorder()

    fetch_with_retry(endpoint, max_attempts=4, backoff_base=0.5, sleep=sleeper)
    assert sleeper.delays == [0.5, 1.0, 2.0]


@pytest.mark.exercise
def test_gives_up_after_max_attempts() -> None:
    endpoint = FakeEndpoint(500)  # always fails
    sleeper = SleepRecorder()

    with pytest.raises(ApiError) as exc:
        fetch_with_retry(endpoint, max_attempts=3, sleep=sleeper)

    assert endpoint.calls == 3, "max_attempts=3 means three calls in total."
    assert exc.value.status_code == 500


@pytest.mark.exercise
def test_does_not_sleep_after_the_final_attempt() -> None:
    """Three attempts means two waits, not three.

    Sleeping before raising adds seconds to every failure and buys nothing --
    there is no further attempt for the server to recover before.
    """
    endpoint = FakeEndpoint(503)
    sleeper = SleepRecorder()

    with pytest.raises(ApiError):
        fetch_with_retry(endpoint, max_attempts=3, backoff_base=0.5, sleep=sleeper)

    assert sleeper.delays == [0.5, 1.0], f"Expected two waits, got {sleeper.delays}"


@pytest.mark.exercise
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_failures_fail_immediately(status: int) -> None:
    """A 404 will still be a 404 on the fifth attempt."""
    endpoint = FakeEndpoint(status)
    sleeper = SleepRecorder()

    with pytest.raises(ApiError) as exc:
        fetch_with_retry(endpoint, max_attempts=5, sleep=sleeper)

    assert endpoint.calls == 1, f"{status} is permanent -- do not retry it."
    assert sleeper.delays == []
    assert exc.value.status_code == status


@pytest.mark.exercise
def test_rejects_a_nonsense_attempt_count() -> None:
    with pytest.raises(ValueError):
        fetch_with_retry(FakeEndpoint(200), max_attempts=0)


@pytest.mark.exercise
def test_a_single_attempt_is_allowed() -> None:
    """max_attempts=1 means "try once, never retry" -- a legitimate policy."""
    endpoint = FakeEndpoint(503)
    sleeper = SleepRecorder()

    with pytest.raises(ApiError):
        fetch_with_retry(endpoint, max_attempts=1, sleep=sleeper)

    assert endpoint.calls == 1
    assert sleeper.delays == []
