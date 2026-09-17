"""Lab 06 - Consuming an API properly.

CONCEPT
-------
Calling an API is one line. Calling an API *reliably* is the part that
actually takes skill, and it is what separates a script that works on your
machine from one you can put on a schedule:

  * Results arrive in pages, not all at once.
  * Servers fail temporarily. A 503 or a 429 (rate limited) means "try again
    in a moment", not "give up".
  * Servers also fail permanently. A 404 or a 401 will still be a 404 or a
    401 on the fifth attempt -- retrying it wastes time and, if you are rate
    limited, makes things worse for everyone.
  * Retrying instantly, in a tight loop, is how you get your key banned.
    Exponential backoff -- wait 0.5s, then 1s, then 2s -- gives the server
    room to recover.

TESTING WITHOUT THE NETWORK
---------------------------
Notice that neither function here calls httpx directly. They take the thing
that does the calling as an argument -- `fetch_page`, `call`, `sleep`. That
is dependency injection, and it is why the tests for this lab run in
milliseconds, offline, and can simulate a server returning 500 three times
in a row, which would be nearly impossible to arrange against a live API.

This is a habit worth taking everywhere: keep the I/O at the edges and the
logic in the middle, where you can test it. A retry policy you cannot test
is a retry policy you do not know works.

To see it against a real API, run:
    python scripts/live_api_demo.py

WORKED EXAMPLE  ->  collect_paginated()
    Walk every page of a paginated endpoint and return the combined results.

YOUR TURN       ->  fetch_with_retry()
    Implement the retry policy: back off on transient failures, fail fast on
    permanent ones, and give up after a bounded number of attempts.

RUN THE TESTS
    pytest tests/test_lab06.py -v
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# A minimal stand-in for an HTTP response.
#
# Using our own tiny type rather than httpx.Response means the tests can
# construct one in a single line, and the lab logic does not care which HTTP
# library you eventually use.
# ---------------------------------------------------------------------------


@dataclass
class ApiResponse:
    """The parts of an HTTP response these labs care about."""

    status_code: int
    payload: dict = field(default_factory=dict)


class ApiError(RuntimeError):
    """Raised when a request ultimately fails.

    Carries the status code so the caller can react to it -- a bare
    RuntimeError("request failed") forces whoever catches it to parse your
    error string, which is a bad interface.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def is_transient(status_code: int) -> bool:
    """Is this status code worth retrying?

    429 means "too many requests" -- the server is explicitly telling you to
    slow down and come back. Anything in the 5xx range is a server-side
    problem that may well be gone in a second. Everything else (400, 401,
    403, 404, ...) is a problem with the request itself and will not fix
    itself, so retrying is pointless.
    """
    return status_code == 429 or 500 <= status_code < 600


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def collect_paginated(
    fetch_page: Callable[[int], dict],
    max_pages: int = 100,
) -> list[dict]:
    """Walk a paginated endpoint and return every result.

    Args:
        fetch_page: A function you supply that takes a page number (starting
            at 1) and returns a dict shaped like:
                {"results": [ ... ], "next_page": 2}      # more to come
                {"results": [ ... ], "next_page": None}   # last page
        max_pages: A hard ceiling on how many pages to request.

    Returns:
        Every item from every page, concatenated in page order.

    Raises:
        ApiError: if the endpoint keeps handing out pages past max_pages.

    Example:
        >>> pages = {1: {"results": [{"id": 1}], "next_page": 2},
        ...          2: {"results": [{"id": 2}], "next_page": None}}
        >>> collect_paginated(lambda p: pages[p])
        [{'id': 1}, {'id': 2}]
    """
    items: list[dict] = []
    page = 1
    pages_fetched = 0

    # A `while True` driven by the server's own "next_page" is the right shape
    # here: you genuinely do not know how many pages there are until the
    # server tells you. The max_pages ceiling is what keeps it from becoming
    # an infinite loop when a buggy endpoint always returns next_page.
    while True:
        if pages_fetched >= max_pages:
            raise ApiError(
                f"Stopped after {max_pages} pages -- the endpoint kept "
                f"advertising another page. Raise max_pages if this is "
                f"genuinely a very large result set."
            )

        body = fetch_page(page)
        pages_fetched += 1

        # .get() with a default, not body["results"], so that a page which
        # omits the key entirely is treated as empty rather than exploding.
        # Be forgiving in what you accept from someone else's server.
        items.extend(body.get("results", []))

        next_page = body.get("next_page")
        if next_page is None:
            return items
        page = next_page


# ===========================================================================
# YOUR TURN
# ===========================================================================


def fetch_with_retry(
    call: Callable[[], ApiResponse],
    max_attempts: int = 4,
    backoff_base: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Call an endpoint, retrying transient failures with exponential backoff.

    Args:
        call: A zero-argument function that performs one request and returns
            an ApiResponse. You call it once per attempt.
        max_attempts: Total attempts including the first. Must be >= 1.
        backoff_base: The first wait, in seconds. Each subsequent wait
            doubles it.
        sleep: The function used to wait. Defaults to time.sleep; the tests
            pass in a recorder instead, so they can assert on the delays
            without actually waiting. Do not call time.sleep directly -- call
            this argument, or the tests will take 4 real seconds.

    Returns:
        The `payload` of the first response with status_code 200.

    Raises:
        ValueError: if max_attempts < 1.
        ApiError: on a permanent failure (raise IMMEDIATELY, do not retry),
            or when every attempt has been used up on transient failures.
            Set .status_code on the error to the last status code seen.

    THE RETRY POLICY, precisely:

        attempt 1 -> 503   transient, and attempts remain  -> sleep(0.5), retry
        attempt 2 -> 429   transient, and attempts remain  -> sleep(1.0), retry
        attempt 3 -> 500   transient, and attempts remain  -> sleep(2.0), retry
        attempt 4 -> 500   transient, no attempts left     -> raise ApiError

        The waits are backoff_base * 2**(n-1) for the n-th failure:
        0.5, 1.0, 2.0, 4.0, ...

        Do NOT sleep after the final attempt. If you are about to raise,
        there is nothing left to wait for -- sleeping there just makes every
        failure take an extra 4 seconds. The tests check this.

        A permanent failure (404, 401, ...) raises on the spot, with no
        sleeping and no further attempts, even if attempts remain.

    Hints:
        * `for attempt in range(1, max_attempts + 1):` gives you a 1-based
          attempt number, which makes the backoff formula read directly.
        * is_transient(status_code) is written for you above.
        * "Is this the last attempt?" is `attempt == max_attempts`.
        * Keep track of the last status code you saw so you can attach it to
          the error you finally raise.

    Check yourself:
        pytest tests/test_lab06.py -v
    """
    raise NotImplementedError("Implement fetch_with_retry -- see the docstring.")
