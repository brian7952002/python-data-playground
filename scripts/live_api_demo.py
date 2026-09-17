"""Call a real public API, and turn the response into a DataFrame.

The lab 06 tests run offline against fakes, which is the right way to test a
retry policy. But at some point you have to point it at a real server, and
this script is where that happens.

It uses Open-Meteo (https://open-meteo.com) because it needs no API key, no
signup and no billing details -- rare, and it makes this script something you
can run the minute you clone the repo.

    python scripts/live_api_demo.py
    python scripts/live_api_demo.py --lat 51.5 --lon -0.13 --days 5

Once you have finished lab 06, this script routes its call through YOUR
fetch_with_retry() and says so. Until then it falls back to a plain request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labs.lab06_api_consume import ApiError, ApiResponse, fetch_with_retry  # noqa: E402

ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast(lat: float, lon: float, days: int) -> dict:
    """Get an hourly forecast, using the lab 06 retry policy when it exists."""

    def call_once() -> ApiResponse:
        # A timeout is not optional. Without one, httpx will wait forever on a
        # server that accepts your connection and then goes quiet -- and your
        # scheduled job hangs instead of failing and retrying.
        response = httpx.get(
            ENDPOINT,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m",
                "forecast_days": days,
                "timezone": "UTC",
            },
            timeout=10.0,
        )
        # Translating httpx's response into the lab's own little type is what
        # lets the same retry logic wrap any HTTP library.
        return ApiResponse(status_code=response.status_code, payload=response.json())

    try:
        payload = fetch_with_retry(call_once, max_attempts=4, backoff_base=0.5)
        print("Fetched via your lab 06 fetch_with_retry(). ")
        return payload
    except NotImplementedError:
        print("Lab 06 not finished yet -- falling back to a single unguarded call.")
        print("Finish fetch_with_retry() and this script will use it automatically.\n")
        response = call_once()
        if response.status_code != 200:
            raise ApiError(f"HTTP {response.status_code}", response.status_code)
        return response.payload


def to_frame(payload: dict) -> pd.DataFrame:
    """Turn the API's column-oriented JSON into a tidy DataFrame.

    Open-Meteo returns parallel arrays -- one for time, one per variable --
    rather than a list of records. That is common in time-series APIs, and
    pd.DataFrame() handles it directly because a dict of equal-length lists is
    exactly a column-oriented table.
    """
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temp_c": hourly["temperature_2m"],
            "humidity_pct": hourly["relative_humidity_2m"],
        }
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, default=33.749, help="Latitude (default: Atlanta)")
    parser.add_argument("--lon", type=float, default=-84.388, help="Longitude")
    parser.add_argument("--days", type=int, default=3, help="Forecast days (1-16)")
    args = parser.parse_args()

    print(f"Requesting {args.days}-day hourly forecast for ({args.lat}, {args.lon})...\n")

    try:
        payload = fetch_forecast(args.lat, args.lon, args.days)
    except (ApiError, httpx.HTTPError) as exc:
        print(f"Request failed: {exc}")
        raise SystemExit(1)

    frame = to_frame(payload)

    print(f"{len(frame)} hourly readings.\n")
    print(frame.head(8).to_string(index=False))

    # The same daily-summary shape lab 05 asks you to build, on live data.
    daily = frame.set_index("timestamp").resample("D").agg(
        temp_min=("temp_c", "min"),
        temp_mean=("temp_c", "mean"),
        temp_max=("temp_c", "max"),
    ).round(1)

    print("\nDaily summary:")
    print(daily.to_string())
    print(
        "\nThis is lab 05's daily_weather_summary() in miniature, on live data. "
        "The technique does not change when the source does -- which is the point."
    )


if __name__ == "__main__":
    main()
