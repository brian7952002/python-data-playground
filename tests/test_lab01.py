"""Tests for Lab 01 - loading and inspecting.

Read these. The tests are the specification: when a docstring is ambiguous,
the test is what decides. Learning to read a test suite as a statement of
requirements is a skill in itself, and it is how you will work out what a
new codebase expects of you.
"""

from __future__ import annotations

import pandas as pd
import pytest

from labs.lab01_loading import clean_sensor, profile_dataframe


# ---------------------------------------------------------------------------
# WORKED EXAMPLE -- these pass on a fresh clone. If they fail, your
# environment or your data is wrong, not your code.
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_profile_reports_shape(raw_orders: pd.DataFrame) -> None:
    profile = profile_dataframe(raw_orders)
    assert profile["n_rows"] == len(raw_orders)
    assert profile["n_columns"] == raw_orders.shape[1]
    assert profile["columns"] == list(raw_orders.columns)


@pytest.mark.demo
def test_profile_finds_the_missing_units(raw_orders: pd.DataFrame) -> None:
    """The raw export lost 74 unit counts. The profile should say so."""
    profile = profile_dataframe(raw_orders)
    assert profile["missing_counts"]["units"] == 74
    assert profile["missing_counts"]["order_id"] == 0


@pytest.mark.demo
def test_profile_finds_the_duplicate_rows(raw_orders: pd.DataFrame) -> None:
    """Twelve orders were written twice by a retry."""
    assert profile_dataframe(raw_orders)["duplicate_rows"] == 12


@pytest.mark.demo
def test_profile_exposes_the_channel_spelling_problem(raw_orders: pd.DataFrame) -> None:
    """Three real channels, eight spellings of them.

    This is the finding that justifies the whole lab: group by this column
    as-is and your revenue splits across eight rows instead of three.
    """
    assert profile_dataframe(raw_orders)["distinct_counts"]["channel"] == 8


@pytest.mark.demo
def test_profile_is_json_serialisable(raw_orders: pd.DataFrame) -> None:
    """numpy scalars do not survive json.dumps -- the profile converts them."""
    import json

    json.dumps(profile_dataframe(raw_orders))  # raises TypeError if it regressed


# ---------------------------------------------------------------------------
# YOUR EXERCISE -- these fail until you implement clean_sensor().
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_clean_sensor_parses_the_timestamp(raw_sensor: pd.DataFrame) -> None:
    result = clean_sensor(raw_sensor)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"]), (
        "timestamp should be datetime64, not object/string. "
        "Use pd.to_datetime()."
    )


@pytest.mark.exercise
def test_clean_sensor_makes_readings_numeric(raw_sensor: pd.DataFrame) -> None:
    result = clean_sensor(raw_sensor)
    assert pd.api.types.is_float_dtype(result["temp_c"])
    assert pd.api.types.is_float_dtype(result["humidity_pct"])


@pytest.mark.exercise
def test_clean_sensor_keeps_the_gaps(raw_sensor: pd.DataFrame) -> None:
    """Missing readings must survive -- lab 05 needs to see the outage.

    Dropping them is the tempting wrong answer. A gap in a time series is
    information: the sensor was down, which is not the same as the
    temperature being zero, and not the same as the hour never existing.
    """
    result = clean_sensor(raw_sensor)
    assert result["temp_c"].isna().sum() == 39, (
        "Expected 39 missing temperature readings to remain. "
        "Do not drop rows with missing readings."
    )


@pytest.mark.exercise
def test_clean_sensor_adds_the_is_missing_flag(raw_sensor: pd.DataFrame) -> None:
    result = clean_sensor(raw_sensor)
    assert "is_missing" in result.columns
    assert result["is_missing"].dtype == bool, "is_missing should be a bool column"
    # The flag must agree with the data it describes, on every single row.
    assert result["is_missing"].equals(result["temp_c"].isna())
    assert result["is_missing"].sum() == 39


@pytest.mark.exercise
def test_clean_sensor_sorts_by_time(raw_sensor: pd.DataFrame) -> None:
    result = clean_sensor(raw_sensor)
    assert result["timestamp"].is_monotonic_increasing


@pytest.mark.exercise
def test_clean_sensor_resets_the_index(raw_sensor: pd.DataFrame) -> None:
    result = clean_sensor(raw_sensor)
    assert list(result.index) == list(range(len(result)))


@pytest.mark.exercise
def test_clean_sensor_drops_duplicate_rows() -> None:
    """Built by hand, because sensor.csv happens to contain no duplicates.

    Testing a rule against data that cannot violate it proves nothing. When
    the real dataset does not exercise a branch, construct a tiny frame that
    does -- three rows you can check by eye beat three thousand you cannot.
    """
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-10-03 02:00:00", "2025-10-03 01:00:00",
                          "2025-10-03 01:00:00"],
            "temp_c": [9.5, 8.0, 8.0],
            "humidity_pct": [70.0, 72.0, 72.0],
        }
    )
    result = clean_sensor(raw)

    assert len(result) == 2, "The two identical 01:00 rows should collapse to one."
    assert list(result["timestamp"].dt.hour) == [1, 2], "…and the result stays sorted."


@pytest.mark.exercise
def test_clean_sensor_does_not_mutate_its_input(raw_sensor: pd.DataFrame) -> None:
    """Functions that quietly edit their arguments cause bugs far from home.

    Start with `df = raw.copy()`. This test exists because the fixture is
    module-scoped: if clean_sensor() mutated it, every later test in this
    file would be running against corrupted data, and the failures would
    point everywhere except at the real cause.
    """
    before_dtype = raw_sensor["timestamp"].dtype
    before_columns = list(raw_sensor.columns)

    clean_sensor(raw_sensor)

    assert raw_sensor["timestamp"].dtype == before_dtype, (
        "clean_sensor() modified the DataFrame it was given. Copy it first."
    )
    assert list(raw_sensor.columns) == before_columns
