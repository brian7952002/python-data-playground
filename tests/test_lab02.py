"""Tests for Lab 02 - selecting and filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from labs.lab02_selection import filter_sensor_readings, select_high_value_orders


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_threshold_is_inclusive(clean_orders: pd.DataFrame) -> None:
    result = select_high_value_orders(clean_orders, 1000)
    assert (result["revenue"] >= 1000).all()


@pytest.mark.demo
def test_channel_filter_narrows_the_result(clean_orders: pd.DataFrame) -> None:
    everything = select_high_value_orders(clean_orders, 500)
    web_only = select_high_value_orders(clean_orders, 500, ["web"])

    assert set(web_only["channel"].unique()) == {"web"}
    assert len(web_only) < len(everything)


@pytest.mark.demo
def test_result_is_sorted_and_reindexed(clean_orders: pd.DataFrame) -> None:
    result = select_high_value_orders(clean_orders, 800)
    assert result["revenue"].is_monotonic_decreasing
    assert list(result.index) == list(range(len(result)))


@pytest.mark.demo
def test_impossible_threshold_gives_an_empty_frame(clean_orders: pd.DataFrame) -> None:
    """Empty is a valid answer, and it must still have the right columns.

    Code downstream will do result["revenue"].sum() on this. That works on an
    empty frame with the column, and raises KeyError on one without it.
    """
    result = select_high_value_orders(clean_orders, 10_000_000)
    assert len(result) == 0
    assert "revenue" in result.columns


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_no_arguments_returns_everything(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor)
    assert len(result) == len(sensor)
    assert list(result.columns) == list(sensor.columns)


@pytest.mark.exercise
def test_min_temp_filters_and_excludes_missing(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, min_temp=10.0)
    assert (result["temp_c"] >= 10.0).all()
    assert result["temp_c"].isna().sum() == 0, (
        "A missing reading is not a match. NaN >= 10.0 is False, so if you "
        "built the mask with a comparison you get this for free -- make sure "
        "you understand why rather than assuming."
    )


@pytest.mark.exercise
def test_max_temp_filters(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, max_temp=5.0)
    assert (result["temp_c"] <= 5.0).all()
    assert len(result) > 0, "There really are readings below 5C in this data."


@pytest.mark.exercise
def test_temperature_range_combines_both_bounds(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, min_temp=4.0, max_temp=8.0)
    assert result["temp_c"].between(4.0, 8.0).all()


@pytest.mark.exercise
def test_hours_filter_uses_the_dt_accessor(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, hours=[6, 7, 8])
    assert set(result["timestamp"].dt.hour.unique()) <= {6, 7, 8}
    # 90 days of data, three readings a day, so roughly 270 rows.
    assert 250 < len(result) <= 270


@pytest.mark.exercise
def test_hours_filter_keeps_missing_readings(sensor: pd.DataFrame) -> None:
    """Subtle, and the reason the temperature rule is worded as it is.

    Filtering by hour alone says nothing about temperature, so a row whose
    sensor dropped out at 07:00 still matches "hour is 7". Only the
    temperature bounds exclude NaN rows. If you wrote a blanket dropna() at
    the top of the function, this test catches it.
    """
    result = filter_sensor_readings(sensor, hours=list(range(24)))
    assert result["temp_c"].isna().sum() == 39


@pytest.mark.exercise
def test_filters_combine(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, min_temp=0.0, max_temp=30.0, hours=[14])
    assert (result["timestamp"].dt.hour == 14).all()
    assert result["temp_c"].between(0.0, 30.0).all()


@pytest.mark.exercise
def test_result_is_sorted_and_reindexed(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, hours=[0, 12])
    assert result["timestamp"].is_monotonic_increasing
    assert list(result.index) == list(range(len(result)))


@pytest.mark.exercise
def test_empty_result_still_has_columns(sensor: pd.DataFrame) -> None:
    result = filter_sensor_readings(sensor, min_temp=999.0)
    assert len(result) == 0
    assert list(result.columns) == list(sensor.columns)
