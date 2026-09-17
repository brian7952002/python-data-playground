"""Tests for Lab 05 - time series."""

from __future__ import annotations

import pandas as pd
import pytest

from labs.lab05_timeseries import daily_revenue_trend, daily_weather_summary


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_every_calendar_day_is_present(clean_orders: pd.DataFrame) -> None:
    """Resampling materialises days with no sales; a groupby would skip them."""
    trend = daily_revenue_trend(clean_orders)
    gaps = trend["date"].diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all()


@pytest.mark.demo
def test_revenue_reconciles(clean_orders: pd.DataFrame) -> None:
    trend = daily_revenue_trend(clean_orders)
    assert trend["revenue"].sum() == pytest.approx(clean_orders["revenue"].sum(), abs=1.0)
    assert trend["n_orders"].sum() == len(clean_orders)


@pytest.mark.demo
def test_rolling_mean_warms_up_with_nans(clean_orders: pd.DataFrame) -> None:
    """The first 6 values of a 7-day mean are NaN, and that is correct.

    There genuinely are not 7 days of history yet. A library that silently
    filled them would be inventing data.
    """
    trend = daily_revenue_trend(clean_orders, window=7)
    assert trend["revenue_ma"].head(6).isna().all()
    assert pd.notna(trend["revenue_ma"].iloc[6])


@pytest.mark.demo
def test_rolling_mean_matches_a_hand_computation(clean_orders: pd.DataFrame) -> None:
    trend = daily_revenue_trend(clean_orders, window=7)
    by_hand = trend["revenue"].head(7).mean()
    assert trend["revenue_ma"].iloc[6] == pytest.approx(by_hand, abs=0.01)


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_one_row_per_day(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor)
    assert len(result) == 90, "The sensor data covers exactly 90 days."
    assert result["date"].is_monotonic_increasing
    assert list(result.index) == list(range(len(result)))


@pytest.mark.exercise
def test_columns_and_order(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor)
    assert list(result.columns) == [
        "date", "temp_min", "temp_mean", "temp_max",
        "humidity_mean", "n_readings", "n_missing", "temp_mean_ma",
    ]


@pytest.mark.exercise
def test_min_mean_max_are_ordered(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor)
    assert (result["temp_min"] <= result["temp_mean"]).all()
    assert (result["temp_mean"] <= result["temp_max"]).all()


@pytest.mark.exercise
def test_aggregates_ignore_missing_readings(sensor: pd.DataFrame) -> None:
    """No day should come back as NaN just because the sensor dropped out.

    This is where numpy and pandas differ: np.mean() of an array holding NaN
    is NaN, but Series.mean() skips NaN by default. Knowing which one you are
    using is the difference between a clean report and a column of NaNs.
    """
    result = daily_weather_summary(sensor)
    assert result["temp_mean"].isna().sum() == 0
    assert result["temp_min"].isna().sum() == 0


@pytest.mark.exercise
def test_counts_add_up_to_a_full_day(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor)
    assert (result["n_readings"] + result["n_missing"] == 24).all()
    assert result["n_readings"].sum() == 2160 - 39
    assert result["n_missing"].sum() == 39


@pytest.mark.exercise
def test_counts_are_integers(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor)
    assert pd.api.types.is_integer_dtype(result["n_readings"])
    assert pd.api.types.is_integer_dtype(result["n_missing"])


@pytest.mark.exercise
def test_the_outage_day_is_visible(sensor: pd.DataFrame) -> None:
    """One day lost 7 consecutive hours. Your summary should surface it.

    That is the practical payoff of the n_missing column: without it, the day
    looks like any other and its mean is quietly biased.
    """
    result = daily_weather_summary(sensor)
    worst = result.sort_values("n_missing", ascending=False).iloc[0]
    assert worst["n_missing"] >= 7


@pytest.mark.exercise
def test_rolling_mean_over_daily_means(sensor: pd.DataFrame) -> None:
    result = daily_weather_summary(sensor, window=7)
    assert result["temp_mean_ma"].head(6).isna().all()
    by_hand = round(result["temp_mean"].head(7).mean(), 2)
    assert result["temp_mean_ma"].iloc[6] == pytest.approx(by_hand, abs=0.02)


@pytest.mark.exercise
def test_the_seasonal_trend_is_downward(sensor: pd.DataFrame) -> None:
    """A sanity check on the result as a whole, not on any one row.

    The data runs from early October into winter, so the smoothed series must
    end colder than it starts. Checks like this catch the class of bug where
    every individual number is plausible but the series is reversed, shifted,
    or grouped by the wrong key.
    """
    result = daily_weather_summary(sensor)
    first_week = result["temp_mean"].head(7).mean()
    last_week = result["temp_mean"].tail(7).mean()
    assert last_week < first_week - 3.0
