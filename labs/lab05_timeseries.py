"""Lab 05 - Time series: resampling and rolling windows.

CONCEPT
-------
Time series data has structure at several scales at once. The sensor data in
this repo has a 24-hour cycle (cold at night, warm mid-afternoon), a slow
seasonal drift (autumn into winter), and random noise on top. Looking at the
raw hourly numbers, you see mostly noise. The techniques below are how you
separate those scales.

RESAMPLING changes the frequency of the data. Going from hourly to daily is
downsampling, and it needs an aggregation: do you want each day's mean, max,
or sum? The answer depends on the quantity. Summing temperatures is
meaningless; summing revenue is exactly right.

    df.set_index("timestamp").resample("D").mean()

ROLLING WINDOWS keep the original frequency but smooth each point using its
neighbours. A 7-day rolling mean of daily revenue removes the day-of-week
sawtooth so you can see the trend underneath.

    daily["revenue"].rolling(window=7).mean()

The first 6 values of a 7-day rolling mean are NaN, because there are not yet
7 observations. That is correct and honest -- `min_periods=1` will fill them
in from partial windows if you would rather have a value, but understand that
those early points are computed from less data.

THE GAP TRAP. Resampling only inserts missing periods when you resample. If
no order was placed on a Tuesday, that Tuesday is simply absent from a
groupby-by-date -- and a 7-day rolling mean over the result silently averages
7 *rows*, not 7 *days*. Resample first; it materialises the empty periods so
your windows mean what you think they mean.

WORKED EXAMPLE  ->  daily_revenue_trend()
    Turn transactional orders into a continuous daily series with a rolling
    average, handling the no-sales-today case correctly.

YOUR TURN       ->  daily_weather_summary()
    Downsample hourly sensor readings to daily statistics, and be explicit
    about the readings the sensor dropped.

RUN THE TESTS
    pytest tests/test_lab05.py -v
"""

from __future__ import annotations

import pandas as pd


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def daily_revenue_trend(orders: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Aggregate orders into a continuous daily revenue series with a trend.

    Args:
        orders: Cleaned orders from labs.load_clean_orders(), with
            "order_date" (datetime64) and "revenue".
        window: Size of the rolling mean, in days.

    Returns:
        A DataFrame with one row per calendar day between the first and last
        order inclusive -- INCLUDING days with no orders -- and these columns:
            date            datetime  -- the day
            revenue         float     -- total revenue that day (0.0 if none)
            n_orders        int       -- orders that day (0 if none)
            revenue_ma      float     -- `window`-day rolling mean of revenue

    Example:
        >>> from labs import load_clean_orders
        >>> trend = daily_revenue_trend(load_clean_orders())
        >>> trend["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()
        True
    """
    # resample() requires a DatetimeIndex, so the date column becomes the
    # index first. This is the step people forget; the error message when you
    # skip it ("Only valid with DatetimeIndex") is at least a clear one.
    indexed = orders.set_index("order_date")

    # "D" = calendar day. Because this is a resample and not a groupby, days
    # with no orders are created as empty periods rather than skipped.
    daily = indexed.resample("D").agg(
        revenue=("revenue", "sum"),
        n_orders=("order_id", "count"),
    )

    # An empty period sums to 0.0 for revenue and counts to 0 -- pandas
    # already does the right thing here, but being explicit documents the
    # intent and protects against a column whose sum of nothing is NaN.
    daily["revenue"] = daily["revenue"].fillna(0.0)
    daily["n_orders"] = daily["n_orders"].fillna(0).astype("int64")

    # The rolling mean is computed AFTER resampling, so `window` really means
    # `window` days -- see "the gap trap" in the module docstring.
    daily["revenue_ma"] = daily["revenue"].rolling(window=window).mean().round(2)
    daily["revenue"] = daily["revenue"].round(2)

    # Move the datetime index back to a column named "date".
    return daily.reset_index().rename(columns={"order_date": "date"})


# ===========================================================================
# YOUR TURN
# ===========================================================================


def daily_weather_summary(sensor: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Downsample hourly sensor readings into daily statistics.

    Args:
        sensor: From labs.load_sensor(). Columns "timestamp" (datetime64),
            "temp_c" (float, NaN where the sensor dropped out) and
            "humidity_pct" (float, same). Readings are hourly, so a complete
            day holds 24 rows.
        window: Size of the rolling mean over daily means, in days.

    Returns:
        A DataFrame with one row per calendar day covered by the data, and
        these columns in this order:

            date           datetime  -- the day (midnight)
            temp_min       float     -- lowest temp_c that day, to 2dp
            temp_mean      float     -- mean temp_c that day, to 2dp
            temp_max       float     -- highest temp_c that day, to 2dp
            humidity_mean  float     -- mean humidity_pct that day, to 2dp
            n_readings     int       -- how many NON-missing temp_c readings
                                        that day (0-24)
            n_missing      int       -- 24 - n_readings; how many the sensor
                                        dropped. Days at the very start or
                                        end of the data may legitimately be
                                        partial -- report what you observe.
            temp_mean_ma   float     -- `window`-day rolling mean of
                                        temp_mean, to 2dp

        Sorted by date ascending, index reset to 0..n-1.

        min/mean/max must IGNORE the missing readings rather than propagating
        NaN. pandas aggregations skip NaN by default -- confirm that for
        yourself rather than taking it on trust, because the behaviour
        differs from numpy's, where np.mean() of an array containing NaN is
        NaN.

    Hints:
        * .set_index("timestamp").resample("D") gets you daily groups.
        * Named aggregation works on a Resampler exactly as it does on a
          GroupBy: .agg(temp_min=("temp_c", "min"), ...).
        * "count" counts non-NaN values -- that is n_readings, for free.
        * "size" would count ALL rows including NaN. The difference between
          "count" and "size" is the whole point of this exercise; make sure
          you can say which one you want and why.
        * .reset_index().rename(columns={"timestamp": "date"}) at the end.

    A question worth answering before you move on:
        The data contains one deliberate 7-hour outage. Find the day it
        happened by sorting your result on n_missing. If you had computed
        temp_mean for that day without noticing, how wrong would it be, and
        in which direction? (The outage is at a specific time of day, and
        temperature has a daily cycle -- so the error is not random.)

    Check yourself:
        pytest tests/test_lab05.py -v
    """
    raise NotImplementedError("Implement daily_weather_summary -- see the docstring.")
