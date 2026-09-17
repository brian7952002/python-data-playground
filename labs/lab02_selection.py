"""Lab 02 - Selecting and filtering rows.

CONCEPT
-------
Filtering in pandas is not a loop. You build a *boolean mask* -- a Series of
True/False the same length as the frame -- and hand it to the frame. This is
the single biggest shift from writing Python loops to writing pandas, and
almost every analysis you ever write starts with it.

    mask = df["revenue"] > 500        # a Series of True/False
    df.loc[mask]                      # the rows where it is True

Three rules that will save you hours:

  1. Combine masks with `&` and `|`, never `and` / `or`. The keywords force a
     whole Series into a single True/False and raise "truth value is
     ambiguous". The operators work element by element, which is what you want.

  2. Parenthesise every condition: `(a > 1) & (b < 2)`. `&` binds tighter than
     `>` in Python, so without parentheses it is parsed as `a > (1 & b) < 2`
     and the error message will not help you.

  3. Use `.loc[mask]` rather than `df[mask]` when you also want to pick
     columns: `df.loc[mask, ["order_id", "revenue"]]` does both in one step,
     and avoids the chained-assignment warnings that bite beginners.

WORKED EXAMPLE  ->  select_high_value_orders()
    Filter orders on a numeric threshold and a set of channels, keep a subset
    of columns, and sort the result.

YOUR TURN       ->  filter_sensor_readings()
    Same skills, different data: filter sensor readings by a temperature
    range and by hour of day, using the .dt accessor.

RUN THE TESTS
    pytest tests/test_lab02.py -v
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def select_high_value_orders(
    orders: pd.DataFrame,
    min_revenue: float,
    channels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return high-value orders, optionally restricted to certain channels.

    Args:
        orders: Cleaned orders, as returned by labs.load_clean_orders().
            Must contain "revenue" and "channel".
        min_revenue: Keep orders whose revenue is >= this value.
        channels: If given, keep only orders in these channels. If None, keep
            every channel.

    Returns:
        A new DataFrame with columns
        ["order_id", "order_date", "customer_id", "channel", "revenue"],
        sorted by revenue descending, with a clean 0..n-1 index.

    Example:
        >>> from labs import load_clean_orders
        >>> top = select_high_value_orders(load_clean_orders(), 2000, ["web"])
        >>> (top["revenue"] >= 2000).all()
        True
    """
    # Step 1: build the mask. Comparing a column to a scalar broadcasts the
    # comparison across every row and gives back a boolean Series.
    mask = orders["revenue"] >= min_revenue

    # Step 2: narrow it further, but only if the caller asked us to.
    # .isin() is the vectorised form of "is this value one of these?" -- it is
    # both faster and clearer than chaining `(col == "a") | (col == "b")`.
    if channels is not None:
        # Note the parentheses around each operand of `&`. They are not
        # optional; see rule 2 in the module docstring.
        mask = mask & (orders["channel"].isin(channels))

    # Step 3: apply the mask and pick columns in one .loc call.
    keep = ["order_id", "order_date", "customer_id", "channel", "revenue"]
    result = orders.loc[mask, keep]

    # Step 4: sort, then reset the index. The original index survives filtering
    # (you would see gaps like 3, 17, 42), which is confusing downstream --
    # drop=True throws the old index away instead of adding it as a column.
    return result.sort_values("revenue", ascending=False).reset_index(drop=True)


# ===========================================================================
# YOUR TURN
# ===========================================================================


def filter_sensor_readings(
    sensor: pd.DataFrame,
    min_temp: float | None = None,
    max_temp: float | None = None,
    hours: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Filter sensor readings by temperature range and hour of day.

    Args:
        sensor: DataFrame with "timestamp" (datetime64), "temp_c" (float,
            may be NaN) and "humidity_pct" (float, may be NaN).
        min_temp: If given, keep only rows where temp_c >= min_temp.
        max_temp: If given, keep only rows where temp_c <= max_temp.
        hours: If given, keep only rows whose timestamp falls in one of these
            hours of the day (0-23). E.g. [6, 7, 8] means the 6am, 7am and
            8am readings, on every day.

    Returns:
        A new DataFrame with the same columns as `sensor`, containing only
        matching rows, sorted by timestamp ascending, with a clean 0..n-1
        index.

        Rows where temp_c is NaN must NEVER be returned when min_temp or
        max_temp was supplied -- a missing reading is not a match. (You may
        get this for free: NaN comparisons are always False. Make sure you
        understand why before moving on -- it is a classic source of silent
        bugs.)

        If every argument is None, return all rows (sorted, reindexed).

    Hints:
        * The .dt accessor exposes datetime parts on a whole column at once:
          `sensor["timestamp"].dt.hour` gives an int Series of 0-23.
          There is also .dt.day, .dt.month, .dt.dayofweek, .dt.date.
        * Start with a mask that is True everywhere, then narrow it:
              mask = pd.Series(True, index=sensor.index)
          Building up a mask conditionally is the standard pattern for
          optional filters -- it is exactly what the worked example does.
        * .isin() handles the `hours` check.

    Check yourself:
        pytest tests/test_lab02.py -v
    """
    raise NotImplementedError("Implement filter_sensor_readings -- see the docstring.")
