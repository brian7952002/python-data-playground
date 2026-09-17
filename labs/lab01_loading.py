"""Lab 01 - Loading data and seeing what you actually got.

CONCEPT
-------
The first thing you do with a new dataset is not analysis. It is finding out
how it is broken. Every real CSV arrives with some combination of: wrong
dtypes, blank cells, duplicate rows, and categorical values that are the same
thing spelled three different ways. If you skip this step, those problems do
not go away -- they quietly corrupt every number you compute afterwards.

A concrete example from this repo's own data: the `channel` column contains
"retail", "Retail" and " retail". Group by it without normalising and you get
three separate rows for one channel, each with a third of the real revenue.
Nothing errors. The report is just wrong.

WORKED EXAMPLE  ->  profile_dataframe()
    Given any DataFrame, produce a structured summary of its shape, dtypes,
    missing values and duplicates. Read it to learn the inspection vocabulary:
    .shape, .dtypes, .isna(), .duplicated(), .nunique().

YOUR TURN       ->  clean_sensor()
    Apply the same thinking to a different dataset. sensor.csv has its own
    problems: text timestamps, blank numeric readings, duplicate rows and
    out-of-order records. Turn it into something the later labs can trust.

RUN THE TESTS
    pytest tests/test_lab01.py -v
"""

from __future__ import annotations

import pandas as pd


# ===========================================================================
# WORKED EXAMPLE -- read this, run it, then write the exercise below.
# ===========================================================================


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Summarise the structure and health of a DataFrame.

    Args:
        df: Any DataFrame.

    Returns:
        A dict with these keys:
            n_rows          int   -- number of rows
            n_columns       int   -- number of columns
            columns         list  -- column names, in order
            dtypes          dict  -- column name -> dtype as a string
            missing_counts  dict  -- column name -> count of missing values
            duplicate_rows  int   -- number of rows that repeat an earlier row
            distinct_counts dict  -- column name -> number of distinct values

    Example:
        >>> from labs import load_raw_orders
        >>> profile = profile_dataframe(load_raw_orders())
        >>> profile["missing_counts"]["units"]
        74
        >>> profile["duplicate_rows"]
        12
        >>> profile["distinct_counts"]["channel"]   # three channels, eight spellings
        8
    """
    # .shape is a (rows, columns) tuple. Unpacking it reads better than
    # indexing it twice, and it is the fastest way to get row count.
    n_rows, n_columns = df.shape

    # .dtypes is a Series indexed by column name. The values are numpy dtype
    # objects, which do not serialise to JSON -- hence the str() conversion.
    # This matters here because the web UI sends this dict over HTTP.
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # .isna() gives a same-shaped boolean frame; .sum() counts True per column
    # because True == 1. This is the standard idiom for "what is missing".
    # int() conversion again for JSON-safety -- these are numpy int64.
    missing_counts = {col: int(count) for col, count in df.isna().sum().items()}

    # .duplicated() marks every row that has appeared before (the FIRST
    # occurrence is False, so this counts extras, not the whole group).
    duplicate_rows = int(df.duplicated().sum())

    # .nunique() excludes NaN by default, which is usually what you want when
    # asking "how many real categories are in this column?"
    distinct_counts = {col: int(df[col].nunique()) for col in df.columns}

    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "duplicate_rows": duplicate_rows,
        "distinct_counts": distinct_counts,
    }


# ===========================================================================
# YOUR TURN -- delete the `raise` and implement this.
# ===========================================================================


def clean_sensor(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw sensor readings into a trustworthy DataFrame.

    `raw` is sensor.csv exactly as pd.read_csv() returns it with no options:
    every column is text or float, nothing is parsed, nothing is sorted.

    Args:
        raw: DataFrame with columns ["timestamp", "temp_c", "humidity_pct"].
            `timestamp` holds strings like "2025-10-03 00:00:00".
            `temp_c` and `humidity_pct` have blanks where the sensor dropped
            out; pandas reads those blanks as NaN.

    Returns:
        A new DataFrame (do not mutate `raw`) with:
            * "timestamp"    -> dtype datetime64[ns]
            * "temp_c"       -> dtype float64, blanks left as NaN
            * "humidity_pct" -> dtype float64, blanks left as NaN
            * "is_missing"   -> NEW bool column, True where temp_c is NaN
            * exact duplicate rows removed
            * rows sorted by timestamp, oldest first
            * a clean 0..n-1 index

    IMPORTANT: do NOT drop the rows with missing readings. A gap in a time
    series is information -- lab 05 needs to see it. The `is_missing` flag
    exists so you can find those gaps without deleting them.

    Hints:
        * pd.to_datetime(series) parses a text column to datetime64.
        * pd.to_numeric(series, errors="coerce") turns unparseable text into
          NaN instead of raising. It is the safe default for dirty numerics.
        * .drop_duplicates(), .sort_values(), .reset_index(drop=True) all
          return new frames by default -- chain them.
        * .isna() gives you the boolean mask for the is_missing column.
        * Start with `df = raw.copy()` so you cannot corrupt the caller's data.

    Check yourself:
        pytest tests/test_lab01.py -v
    """
    raise NotImplementedError("Implement clean_sensor -- see the docstring above.")
