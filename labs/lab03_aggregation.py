"""Lab 03 - Grouping and aggregating.

CONCEPT
-------
"Split, apply, combine" is the engine of data analytics. You split rows into
groups by some key, apply a reduction to each group, and combine the results
into a new frame. In SQL this is GROUP BY; in pandas it is .groupby().

If you know SQL, the mapping is direct:

    SELECT category,
           COUNT(*)      AS n_orders,
           SUM(revenue)  AS total_revenue
    FROM orders
    GROUP BY category

    orders.groupby("category").agg(
        n_orders=("order_id", "count"),
        total_revenue=("revenue", "sum"),
    )

That keyword form is called *named aggregation* and it is the one to learn.
The older styles -- passing a dict, or a list of functions -- produce
MultiIndex columns like ("revenue", "sum") that you then have to flatten.
Named aggregation gives you flat, readable column names on the first try.

One thing that trips up people coming from SQL: .groupby() puts the grouping
key in the *index*, not in a column. `.reset_index()` moves it back to being
a column, which is almost always what you want when the result is a report.
`as_index=False` in the groupby call does the same thing inline.

WORKED EXAMPLE  ->  revenue_by_category()
    Group orders by category, compute several aggregates at once, rank them.

YOUR TURN       ->  summarise_customers()
    Group by a different key and build a per-customer summary, including
    date aggregates and a distinct count. This is the shape of table that
    feeds a churn model -- lab 08 uses exactly these ideas.

RUN THE TESTS
    pytest tests/test_lab03.py -v
"""

from __future__ import annotations

import pandas as pd


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def revenue_by_category(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarise orders by product category, best-selling first.

    Args:
        orders: Cleaned orders from labs.load_clean_orders().

    Returns:
        A DataFrame with one row per category and these columns:
            category          str    -- the group key, as a real column
            n_orders          int    -- how many orders in this category
            total_units       int    -- units sold
            total_revenue     float  -- revenue, rounded to 2dp
            avg_order_value   float  -- mean revenue per order, rounded to 2dp
            revenue_share     float  -- this category's fraction of all
                                        revenue, rounded to 4dp (0.0-1.0)
        Sorted by total_revenue descending, index reset to 0..n-1.

    Example:
        >>> from labs import load_clean_orders
        >>> summary = revenue_by_category(load_clean_orders())
        >>> round(summary["revenue_share"].sum(), 2)
        1.0
    """
    # Named aggregation: each keyword becomes an output column, and the value
    # is a (source_column, function) tuple. Everything is computed in a single
    # pass over the groups, which is why this is far faster than looping.
    grouped = orders.groupby("category").agg(
        n_orders=("order_id", "count"),
        total_units=("units", "sum"),
        total_revenue=("revenue", "sum"),
        avg_order_value=("revenue", "mean"),
    )

    # The category is currently the index. Move it back into a column so the
    # result is a plain table you could write to CSV or send as JSON.
    grouped = grouped.reset_index()

    # A share-of-total column needs the grand total, which you compute from
    # the grouped frame -- not from the original, because they can differ if
    # any rows were excluded upstream.
    total = grouped["total_revenue"].sum()
    grouped["revenue_share"] = grouped["total_revenue"] / total

    # Round only at the end. Rounding intermediates makes shares stop summing
    # to 1 and is one of the most common sources of "my numbers are slightly
    # off" in analytics work.
    grouped["total_revenue"] = grouped["total_revenue"].round(2)
    grouped["avg_order_value"] = grouped["avg_order_value"].round(2)
    grouped["revenue_share"] = grouped["revenue_share"].round(4)

    return grouped.sort_values("total_revenue", ascending=False).reset_index(drop=True)


# ===========================================================================
# YOUR TURN
# ===========================================================================


def summarise_customers(orders: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-customer summary of ordering behaviour.

    Args:
        orders: Cleaned orders from labs.load_clean_orders(). Relevant
            columns: "customer_id", "order_id", "order_date" (datetime64),
            "revenue", "category".

    Returns:
        A DataFrame with one row per customer that placed at least one order,
        and these columns in this order:

            customer_id       str       -- the group key, as a real column
            n_orders          int       -- number of orders
            total_revenue     float     -- sum of revenue, rounded to 2dp
            avg_order_value   float     -- mean revenue per order, to 2dp
            first_order_date  datetime  -- earliest order_date
            last_order_date   datetime  -- latest order_date
            n_categories      int       -- how many DISTINCT categories they
                                           have bought from

        Sorted by total_revenue descending, index reset to 0..n-1.

        Customers with no orders simply do not appear -- you are grouping the
        orders table, so a customer with nothing to group is absent. (Getting
        them back requires a join, which is lab 04.)

    Hints:
        * Named aggregation takes any function name pandas knows as a string:
          "count", "sum", "mean", "min", "max", "nunique", "std", "median".
        * "min" and "max" work on datetime columns and give you datetimes
          back -- that is how first_order_date and last_order_date work.
        * "nunique" counts distinct values -- that is n_categories.
        * Do NOT round first_order_date / last_order_date; they stay as
          datetimes.
        * Reorder columns at the end with df[[...]] if you need to.

    Why this matters:
        This table is the foundation of RFM analysis (Recency, Frequency,
        Monetary) -- one of the most widely used customer-analytics
        techniques in industry. last_order_date gives you Recency, n_orders
        gives you Frequency, total_revenue gives you Monetary. Lab 08 turns
        exactly these three into features for a churn model.

    Check yourself:
        pytest tests/test_lab03.py -v
    """
    raise NotImplementedError("Implement summarise_customers -- see the docstring.")
