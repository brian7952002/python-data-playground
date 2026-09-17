"""Lab 04 - Joining tables.

CONCEPT
-------
Real analysis almost never happens on one table. The facts (orders) live in
one place, the attributes (which region a customer is in) in another, and you
have to put them together before you can answer anything interesting.

pd.merge() is pandas' JOIN. The parameters that matter:

    how="inner"   keep only rows that matched on both sides (the default)
    how="left"    keep every row from the left frame, NaN where no match
    how="outer"   keep everything from both sides
    on="key"      the column to join on, when both sides name it the same
    left_on/right_on   when they don't

Two habits that separate a careful analyst from a sloppy one:

  1. `validate=` states the relationship you believe holds and makes pandas
     check it: "many_to_one", "one_to_one", "one_to_many". If your belief is
     wrong -- say the right-hand key is not actually unique -- you get a
     loud MergeError instead of a silently duplicated result set. A join that
     accidentally fans out is the classic way to double your revenue numbers.

  2. `indicator=True` adds a "_merge" column saying where each row came from
     ("both", "left_only", "right_only"). Check it before dropping it. If you
     expected every order to match a customer and 40 didn't, you want to know
     that, not discover it three steps later.

The direction of a left join is a decision, not a detail. Joining orders to
customers keeps one row per order. Joining customers to an order summary
keeps one row per customer -- including customers who never ordered. Those
answer different questions.

WORKED EXAMPLE  ->  attach_customer_attributes()
    A many-to-one left join that enriches each order with its customer's
    region and segment, with both safety checks in place.

YOUR TURN       ->  customer_report()
    Join in the other direction so that customers with zero orders survive,
    and fill their missing aggregates sensibly.

RUN THE TESTS
    pytest tests/test_lab04.py -v
"""

from __future__ import annotations

import pandas as pd


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def attach_customer_attributes(
    orders: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """Add each order's customer region and segment to the orders table.

    Args:
        orders: Cleaned orders from labs.load_clean_orders().
        customers: Customers from labs.load_customers(). One row per
            customer_id, with "region", "segment" and "signup_date".

    Returns:
        A DataFrame with every column of `orders` plus "region", "segment"
        and "signup_date". Exactly as many rows as `orders` -- a many-to-one
        join must never change the row count of the left side.

    Raises:
        ValueError: if any order references a customer that does not exist.

    Example:
        >>> from labs import load_clean_orders, load_customers
        >>> enriched = attach_customer_attributes(load_clean_orders(), load_customers())
        >>> len(enriched) == len(load_clean_orders())
        True
    """
    # validate="many_to_one": many orders, one customer each. If customers
    # had a duplicate customer_id, this raises instead of quietly producing
    # more rows than we started with.
    merged = pd.merge(
        orders,
        customers[["customer_id", "region", "segment", "signup_date"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # A left join fills non-matches with NaN rather than dropping them, so the
    # row count alone will not tell you something went wrong. The indicator
    # column will. Check it explicitly, then discard it.
    unmatched = merged["_merge"] == "left_only"
    if unmatched.any():
        missing_ids = sorted(merged.loc[unmatched, "customer_id"].unique())
        raise ValueError(
            f"{int(unmatched.sum())} orders reference unknown customers: "
            f"{missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}"
        )

    return merged.drop(columns="_merge")


# ===========================================================================
# YOUR TURN
# ===========================================================================


def customer_report(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Build a per-customer report that includes customers who never ordered.

    Lab 03's summarise_customers() silently dropped anyone with no orders,
    because you cannot group rows that do not exist. That is a real bug in a
    real report: "our customers average 4.2 orders" is a very different claim
    if a third of your customers ordered zero times and were excluded.

    This function fixes it by joining from the customers side.

    Args:
        customers: From labs.load_customers(). One row per customer_id.
        orders: Cleaned orders from labs.load_clean_orders().

    Returns:
        A DataFrame with EXACTLY one row per customer in `customers` --
        the same row count, no matter how many ordered -- and these columns:

            customer_id      str      -- from customers
            region           str      -- from customers
            segment          str      -- from customers
            n_orders         int      -- 0 for customers with no orders
            total_revenue    float    -- 0.0 for customers with no orders,
                                         otherwise summed and rounded to 2dp
            has_ordered      bool     -- True when n_orders > 0

        Sorted by total_revenue descending, then customer_id ascending as a
        tie-break (so the zero-revenue customers come back in a stable,
        predictable order rather than whatever the join happened to produce).
        Index reset to 0..n-1.

        Note the dtypes: n_orders must be a real integer type, not float.
        This is the trap in this exercise. A left join introduces NaN, NaN
        forces the column to float64, and .fillna(0) alone leaves it float --
        so you get 0.0 orders, and an "n_orders" column you cannot use as an
        index or a count. Fill first, then cast.

    Hints:
        * Aggregate the orders first (one row per customer), THEN merge that
          summary onto `customers` with how="left". Do not merge the raw
          orders -- that fans out to one row per order.
        * validate="one_to_one" is the right assertion for this merge, and it
          will catch you if your aggregation accidentally produced duplicates.
        * .fillna({"n_orders": 0, "total_revenue": 0.0}) fills per column.
        * .astype("int64") after filling.
        * .sort_values() takes a list of columns and a list of ascending
          flags: sort_values(["a", "b"], ascending=[False, True]).

    Check yourself:
        pytest tests/test_lab04.py -v
    """
    raise NotImplementedError("Implement customer_report -- see the docstring.")
