"""Tests for Lab 03 - grouping and aggregating."""

from __future__ import annotations

import pandas as pd
import pytest

from labs.lab03_aggregation import revenue_by_category, summarise_customers


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_one_row_per_category(clean_orders: pd.DataFrame) -> None:
    result = revenue_by_category(clean_orders)
    assert len(result) == clean_orders["category"].nunique()
    assert result["category"].is_unique


@pytest.mark.demo
def test_totals_reconcile_with_the_source(clean_orders: pd.DataFrame) -> None:
    """The first thing to check on any aggregate: does it add up?

    An aggregation that loses or duplicates rows is the most common and most
    expensive bug in analytics work, and it never announces itself.
    """
    result = revenue_by_category(clean_orders)
    assert result["n_orders"].sum() == len(clean_orders)
    assert result["total_revenue"].sum() == pytest.approx(
        clean_orders["revenue"].sum(), abs=0.05
    )


@pytest.mark.demo
def test_shares_sum_to_one(clean_orders: pd.DataFrame) -> None:
    result = revenue_by_category(clean_orders)
    assert result["revenue_share"].sum() == pytest.approx(1.0, abs=0.001)


@pytest.mark.demo
def test_ranked_by_revenue(clean_orders: pd.DataFrame) -> None:
    result = revenue_by_category(clean_orders)
    assert result["total_revenue"].is_monotonic_decreasing
    assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_one_row_per_ordering_customer(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    assert len(result) == clean_orders["customer_id"].nunique()
    assert result["customer_id"].is_unique


@pytest.mark.exercise
def test_has_the_required_columns_in_order(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    assert list(result.columns) == [
        "customer_id", "n_orders", "total_revenue", "avg_order_value",
        "first_order_date", "last_order_date", "n_categories",
    ]


@pytest.mark.exercise
def test_counts_and_totals_reconcile(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    assert result["n_orders"].sum() == len(clean_orders)
    assert result["total_revenue"].sum() == pytest.approx(
        clean_orders["revenue"].sum(), abs=0.5
    )


@pytest.mark.exercise
def test_dates_are_still_datetimes(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    assert pd.api.types.is_datetime64_any_dtype(result["first_order_date"])
    assert pd.api.types.is_datetime64_any_dtype(result["last_order_date"])
    assert (result["first_order_date"] <= result["last_order_date"]).all()


@pytest.mark.exercise
def test_distinct_category_count_is_bounded(clean_orders: pd.DataFrame) -> None:
    """n_categories counts DISTINCT categories, so it cannot exceed either
    the number of categories that exist or the customer's own order count."""
    result = summarise_customers(clean_orders)
    n_all = clean_orders["category"].nunique()
    assert (result["n_categories"] >= 1).all()
    assert (result["n_categories"] <= n_all).all()
    assert (result["n_categories"] <= result["n_orders"]).all()


@pytest.mark.exercise
def test_average_order_value_is_consistent(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    recomputed = (result["total_revenue"] / result["n_orders"]).round(2)

    # A cent of tolerance, because the two routes round at different points:
    # the mean of unrounded revenues is not always the rounded total divided
    # by the count. Neither is wrong -- but an exact-equality assertion here
    # would fail for a correct implementation, which makes it a bad test.
    assert (result["avg_order_value"] - recomputed).abs().max() <= 0.011


@pytest.mark.exercise
def test_sorted_by_revenue_and_reindexed(clean_orders: pd.DataFrame) -> None:
    result = summarise_customers(clean_orders)
    assert result["total_revenue"].is_monotonic_decreasing
    assert list(result.index) == list(range(len(result)))


@pytest.mark.exercise
def test_spot_check_one_customer(clean_orders: pd.DataFrame) -> None:
    """Pick one customer and verify the aggregate against the raw rows.

    Reconciling totals proves the whole is right; spot-checking one group
    proves the parts are. Do both -- an aggregate can total correctly while
    assigning the wrong values to the wrong groups.
    """
    result = summarise_customers(clean_orders)
    target = result.iloc[0]["customer_id"]
    theirs = clean_orders[clean_orders["customer_id"] == target]
    row = result[result["customer_id"] == target].iloc[0]

    assert row["n_orders"] == len(theirs)
    assert row["total_revenue"] == pytest.approx(theirs["revenue"].sum(), abs=0.01)
    assert row["first_order_date"] == theirs["order_date"].min()
    assert row["last_order_date"] == theirs["order_date"].max()
    assert row["n_categories"] == theirs["category"].nunique()
