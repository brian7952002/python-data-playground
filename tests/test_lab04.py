"""Tests for Lab 04 - joining tables."""

from __future__ import annotations

import pandas as pd
import pytest

from labs.lab04_joins import attach_customer_attributes, customer_report


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_row_count_is_preserved(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    """A many-to-one join must not change the left side's row count.

    If it does, your right-hand key was not unique and you have just
    multiplied your revenue. This assertion is worth writing in production
    code, not just in tests.
    """
    result = attach_customer_attributes(clean_orders, customers)
    assert len(result) == len(clean_orders)


@pytest.mark.demo
def test_attributes_are_attached(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    result = attach_customer_attributes(clean_orders, customers)
    for col in ("region", "segment", "signup_date"):
        assert col in result.columns
        assert result[col].isna().sum() == 0


@pytest.mark.demo
def test_unknown_customer_raises(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    """Drop a customer from the lookup table and the join should complain.

    The failure mode this prevents: a silent left join leaves NaN regions,
    a later groupby drops them, and the regional report quietly under-counts.
    """
    truncated = customers.iloc[1:]  # remove the first customer
    orphaned_id = customers.iloc[0]["customer_id"]

    if orphaned_id not in set(clean_orders["customer_id"]):
        pytest.skip("That customer placed no orders, so nothing is orphaned.")

    with pytest.raises(ValueError, match="unknown customers"):
        attach_customer_attributes(clean_orders, truncated)


@pytest.mark.demo
def test_duplicate_keys_are_caught_by_validate(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """validate="many_to_one" turns a silent fan-out into a loud error."""
    doubled = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        attach_customer_attributes(clean_orders, doubled)


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_every_customer_appears(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    """The whole point: customers with zero orders must survive the join."""
    result = customer_report(customers, clean_orders)
    assert len(result) == len(customers)
    assert set(result["customer_id"]) == set(customers["customer_id"])


@pytest.mark.exercise
def test_non_orderers_are_present_and_zeroed(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    result = customer_report(customers, clean_orders)
    ordered_ids = set(clean_orders["customer_id"].unique())
    silent = result[~result["customer_id"].isin(ordered_ids)]

    assert len(silent) > 0, "This dataset does have customers who never ordered."
    assert (silent["n_orders"] == 0).all()
    assert (silent["total_revenue"] == 0.0).all()
    assert (~silent["has_ordered"]).all()


@pytest.mark.exercise
def test_n_orders_is_an_integer_not_a_float(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """The trap in this exercise.

    A left join introduces NaN, NaN forces the column to float64, and
    .fillna(0) on its own leaves it float. You end up reporting "0.0 orders",
    and the column cannot be used as a count or an index. Fill, then cast.
    """
    result = customer_report(customers, clean_orders)
    assert pd.api.types.is_integer_dtype(result["n_orders"]), (
        f"n_orders is {result['n_orders'].dtype}, expected an integer dtype. "
        f"Fill the NaNs first, then .astype('int64')."
    )


@pytest.mark.exercise
def test_has_ordered_is_a_real_bool(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    result = customer_report(customers, clean_orders)
    assert result["has_ordered"].dtype == bool
    assert result["has_ordered"].equals(result["n_orders"] > 0)


@pytest.mark.exercise
def test_columns_and_order(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    result = customer_report(customers, clean_orders)
    assert list(result.columns) == [
        "customer_id", "region", "segment", "n_orders", "total_revenue", "has_ordered",
    ]


@pytest.mark.exercise
def test_totals_still_reconcile(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    result = customer_report(customers, clean_orders)
    assert result["n_orders"].sum() == len(clean_orders)
    assert result["total_revenue"].sum() == pytest.approx(
        clean_orders["revenue"].sum(), abs=0.5
    )


@pytest.mark.exercise
def test_sorted_by_revenue_then_customer_id(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """The tie-break matters: without it the zero-revenue block comes back in
    whatever order the join produced, which can change between pandas
    versions. Deterministic output is what makes a report diffable."""
    result = customer_report(customers, clean_orders)
    expected = result.sort_values(
        ["total_revenue", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.exercise
def test_the_bug_this_fixes(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    """Demonstrates why lab 03's version was misleading.

    Averaging orders per customer over only the customers who ordered gives a
    materially higher number than averaging over all customers. Both are
    computable; only one answers "how often does a customer order?".
    """
    result = customer_report(customers, clean_orders)
    honest_mean = result["n_orders"].mean()
    flattering_mean = result.loc[result["has_ordered"], "n_orders"].mean()
    assert honest_mean < flattering_mean
