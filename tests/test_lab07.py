"""Tests for Lab 07 - building an API endpoint.

The validation tests are grouped into a class so you can work on the model
alone before touching the handler:

    pytest tests/test_lab07.py::TestRegionQueryValidation -v
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from labs.lab07_api_build import (
    CategoryQuery,
    RegionQuery,
    run_category_query,
    run_region_query,
)


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.mark.demo
def test_defaults_are_applied() -> None:
    query = CategoryQuery()
    assert query.limit == 10
    assert query.category is None
    assert query.sort_by == "total_revenue"


@pytest.mark.demo
@pytest.mark.parametrize("bad_limit", [0, -1, 101, 10_000])
def test_limit_bounds_are_enforced(bad_limit: int) -> None:
    """An unbounded limit is a denial of service you wrote yourself."""
    with pytest.raises(ValidationError):
        CategoryQuery(limit=bad_limit)


@pytest.mark.demo
def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown category"):
        CategoryQuery(category="sunglasses")


@pytest.mark.demo
def test_unknown_sort_column_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CategoryQuery(sort_by="revenue; DROP TABLE orders")


@pytest.mark.demo
def test_handler_respects_the_limit(clean_orders: pd.DataFrame) -> None:
    rows = run_category_query(clean_orders, CategoryQuery(limit=2))
    assert len(rows) == 2
    assert rows[0]["total_revenue"] >= rows[1]["total_revenue"]


@pytest.mark.demo
def test_handler_returns_plain_dicts(clean_orders: pd.DataFrame) -> None:
    """The handler's output has to survive json.dumps -- it is an API."""
    import json

    json.dumps(run_category_query(clean_orders, CategoryQuery()))


# ---------------------------------------------------------------------------
# YOUR EXERCISE -- part 1: the model
# ---------------------------------------------------------------------------


@pytest.mark.exercise
class TestRegionQueryValidation:
    """Work on RegionQuery until this whole class passes."""

    def test_defaults(self) -> None:
        query = RegionQuery()
        assert query.region is None
        assert query.segment is None
        assert query.limit == 10
        assert query.min_orders == 0

    def test_accepts_a_valid_region(self) -> None:
        assert RegionQuery(region="North").region == "North"

    def test_rejects_an_unknown_region(self) -> None:
        with pytest.raises(ValidationError):
            RegionQuery(region="Atlantis")

    def test_region_is_case_sensitive(self) -> None:
        """The data holds "North", not "north".

        You could choose to normalise case instead of rejecting -- that is a
        real API design decision. This spec chooses strictness, because
        silently accepting "north" and returning results for "North" makes
        the contract fuzzy.
        """
        with pytest.raises(ValidationError):
            RegionQuery(region="north")

    def test_accepts_valid_segments(self) -> None:
        for segment in ("consumer", "smb", "enterprise"):
            assert RegionQuery(segment=segment).segment == segment

    def test_rejects_an_unknown_segment(self) -> None:
        with pytest.raises(ValidationError):
            RegionQuery(segment="government")

    @pytest.mark.parametrize("bad_limit", [0, -5, 51, 1000])
    def test_limit_bounds(self, bad_limit: int) -> None:
        with pytest.raises(ValidationError):
            RegionQuery(limit=bad_limit)

    def test_limit_accepts_its_boundaries(self) -> None:
        """ge and le are inclusive. Off-by-one at a boundary is the classic
        validation bug, so the spec pins both ends."""
        assert RegionQuery(limit=1).limit == 1
        assert RegionQuery(limit=50).limit == 50

    def test_min_orders_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            RegionQuery(min_orders=-1)


# ---------------------------------------------------------------------------
# YOUR EXERCISE -- part 2: the handler
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_returns_a_row_per_region(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery())
    assert len(rows) == 4, "There are four regions in the data."
    assert {row["region"] for row in rows} == {"North", "South", "East", "West"}


@pytest.mark.exercise
def test_row_shape(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery())
    assert set(rows[0]) == {"region", "n_orders", "n_customers", "total_revenue"}


@pytest.mark.exercise
def test_sorted_by_revenue(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery())
    revenues = [row["total_revenue"] for row in rows]
    assert revenues == sorted(revenues, reverse=True)


@pytest.mark.exercise
def test_totals_reconcile(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery())
    assert sum(row["n_orders"] for row in rows) == len(clean_orders)
    assert sum(row["total_revenue"] for row in rows) == pytest.approx(
        clean_orders["revenue"].sum(), abs=1.0
    )


@pytest.mark.exercise
def test_region_filter(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery(region="East"))
    assert len(rows) == 1
    assert rows[0]["region"] == "East"


@pytest.mark.exercise
def test_segment_filter_shrinks_the_numbers(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    everyone = run_region_query(clean_orders, customers, RegionQuery())
    enterprise = run_region_query(
        clean_orders, customers, RegionQuery(segment="enterprise")
    )
    total_all = sum(row["n_orders"] for row in everyone)
    total_ent = sum(row["n_orders"] for row in enterprise)
    assert 0 < total_ent < total_all


@pytest.mark.exercise
def test_limit_truncates(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery(limit=2))
    assert len(rows) == 2


@pytest.mark.exercise
def test_min_orders_applies_after_grouping(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """min_orders is a threshold on a GROUP's count.

    Applying it before the groupby would filter individual orders, which is
    meaningless -- a single order does not have a count. Set it high enough
    to exclude everything and the result must be empty, not unchanged.
    """
    rows = run_region_query(clean_orders, customers, RegionQuery(min_orders=1_000_000))
    assert rows == []


@pytest.mark.exercise
def test_n_customers_counts_distinct_people(
    clean_orders: pd.DataFrame, customers: pd.DataFrame
) -> None:
    rows = run_region_query(clean_orders, customers, RegionQuery())
    for row in rows:
        assert row["n_customers"] < row["n_orders"], (
            "Customers order more than once, so distinct customers must be "
            "fewer than orders. If these are equal you counted rows, not "
            "distinct customers -- use nunique, not count."
        )
