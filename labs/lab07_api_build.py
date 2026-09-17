"""Lab 07 - Building an API endpoint that defends itself.

CONCEPT
-------
The moment your analysis is reachable over HTTP, the inputs stop being
yours. Somebody will send limit=999999999, sort=DROP TABLE, region=null, or
just leave a parameter off entirely. An endpoint that assumes well-formed
input is an endpoint that returns a 500 and a stack trace.

pydantic turns that problem into a declaration. You describe the shape of a
valid request once, and it enforces it:

    class Query(BaseModel):
        limit: int = Field(default=10, ge=1, le=100)

FastAPI wires this in automatically: bad input never reaches your function,
and the caller gets a 422 with a precise, machine-readable explanation of
which field was wrong and why. You write zero validation code in the handler.

Two distinctions worth holding onto:

  * `Field(ge=1, le=100)` is for constraints pydantic already knows how to
    express -- ranges, lengths, patterns. Prefer it; it is declarative and it
    shows up in your generated OpenAPI docs.
  * `@field_validator` is for rules pydantic cannot guess -- "this region must
    be one we actually have data for", "these two fields must agree". Use it
    when Field cannot say what you mean.

And one habit: validate at the boundary, not in the middle. Once a request
has been parsed into a model, the rest of your code can trust it completely
and stop writing defensive `if x is None` checks everywhere.

SEE IT RUNNING
    Start the playground (python -m uvicorn app.main:app --reload) and open
    http://127.0.0.1:8000/docs -- FastAPI generates interactive documentation
    from these models. Try sending limit=0 and read the 422 it gives back.

WORKED EXAMPLE  ->  CategoryQuery + run_category_query()
    A validated query model and the handler that answers it.

YOUR TURN       ->  RegionQuery + run_region_query()
    The twin: a model with a custom validator, and a handler that joins two
    tables before aggregating.

RUN THE TESTS
    pytest tests/test_lab07.py -v
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# The categories and regions that actually exist in the data. Validating
# against these means a typo comes back as a clear 422 listing the valid
# options, instead of a 200 with an empty result set that looks like a real
# answer meaning "no sales in that category".
VALID_CATEGORIES = ["footwear", "outerwear", "apparel", "packs", "hardware"]
VALID_REGIONS = ["North", "South", "East", "West"]


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


class CategoryQuery(BaseModel):
    """Validated parameters for the category-revenue endpoint."""

    # Optional filter. `| None` with a default of None is how you say
    # "this parameter may be omitted" -- without the default it is required.
    category: str | None = Field(
        default=None,
        description="Restrict to one category. Omit for all categories.",
    )

    # ge/le are enforced by pydantic before your handler runs. The ceiling of
    # 100 is not arbitrary politeness -- an unbounded limit is a denial of
    # service you wrote yourself.
    limit: int = Field(default=10, ge=1, le=100, description="Max rows to return.")

    # Literal restricts a string to an exact set of values, and FastAPI turns
    # it into a dropdown in the generated docs.
    sort_by: Literal["total_revenue", "n_orders"] = Field(
        default="total_revenue",
        description="Which column to rank by.",
    )

    min_revenue: float = Field(default=0.0, ge=0.0)

    @field_validator("category")
    @classmethod
    def category_must_exist(cls, value: str | None) -> str | None:
        """Reject categories that are not in the dataset.

        Field() cannot express this, because the valid set lives in our data
        rather than in the type. Returning the value is mandatory -- a
        validator that forgets to return turns the field into None.
        """
        if value is not None and value not in VALID_CATEGORIES:
            raise ValueError(
                f"Unknown category {value!r}. Valid options: {VALID_CATEGORIES}"
            )
        return value


def run_category_query(orders: pd.DataFrame, query: CategoryQuery) -> list[dict]:
    """Answer a CategoryQuery against the orders table.

    Args:
        orders: Cleaned orders from labs.load_clean_orders().
        query: An ALREADY-VALIDATED CategoryQuery. Because validation happened
            at the boundary, this function does no defensive checking at all.

    Returns:
        A list of plain dicts (JSON-serialisable), one per category, each with
        "category", "n_orders" and "total_revenue".

    Example:
        >>> from labs import load_clean_orders
        >>> rows = run_category_query(load_clean_orders(), CategoryQuery(limit=2))
        >>> len(rows)
        2
    """
    working = orders

    # query.category is either a valid category or None -- the model has
    # already guaranteed it. No need to check for "", whitespace, or a typo.
    if query.category is not None:
        working = working[working["category"] == query.category]

    summary = (
        working.groupby("category")
        .agg(n_orders=("order_id", "count"), total_revenue=("revenue", "sum"))
        .reset_index()
    )
    summary["total_revenue"] = summary["total_revenue"].round(2)
    summary = summary[summary["total_revenue"] >= query.min_revenue]

    summary = summary.sort_values(query.sort_by, ascending=False).head(query.limit)

    # to_dict("records") is the standard bridge from DataFrame to JSON. Note
    # that numpy scalars can trip up some serialisers -- here the .round() and
    # the int counts come back as Python-friendly types, but on a frame with
    # exotic dtypes you may need .astype(object) first.
    return summary.to_dict("records")


# ===========================================================================
# YOUR TURN
# ===========================================================================


class RegionQuery(BaseModel):
    """Validated parameters for the region-revenue endpoint.

    YOUR TASK: declare the fields below. Delete this docstring's TODO list as
    you go. The tests construct RegionQuery(...) directly and assert that
    invalid input raises pydantic.ValidationError, so the constraints have to
    live on the model -- not in run_region_query().

    Fields to declare:

        region: str | None
            Default None (meaning "all regions"). If given, must be one of
            VALID_REGIONS. That set-membership check cannot be expressed with
            Field(), so it needs a @field_validator -- copy the shape of
            category_must_exist above, including the @classmethod decorator
            and the `return value` at the end.

        segment: Literal["consumer", "smb", "enterprise"] | None
            Default None. Literal does the validation for you here, so no
            custom validator is needed. Notice that you get to choose between
            Literal and a validator depending on whether the valid set is
            known at import time -- regions come from the data, segments are
            fixed by the business.

        limit: int
            Default 10, minimum 1, maximum 50. Use Field(ge=..., le=...).

        min_orders: int
            Default 0, minimum 0. Filters out regions with too few orders.

    Hints:
        * `from typing import Literal` is already imported at the top.
        * A validator's method name is arbitrary; only the field name passed
          to @field_validator matters.
        * Run `pytest tests/test_lab07.py::TestRegionQueryValidation -v` to
          work on the model alone before touching the handler.
    """

    # TODO: replace this line with your field declarations.
    ...


def run_region_query(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    query: RegionQuery,
) -> list[dict]:
    """Answer a RegionQuery by joining orders to customers, then aggregating.

    Args:
        orders: Cleaned orders from labs.load_clean_orders().
        customers: From labs.load_customers(), with "region" and "segment".
        query: An already-validated RegionQuery.

    Returns:
        A list of plain dicts, one per region that survives the filters, each
        with exactly these keys:

            region           str    -- the region name
            n_orders         int    -- orders from customers in that region
            n_customers      int    -- DISTINCT customers who ordered
            total_revenue    float  -- summed revenue, rounded to 2dp

        Sorted by total_revenue descending, and truncated to query.limit rows.

        Filtering order matters, and this is the part to think about rather
        than guess:
          1. Filter to query.region and query.segment FIRST, at the order
             level, before you aggregate.
          2. Then group by region.
          3. Then drop regions with n_orders < query.min_orders.
        Applying min_orders before the grouping would filter individual
        orders, which is meaningless -- an order does not have a count.

    Hints:
        * Lab 04's attach_customer_attributes() does the join you need. You
          may import and reuse it:
              from labs.lab04_joins import attach_customer_attributes
          Reusing your own earlier work is the point of building a library.
        * "nunique" on customer_id gives n_customers.
        * .head(n) truncates after sorting.
        * .to_dict("records") converts the final frame to a list of dicts.

    Check yourself:
        pytest tests/test_lab07.py -v
    """
    raise NotImplementedError("Implement run_region_query -- see the docstring.")
