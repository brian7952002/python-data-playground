"""The catalogue of labs.

The web UI reads this to render the lab list, and the test runner reads it to
know which test file belongs to which lab. Keeping it in one place means
adding a lab is a single edit here plus two new files.

To add your own lab:
    1. Write labs/lab09_yourthing.py with a demo_* and an exercise function.
    2. Write tests/test_lab09.py with @pytest.mark.demo / @pytest.mark.exercise.
    3. Append a Lab(...) entry below.
The UI picks it up on the next page load -- there is nothing else to register.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Lab:
    """One lab: a concept, a worked example, and a twin exercise."""

    id: str                 # "lab01", used in URLs
    number: int             # display order
    title: str
    track: str              # "analytics" | "apis" | "data science"
    concept: str            # one-line summary shown on the card
    module: str             # importable module path
    test_file: str          # path to its pytest file, relative to repo root
    worked_example: str     # function name of the implemented example
    exercise: str           # function name of the stub you implement
    dataset: str            # which data it works on

    def to_dict(self) -> dict:
        return asdict(self)


LABS: list[Lab] = [
    Lab(
        id="lab01",
        number=1,
        title="Loading and inspecting messy data",
        track="analytics",
        concept="Find out how a dataset is broken before you trust a single number from it.",
        module="labs.lab01_loading",
        test_file="tests/test_lab01.py",
        worked_example="profile_dataframe",
        exercise="clean_sensor",
        dataset="orders.csv, sensor.csv",
    ),
    Lab(
        id="lab02",
        number=2,
        title="Selecting and filtering rows",
        track="analytics",
        concept="Boolean masks instead of loops -- the core shift from Python to pandas.",
        module="labs.lab02_selection",
        test_file="tests/test_lab02.py",
        worked_example="select_high_value_orders",
        exercise="filter_sensor_readings",
        dataset="orders.csv, sensor.csv",
    ),
    Lab(
        id="lab03",
        number=3,
        title="Grouping and aggregating",
        track="analytics",
        concept="Split, apply, combine -- GROUP BY for people who know SQL.",
        module="labs.lab03_aggregation",
        test_file="tests/test_lab03.py",
        worked_example="revenue_by_category",
        exercise="summarise_customers",
        dataset="orders.csv",
    ),
    Lab(
        id="lab04",
        number=4,
        title="Joining tables",
        track="analytics",
        concept="Merges that fail loudly instead of silently doubling your revenue.",
        module="labs.lab04_joins",
        test_file="tests/test_lab04.py",
        worked_example="attach_customer_attributes",
        exercise="customer_report",
        dataset="orders.csv, customers.csv",
    ),
    Lab(
        id="lab05",
        number=5,
        title="Time series: resampling and rolling windows",
        track="analytics",
        concept="Separate the daily cycle from the trend from the noise.",
        module="labs.lab05_timeseries",
        test_file="tests/test_lab05.py",
        worked_example="daily_revenue_trend",
        exercise="daily_weather_summary",
        dataset="orders.csv, sensor.csv",
    ),
    Lab(
        id="lab06",
        number=6,
        title="Consuming an API properly",
        track="apis",
        concept="Pagination, backoff, and knowing which failures are worth retrying.",
        module="labs.lab06_api_consume",
        test_file="tests/test_lab06.py",
        worked_example="collect_paginated",
        exercise="fetch_with_retry",
        dataset="none -- runs offline against fakes",
    ),
    Lab(
        id="lab07",
        number=7,
        title="Building an API endpoint that defends itself",
        track="apis",
        concept="Validate at the boundary with pydantic, then trust your inputs.",
        module="labs.lab07_api_build",
        test_file="tests/test_lab07.py",
        worked_example="run_category_query",
        exercise="run_region_query",
        dataset="orders.csv, customers.csv",
    ),
    Lab(
        id="lab08",
        number=8,
        title="Features, labels, and honest evaluation",
        track="data science",
        concept="Leakage-free cutoffs, and why accuracy is usually a lie.",
        module="labs.lab08_modeling",
        test_file="tests/test_lab08.py",
        worked_example="build_customer_features",
        exercise="evaluate_binary_classifier",
        dataset="orders.csv, customers.csv",
    ),
]

LABS_BY_ID: dict[str, Lab] = {lab.id: lab for lab in LABS}
