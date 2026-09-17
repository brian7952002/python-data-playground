"""Shared pytest fixtures.

conftest.py is discovered automatically by pytest -- you never import it.
Anything defined here with @pytest.fixture is available by name as an
argument to any test in this directory or below.

The fixtures are module-scoped: each dataset is loaded once per test module
rather than once per test. Loading a CSV eight times to run eight tests is
wasted time, and slow tests are tests you stop running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Put the repo root on sys.path so `import labs` works no matter where pytest
# was launched from. The alternative is packaging the project properly with a
# pyproject.toml and `pip install -e .`, which is what you would do for real
# library code -- this is the lighter-weight route for a learning repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs import (  # noqa: E402  (import after sys.path edit, deliberately)
    dataset_path,
    load_clean_orders,
    load_customers,
    load_sensor,
    load_raw_orders,
)


@pytest.fixture(scope="module")
def raw_orders() -> pd.DataFrame:
    """orders.csv exactly as it sits on disk, mess included."""
    return load_raw_orders()


@pytest.fixture(scope="module")
def clean_orders() -> pd.DataFrame:
    """orders.csv with lab 01's problems fixed and a revenue column added."""
    return load_clean_orders()


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    """customers.csv with signup_date parsed."""
    return load_customers()


@pytest.fixture(scope="module")
def raw_sensor() -> pd.DataFrame:
    """sensor.csv with NO parsing at all -- timestamps are still strings.

    This is what lab 01's clean_sensor() receives.
    """
    return pd.read_csv(dataset_path("sensor.csv"))


@pytest.fixture(scope="module")
def sensor() -> pd.DataFrame:
    """sensor.csv with timestamps parsed but gaps left intact."""
    return load_sensor()
