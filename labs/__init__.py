"""Shared helpers for every lab.

Keep this module small. Its only job is to answer "where is the data?" in one
place, so that no lab has to hardcode a path and every lab works no matter
which directory you run pytest from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# __file__ is this file; .parent is labs/; .parent.parent is the repo root.
# Resolving to an absolute path first means this works when pytest is invoked
# from a subdirectory, from your editor, or from the web UI's subprocess.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def dataset_path(name: str) -> Path:
    """Return the path to a dataset in data/, checking that it exists.

    Failing loudly here -- with a message that says what to run -- beats a
    FileNotFoundError from deep inside pandas that doesn't tell you the fix.
    """
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {name!r} not found at {path}.\n"
            f"Regenerate the datasets with:  python scripts/generate_data.py"
        )
    return path


def load_raw_orders() -> pd.DataFrame:
    """Load orders.csv exactly as it sits on disk -- mess included.

    Deliberately does NO cleaning and NO date parsing. Every column comes back
    as pandas inferred it from the raw text. Lab 01 is about seeing that mess
    clearly before you touch it; the later labs clean it first.
    """
    return pd.read_csv(dataset_path("orders.csv"))


def load_clean_orders() -> pd.DataFrame:
    """Load orders.csv with the lab-01 problems already fixed.

    This is the loader labs 02-08 use, so that each lab exercises one new idea
    instead of re-solving cleaning every time. What it fixes:

      * `order_date` parsed to real datetime64, not strings
      * `units` coerced to a nullable integer, with missing values dropped
      * `channel` stripped of whitespace and lowercased
      * exact duplicate rows removed
      * a `revenue` column added (units * unit_price)

    Read the body -- this is the worked reference for the lab 01 exercise.
    """
    df = pd.read_csv(dataset_path("orders.csv"))

    # parse_dates on read_csv would also work; doing it explicitly keeps the
    # step visible. errors="raise" means a bad date fails loudly, not silently.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")

    # Blank units came in as NaN, which forces the column to float. Coerce,
    # drop the rows we cannot use, then narrow back to an integer type.
    df["units"] = pd.to_numeric(df["units"], errors="coerce")
    df = df.dropna(subset=["units"])
    df["units"] = df["units"].astype("int64")

    # " retail" and "Retail" and "retail" are the same channel. Normalise the
    # text so that a groupby doesn't split one channel into three.
    df["channel"] = df["channel"].str.strip().str.lower()

    # The duplicates are whole-row repeats of a double-written order.
    df = df.drop_duplicates()

    # A derived column you will want in almost every later lab.
    df["revenue"] = df["units"] * df["unit_price"]

    # Dropping rows leaves gaps in the index; reset so downstream positional
    # work (.iloc, merges, concat) behaves predictably.
    return df.reset_index(drop=True)


def load_customers() -> pd.DataFrame:
    """Load customers.csv with signup_date parsed as a datetime."""
    return pd.read_csv(dataset_path("customers.csv"), parse_dates=["signup_date"])


def load_sensor() -> pd.DataFrame:
    """Load sensor.csv with the timestamp parsed, but gaps left intact.

    The missing readings are the point of lab 05 -- do not fill them here.
    """
    return pd.read_csv(dataset_path("sensor.csv"), parse_dates=["timestamp"])
