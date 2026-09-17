"""Generate the playground's datasets.

Everything here is synthetic but *deliberately structured*: regions have
different average basket sizes, some products are seasonal, and a subset of
customers go quiet partway through the year. That matters because a modelling
lab run against pure noise teaches you nothing -- you want to be able to find
real signal and be rewarded for finding it.

The generator is seeded, so it produces identical files on every run. The CSVs
are committed to the repo, so you normally never need to run this. Run it only
if you want to regenerate or tweak the data:

    python scripts/generate_data.py
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

# A fixed seed makes the whole dataset reproducible. Reproducibility is not a
# nicety in data work -- it is how you tell a real result from a fluke.
SEED = 6400
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The data covers a year ending on a fixed date, so "days since" style features
# are stable no matter when you clone the repo.
END_DATE = date(2025, 12, 31)
START_DATE = END_DATE - timedelta(days=364)

REGIONS = ["North", "South", "East", "West"]
# Regional multipliers on basket size -- signal the aggregation labs surface.
REGION_BASKET_MULT = {"North": 1.00, "South": 0.82, "East": 1.24, "West": 0.95}

SEGMENTS = ["consumer", "smb", "enterprise"]
SEGMENT_WEIGHTS = [0.62, 0.28, 0.10]
SEGMENT_ORDER_RATE = {"consumer": 0.9, "smb": 2.4, "enterprise": 6.1}
SEGMENT_BASKET_MULT = {"consumer": 1.0, "smb": 2.1, "enterprise": 5.4}

PRODUCTS = [
    # (product, category, base_unit_price, is_seasonal)
    ("Trail Runner", "footwear", 128.00, False),
    ("Approach Shoe", "footwear", 154.00, False),
    ("Down Jacket", "outerwear", 289.00, True),
    ("Rain Shell", "outerwear", 199.00, True),
    ("Base Layer", "apparel", 64.00, True),
    ("Hiking Sock", "apparel", 18.50, False),
    ("Daypack 22L", "packs", 112.00, False),
    ("Expedition Pack 65L", "packs", 340.00, False),
    ("Trekking Poles", "hardware", 96.00, False),
    ("Headlamp", "hardware", 42.00, False),
]

# Channel values are intentionally inconsistent in casing and whitespace. Real
# exported data looks exactly like this, and lab 01 asks you to notice it.
CHANNEL_RAW = ["web", "Web", "WEB ", "retail", "Retail", " retail", "partner", "Partner"]
CHANNEL_WEIGHTS = [0.30, 0.10, 0.04, 0.22, 0.12, 0.04, 0.13, 0.05]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Two separate newline decisions, and both matter:
    #
    #   newline=""            stops Python's own translation layer, which would
    #                         otherwise turn the writer's \r\n into \r\r\n on
    #                         Windows. This is the documented way to open a
    #                         file for the csv module.
    #   lineterminator="\n"   makes csv write LF rather than its default CRLF.
    #
    # Without the second one, this script produces different bytes on Windows
    # than on Linux, and the CI job that regenerates the data and diffs it
    # would fail on every push. Text-file line endings are a genuinely common
    # source of "works on my machine".
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path.name}  ({len(rows)} rows)")


def generate_customers(rng: np.random.Generator) -> list[dict]:
    """400 customers, each with a region, a segment and a signup date."""
    customers = []
    for i in range(400):
        # Signup spread over two years so that tenure is a usable feature.
        days_ago = int(rng.integers(0, 730))
        customers.append(
            {
                "customer_id": f"C{1000 + i}",
                "region": str(rng.choice(REGIONS, p=[0.28, 0.24, 0.26, 0.22])),
                "segment": str(rng.choice(SEGMENTS, p=SEGMENT_WEIGHTS)),
                "signup_date": (END_DATE - timedelta(days=days_ago)).isoformat(),
            }
        )
    return customers


def generate_orders(rng: np.random.Generator, customers: list[dict]) -> list[dict]:
    """One row per order.

    Two pieces of deliberate signal live in here:
      1. Enterprise customers order far more often and in bigger quantities.
      2. About a third of customers go quiet partway through the year -- that
         is the churn signal lab 08 asks you to predict from recency.
    """
    orders: list[dict] = []
    order_seq = 50000
    span_days = (END_DATE - START_DATE).days

    for cust in customers:
        rate = SEGMENT_ORDER_RATE[cust["segment"]]
        # Poisson is the natural distribution for "how many events in a period".
        n_orders = int(rng.poisson(rate * 4))
        if n_orders == 0:
            continue

        # Dormant customers have their orders squeezed into the earlier part of
        # the year, so recency becomes genuinely predictive.
        dormant = rng.random() < 0.35
        latest_allowed = int(span_days * rng.uniform(0.25, 0.62)) if dormant else span_days

        for _ in range(n_orders):
            offset = int(rng.integers(0, max(latest_allowed, 1)))
            order_date = START_DATE + timedelta(days=offset)
            product, category, base_price, seasonal = PRODUCTS[int(rng.integers(0, len(PRODUCTS)))]

            # Seasonal products cost more and move more in the cold months.
            month = order_date.month
            if seasonal and month in (10, 11, 12, 1, 2):
                season_mult, qty_bonus = 1.10, 1
            elif seasonal:
                season_mult, qty_bonus = 0.78, 0  # off-season discounting
            else:
                season_mult, qty_bonus = 1.0, 0

            basket_mult = REGION_BASKET_MULT[cust["region"]]
            seg_mult = SEGMENT_BASKET_MULT[cust["segment"]]
            units = max(1, int(rng.poisson(1.6 * basket_mult * seg_mult)) + qty_bonus)
            unit_price = round(base_price * season_mult * float(rng.uniform(0.96, 1.04)), 2)

            order_seq += 1
            orders.append(
                {
                    "order_id": f"ORD-{order_seq}",
                    "order_date": order_date.isoformat(),
                    "customer_id": cust["customer_id"],
                    "product": product,
                    "category": category,
                    "units": units,
                    "unit_price": unit_price,
                    "channel": str(rng.choice(CHANNEL_RAW, p=CHANNEL_WEIGHTS)),
                }
            )

    # --- Deliberate mess, so lab 01 has something real to find -------------
    # 1. About 2.5% of `units` are blank, the way a bad CSV export loses them.
    blank_idx = rng.choice(len(orders), size=int(len(orders) * 0.025), replace=False)
    for i in blank_idx:
        orders[int(i)]["units"] = ""

    # 2. Twelve rows are exact duplicates (a retry that double-wrote).
    dupe_idx = rng.choice(len(orders), size=12, replace=False)
    for i in dupe_idx:
        orders.append(dict(orders[int(i)]))

    orders.sort(key=lambda r: (r["order_date"], r["order_id"]))
    return orders


def generate_sensor(rng: np.random.Generator) -> list[dict]:
    """Hourly sensor readings for 90 days, with daily and seasonal cycles.

    Time series data has structure at several scales at once. Here there is a
    24-hour cycle, a slow seasonal drift, and noise -- exactly what resampling
    and rolling windows exist to separate. Some readings are missing, because
    real sensors drop out.
    """
    rows = []
    start = datetime(2025, 10, 3, 0, 0)
    n_hours = 90 * 24

    for h in range(n_hours):
        ts = start + timedelta(hours=h)

        # Daily cycle: coldest around 03:00, warmest around 15:00.
        daily = -6.5 * np.cos(2 * np.pi * (ts.hour - 3) / 24)
        # Seasonal drift across the 90 days (autumn into winter).
        seasonal = -8.0 * (h / n_hours)
        temp = 14.0 + daily + seasonal + rng.normal(0, 1.1)

        # Humidity moves roughly opposite to temperature.
        humidity = float(np.clip(68 - 1.9 * daily + rng.normal(0, 4.0), 15, 100))

        # ~1.5% dropout, plus one hard 7-hour outage to make gap handling real.
        dropped = rng.random() < 0.015
        outage = 1200 <= h < 1207
        missing = dropped or outage

        rows.append(
            {
                "timestamp": ts.isoformat(sep=" "),
                "temp_c": "" if missing else round(float(temp), 2),
                "humidity_pct": "" if missing else round(humidity, 1),
            }
        )
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)

    customers = generate_customers(rng)
    _write_csv(
        DATA_DIR / "customers.csv",
        ["customer_id", "region", "segment", "signup_date"],
        [[c["customer_id"], c["region"], c["segment"], c["signup_date"]] for c in customers],
    )

    orders = generate_orders(rng, customers)
    _write_csv(
        DATA_DIR / "orders.csv",
        ["order_id", "order_date", "customer_id", "product", "category",
         "units", "unit_price", "channel"],
        [
            [o["order_id"], o["order_date"], o["customer_id"], o["product"],
             o["category"], o["units"], o["unit_price"], o["channel"]]
            for o in orders
        ],
    )

    sensor = generate_sensor(rng)
    _write_csv(
        DATA_DIR / "sensor.csv",
        ["timestamp", "temp_c", "humidity_pct"],
        [[s["timestamp"], s["temp_c"], s["humidity_pct"]] for s in sensor],
    )


if __name__ == "__main__":
    main()
