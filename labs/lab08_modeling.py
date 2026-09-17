"""Lab 08 - Features, labels, and honest evaluation.

CONCEPT
-------
This is the bridge from analytics into data science, and it rests on two
ideas that matter more than any algorithm you will ever pick.

1. A SUPERVISED PROBLEM IS A TABLE PLUS A CUTOFF.
   "Predict churn" is not a well-posed question until you say: as of WHEN,
   using WHAT, to predict WHICH following period. Here:

       as_of = 2025-11-01, horizon = 60 days

       features  <- everything that happened STRICTLY BEFORE as_of
       label     <- did they order in the 60 days AFTER as_of?

   Every feature must be computable on the as_of date, using only what was
   knowable then. Break that rule and you get LEAKAGE: a model that scores
   beautifully in your notebook and is worthless in production, because it
   was quietly reading the answer. The single most common way beginners leak
   is by computing an aggregate over the whole dataset and then splitting.
   Do the cutoff first, always.

2. ACCURACY IS USUALLY A LIE.
   Under the cutoff below, about 64% of customers count as churned. So a
   model that predicts "churned" for everyone -- no features, no fitting,
   one line -- scores about 64% accuracy. Report that number on its own and
   it sounds like a working model. It has learned nothing.

   This is class imbalance, and it is the normal case rather than an edge
   case: fraud, disease screening, equipment failure and ad clicks are all
   far more skewed than this. The first question to ask of any accuracy
   figure is "what does the constant predictor score?".

   Precision and recall separate the two ways of being wrong:
       precision = of those I FLAGGED, how many really churned?
       recall    = of those who really churned, how many did I CATCH?
   You trade one against the other by moving the threshold. AUC summarises
   that trade-off across every possible threshold at once -- which is why it
   does not depend on the threshold you happened to choose.

WHY NO scikit-learn
-------------------
You are going to write the metrics yourself. `roc_auc_score(y, p)` is one
line and teaches you nothing about what it measures; implementing it once
means you will never again misread a model report, and you will understand
why AUC is exactly "the probability that a randomly chosen positive scores
higher than a randomly chosen negative". See requirements.txt for the full
reasoning (and a Windows DLL story).

WORKED EXAMPLE  ->  build_customer_features()
    Construct a leakage-free feature table with a time cutoff, plus a
    deliberately simple baseline score to evaluate.

YOUR TURN       ->  evaluate_binary_classifier()
    Implement the confusion matrix, precision, recall, F1 and AUC from
    scratch, including the edge cases that make library implementations
    longer than you would expect.

RUN THE TESTS
    pytest tests/test_lab08.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The cutoff and horizon are module constants rather than magic numbers buried
# in a function, because they are modelling decisions. Anyone reading this
# file should be able to see, in one place, what question is being asked.
AS_OF = pd.Timestamp("2025-11-01")
HORIZON_DAYS = 60


# ===========================================================================
# WORKED EXAMPLE
# ===========================================================================


def build_customer_features(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    as_of: pd.Timestamp = AS_OF,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Build a leakage-free churn feature table with labels.

    Args:
        orders: Cleaned orders from labs.load_clean_orders().
        customers: From labs.load_customers().
        as_of: The cutoff. Features use orders strictly before this.
        horizon_days: How far after the cutoff to look for the label.

    Returns:
        One row per customer who ordered at least once before `as_of`:
            customer_id      str
            recency_days     int    -- days from last order to as_of
            frequency        int    -- orders before as_of
            monetary         float  -- revenue before as_of
            avg_order_value  float
            tenure_days      int    -- days from signup to as_of
            churned          int    -- 1 if NO order in the horizon, else 0
            baseline_score   float  -- a naive risk score in [0, 1]

        Customers with no pre-cutoff history are excluded: there is nothing
        to build features from, so they are a different problem (cold start).

    Example:
        >>> from labs import load_clean_orders, load_customers
        >>> f = build_customer_features(load_clean_orders(), load_customers())
        >>> set(f["churned"].unique()) <= {0, 1}
        True
    """
    horizon_end = as_of + pd.Timedelta(days=horizon_days)

    # --- Split by time FIRST. Everything downstream depends on this. -------
    past = orders[orders["order_date"] < as_of]
    future = orders[
        (orders["order_date"] >= as_of) & (orders["order_date"] < horizon_end)
    ]

    # --- Features: computed only from `past` ------------------------------
    features = past.groupby("customer_id").agg(
        frequency=("order_id", "count"),
        monetary=("revenue", "sum"),
        last_order=("order_date", "max"),
    )

    # Recency as an integer number of days. .dt.days on a Timedelta column
    # extracts whole days; without it you would be handing a timedelta64 to a
    # model, which is not a number it can use.
    features["recency_days"] = (as_of - features["last_order"]).dt.days
    features["avg_order_value"] = (features["monetary"] / features["frequency"]).round(2)
    features["monetary"] = features["monetary"].round(2)
    features = features.drop(columns="last_order").reset_index()

    # Tenure comes from the customers table, so it needs a join. signup_date
    # is knowable at as_of, so using it is legitimate -- contrast that with
    # anything derived from `future`, which is not.
    features = features.merge(
        customers[["customer_id", "signup_date"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    features["tenure_days"] = (as_of - features["signup_date"]).dt.days
    features = features.drop(columns="signup_date")

    # --- Label: computed only from `future` -------------------------------
    # A set of the customers seen in the horizon; ~isin() gives "not seen",
    # which is the definition of churned here.
    active_later = set(future["customer_id"].unique())
    features["churned"] = (~features["customer_id"].isin(active_later)).astype("int64")

    # --- A deliberately naive baseline ------------------------------------
    # "The longer since your last order, the more likely you have gone."
    # No fitting, no training data, one feature. It exists so you have
    # something real to evaluate in the exercise -- and so that when you
    # later train an actual model, you know what score it has to beat.
    # A model that cannot beat a one-line heuristic is not worth shipping.
    max_recency = features["recency_days"].max()
    features["baseline_score"] = (features["recency_days"] / max_recency).round(4)

    ordered_columns = [
        "customer_id", "recency_days", "frequency", "monetary",
        "avg_order_value", "tenure_days", "churned", "baseline_score",
    ]
    return features[ordered_columns].reset_index(drop=True)


# ===========================================================================
# YOUR TURN
# ===========================================================================


def evaluate_binary_classifier(
    y_true: np.ndarray | pd.Series | list,
    y_score: np.ndarray | pd.Series | list,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a binary classifier from scratch.

    Args:
        y_true: The actual labels, 0 or 1. Length n.
        y_score: Predicted scores in [0, 1] -- NOT hard 0/1 predictions.
            Higher means "more likely to be class 1". Length n.
        threshold: Scores >= threshold are predicted positive.
            Note >=, not >. The tests depend on it.

    Returns:
        A dict with exactly these keys:

            tp, fp, tn, fn   int    -- the confusion matrix
            accuracy         float  -- (tp + tn) / n
            precision        float  -- tp / (tp + fp)
            recall           float  -- tp / (tp + fn)
            f1               float  -- harmonic mean of precision and recall
            roc_auc          float  -- see below
            n                int    -- number of observations
            positive_rate    float  -- fraction of y_true that is 1

        Every float rounded to 4dp. (Round at the very end, not between
        steps -- rounding precision and recall before computing F1 from them
        will give you a subtly different number.)

    Raises:
        ValueError: if y_true and y_score have different lengths, or if
            either is empty.

    EDGE CASES -- these are most of the work, and they are why the real
    implementations are longer than the formulas suggest:

        * precision when tp + fp == 0 (you flagged nothing): return 0.0.
        * recall when tp + fn == 0 (there were no positives): return 0.0.
        * f1 when precision + recall == 0: return 0.0, do not divide by zero.
        * roc_auc when either class is absent entirely: return float("nan").
          AUC asks "does a positive outscore a negative?" -- with no
          positives or no negatives there is no such pair, and the question
          is genuinely undefined. NaN is the honest answer; 0.0 and 0.5 are
          both lies. (Use math.isnan or np.isnan to check for it -- remember
          that nan != nan, so `x == float("nan")` is always False.)

    COMPUTING AUC WITHOUT A LIBRARY

        The definition: AUC is the probability that a randomly chosen
        positive scores higher than a randomly chosen negative, counting
        ties as half.

        The naive route is a double loop over every (positive, negative)
        pair. That is O(n_pos * n_neg) and it is a perfectly good first
        implementation -- write it first if it helps you believe the result.

        The fast route is the Mann-Whitney U identity. Rank every score from
        1 to n, giving tied scores their AVERAGE rank (this is what handles
        the "ties count as half" rule, and getting it wrong is the classic
        bug here). Then:

            auc = (sum_of_ranks_of_positives - n_pos * (n_pos + 1) / 2)
                  / (n_pos * n_neg)

        pandas will do the ranking for you, average-ties and all:

            ranks = pd.Series(y_score).rank(method="average")

        Implement it whichever way you like -- the tests only check the
        answer. But if you write the double loop first and then the ranked
        version, assert they agree on random data. Convincing yourself that
        two very different-looking algorithms compute the same quantity is
        one of the genuinely useful habits in this field.

    Hints:
        * np.asarray(y_true) accepts lists, Series and arrays alike, so you
          can normalise all three inputs in one line.
        * With boolean arrays, `(pred & actual).sum()` counts tp directly --
          no loop needed. Vectorised counting is the pandas/numpy way.
        * int() and float() around the numpy scalars, so the dict is
          JSON-serialisable and the web UI can display it.

    When it passes, run this to see the baseline evaluated:
        python scripts/run_lab08_baseline.py

    Check yourself:
        pytest tests/test_lab08.py -v
    """
    raise NotImplementedError("Implement evaluate_binary_classifier -- see the docstring.")
