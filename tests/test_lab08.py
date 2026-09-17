"""Tests for Lab 08 - features, labels and evaluation.

The metric tests use tiny hand-computable inputs. That is deliberate: when a
test on four rows fails you can work out the right answer on paper in ten
seconds, which you cannot do when it fails on four hundred.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from labs.lab08_modeling import AS_OF, build_customer_features, evaluate_binary_classifier


# ---------------------------------------------------------------------------
# WORKED EXAMPLE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def features(clean_orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    return build_customer_features(clean_orders, customers)


@pytest.mark.demo
def test_one_row_per_customer_with_history(features: pd.DataFrame) -> None:
    assert features["customer_id"].is_unique
    assert len(features) > 300


@pytest.mark.demo
def test_labels_are_binary(features: pd.DataFrame) -> None:
    assert set(features["churned"].unique()) <= {0, 1}


@pytest.mark.demo
def test_the_classes_are_imbalanced(features: pd.DataFrame) -> None:
    """About 64% of customers churn under this cutoff and horizon.

    So the do-nothing model -- "everybody churns", no features, no fitting --
    scores roughly 64% accuracy. Quote that number without saying what it is
    measured against and it sounds like a working model. It is the trap that
    precision and recall exist to expose, and it is why the exercise makes
    you compute the full confusion matrix rather than one headline number.

    Note that the rate is a consequence of the choices in AS_OF and
    HORIZON_DAYS, not a fact about the customers. Shorten the horizon and
    "churn" gets more common, because fewer people happen to order in a
    narrower window. The label is something you define, not something you
    discover -- change those constants and watch this number move.
    """
    churn_rate = features["churned"].mean()
    assert 0.55 < churn_rate < 0.75


@pytest.mark.demo
def test_features_are_non_negative(features: pd.DataFrame) -> None:
    assert (features["recency_days"] >= 0).all()
    assert (features["frequency"] >= 1).all()
    assert (features["monetary"] > 0).all()


@pytest.mark.demo
def test_no_leakage_from_the_future(
    clean_orders: pd.DataFrame, features: pd.DataFrame
) -> None:
    """The features must be computable on the as_of date and no later.

    Recency is measured from the last order BEFORE the cutoff, so the most
    recent possible value is 0 days and every customer's last pre-cutoff
    order must actually pre-date the cutoff.
    """
    past = clean_orders[clean_orders["order_date"] < AS_OF]
    last_seen = past.groupby("customer_id")["order_date"].max()
    expected_recency = (AS_OF - last_seen).dt.days

    merged = features.set_index("customer_id")["recency_days"]
    pd.testing.assert_series_equal(
        merged.sort_index(), expected_recency.sort_index(),
        check_names=False, check_dtype=False,
    )


@pytest.mark.demo
def test_baseline_score_is_a_probability(features: pd.DataFrame) -> None:
    assert features["baseline_score"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# YOUR EXERCISE
# ---------------------------------------------------------------------------


@pytest.mark.exercise
def test_perfect_classifier() -> None:
    result = evaluate_binary_classifier([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert result["tp"] == 2
    assert result["tn"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["roc_auc"] == 1.0


@pytest.mark.exercise
def test_perfectly_wrong_classifier() -> None:
    result = evaluate_binary_classifier([1, 0], [0.1, 0.9])
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (0, 1, 0, 1)
    assert result["accuracy"] == 0.0
    assert result["roc_auc"] == 0.0


@pytest.mark.exercise
def test_threshold_is_inclusive() -> None:
    """A score exactly equal to the threshold counts as positive.

    This is arbitrary -- some libraries choose >, some >=. What is not
    arbitrary is that your implementation and its documentation agree.
    """
    result = evaluate_binary_classifier([1, 1], [0.5, 0.5], threshold=0.5)
    assert result["tp"] == 2
    assert result["fn"] == 0


@pytest.mark.exercise
def test_a_realistic_mixed_case() -> None:
    """y_true = [1, 1, 0, 0], scores = [0.9, 0.4, 0.6, 0.1], threshold 0.5.

    Predictions are [1, 0, 1, 0], so tp=1, fn=1, fp=1, tn=1.

    AUC by hand -- positives score {0.9, 0.4}, negatives {0.6, 0.1}:
        0.9 > 0.6  win        0.4 < 0.6  loss
        0.9 > 0.1  win        0.4 > 0.1  win
    Three wins out of four pairs, so AUC = 0.75.
    """
    result = evaluate_binary_classifier([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["roc_auc"] == 0.75


@pytest.mark.exercise
def test_f1_is_the_harmonic_mean_not_the_average() -> None:
    """precision 1.0, recall 0.5. The arithmetic mean is 0.75; F1 is 0.6667.

    F1 punishes imbalance between the two, which is the entire reason to use
    it. If you get 0.75 here you averaged them.
    """
    result = evaluate_binary_classifier([1, 1, 0, 0], [0.9, 0.2, 0.3, 0.1])
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(0.6667, abs=0.0001)


@pytest.mark.exercise
def test_ties_score_half() -> None:
    """Identical scores cannot separate the classes: AUC is exactly 0.5.

    This is the test that catches a ranking implementation that does not use
    average ranks for ties -- the most common bug in a hand-rolled AUC.
    """
    result = evaluate_binary_classifier([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
    assert result["roc_auc"] == 0.5


@pytest.mark.exercise
def test_partial_ties() -> None:
    """One tied pair out of two: 1 win + 0.5 for the tie, over 2 pairs."""
    result = evaluate_binary_classifier([1, 0, 0], [0.8, 0.8, 0.1])
    assert result["roc_auc"] == 0.75


@pytest.mark.exercise
def test_nothing_flagged_gives_zero_precision() -> None:
    """Dividing by tp + fp == 0 must not raise, and must not be 1.0."""
    result = evaluate_binary_classifier([1, 1, 0], [0.1, 0.2, 0.05])
    assert result["tp"] == 0 and result["fp"] == 0
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


@pytest.mark.exercise
def test_auc_is_nan_when_a_class_is_absent() -> None:
    """With no positives there is no positive/negative pair to rank, so AUC
    is genuinely undefined. NaN says that; 0.5 would be a quiet lie."""
    result = evaluate_binary_classifier([0, 0, 0], [0.1, 0.5, 0.9])
    assert math.isnan(result["roc_auc"])
    assert result["recall"] == 0.0


@pytest.mark.exercise
def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        evaluate_binary_classifier([1, 0, 1], [0.5, 0.5])


@pytest.mark.exercise
def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_binary_classifier([], [])


@pytest.mark.exercise
def test_accepts_series_and_arrays(features: pd.DataFrame) -> None:
    """Real calls pass pandas Series, not lists. np.asarray handles all three."""
    result = evaluate_binary_classifier(
        features["churned"], features["baseline_score"], threshold=0.5
    )
    assert result["n"] == len(features)
    assert result["positive_rate"] == pytest.approx(features["churned"].mean(), abs=0.001)


@pytest.mark.exercise
def test_the_baseline_actually_beats_random(features: pd.DataFrame) -> None:
    """The payoff. "Days since last order" alone should rank churners well
    above random guessing -- if your AUC comes out near 0.5, or below it,
    check whether you inverted the comparison somewhere."""
    result = evaluate_binary_classifier(features["churned"], features["baseline_score"])
    assert result["roc_auc"] > 0.75


@pytest.mark.exercise
def test_output_is_json_serialisable(features: pd.DataFrame) -> None:
    """numpy scalars do not survive json.dumps, and the web UI shows these."""
    import json

    result = evaluate_binary_classifier(features["churned"], features["baseline_score"])
    json.dumps(result)
