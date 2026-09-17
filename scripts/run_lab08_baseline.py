"""Evaluate the lab 08 baseline churn model with your own metrics.

Run this once you have finished evaluate_binary_classifier():

    python scripts/run_lab08_baseline.py

It builds the feature table, scores every customer with the one-line
recency heuristic, and prints the evaluation at several thresholds so you can
watch precision and recall trade against each other.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labs import load_clean_orders, load_customers  # noqa: E402
from labs.lab08_modeling import (  # noqa: E402
    AS_OF,
    HORIZON_DAYS,
    build_customer_features,
    evaluate_binary_classifier,
)


def main() -> None:
    features = build_customer_features(load_clean_orders(), load_customers())

    print(f"Cutoff {AS_OF.date()}, horizon {HORIZON_DAYS} days")
    print(f"{len(features)} customers with pre-cutoff history")
    print(f"Churn rate: {features['churned'].mean():.1%}\n")

    try:
        evaluate_binary_classifier([1, 0], [0.9, 0.1])
    except NotImplementedError:
        print("evaluate_binary_classifier() is not implemented yet.")
        print("Finish lab 08's exercise, then run this again.")
        raise SystemExit(1)

    # The do-nothing model: predict "churned" for everybody. Its accuracy is
    # the number every real model has to beat, and it is usually higher than
    # people expect.
    everyone = evaluate_binary_classifier(
        features["churned"], [1.0] * len(features), threshold=0.5
    )
    print("Constant predictor -- 'everybody churns', no features, no fitting:")
    print(f"  accuracy {everyone['accuracy']:.3f}   recall {everyone['recall']:.3f}"
          f"   precision {everyone['precision']:.3f}")
    print("  It catches every churner, and is useless: it also flags everyone else.\n")

    print("Recency baseline, across thresholds:")
    print(f"  {'thresh':>7}  {'acc':>6}  {'prec':>6}  {'recall':>7}  {'f1':>6}"
          f"  {'flagged':>8}")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        result = evaluate_binary_classifier(
            features["churned"], features["baseline_score"], threshold=threshold
        )
        flagged = result["tp"] + result["fp"]
        print(f"  {threshold:>7.1f}  {result['accuracy']:>6.3f}  {result['precision']:>6.3f}"
              f"  {result['recall']:>7.3f}  {result['f1']:>6.3f}  {flagged:>8}")

    auc = evaluate_binary_classifier(features["churned"], features["baseline_score"])["roc_auc"]
    print(f"\n  AUC: {auc:.4f}  (threshold-independent -- it is the same for every row above)")

    print(
        "\nRead the table top to bottom and watch the trade: a low threshold flags "
        "nearly everyone, so recall is high and precision is poor. Raise it and "
        "that reverses. There is no 'correct' row -- the right threshold depends on "
        "what a false positive costs you versus a false negative.\n"
        "\nNext: beat this baseline. Add frequency and monetary as features, fit "
        "something, and compare AUC. If you cannot beat a one-line heuristic, the "
        "model is not worth shipping -- see docs/going-further.md."
    )


if __name__ == "__main__":
    main()
