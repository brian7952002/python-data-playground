# Going further

The eight labs cover the mechanics. This file is about what to do once they pass —
the moves that turn "I can use pandas" into "I can do the job".

---

## 1. Check your metrics against scikit-learn

The first thing to do after lab 08 passes. On a machine that allows scipy:

```bash
pip install scikit-learn
```

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from labs import load_clean_orders, load_customers
from labs.lab08_modeling import build_customer_features, evaluate_binary_classifier

features = build_customer_features(load_clean_orders(), load_customers())
y, scores = features["churned"], features["baseline_score"]
predictions = (scores >= 0.5).astype(int)

mine = evaluate_binary_classifier(y, scores)
print(mine["roc_auc"], roc_auc_score(y, scores))
print(mine["precision"], precision_score(y, predictions, zero_division=0))
print(mine["recall"], recall_score(y, predictions))
print(mine["f1"], f1_score(y, predictions))
```

They should agree to four decimal places. If they do, you now know exactly what
those functions do — which is a different thing from knowing how to call them.

Then look at the edge cases. `precision_score` has a `zero_division` parameter
because sklearn's authors hit the same "you flagged nothing" problem you did, and
chose to make the answer configurable rather than pick one. Reading how a mature
library handles the awkward cases you just handled yourself is one of the fastest
ways to improve your judgement about API design.

---

## 2. Beat the baseline

The recency heuristic scores around 0.86 AUC. Beat it properly:

1. **Add features.** `frequency`, `monetary`, `avg_order_value`, `tenure_days` are
   already in the table. Ratios often beat raw values — orders per month of
   tenure says more than either number alone.
2. **Split before you fit.** `build_customer_features()` gives you one table; you
   need to hold out part of it. For churn, a *time-based* split is more honest
   than a random one: fit on customers whose history ends earlier, test on later
   ones. A random split quietly assumes you will always be predicting the past.
3. **Fit something simple.** Logistic regression first, always. It trains in
   milliseconds, its coefficients tell you which features matter and in which
   direction, and it is a genuinely strong baseline. Reach for gradient boosting
   only when you can say what logistic regression got wrong.
4. **Compare on AUC, not accuracy.** You know why now.

If your model cannot beat one line of arithmetic, that is a finding, not a
failure. Report it.

---

## 3. Write a lab of your own

The highest-value thing in this repo is the *format*, not the content. Pick
something you do not understand yet, and write the lab for it:

- **Window functions** — `rank()`, `cumsum()`, `shift()`, `pct_change()`. If you
  know SQL window functions, this is the pandas mapping, and it is where a lot of
  real analytics lives.
- **Reshaping** — `pivot_table()`, `melt()`, `stack()`/`unstack()`. Long versus
  wide is the shape decision underneath most "I can't get this chart to work".
- **Categoricals** — `astype("category")`, and what it does to memory and to
  groupby ordering.
- **`.apply()` and why to avoid it** — write the loop-based version and the
  vectorised version of the same transform, and time them both. Feel the
  difference rather than being told about it.

Writing the *tests* is where the learning is concentrated. You cannot write a test
for behaviour you have not fully pinned down, which is exactly why doing it forces
you to.

---

## 4. Point it at real data

Synthetic data is honest about being synthetic: no encoding surprises, no
timezone confusion, no column that is 97% null for reasons nobody remembers.

Swap in something real and watch which assumptions break:

- Your own bank or card exports (CSV, genuinely messy, and you know the ground truth)
- [data.gov](https://data.gov) or your city's open data portal
- The GitHub API — paginated, rate-limited, and free; a perfect target for lab 06's
  retry logic against a server that really does return 429

The techniques do not change when the source does. The *cleaning* gets much harder,
which is why lab 01 comes first.

---

## 5. The things this repo deliberately left out

Worth learning next, roughly in order of payoff:

| Topic | Why it matters |
| --- | --- |
| **SQL from Python** | `pd.read_sql()`, SQLAlchemy connections. Most real data lives in a database, not a CSV — and you already know SQL |
| **Plotting** | matplotlib and seaborn. A chart is often the deliverable |
| **Jupyter** | The notebook workflow, and its failure modes — out-of-order execution and hidden state |
| **`pyproject.toml`** | Packaging a project properly instead of the `sys.path` insert in `tests/conftest.py` |
| **Type checking** | `mypy` or `pyright` over these labs. The type hints are already there |
| **Polars / DuckDB** | Where a lot of the field is heading for larger-than-memory work |

---

## 6. Continuous integration

`.github/workflows/tests.yml` runs the worked-example tests on every push, so the
repo stays green while your exercises are still in progress.

Once you have finished the labs, change the workflow to run the whole suite:

```yaml
- run: pytest          # instead of: pytest -m demo
```

Having your own repository go red when you break something — and knowing exactly
which commit did it — is the single habit that most changes how it feels to work
on code over months rather than days.
