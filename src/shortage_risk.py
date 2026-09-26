"""Phase 1: discontinuation risk classification.

Among drugs already on the FDA shortage list, which features predict
whether the entry is a permanent discontinuation ("To Be Discontinued")
rather than an active/temporary shortage ("Current")? This is NOT a
"will this drug go into shortage" model - we don't have a control group
of non-shortage drugs for that. It's narrower: given a drug is already
listed, what makes a permanent supply loss more likely.

A note on methodology, kept here rather than only in the README because
it changes how the numbers below should be read:

The first version of this model included `days_since_posted` (how long
the entry has sat on the shortage list) as a feature and hit AUC 1.000.
That is not a real result - `days_since_posted` is entangled with the
label almost by definition (FDA tends to resolve the "To Be
Discontinued" bookkeeping faster than open-ended "Current" shortages),
so a model "predicting" discontinuation from it is closer to reading
the FDA's own paperwork clock than finding a supply-chain risk factor.
It's dropped from the model below and reported separately as a
descriptive note, not a predictive feature.

With `days_since_posted` removed, a plain random row split still gave
AUC ~0.986 on categorical features alone (route, dosage form,
therapeutic category) plus supplier concentration. That also needed a
second look: with only ~140 unique combinations of those three
categorical fields across ~1,600 rows, a random split lets the same
combination appear in both train and test, so part of that 0.986 is
the model looking up a combination it already saw rather than
generalizing. `evaluate_grouped()` below splits by *combination*
instead of by row - a combination in the test fold was never seen in
training - which is the honest test of whether these fields carry
real signal. It comes back lower (average AUC ~0.90, see
`grouped_auc_summary()`) but still clearly informative, which is the
finding actually worth reporting: route/dosage form/therapeutic
category do carry a real, moderately strong, generalizable signal
about discontinuation risk - just not the near-perfect signal the
naive split implied.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_prep import add_supplier_concentration, load_shortages

CAT_FEATURES = ["route", "dosage_form", "therapeutic_category"]
MODEL_FEATURES = CAT_FEATURES + ["suppliers_in_shortage_list"]
# kept out of the model on purpose - see module docstring
DESCRIPTIVE_ONLY = ["days_since_posted"]


def build_dataset(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = load_shortages(path)
    df = add_supplier_concentration(df)
    # Resolved is its own small, different category (only 7 rows) - not
    # part of the Current-vs-Discontinuation comparison this phase asks,
    # so it's dropped here rather than forced into either class
    df = df[df["status"] != "Resolved"].copy()
    for c in CAT_FEATURES:
        df[c] = df[c].fillna("Unknown")
    num_cols = ["suppliers_in_shortage_list"] + DESCRIPTIVE_ONLY
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df["combo"] = df[CAT_FEATURES].astype(str).agg("|".join, axis=1)
    return df


def make_pipeline(model="gbm"):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", "passthrough", ["suppliers_in_shortage_list"]),
    ])
    clf = (GradientBoostingClassifier(random_state=0) if model == "gbm"
           else LogisticRegression(max_iter=1000, class_weight="balanced"))
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(df, model="gbm", test_size=0.25, seed=0):
    """Plain random row split. Reported for comparison only - see
    module docstring for why this overstates true generalization."""
    X = df[MODEL_FEATURES]
    y = df["is_discontinuation"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=False)
    return pipe, auc, report, (X_test, y_test, proba)


def evaluate_grouped(df, model="gbm", test_frac=0.25, seed=0):
    """Split by categorical combination, not by row: every combination
    in the test fold is unseen during training. This is the honest
    estimate of whether route/dosage form/therapeutic category
    generalize to a drug profile the model hasn't encountered before."""
    rng = np.random.RandomState(seed)
    combos = df["combo"].unique().copy()
    rng.shuffle(combos)
    test_combos = set(combos[: int(len(combos) * test_frac)])
    test_mask = df["combo"].isin(test_combos)
    train, test = df[~test_mask], df[test_mask]

    pipe = make_pipeline(model)
    pipe.fit(train[MODEL_FEATURES], train["is_discontinuation"])
    proba = pipe.predict_proba(test[MODEL_FEATURES])[:, 1]
    auc = roc_auc_score(test["is_discontinuation"], proba)
    return pipe, auc, (train, test, proba)


def grouped_auc_summary(df, model="gbm", n_splits=5):
    """Average + spread of the combo-grouped AUC across several
    random combo splits, since any single split's AUC depends a lot
    on which combinations happen to land in the test fold."""
    aucs = [evaluate_grouped(df, model=model, seed=s)[1] for s in range(n_splits)]
    return pd.Series(aucs, name="grouped_auc")


def combo_purity(df):
    """How many of the ~140 categorical combinations are 'pure'
    (every row with that combination shares the same status) versus
    'mixed'. High purity is *why* the naive random split (evaluate())
    scores so high - it's evidence for the overlap explanation, not a
    separate finding."""
    g = df.groupby(CAT_FEATURES)["status"].nunique()
    pure = (g == 1).sum()
    mixed = (g > 1).sum()
    rows_in_pure = df.set_index(CAT_FEATURES).index.isin(g[g == 1].index).sum()
    return {
        "unique_combos": len(g),
        "pure_combos": int(pure),
        "mixed_combos": int(mixed),
        "rows_in_pure_combos": int(rows_in_pure),
        "total_rows": len(df),
    }


def top_features(pipe, n=10):
    """Feature importance for the GBM model, mapped back to readable names."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    names = cat_names + ["suppliers_in_shortage_list"]
    imp = pd.Series(clf.feature_importances_, index=names)
    return imp.sort_values(ascending=False).head(n)


def discontinuation_rate_by(df, col, min_rows=10):
    """Descriptive table: discontinuation rate per category value, for
    the categories with enough rows to be meaningful."""
    ct = df.groupby(col)["is_discontinuation"].agg(["mean", "count"])
    ct = ct[ct["count"] >= min_rows].sort_values("mean", ascending=False)
    ct.columns = ["discontinuation_rate", "n_rows"]
    return ct


if __name__ == "__main__":
    df = build_dataset()
    print(f"dataset: {len(df)} rows, {df['is_discontinuation'].mean():.1%} discontinuation rate")
    print(f"(days_since_posted excluded from the model - see module docstring)")

    print("\n--- naive random row split (inflated - see docstring) ---")
    pipe, auc, report, _ = evaluate(df, model="gbm")
    print(f"GBM test AUC: {auc:.3f}")

    print("\n--- combo purity check (why the naive split is inflated) ---")
    purity = combo_purity(df)
    for k, v in purity.items():
        print(f"  {k}: {v}")

    print("\n--- honest estimate: combo-grouped split (unseen combos) ---")
    summary = grouped_auc_summary(df, n_splits=5)
    print(summary.to_string())
    print(f"mean: {summary.mean():.3f}   std: {summary.std():.3f}")

    print("\ntop features (fit on full data):")
    full_pipe = make_pipeline("gbm")
    full_pipe.fit(df[MODEL_FEATURES], df["is_discontinuation"])
    print(top_features(full_pipe))

    print("\ndiscontinuation rate by therapeutic category (>=10 rows):")
    print(discontinuation_rate_by(df, "therapeutic_category"))

    print("\ndiscontinuation rate by dosage form (>=10 rows):")
    print(discontinuation_rate_by(df, "dosage_form"))
