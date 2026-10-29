"""Phase 1: discontinuation risk classification.

Among drugs already on the FDA shortage list, which features predict
whether the listing is a permanent discontinuation ("To Be
Discontinued") vs. an active/temporary shortage ("Current")? Note this
is NOT a "will this drug go into shortage" model - there's no control
group of non-shortage drugs for that. It's narrower: given a drug is
already listed, what makes a permanent loss more likely.

Methodology note, because it changes how the numbers below should be
read. First pass included `days_since_posted` as a feature and hit AUC
1.000 - not a real result, FDA tends to close out "To Be Discontinued"
paperwork faster than open-ended "Current" shortages, so the model was
basically reading FDA's own clock, not finding a risk factor. Dropped
it from the model (kept as a descriptive note below, see
DESCRIPTIVE_ONLY).

With that removed, a plain random split still gave AUC ~0.986 on
route/dosage form/therapeutic category + supplier concentration alone.
Also inflated: with only ~140 unique category combinations across
~1,600 rows, a random split lets the same combination land in both
train and test, so the model is partly just recognizing a combo it's
already seen. `evaluate_grouped()` splits by combination instead - a
combo in the test fold was never trained on - and that's the number
that matters: mean AUC ~0.90 (see `grouped_auc_summary()`). Lower, but
still clearly informative - real, moderately strong, generalizable
signal, just not the near-perfect signal the naive split implied.
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
DESCRIPTIVE_ONLY = ["days_since_posted"]  # kept out of the model, see docstring


def build_dataset(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = load_shortages(path)
    df = add_supplier_concentration(df)
    # "Resolved" is its own small category (7 rows), not part of the
    # Current-vs-Discontinuation question this phase asks
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
    # plain random row split - inflated, see module docstring. kept
    # around for comparison against evaluate_grouped()
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
    # split by category combo, not row - every combo in the test fold
    # is unseen during training, so this is the honest number
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
    # any one combo split's AUC is noisy depending on which combos land
    # in test, so average over a few
    aucs = [evaluate_grouped(df, model=model, seed=s)[1] for s in range(n_splits)]
    return pd.Series(aucs, name="grouped_auc")


def combo_purity(df):
    # how many of the ~140 combos are "pure" (every row same status) vs
    # mixed - high purity is why the naive split scores so high
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
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    names = cat_names + ["suppliers_in_shortage_list"]
    imp = pd.Series(clf.feature_importances_, index=names)
    return imp.sort_values(ascending=False).head(n)


def discontinuation_rate_by(df, col, min_rows=10):
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
