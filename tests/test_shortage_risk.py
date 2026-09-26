import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shortage_risk import (
    CAT_FEATURES,
    MODEL_FEATURES,
    build_dataset,
    combo_purity,
    discontinuation_rate_by,
    evaluate,
    evaluate_grouped,
    grouped_auc_summary,
    make_pipeline,
    top_features,
)


@pytest.fixture(scope="module")
def df():
    return build_dataset()


def test_build_dataset_shape(df):
    assert len(df) > 1000
    # Resolved rows dropped, only the two statuses remain
    assert set(df["status"].unique()) == {"Current", "To Be Discontinued"}


def test_build_dataset_no_missing_model_features(df):
    assert df[MODEL_FEATURES].isna().sum().sum() == 0


def test_combo_column_present(df):
    assert "combo" in df.columns
    assert df["combo"].nunique() > 1


def test_make_pipeline_fits_and_predicts(df):
    pipe = make_pipeline("gbm")
    pipe.fit(df[MODEL_FEATURES], df["is_discontinuation"])
    proba = pipe.predict_proba(df[MODEL_FEATURES])[:, 1]
    assert len(proba) == len(df)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluate_returns_valid_auc(df):
    _, auc, report, _ = evaluate(df, model="gbm")
    assert 0.5 < auc <= 1.0
    assert report is not None


def test_evaluate_grouped_is_lower_than_or_near_naive_split(df):
    """The whole point of Phase 1's methodology fix: the combo-grouped
    (honest) AUC should not exceed the naive random-split AUC by much,
    and in practice comes in noticeably lower, because it removes the
    train/test combo overlap that inflates the naive split."""
    _, naive_auc, _, _ = evaluate(df, model="gbm")
    _, grouped_auc, _ = evaluate_grouped(df, model="gbm", seed=0)
    assert 0.5 < grouped_auc <= 1.0
    assert grouped_auc <= naive_auc + 0.01


def test_grouped_auc_summary_shape(df):
    summary = grouped_auc_summary(df, n_splits=3)
    assert len(summary) == 3
    assert summary.between(0.5, 1.0).all()


def test_combo_purity_accounts_for_all_rows(df):
    purity = combo_purity(df)
    assert purity["pure_combos"] + purity["mixed_combos"] == purity["unique_combos"]
    assert purity["rows_in_pure_combos"] <= purity["total_rows"]
    assert purity["rows_in_pure_combos"] > 0


def test_top_features_includes_all_model_inputs(df):
    pipe = make_pipeline("gbm")
    pipe.fit(df[MODEL_FEATURES], df["is_discontinuation"])
    feats = top_features(pipe, n=50)
    assert "suppliers_in_shortage_list" in feats.index
    assert any(name.startswith("route_") for name in feats.index)


def test_discontinuation_rate_by_respects_min_rows(df):
    table = discontinuation_rate_by(df, "therapeutic_category", min_rows=10)
    assert (table["n_rows"] >= 10).all()
    assert table["discontinuation_rate"].between(0, 1).all()
    # should be sorted descending by rate
    assert table["discontinuation_rate"].is_monotonic_decreasing
