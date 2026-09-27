import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recall_overlay import (
    build_overlay_dataset,
    discontinuation_rate_by_recall,
    overlap_summary,
    scale_check,
    top_recalled_manufacturers,
)


@pytest.fixture(scope="module")
def df():
    return build_overlay_dataset()


def test_build_overlay_dataset_shape(df):
    assert len(df) > 1500
    assert {"recall_count", "recent_recall_count", "has_recall", "has_recent_recall", "listing_count"} <= set(df.columns)


def test_recall_count_non_negative_and_consistent_with_flag(df):
    assert (df["recall_count"] >= 0).all()
    assert (df["recent_recall_count"] >= 0).all()
    assert ((df["recall_count"] > 0) == df["has_recall"]).all()
    assert ((df["recent_recall_count"] > 0) == df["has_recent_recall"]).all()


def test_recent_recall_count_never_exceeds_total(df):
    assert (df["recent_recall_count"] <= df["recall_count"]).all()


def test_overlap_summary_sane(df):
    summary = overlap_summary(df)
    assert summary["manufacturers_with_recall_match"] <= summary["unique_manufacturers"]
    assert summary["rows_with_recall_match"] <= summary["rows_total"]
    assert summary["manufacturers_with_recall_match"] > 0  # real data should have some overlap


def test_discontinuation_rate_by_recall_returns_both_groups(df):
    rates, p_value = discontinuation_rate_by_recall(df, recent_only=False)
    assert set(rates.index) == {True, False}
    assert 0.0 <= p_value <= 1.0


def test_recall_history_associated_with_lower_not_higher_discontinuation(df):
    # the actual, checked finding: recall history correlates with LOWER
    # discontinuation rate in this data, likely a manufacturer-scale
    # effect - this test guards against silently regressing to the
    # naive (wrong) assumption if the data or logic changes
    rates, p_value = discontinuation_rate_by_recall(df, recent_only=False)
    assert rates.loc[True, "mean"] < rates.loc[False, "mean"]
    assert p_value < 0.05


def test_listing_count_higher_for_recalled_manufacturers(df):
    # supports the scale-effect explanation
    median_by_recall = df.groupby("has_recall")["listing_count"].median()
    assert median_by_recall.loc[True] > median_by_recall.loc[False]


def test_scale_check_respects_min_listings(df):
    table = scale_check(df, min_listings=10)
    assert len(table) in (1, 2)  # may collapse to one group in small samples


def test_top_recalled_manufacturers_sorted_descending(df):
    top = top_recalled_manufacturers(df, n=10)
    assert list(top["recall_count"]) == sorted(top["recall_count"], reverse=True)
    assert (top["recall_count"] > 0).all()
