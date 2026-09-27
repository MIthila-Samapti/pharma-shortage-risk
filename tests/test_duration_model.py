import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from duration_model import (
    build_survival_dataset,
    elapsed_age_by,
    fit_km,
    km_by_group,
    median_survival_days,
    survival_probability_at,
)


@pytest.fixture(scope="module")
def df():
    return build_survival_dataset()


def test_build_survival_dataset_shape(df):
    assert len(df) > 1500
    assert set(df["status"].unique()) == {"Current", "To Be Discontinued", "Resolved"}


def test_duration_and_event_present_and_valid(df):
    assert df["duration"].isna().sum() == 0
    assert (df["duration"] >= 0).all()
    assert set(df["event"].unique()) <= {0, 1}


def test_event_matches_status(df):
    # event should be 1 exactly for non-Current rows
    assert (df.loc[df["status"] != "Current", "event"] == 1).all()
    assert (df.loc[df["status"] == "Current", "event"] == 0).all()


def test_duration_equals_days_since_posted(df):
    assert (df["duration"] == df["days_since_posted"]).all()


def test_supplier_tier_buckets_present(df):
    assert df["supplier_tier"].notna().any()
    assert set(df["supplier_tier"].cat.categories) == {
        "1-2 (concentrated)", "3-8 (moderate)", "9+ (many suppliers)",
    }


def test_km_curve_is_monotonically_non_increasing(df):
    km = fit_km(df)
    sf = km.survival_function_.iloc[:, 0].values
    assert np.all(np.diff(sf) <= 1e-9)


def test_km_starts_near_one_and_censoring_matches_event_rate(df):
    km = fit_km(df)
    assert km.survival_function_.iloc[0, 0] == pytest.approx(1.0)
    # long-run floor of the curve should roughly equal the fraction of
    # rows that are still censored (never reach a final determination)
    floor = km.survival_function_.iloc[-1, 0]
    assert floor == pytest.approx(1 - df["event"].mean(), abs=0.05)


def test_survival_probability_at_is_between_zero_and_one(df):
    km = fit_km(df)
    for d in [0, 30, 365, 3000]:
        p = survival_probability_at(km, d)
        assert 0.0 <= p <= 1.0


def test_median_survival_days_returns_none_when_not_reached(df):
    # the overall curve never drops below 50% in this dataset - most
    # listings are still undecided
    km = fit_km(df)
    assert median_survival_days(km) is None


def test_low_supplier_tier_reaches_determination_faster(df):
    # cross-validates the Phase 1 finding: low supplier concentration
    # should be associated with much faster resolution into a final
    # determination than high supplier concentration
    fits, test = km_by_group(df, "supplier_tier", min_group_size=30)
    low = fits["1-2 (concentrated)"]
    high = fits["9+ (many suppliers)"]
    assert survival_probability_at(low, 365) < survival_probability_at(high, 365)
    assert test.p_value < 0.01


def test_km_by_group_respects_min_group_size(df):
    fits, _ = km_by_group(df, "therapeutic_category", min_group_size=30)
    counts = df["therapeutic_category"].value_counts()
    for g in fits:
        assert counts[g] >= 30


def test_elapsed_age_by_only_uses_current_rows(df):
    table = elapsed_age_by(df, "supplier_tier").set_index("group")["n_open"]
    current_counts = df[df.status == "Current"]["supplier_tier"].value_counts()
    for group, n in table.items():
        assert n <= current_counts[group]
    assert table.notna().all()
