"""Logic tests for concentration_map.py using small, hand-built data
shaped like fetch_ndc_directory.py's output - no network access and no
dependency on a real NDC snapshot being present on disk.

These exist because I could not reach api.fda.gov from the environment
this was built in to pull live data for a sanity check before this was
reviewed. They prove the filtering/grouping/flagging logic is correct
on known inputs; they do not substitute for checking real output once
the fetch step actually runs against live FDA data.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from concentration_map import (
    at_risk_table,
    build_concentration_table,
    filter_active_finished_drugs,
)

FUTURE = "2099-01-01"
PAST = "2020-01-01"


def _row(product_ndc, generic_name, labeler_name, dosage_form="tablet",
         product_type="HUMAN PRESCRIPTION DRUG", finished="true",
         marketing_end_date=""):
    return dict(
        product_ndc=product_ndc, generic_name=generic_name, brand_name="",
        labeler_name=labeler_name, dosage_form=dosage_form, route="ORAL",
        product_type=product_type, marketing_category="ANDA",
        finished=finished, marketing_start_date="2015-01-01",
        marketing_end_date=marketing_end_date, active_ingredients="",
    )


@pytest.fixture
def ndc_df():
    rows = [
        # drug A: two labeler-name variants of the SAME company -> should
        # normalize to one -> single-source
        _row("0001-01", "acetaminophen", "Acme Pharma"),
        _row("0001-02", "acetaminophen", "Acme Pharma, Inc."),
        # drug B: two genuinely different companies -> dual-source
        _row("0002-01", "lisinopril", "Pfizer Inc."),
        _row("0002-02", "lisinopril", "Teva Pharmaceuticals"),
        # drug C: five companies -> diversified
        *[_row(f"0003-0{i}", "ibuprofen", f"Company {i}") for i in range(5)],
        # drug D: single-source, but ALSO already shortage-listed
        _row("0004-01", "cisplatin", "Solo Manufacturer", dosage_form="injection"),
        # drug E: stale listing (marketing ended in the past) -> excluded
        _row("0005-01", "oldmed", "Some Company", marketing_end_date=PAST),
        # drug E, still-active variant, so the group isn't entirely filtered out
        _row("0005-02", "oldmed", "Some Company", marketing_end_date=FUTURE),
        # drug F: unfinished bulk listing -> excluded
        _row("0006-01", "bulkchem", "Bulk Co", finished="false"),
        # drug G: veterinary, not human -> excluded
        _row("0007-01", "vetdrug", "Vet Co", product_type="VETERINARY"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def shortages_df():
    return pd.DataFrame([
        {"generic_name": "cisplatin", "status": "Current"},
        {"generic_name": "some other drug", "status": "Current"},
    ])


def test_filter_drops_unfinished_veterinary_and_stale_listings(ndc_df):
    active = filter_active_finished_drugs(ndc_df, as_of="2026-01-01")
    kept_ndcs = set(active["product_ndc"])
    assert "0006-01" not in kept_ndcs  # unfinished
    assert "0007-01" not in kept_ndcs  # veterinary
    assert "0005-01" not in kept_ndcs  # expired marketing_end_date
    assert "0005-02" in kept_ndcs      # same drug, still-active listing kept


def test_labeler_name_variants_of_the_same_company_count_as_one(ndc_df):
    active = filter_active_finished_drugs(ndc_df, as_of="2026-01-01")
    table = build_concentration_table(active)
    row = table[table["generic_name"] == "acetaminophen"].iloc[0]
    assert row["labeler_count"] == 1
    assert row["risk_tier"] == "single-source"


def test_tiers_assigned_correctly(ndc_df):
    active = filter_active_finished_drugs(ndc_df, as_of="2026-01-01")
    table = build_concentration_table(active)
    tiers = table.set_index("generic_name")["risk_tier"]
    assert tiers["lisinopril"] == "dual-source"
    assert tiers["ibuprofen"] == "diversified"
    assert tiers["cisplatin"] == "single-source"


def test_already_shortage_listed_drugs_are_excluded_from_at_risk(ndc_df, shortages_df):
    active = filter_active_finished_drugs(ndc_df, as_of="2026-01-01")
    table = build_concentration_table(active, shortages_df)
    flagged = at_risk_table(table)
    flagged_names = set(flagged["generic_name"])
    # single-source, but ALREADY shortage-listed - Phase 1's model already
    # covers this one, so it should not appear in the new leading-indicator list
    assert "cisplatin" not in flagged_names
    # single-source, NOT shortage-listed - this is exactly what the new list is for
    assert "acetaminophen" in flagged_names
    # diversified - not fragile, should never appear regardless of shortage status
    assert "ibuprofen" not in flagged_names


def test_dual_source_not_shortage_listed_is_flagged(ndc_df, shortages_df):
    active = filter_active_finished_drugs(ndc_df, as_of="2026-01-01")
    table = build_concentration_table(active, shortages_df)
    flagged = at_risk_table(table)
    assert "lisinopril" in set(flagged["generic_name"])
