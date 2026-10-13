import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from build_site_data import build, listing_age_note, supplier_chip
from data_prep import latest_snapshot_path, snapshot_date_from_path


@pytest.fixture(scope="module")
def built():
    return build("data/raw/fda_shortages_2026-09-22.csv")


def test_one_maker_is_not_described_as_one_other_supplier():
    chip = supplier_chip(1)
    assert "other supplier" not in chip
    assert chip.startswith("1 maker listed")


def test_supplier_chip_counts_makers_as_listed():
    assert supplier_chip(2) == "2 makers listed for this drug"


def test_snapshot_date_comes_from_the_file_name():
    assert snapshot_date_from_path("data/raw/fda_shortages_2026-09-22.csv") == pd.Timestamp("2026-09-22")
    assert snapshot_date_from_path("data/raw/fda_recalls_cder.csv") is None


def test_latest_snapshot_is_the_newest_dated_file(tmp_path):
    for name in ["fda_shortages_2026-09-22.csv", "fda_shortages_2026-10-05.csv", "fetch_log.csv", "fda_recalls_cder.csv"]:
        (tmp_path / name).write_text("x")
    assert latest_snapshot_path(str(tmp_path)).endswith("fda_shortages_2026-10-05.csv")


def test_site_date_is_the_data_date_not_the_build_date(built):
    _, snapshot_date = built
    assert snapshot_date == pd.Timestamp("2026-09-22")


def test_one_entry_per_drug_with_valid_scores(built):
    records, _ = built
    names = [r["generic_name"] for r in records]
    assert len(names) == len(set(names)) == 70
    assert all(0 <= r["risk_score"] <= 100 for r in records)


def test_no_entry_repeats_the_withdrawn_claims(built):
    records, _ = built
    text = " ".join(" ".join(r["reasons"]) + " " + (r["duration_context"] or "") for r in records)
    assert "260 days" not in text
    assert "other supplier" not in text
    assert "final determination" not in text


def test_listing_age_note_states_when_fda_first_posted_it():
    note = listing_age_note(pd.Timestamp("2021-11-03"), pd.Timestamp("2026-09-22"))
    assert note == "On FDA's shortage list since November 2021 (about 5 years)."
    assert listing_age_note(pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-22")).endswith("(under a year).")
    assert listing_age_note(pd.NaT, pd.Timestamp("2026-09-22")) is None
