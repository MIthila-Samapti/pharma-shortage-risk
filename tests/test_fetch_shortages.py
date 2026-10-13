import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fetch_shortages as fs


def record(ndc="0409-1234-01", status="Current", name="Example Injection"):
    """One record shaped like openFDA's drug/shortages feed."""
    return {
        "generic_name": name,
        "company_name": "Example Pharma",
        "package_ndc": ndc,
        "dosage_form": "Injection",
        "therapeutic_category": ["Oncology", "Anesthesia"],
        "status": status,
        "update_type": "Revised",
        "initial_posting_date": "01/15/2024",
        "update_date": "09/01/2026",
        "presentation": "10 mL vial",
        "availability": "Limited",
        "openfda": {
            "brand_name": ["EXAMPLE"],
            "manufacturer_name": ["Example Pharma, Inc."],
            "product_ndc": ["0409-9999", "0409-1234"],
            "route": ["INTRAVENOUS"],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
        },
    }


def many(n, **kw):
    return [fs.record_to_row(record(ndc=f"0409-1234-{i:04d}", **kw)) for i in range(n)]


def test_row_has_the_17_base_columns_first():
    row = fs.record_to_row(record())
    assert list(row)[:17] == fs.BASE_COLUMNS


def test_row_picks_the_product_ndc_that_matches_the_package():
    assert fs.record_to_row(record())["product_ndc"] == "0409-1234"


def test_row_keeps_first_category_and_the_full_list():
    row = fs.record_to_row(record())
    assert row["therapeutic_category"] == "Oncology"
    assert row["therapeutic_category_all"] == "Anesthesia; Oncology"


def test_row_falls_back_to_company_name_without_openfda_block():
    rec = record()
    del rec["openfda"]
    row = fs.record_to_row(rec)
    assert row["manufacturer_name"] == "Example Pharma"
    assert row["brand_name"] == ""


def test_extra_fda_fields_are_kept_not_dropped():
    assert fs.record_to_row(record())["availability"] == "Limited"


def test_csv_is_the_same_whatever_order_fda_sends_rows_in():
    rows = many(5)
    assert fs.rows_to_csv(rows) == fs.rows_to_csv(list(reversed(rows)))


def test_first_run_saves_a_snapshot_and_a_log_line(tmp_path):
    line = fs.save_snapshot(many(600), "2026-10-05", raw_dir=str(tmp_path))
    assert line["data_changed"] == "yes"
    assert (tmp_path / "fda_shortages_2026-10-05.csv").exists()
    assert line["records"] == 600 and line["current"] == 600


def test_unchanged_list_writes_no_new_snapshot_but_still_logs(tmp_path):
    rows = many(600)
    fs.save_snapshot(rows, "2026-10-05", raw_dir=str(tmp_path))
    line = fs.save_snapshot(rows, "2026-10-06", raw_dir=str(tmp_path))
    assert line["data_changed"] == "no"
    assert line["snapshot_file"] == "fda_shortages_2026-10-05.csv"
    assert not (tmp_path / "fda_shortages_2026-10-06.csv").exists()
    with open(tmp_path / "fetch_log.csv", newline="") as f:
        assert [l["checked_on"] for l in csv.DictReader(f)] == ["2026-10-05", "2026-10-06"]


def test_changed_list_saves_a_new_snapshot_and_keeps_the_old_one(tmp_path):
    fs.save_snapshot(many(600), "2026-10-05", raw_dir=str(tmp_path))
    before = (tmp_path / "fda_shortages_2026-10-05.csv").read_text()
    line = fs.save_snapshot(many(599) + many(1, status="To Be Discontinued"), "2026-10-06", raw_dir=str(tmp_path))
    assert line["data_changed"] == "yes"
    assert (tmp_path / "fda_shortages_2026-10-06.csv").exists()
    assert (tmp_path / "fda_shortages_2026-10-05.csv").read_text() == before


def test_same_day_rerun_keeps_one_log_line(tmp_path):
    rows = many(600)
    fs.save_snapshot(rows, "2026-10-05", raw_dir=str(tmp_path))
    fs.save_snapshot(rows, "2026-10-05", raw_dir=str(tmp_path))
    with open(tmp_path / "fetch_log.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_a_short_download_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(RuntimeError):
        fs.save_snapshot(many(20), "2026-10-05", raw_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_records_missing_name_or_status_are_refused(tmp_path):
    rows = many(600)
    rows[0]["status"] = ""
    with pytest.raises(RuntimeError):
        fs.save_snapshot(rows, "2026-10-05", raw_dir=str(tmp_path))


def test_new_snapshot_loads_with_the_existing_data_prep(tmp_path):
    from data_prep import latest_snapshot_path, load_shortages

    fs.save_snapshot(many(600), "2026-10-05", raw_dir=str(tmp_path))
    path = latest_snapshot_path(str(tmp_path))
    df = load_shortages(path)
    assert len(df) == 600
    # ages are measured to the snapshot's own date, not a fixed one
    assert df["days_since_posted"].iloc[0] == 994  # 2024-01-15 to 2026-10-05
