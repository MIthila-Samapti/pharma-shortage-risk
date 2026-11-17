"""Tests for fetch_ndc_directory.py's parsing logic, using a record
shaped like openFDA's real documented /drug/ndc.json schema
(https://open.fda.gov/apis/drug/ndc/example-api-returns/). No network
access needed or used.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fetch_ndc_directory import record_to_row, check_rows

SAMPLE_RECORD = {
    "product_ndc": "0002-1433",
    "generic_name": "insulin lispro",
    "brand_name": "Humalog",
    "labeler_name": "Eli Lilly and Company",
    "dosage_form": "INJECTION, SOLUTION",
    "route": ["SUBCUTANEOUS"],
    "product_type": "HUMAN PRESCRIPTION DRUG",
    "marketing_category": "NDA",
    "finished": True,
    "marketing_start_date": "19960101",
    "active_ingredients": [{"name": "INSULIN LISPRO", "strength": "100 [iU]/mL"}],
}


def test_record_to_row_pulls_expected_fields():
    row = record_to_row(SAMPLE_RECORD)
    assert row["product_ndc"] == "0002-1433"
    assert row["generic_name"] == "insulin lispro"
    assert row["labeler_name"] == "Eli Lilly and Company"
    assert row["route"] == "SUBCUTANEOUS"
    assert row["finished"] == "True"
    assert row["active_ingredients"] == "INSULIN LISPRO 100 [iU]/mL"


def test_record_to_row_handles_missing_marketing_end_date():
    # most listings have no marketing_end_date at all - must not KeyError
    row = record_to_row(SAMPLE_RECORD)
    assert row["marketing_end_date"] == ""


def test_record_to_row_sorts_multiple_ingredients_consistently():
    rec = dict(SAMPLE_RECORD, active_ingredients=[
        {"name": "B INGREDIENT", "strength": "5 MG"},
        {"name": "A INGREDIENT", "strength": "10 MG"},
    ])
    row = record_to_row(rec)
    assert row["active_ingredients"] == "A INGREDIENT 10 MG; B INGREDIENT 5 MG"


def test_check_rows_rejects_a_pull_with_no_product_ndc():
    import pytest
    bad = [dict(record_to_row(SAMPLE_RECORD), product_ndc="")] * 60000
    with pytest.raises(RuntimeError):
        check_rows(bad)


def test_check_rows_rejects_a_pull_that_is_too_small():
    import pytest
    small = [record_to_row(SAMPLE_RECORD)] * 10
    with pytest.raises(RuntimeError):
        check_rows(small)
