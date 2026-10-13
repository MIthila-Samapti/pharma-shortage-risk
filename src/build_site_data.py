"""Builds the scored dataset the RxContinuity website runs on.

Scores every currently-listed ("Current") shortage with the Phase 1
model and attaches plain-English reasons. Writes docs/data/drugs.js - a
plain JS file (not JSON fetched over CORS) so the static site can just
<script src> it with no server needed, including for local file://
preview.

What the score is, and is not: FDA's feed mixes two kinds of record -
drugs short right now ("Current") and notices that a company will stop
making a product ("To Be Discontinued"). The model is trained to tell
the two apart, so the score says how closely a current shortage
resembles a discontinuation notice. It is not a forecast of how the
shortage will end: one snapshot follows no listing over time.

Re-running this script picks up the newest data/raw/fda_shortages_*.csv
snapshot (src/fetch_shortages.py adds one whenever FDA's list changes).
The date written to the site is the date of that snapshot - the day the
data was actually pulled from FDA - not the day this script ran.
"""

import json

import pandas as pd

from data_prep import latest_snapshot_path, snapshot_date_from_path
from shortage_risk import (
    MODEL_FEATURES,
    build_dataset,
    discontinuation_rate_by,
    make_pipeline,
)


def supplier_chip(n):
    """`suppliers_in_shortage_list` counts every maker listed for the
    drug, including the maker of the row itself - so 1 means no other
    maker is listed, not "one other"."""
    n = int(n)
    if n == 1:
        return "1 maker listed for this drug (no other maker on FDA's list)"
    if n <= 8:
        return f"{n} makers listed for this drug"
    return f"{n} makers listed for this drug (many suppliers)"


def reason_chips(row, dosage_rates, category_rates):
    chips = [supplier_chip(row["suppliers_in_shortage_list"])]

    dosage = row["dosage_form"]
    if dosage in dosage_rates.index:
        rate = dosage_rates.loc[dosage, "discontinuation_rate"]
        chips.append(f"{dosage} dosage form ({rate:.0%} of these listings are discontinuation notices)")
    else:
        chips.append(f"{dosage} dosage form")

    cat = row["therapeutic_category"]
    if cat in category_rates.index:
        rate = category_rates.loc[cat, "discontinuation_rate"]
        chips.append(f"{cat} category ({rate:.0%} of these listings are discontinuation notices)")
    else:
        chips.append(f"{cat} category")

    return chips


def listing_age_note(first_posted, snapshot_date):
    """One true, checkable line per drug: when FDA first posted a current
    listing for it. Replaces an earlier "typically reach a final
    determination within ~260 days" line, which measured how long
    discontinuation notices stay on FDA's list (FDA removes them after a
    year), not how long shortages last."""
    if pd.isna(first_posted):
        return None
    years = (snapshot_date - first_posted).days / 365.25
    since = first_posted.strftime("%B %Y")
    if years < 1:
        return f"On FDA's shortage list since {since} (under a year)."
    return f"On FDA's shortage list since {since} (about {years:.0f} year{'s' if round(years) != 1 else ''})."


def build(path=None):
    path = path or latest_snapshot_path()
    snapshot_date = snapshot_date_from_path(path)

    df = build_dataset(path)
    pipe = make_pipeline("gbm")
    pipe.fit(df[MODEL_FEATURES], df["is_discontinuation"])

    dosage_rates = discontinuation_rate_by(df, "dosage_form", min_rows=10)
    category_rates = discontinuation_rate_by(df, "therapeutic_category", min_rows=10)

    current = df[df["status"] == "Current"].copy()
    proba = pipe.predict_proba(current[MODEL_FEATURES])[:, 1]
    current["risk_score"] = (proba * 100).round().astype(int)
    first_posted = current.groupby("generic_name")["initial_posting_date"].min()

    records = []
    for _, row in current.iterrows():
        records.append({
            "generic_name": row["generic_name"],
            "brand_name": row["brand_name"] if isinstance(row["brand_name"], str) else None,
            "route": row["route"],
            "dosage_form": row["dosage_form"],
            "therapeutic_category": row["therapeutic_category"],
            "suppliers_in_shortage_list": int(row["suppliers_in_shortage_list"]),
            "risk_score": int(row["risk_score"]),
            "reasons": reason_chips(row, dosage_rates, category_rates),
            "duration_context": listing_age_note(first_posted.get(row["generic_name"]), snapshot_date),
        })

    # de-duplicate by generic_name, keeping the highest-scoring record (a
    # generic name can have multiple manufacturer/package rows)
    by_name = {}
    for r in records:
        key = r["generic_name"]
        if key not in by_name or r["risk_score"] > by_name[key]["risk_score"]:
            by_name[key] = r
    records = sorted(by_name.values(), key=lambda r: (-r["risk_score"], r["generic_name"]))

    return records, snapshot_date


if __name__ == "__main__":
    import os
    records, snapshot_date = build()
    as_of = snapshot_date.date().isoformat()
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/drugs.js", "w") as f:
        f.write("// Auto-generated by src/build_site_data.py - do not hand-edit\n")
        f.write("window.RX_DRUGS = ")
        json.dump(records, f, indent=1)
        f.write(";\n")
        # the date FDA's list was pulled, not the date this file was built
        f.write('window.RX_DATA_AS_OF = "' + as_of + '";\n')
    print(f"wrote docs/data/drugs.js: {len(records)} drugs, FDA data as of {as_of}")
    print(f"score range: {min(r['risk_score'] for r in records)} - {max(r['risk_score'] for r in records)}")
