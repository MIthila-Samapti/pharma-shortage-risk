"""Load + clean the FDA shortage and recall snapshots."""

import os
import re

import pandas as pd
import numpy as np

SNAPSHOT_DATE = pd.Timestamp("2026-09-22")
RAW_DIR = "data/raw"
_SNAPSHOT_NAME = re.compile(r"^fda_shortages_(\d{4}-\d{2}-\d{2})\.csv$")


def snapshot_date_from_path(path):
    m = _SNAPSHOT_NAME.match(os.path.basename(str(path)))
    return pd.Timestamp(m.group(1)) if m else None


def latest_snapshot_path(raw_dir=RAW_DIR):
    # daily job drops a new file only when the list actually changes
    dated = sorted(n for n in os.listdir(raw_dir) if _SNAPSHOT_NAME.match(n))
    if not dated:
        raise FileNotFoundError(f"no fda_shortages_YYYY-MM-DD.csv snapshot in {raw_dir}")
    return os.path.join(raw_dir, dated[-1])


def load_shortages(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = pd.read_csv(path)
    # ages measured relative to THIS snapshot's own date, not a hardcoded
    # one, otherwise every later snapshot reports stale ages
    snapshot_date = snapshot_date_from_path(path) or SNAPSHOT_DATE
    df["initial_posting_date"] = pd.to_datetime(df["initial_posting_date"], errors="coerce")
    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    df["discontinued_date"] = pd.to_datetime(df["discontinued_date"], errors="coerce")

    df["days_since_posted"] = (snapshot_date - df["initial_posting_date"]).dt.days

    df["is_discontinuation"] = (df["status"] == "To Be Discontinued").astype(int)
    df["is_resolved"] = (df["status"] == "Resolved").astype(int)

    return df


def load_recalls(path="data/raw/fda_recalls_cder.csv"):
    df = pd.read_csv(path)
    df["recall_initiation_date"] = pd.to_datetime(df["recall_initiation_date"], errors="coerce")
    df["center_classification_date"] = pd.to_datetime(df["center_classification_date"], errors="coerce")
    df["termination_date"] = pd.to_datetime(df["termination_date"], errors="coerce")
    return df


def add_supplier_concentration(df):
    # distinct manufacturers shortage-listed under the same generic name.
    # not true sole-source status (would need the full NDC directory) but
    # close enough to be a useful fragility signal
    counts = (
        df.groupby("generic_name")["manufacturer_name"]
        .nunique()
        .rename("suppliers_in_shortage_list")
    )
    return df.merge(counts, on="generic_name", how="left")


def normalize_firm_name(name):
    # manufacturer names are never spelled the same across the two feeds
    # ("Hospira, Inc." vs "Hospira, Inc., a Pfizer Company"), so strip the
    # common suffixes before matching
    if not isinstance(name, str):
        return ""
    name = name.lower()
    for suffix in [", inc.", ", inc", " inc.", " inc", ", llc", " llc",
                   ", ltd.", " ltd.", ", l.p.", " corporation", " corp.",
                   " corp", " co.", " company", " pharmaceuticals",
                   " pharmaceutical", " laboratories", " labs"]:
        name = name.replace(suffix, "")
    return name.split(",")[0].strip()


if __name__ == "__main__":
    shortages = load_shortages()
    recalls = load_recalls()
    shortages = add_supplier_concentration(shortages)

    print(f"shortages: {len(shortages)} rows")
    print(shortages["status"].value_counts())
    print(f"\nrecalls: {len(recalls)} rows, "
          f"{recalls['recall_initiation_date'].min().date()} to "
          f"{recalls['recall_initiation_date'].max().date()}")
    print(f"\nmedian days_since_posted (Current): "
          f"{shortages[shortages.status=='Current']['days_since_posted'].median():.0f}")
    print(f"median days_since_posted (To Be Discontinued): "
          f"{shortages[shortages.status=='To Be Discontinued']['days_since_posted'].median():.0f}")
    print(f"\ntop 10 generic names by supplier count in shortage list:")
    print(shortages.drop_duplicates("generic_name")
          .nlargest(10, "suppliers_in_shortage_list")
          [["generic_name", "suppliers_in_shortage_list"]])
