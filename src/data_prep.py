"""Load and clean the FDA shortage + recall snapshots."""

import os
import re

import pandas as pd
import numpy as np

SNAPSHOT_DATE = pd.Timestamp("2026-09-22")
RAW_DIR = "data/raw"
_SNAPSHOT_NAME = re.compile(r"^fda_shortages_(\d{4}-\d{2}-\d{2})\.csv$")


def snapshot_date_from_path(path):
    """The date in a snapshot's file name (fda_shortages_YYYY-MM-DD.csv),
    or None if the name doesn't carry one."""
    m = _SNAPSHOT_NAME.match(os.path.basename(str(path)))
    return pd.Timestamp(m.group(1)) if m else None


def latest_snapshot_path(raw_dir=RAW_DIR):
    """Path of the newest dated shortage snapshot in data/raw. The daily
    job adds a new file whenever FDA's list changes, so "newest" is what
    the site should show."""
    dated = sorted(n for n in os.listdir(raw_dir) if _SNAPSHOT_NAME.match(n))
    if not dated:
        raise FileNotFoundError(f"no fda_shortages_YYYY-MM-DD.csv snapshot in {raw_dir}")
    return os.path.join(raw_dir, dated[-1])


def load_shortages(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = pd.read_csv(path)
    # listing ages are measured to the day the snapshot was taken, not to
    # a fixed date - otherwise every newer snapshot would report ages as
    # of 2026-09-22
    snapshot_date = snapshot_date_from_path(path) or SNAPSHOT_DATE
    df["initial_posting_date"] = pd.to_datetime(df["initial_posting_date"], errors="coerce")
    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    df["discontinued_date"] = pd.to_datetime(df["discontinued_date"], errors="coerce")

    # a handful of rows have no posting date at all - can't use them for
    # duration/timing features, but they're still fine for the classification
    # phase, so keep them and let downstream code decide what it needs
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
    """How many distinct manufacturers show up in the shortage list for the
    same generic name. Not the same as true market sole-source status (we'd
    need the full NDC directory for that) - this only counts suppliers that
    are ALSO currently shortage-listed for that generic, which is a narrower
    but still useful fragility signal: multiple suppliers of the same drug
    struggling at once vs. just one."""
    counts = (
        df.groupby("generic_name")["manufacturer_name"]
        .nunique()
        .rename("suppliers_in_shortage_list")
    )
    return df.merge(counts, on="generic_name", how="left")


def normalize_firm_name(name):
    """Loose normalization for matching manufacturer names across the two
    datasets - they're never spelled identically (e.g. 'Hospira, Inc.' vs
    'Hospira, Inc., a Pfizer Company')."""
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
