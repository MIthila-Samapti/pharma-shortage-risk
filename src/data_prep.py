"""Load and clean the FDA shortage + recall snapshots."""

import pandas as pd
import numpy as np

SNAPSHOT_DATE = pd.Timestamp("2026-09-22")


def load_shortages(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = pd.read_csv(path)
    df["initial_posting_date"] = pd.to_datetime(df["initial_posting_date"], errors="coerce")
    df["update_date"] = pd.to_datetime(df["update_date"], errors="coerce")
    df["discontinued_date"] = pd.to_datetime(df["discontinued_date"], errors="coerce")

    # a handful of rows have no posting date at all - can't use them for
    # duration/timing features, but they're still fine for the classification
    # phase, so keep them and let downstream code decide what it needs
    df["days_since_posted"] = (SNAPSHOT_DATE - df["initial_posting_date"]).dt.days

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
