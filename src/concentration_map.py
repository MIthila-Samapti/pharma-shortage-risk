"""Flags drugs with few active US manufacturers, before they go short.

`data_prep.add_supplier_concentration()` already found that supplier
concentration is the strongest signal in this project (Phase 1 and
Phase 2 both land on it independently) - but only among drugs that are
*already* shortage-listed. This module answers the harder, more useful
question the project's own README and site Limitations section both
call out as missing: of drugs NOT currently in shortage, which ones
would be most exposed if one maker stopped?

Grouping key: (generic_name, dosage_form). Not strength - different
strengths of the same drug in the same form are normally made by the
same set of companies, so splitting by strength would undercount real
redundancy. Not route alone - an oral shortage isn't eased by an IV
version of the same drug, so dosage_form is kept as part of the key
alongside generic_name, matching how a pharmacist would actually judge
substitutability.

Honest caveats, worth repeating wherever this output is shown:

- "Labeler" is not the same as "manufacturer." The NDC directory lists
  whoever is legally responsible for the listing - that can be a
  repackager or private-label distributor reselling someone else's
  product, not a second independent production line. A low labeler
  count is a real signal; a labeler count of exactly 1 is the most
  reliable tier (least likely to hide a repackager pair), 2 is next
  most reliable, and anything beyond isn't filtered for this at all.
- NDC listings aren't verified as currently-marketed. `finished=true`
  and an empty/future `marketing_end_date` filter out some stale
  listings, but FDA doesn't guarantee every remaining row represents a
  product actually being made and sold today.
- This is a structural fragility score, not a shortage forecast. It
  says "few alternative sources exist," not "this will go short." A
  drug can sit at labeler_count=1 for years with no problem, same as
  the shortage list's own "Current" listings can stay open for years
  (see Phase 2's duration findings).
"""

import gzip
import os

import pandas as pd

from data_prep import load_shortages, normalize_firm_name

NDC_PATH = "data/raw/fda_ndc_directory_latest.csv.gz"

# excludes bulk/unfinished listings, animal drugs, and a few marketing
# categories that aren't finished retail drug products
KEEP_PRODUCT_TYPES = {"HUMAN PRESCRIPTION DRUG", "HUMAN OTC DRUG"}


def load_ndc_directory(path=NDC_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run src/fetch_ndc_directory.py first"
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    df["marketing_end_date"] = pd.to_datetime(df["marketing_end_date"], errors="coerce")
    df["marketing_start_date"] = pd.to_datetime(df["marketing_start_date"], errors="coerce")
    return df


def filter_active_finished_drugs(df, as_of=None):
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz=None)
    df = df.copy()
    # defensive: works whether or not the caller already ran this through
    # load_ndc_directory's date parsing - a no-op if it's already datetime
    df["marketing_end_date"] = pd.to_datetime(df["marketing_end_date"], errors="coerce")
    keep = (
        (df["finished"].astype(str).str.lower() == "true")
        & (df["product_type"].isin(KEEP_PRODUCT_TYPES))
        & (df["generic_name"].notna())
        & (df["generic_name"].astype(str).str.strip() != "")
        # no end date, or an end date that hasn't happened yet
        & (df["marketing_end_date"].isna() | (df["marketing_end_date"] >= as_of))
    )
    return df.loc[keep].copy()


def build_concentration_table(ndc_df, shortages_df=None):
    """One row per (generic_name, dosage_form), with the number of
    distinct normalized labelers and whether it's currently
    shortage-listed."""
    df = ndc_df.copy()
    df["generic_name"] = df["generic_name"].astype(str).str.strip().str.lower()
    df["dosage_form"] = df["dosage_form"].astype(str).str.strip().str.lower()
    df["labeler_norm"] = df["labeler_name"].map(normalize_firm_name)
    df = df[df["labeler_norm"] != ""]

    grouped = (
        df.groupby(["generic_name", "dosage_form"])
        .agg(
            labeler_count=("labeler_norm", "nunique"),
            labelers=("labeler_norm", lambda s: "; ".join(sorted(set(s)))),
            route=("route", "first"),
            product_type=("product_type", "first"),
        )
        .reset_index()
    )

    if shortages_df is not None:
        shortage_names = set(
            shortages_df.loc[shortages_df["status"] == "Current", "generic_name"]
            .astype(str).str.strip().str.lower()
        )
        grouped["currently_shortage_listed"] = grouped["generic_name"].isin(shortage_names)
    else:
        grouped["currently_shortage_listed"] = False

    def tier(n):
        if n == 1:
            return "single-source"
        if n == 2:
            return "dual-source"
        return "diversified"

    grouped["risk_tier"] = grouped["labeler_count"].map(tier)
    return grouped


def at_risk_table(concentration_df):
    """The new, leading-indicator list this module exists to produce:
    single/dual-source drugs that are NOT already on the shortage
    list (those are already covered by the Phase 1 model)."""
    at_risk = concentration_df[
        (concentration_df["risk_tier"].isin(["single-source", "dual-source"]))
        & (~concentration_df["currently_shortage_listed"])
    ].copy()
    at_risk = at_risk.sort_values(["labeler_count", "generic_name"])
    return at_risk


def build(ndc_path=NDC_PATH, shortages_path=None):
    ndc_df = load_ndc_directory(ndc_path)
    active = filter_active_finished_drugs(ndc_df)
    shortages_df = load_shortages(shortages_path) if shortages_path else None
    concentration = build_concentration_table(active, shortages_df)
    flagged = at_risk_table(concentration)
    return concentration, flagged


if __name__ == "__main__":
    from data_prep import latest_snapshot_path

    concentration, flagged = build(shortages_path=latest_snapshot_path())
    print(f"{len(concentration)} (generic_name, dosage_form) profiles scored")
    print(concentration["risk_tier"].value_counts())
    print(f"\n{len(flagged)} flagged as at-risk and not already shortage-listed")
    print(flagged.head(15)[["generic_name", "dosage_form", "labeler_count", "labelers"]])
