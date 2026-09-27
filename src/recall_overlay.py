"""Phase 3: recall overlay.

The original question for this phase was whether manufacturers with a
recent FDA recall show up more often among current shortages - the
assumption being that recall trouble is a warning sign of a
manufacturer struggling to keep a product on the market.

The data says the opposite. Manufacturers with any recall on record
have a LOWER discontinuation rate than manufacturers with none (25%
vs 32%), and it's more pronounced for recalls in the last two years
(22% vs 35%, chi-square p < 0.0001). That's not what the original
premise expected, so it's worth checking why before reporting it as a
finding rather than an error.

The likely explanation: recall history is entangled with manufacturer
scale. A firm shows up in the CDER recall feed mostly by having more
products and more manufacturing volume, which also gives it more
capacity to keep supplying a struggling product rather than
discontinuing it outright. Manufacturers matched to a recall show up
on a median of 42 shortage-list entries elsewhere versus 16 for
unmatched manufacturers - a real, measurable difference in scale as
proxied by how many products under that name are currently
shortage-listed. Restricting to only manufacturers already at that
larger scale (10+ shortage-list entries) shrinks the gap (30.7% vs
23.4%) but doesn't erase it, so scale explains part of this, not all
of it - a firm under active FDA scrutiny for a recall may also be
under more pressure to keep supplying rather than quietly walking
away from a product.

Either way, the practical finding survives: a recall on a
manufacturer's record is not itself a red flag for that product's
shortage turning into a permanent discontinuation - if anything it is
mildly reassuring, and should not be read as a risk multiplier on top
of the Phase 1 model without this caveat.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from data_prep import add_supplier_concentration, load_recalls, load_shortages, normalize_firm_name, SNAPSHOT_DATE

RECENT_WINDOW_DAYS = 730  # 2 years


def build_overlay_dataset(
    shortages_path="data/raw/fda_shortages_2026-09-22.csv",
    recalls_path="data/raw/fda_recalls_cder.csv",
):
    shortages = add_supplier_concentration(load_shortages(shortages_path))
    recalls = load_recalls(recalls_path)

    shortages["mfr_norm"] = shortages["manufacturer_name"].apply(normalize_firm_name)
    recalls["firm_norm"] = recalls["firm_legal_name"].apply(normalize_firm_name)

    recent_cutoff = SNAPSHOT_DATE - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    recall_counts = recalls.groupby("firm_norm").size().rename("recall_count")
    recent_counts = (
        recalls[recalls["recall_initiation_date"] >= recent_cutoff]
        .groupby("firm_norm").size().rename("recent_recall_count")
    )
    worst_class = (
        recalls.groupby("firm_norm")["classification"]
        .apply(lambda s: s.astype(str).min())
        .rename("worst_recall_class")
    )

    shortages = shortages.merge(recall_counts, left_on="mfr_norm", right_index=True, how="left")
    shortages = shortages.merge(recent_counts, left_on="mfr_norm", right_index=True, how="left")
    shortages = shortages.merge(worst_class, left_on="mfr_norm", right_index=True, how="left")
    shortages["recall_count"] = shortages["recall_count"].fillna(0).astype(int)
    shortages["recent_recall_count"] = shortages["recent_recall_count"].fillna(0).astype(int)
    shortages["has_recall"] = shortages["recall_count"] > 0
    shortages["has_recent_recall"] = shortages["recent_recall_count"] > 0

    # scale proxy: how many shortage-list rows (any drug) carry this
    # manufacturer's name - a stand-in for how large/diversified a
    # manufacturer is, since that isn't otherwise in this data
    listing_counts = shortages.groupby("mfr_norm").size().rename("listing_count")
    shortages = shortages.merge(listing_counts, left_on="mfr_norm", right_index=True, how="left")

    return shortages


def overlap_summary(df):
    matched = df[df["has_recall"]]
    return {
        "unique_manufacturers": df["mfr_norm"].nunique(),
        "manufacturers_with_recall_match": matched["mfr_norm"].nunique(),
        "rows_total": len(df),
        "rows_with_recall_match": len(matched),
    }


def discontinuation_rate_by_recall(df, recent_only=False):
    """Compares discontinuation rate for shortage-list rows whose
    manufacturer has (vs doesn't have) a recall on record, with a
    chi-square test for whether the difference is real. Resolved rows
    are excluded, same as Phase 1's classification target."""
    col = "has_recent_recall" if recent_only else "has_recall"
    sub = df[df["status"] != "Resolved"]
    rates = sub.groupby(col)["is_discontinuation"].agg(["mean", "count"])
    ct = pd.crosstab(sub[col], sub["is_discontinuation"])
    _, p_value, _, _ = chi2_contingency(ct)
    return rates, p_value


def scale_check(df, min_listings=10):
    """Does the recall/discontinuation relationship survive when
    restricted to manufacturers already at a larger scale? If the gap
    shrinks a lot, scale is most of the explanation; if it survives,
    something besides scale is going on too."""
    large = df[df["listing_count"] >= min_listings]
    sub = large[large["status"] != "Resolved"]
    return sub.groupby(sub["recall_count"] > 0)["is_discontinuation"].agg(["mean", "count"])


def top_recalled_manufacturers(df, n=10):
    matched = df[df["has_recall"]]
    return (
        matched.drop_duplicates("mfr_norm")
        .nlargest(n, "recall_count")
        [["mfr_norm", "recall_count", "recent_recall_count", "worst_recall_class", "listing_count"]]
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    df = build_overlay_dataset()

    print("--- overlap ---")
    for k, v in overlap_summary(df).items():
        print(f"  {k}: {v}")

    print("\n--- discontinuation rate: any recall on record ---")
    rates, p = discontinuation_rate_by_recall(df, recent_only=False)
    print(rates.to_string())
    print(f"chi-square p-value: {p:.4f}")

    print("\n--- discontinuation rate: recall within last 2 years ---")
    rates_recent, p_recent = discontinuation_rate_by_recall(df, recent_only=True)
    print(rates_recent.to_string())
    print(f"chi-square p-value: {p_recent:.2e}")

    print("\n--- is this just a manufacturer-scale effect? ---")
    print("median shortage-list entries per manufacturer, by recall status:")
    print(df.groupby("has_recall")["listing_count"].median().to_string())
    print("\nsame comparison, restricted to manufacturers with 10+ listings already:")
    print(scale_check(df).to_string())

    print("\n--- top recalled manufacturers on the shortage list ---")
    print(top_recalled_manufacturers(df).to_string(index=False))
