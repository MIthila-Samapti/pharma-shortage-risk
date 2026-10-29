"""Phase 2: shortage listing survival analysis.

Original plan: Kaplan-Meier curve for time from a shortage first
appearing on the FDA feed to it getting marked a permanent
discontinuation, using `discontinued_date - initial_posting_date` as
duration and `status == 'Current'` as censoring.

That fell apart on inspection. For 440 of 443 "To Be Discontinued"
rows, `discontinued_date` is the literal same day as
`initial_posting_date` (and `update_date` too). That's not a shortage
that sat open for months before getting flagged - it's a brand-new
feed entry posted the same day the discontinuation went public, with
no separate record of whatever shortage period came before it (if
any). `initial_posting_date` on one of these rows means "when the
discontinuation notice went up," not "when the shortage started." One
snapshot of the feed can't reconstruct that history - there's no
repeated pull tracking the same product over time here.

So the phase got reframed around what a single snapshot can actually
answer. For every row we know `days_since_posted` as of the snapshot,
and whether it's already hit a terminal status (Discontinued/Resolved)
or is still open (Current - right-censored). That supports a narrower,
honest question: of listings open at least X days, what share have
already resolved to a final determination vs. still pending? Not "how
long until an open shortage gets discontinued" - this data can't say
that - but "given how long a listing's been visible, how likely is it
to already carry a final call." Still uses censoring correctly for the
Current rows, just scoped to what's actually observable.
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from data_prep import add_supplier_concentration, load_shortages

CAT_FEATURES = ["route", "dosage_form", "therapeutic_category"]


def build_survival_dataset(path="data/raw/fda_shortages_2026-09-22.csv"):
    df = load_shortages(path)
    df = add_supplier_concentration(df)

    # event = already has a final determination as of the snapshot;
    # Current = censored, still open
    df["event"] = (df["status"] != "Current").astype(int)
    df["duration"] = df["days_since_posted"]

    before = len(df)
    df = df[df["duration"].notna() & (df["duration"] >= 0)].copy()
    dropped = before - len(df)
    if dropped:
        df.attrs["rows_dropped_no_duration"] = dropped

    for c in CAT_FEATURES:
        df[c] = df[c].fillna("Unknown")

    df["supplier_tier"] = pd.cut(
        df["suppliers_in_shortage_list"],
        bins=[0, 2, 8, 100],
        labels=["1-2 (concentrated)", "3-8 (moderate)", "9+ (many suppliers)"],
    )

    return df


def fit_km(df, label="All listings"):
    km = KaplanMeierFitter()
    km.fit(df["duration"], event_observed=df["event"], label=label)
    return km


def median_survival_days(km):
    # None if the curve never drops below 50% - most of this group is
    # still undecided as of the snapshot
    m = km.median_survival_time_
    return None if np.isinf(m) else m


def km_by_group(df, group_col, min_group_size=30, label_prefix=""):
    # separate KM curve per group with enough rows, plus a log-rank
    # test across groups to check the differences aren't just noise
    counts = df[group_col].value_counts()
    keep_groups = counts[counts >= min_group_size].index.tolist()
    sub = df[df[group_col].isin(keep_groups)]

    fits = {}
    for g in keep_groups:
        gdf = sub[sub[group_col] == g]
        fits[g] = fit_km(gdf, label=f"{label_prefix}{g} (n={len(gdf)})")

    test = None
    if len(keep_groups) >= 2:
        test = multivariate_logrank_test(sub["duration"], sub[group_col], sub["event"])
    return fits, test


def survival_probability_at(km, days):
    return float(km.survival_function_at_times(days).iloc[0])


def elapsed_age_by(df, col, min_rows=30):
    # descriptive, not censoring-based: for listings still Current, how
    # long have they already been open, by group? complements the KM
    # curve - this is today's backlog, not a time-to-event estimate
    current = df[df["status"] == "Current"]
    rows = []
    for g, gdf in current.groupby(col, observed=True):
        if len(gdf) < min_rows:
            continue
        rows.append({
            "group": g,
            "n_open": len(gdf),
            "median_days_open_so_far": gdf["duration"].median(),
            "p75_days_open_so_far": gdf["duration"].quantile(0.75),
        })
    return pd.DataFrame(rows).sort_values("n_open", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = build_survival_dataset()
    print(f"dataset: {len(df)} rows "
          f"({df['event'].sum()} already at a final determination, "
          f"{(1 - df['event'].mean()):.1%} still open/censored)")
    if df.attrs.get("rows_dropped_no_duration"):
        print(f"({df.attrs['rows_dropped_no_duration']} rows dropped - no usable duration)")

    print("\n--- overall: time listed until a final determination is reached ---")
    km_all = fit_km(df)
    med = median_survival_days(km_all)
    print(f"median: {'not reached (most listings still undecided)' if med is None else f'{med:.0f} days'}")
    for d in [30, 90, 180, 365, 730]:
        print(f"  P(still undecided after {d} days on the feed): {survival_probability_at(km_all, d):.3f}")

    print("\n--- by supplier tier (cross-validates the Phase 1 finding) ---")
    fits, test = km_by_group(df, "supplier_tier", min_group_size=30)
    for g, km in fits.items():
        m = median_survival_days(km)
        s365 = survival_probability_at(km, 365)
        print(f"  {g}: median = {'not reached' if m is None else f'{m:.0f} days'}, "
              f"P(still undecided after 1yr) = {s365:.3f}")
    if test is not None:
        print(f"  log-rank test across tiers: p = {test.p_value:.2e}")

    print("\n--- by therapeutic category (n>=30) ---")
    fits, test = km_by_group(df, "therapeutic_category", min_group_size=30)
    for g, km in fits.items():
        m = median_survival_days(km)
        print(f"  {g}: median = {'not reached' if m is None else f'{m:.0f} days'}")
    if test is not None:
        print(f"  log-rank test across categories: p = {test.p_value:.2e}")

    print("\n--- descriptive: how long has today's OPEN backlog already been running? ---")
    print("(Current listings only, elapsed time so far - not censoring-based)")
    print(elapsed_age_by(df, "supplier_tier").to_string(index=False))
    print()
    print(elapsed_age_by(df, "therapeutic_category").to_string(index=False))
