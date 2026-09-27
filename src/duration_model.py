"""Phase 2: shortage listing survival analysis.

The original plan for this phase was a straightforward one: fit a
Kaplan-Meier curve for "time from a shortage first appearing on the FDA
feed to it being marked a permanent discontinuation," using
`discontinued_date - initial_posting_date` as the duration and
`status == 'Current'` as right-censoring.

That doesn't hold up. Checking it against the raw data: for 440 of the
443 "To Be Discontinued" rows, `discontinued_date` is literally the
same day as `initial_posting_date` (and `update_date` too). That's not
a shortage that sat open for months and then got a discontinuation
flag added - it's a brand-new feed entry, posted the same day the
discontinuation became public, with no separate record of whatever
shortage period (if any) came before it. `initial_posting_date` for a
"To Be Discontinued" row means "when this discontinuation notice went
up," not "when this drug's shortage began." With only one snapshot of
the feed (no historical pulls to track the same product across time),
there's no way to reconstruct a true onset-to-discontinuation duration
for these rows - the underlying feed doesn't carry that history, and
this project doesn't have the many-months-apart repeated pulls that
would.

So the phase is reframed around what's actually observable from a
single snapshot: **current status data**. For every row, we know
exactly how long it's been listed as of the snapshot
(`days_since_posted`, computed identically for every status), and we
know whether it has, by the snapshot date, already reached a terminal
determination (`To Be Discontinued` or `Resolved`) or is still
undecided (`Current` - right-censored, since we don't know if or when
it will end). That supports an honest, narrower question:

  Of listings that have been on the feed for at least X days, what
  share have already been resolved into a final determination, versus
  how many are still open and undecided?

That's not "how long until an open shortage gets discontinued" (this
data can't answer that). It's "given how long a listing has been
visible, how likely is it to already carry a final determination" -
still useful, still uses right-censoring correctly for the `Current`
rows, just honestly scoped.
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

    # event = this listing already carries a final determination as of
    # the snapshot (discontinued or resolved); Current = censored, still
    # undecided as of the snapshot
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
    """Median time until a listing reaches a final determination -
    None if the curve never drops below 50% (most listings in this
    group are still undecided as of the snapshot)."""
    m = km.median_survival_time_
    return None if np.isinf(m) else m


def km_by_group(df, group_col, min_group_size=30, label_prefix=""):
    """Fit a separate KM curve per group with enough rows, plus a
    multivariate log-rank test across groups (are the "time to
    determination" distributions actually different, or could this be
    noise?)."""
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
    """P(still undecided / no final determination yet, at `days` since
    posting) - the KM curve's height at that point."""
    return float(km.survival_function_at_times(days).iloc[0])


def elapsed_age_by(df, col, min_rows=30):
    """Descriptive only: for listings still 'Current' (open, undecided)
    as of the snapshot, how long have they already been sitting on the
    feed, by group? Unlike the KM curve above, this isn't about
    censoring - every one of these rows genuinely has this exact
    elapsed time so far. It answers a different, complementary
    question: among today's open backlog, which categories/tiers skew
    toward long-running, chronic listings."""
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
