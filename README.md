# Pharma Shortage Risk Intelligence

A methodology-focused project on FDA drug shortages: which currently-listed
shortages are more likely to be permanent discontinuations rather than
temporary supply gaps, how long an active shortage has typically been
going on, and whether manufacturers with recent FDA recalls show up more
often in the shortage list.

Data comes from the FDA's public [openFDA](https://open.fda.gov/) API
(drug shortages + CDER recalls) - see `data/README.md` for exact sources
and honest notes on what each dataset can and can't tell you.

This is a companion project to
[retail-demand-forecasting](https://github.com/MIthila-Samapti/retail-demand-forecasting)
and
[supply-chain-risk-intelligence](https://github.com/MIthila-Samapti/supply-chain-risk-intelligence).
Those two are retail/CPG focused; this one applies the same disruption-risk
thinking to pharmaceutical supply chains, on public healthcare data instead
of retail sales data.

## Correction, October 2026

A week after launch I re-checked my own results. Three things below were
wrong or overstated, and I would rather say so here than leave them.

- **A plain count does the model's job.** The Phase 1 model scores about
  0.93 AUC on unseen drug profiles. Counting how many makers are listed for
  the drug, with no model at all, scores about 0.92 on the same test (0.935
  on all rows). The reason is simple: a discontinuation notice is filed by
  one company for its own product, so "one maker listed" is nearly always a
  notice (331 of 348 such rows).
- **The model separates two kinds of FDA record. It does not predict how a
  shortage ends.** Of 238 drugs in the snapshot, 180 have a discontinuation
  notice, 70 are in current shortage, and only 12 are on both lists. Nothing
  here follows a listing over time.
- **The "~260 days" duration finding is an artifact.** FDA removes
  discontinuation notices from its list after one year (and resolved
  shortages after six months), so those rows are young by construction: 98%
  are under a year old, median 182 days. Current shortages with one or two
  makers listed have a median age of 1,735 days.

The site also said "updated daily" while the daily job only re-scored the
2026-09-22 file. The job now downloads FDA's list each day
(`src/fetch_shortages.py`), keeps a dated copy whenever it changes, and the
site shows the date the data was pulled.

What still holds: current shortages are chronic (median age 4.8 years, 99%
older than one year), and shortages and discontinuations are close to two
separate sets of drugs. The results sections below are kept as written, so
read them with this correction in mind.

## Why this dataset is different from the retail ones

The retail projects had transaction-level history to backtest against. The
FDA shortage list is different: it only shows what's short *today* - once a
shortage resolves, it drops off the feed with no record of how long it
lasted. That shapes the whole approach here:

- Most "durations" in this project are how long a shortage **has already
  been ongoing** as of the snapshot date, not how long it ultimately lasted
  - most of the data is right-censored, and that's handled explicitly with
    survival analysis rather than pretending it's a clean regression target.
- There's no "control group" of drugs that are NOT in shortage, since
  pulling the full NDC directory (2M+ products) was out of scope for this
  project. So this isn't a "will drug X go into shortage" predictor - it's
  a "given a drug is already shortage-listed, what predicts whether it
  becomes a permanent discontinuation vs. a resolvable gap" model. That's a
  narrower, honest claim.

## Planned phases

1. **Data prep** - clean the shortage and recall snapshots, basic EDA (done)
2. **Discontinuation risk classification** - which shortage-list features
   predict permanent discontinuation vs. an active/temporary shortage (done,
   see results below)
3. **Shortage duration (survival analysis)** - Kaplan-Meier estimate of how
   long shortages have been active, with censoring handled properly (done,
   see results below)
4. **Recall overlay** - cross-reference CDER recall records against the
   shortage list by manufacturer to see whether recently-recalled firms are
   over-represented among current shortages (done, see results below)

## Results so far (Phase 1: discontinuation risk)

Full notebook: [`notebooks/01_discontinuation_risk.ipynb`](notebooks/01_discontinuation_risk.ipynb).

A first pass at this model included how long an entry had sat on the
shortage list as a feature and scored a perfect AUC of 1.000 - a sign of a
methodology problem, not a good result. That feature is entangled with the
label almost by definition (FDA's own bookkeeping tends to close out "to be
discontinued" entries faster than open-ended "current" ones), so it was
dropped from the model.

With it removed, a plain random train/test split still scored AUC 0.986 on
categorical features alone. That also needed a second look: there are only
~140 unique combinations of route, dosage form, and therapeutic category
across ~1,600 rows, so a random split lets the same combination land in
both train and test - part of that score is the model recognizing a
combination it already saw, not generalizing to a new drug profile.

Splitting by *combination* instead - so every combination in the test fold
is unseen during training - gives a more honest estimate: **mean AUC ~0.93
(std ~0.06)** across five different random combo splits. Lower than the
naive number, but still a real, moderately strong signal.

The main driver turns out to be **supplier concentration**: how many other
manufacturers of the same generic drug are also currently listed as short.
A drug with few or no alternative suppliers also on the shortage list is
structurally more fragile - one company's decision to discontinue is
effectively the market's supply for that product. Therapeutic category and
dosage form also show a clear, sensible pattern (anesthesia and psychiatry
drugs are rarely permanently discontinued once short, ~5%; transplant,
antiviral, and renal drugs are far more likely to be, though on small
sample sizes; injectable dosage forms are less likely to be discontinued
than tablets or solutions) but carry much less weight in the model than
supplier concentration does.

## Results so far (Phase 2: shortage duration)

Full notebook: [`notebooks/02_shortage_duration.ipynb`](notebooks/02_shortage_duration.ipynb).

The original plan was a Kaplan-Meier curve for time from a shortage's onset
to being marked a permanent discontinuation. Checking that against the raw
data first: for 440 of 443 "To Be Discontinued" rows, `discontinued_date`
is the exact same day as `initial_posting_date`. That means a
discontinuation notice is posted as a brand-new feed entry the day it
becomes public, not an update to a `Current` entry that had been open for
a while - so with only one snapshot of the feed, there's no way to recover
how long any shortage period before it actually lasted. That approach was
dropped rather than reported.

Reframed instead around **current status data**: every row has a real,
non-degenerate `days_since_posted`, and either has already reached a final
determination (discontinued or resolved) by the snapshot date, or is still
`Current` - undecided, right-censored. That supports an honest question:
of listings visible for at least X days, what share have already reached a
final determination versus are still open?

**Supplier concentration** - already the dominant driver in Phase 1's
classification model - shows up again here, independently: shortages with
only 1-2 other suppliers reach a final determination in a median of
**~260 days**, and by one year only ~26% are still undecided. Shortages
with 3+ other suppliers barely move off "still undecided" in the same
window - the large majority just stay `Current`, open-ended, for years
(log-rank test across tiers, p << 0.001). Two independent methods -
classification and survival analysis - point at the same driver.

A separate, complementary finding, not about censoring at all: among
listings still marked `Current` today, the median one has **already been
on the feed for 1,400-2,000+ days - roughly 4 to 6 years** - regardless of
supplier tier or drug category. Most active FDA drug shortages are
chronic, not transient.

## Results so far (Phase 3: recall overlay)

Full notebook: [`notebooks/03_recall_overlay.ipynb`](notebooks/03_recall_overlay.ipynb).

The starting question was whether manufacturers with FDA recalls on record
show up more often among shortages that turn into permanent
discontinuations - the assumption being that recall trouble is a warning
sign. Matching manufacturer names between the two datasets (55 of 145
shortage-list manufacturers, covering 961 of 1,603 rows, match at least one
CDER recall record), the data says the opposite: manufacturers with any
recall on record have a **lower** discontinuation rate (25% vs 32%), wider
still for a recall in the last two years (22% vs 35%) - both statistically
significant (chi-square p < 0.01 and p < 0.0001).

That contradicted the starting assumption, so it got checked rather than
reported as-is. Manufacturers matched to a recall show up on a median of 42
shortage-list entries elsewhere, versus 16 for unmatched manufacturers -
recall history is entangled with manufacturer scale, using "how many other
shortage listings carry this name" as a stand-in for company size (this
data has no direct revenue or headcount figure). Restricting to
manufacturers already at that larger scale shrinks the gap (31% vs 23%) but
doesn't erase it, so scale explains part of this, not all of it.

Practical takeaway: a manufacturer's recall history should not be treated
as a risk-increasing signal on top of the Phase 1 model - the data doesn't
support that direction.

## Tools

Python, pandas, scikit-learn, lifelines (survival analysis), scipy,
matplotlib, pytest, Jupyter notebooks.
