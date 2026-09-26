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
   long shortages have been active, with censoring handled properly
4. **Recall overlay** - cross-reference CDER recall records against the
   shortage list by manufacturer to see whether recently-recalled firms are
   over-represented among current shortages

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

## Tools

Python, pandas, scikit-learn, lifelines (survival analysis), matplotlib,
pytest, Jupyter notebooks.
