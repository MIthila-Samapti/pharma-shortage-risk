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
   (route, dosage form, therapeutic category, supplier concentration, days
   posted) predict permanent discontinuation vs. an active/temporary
   shortage
3. **Shortage duration (survival analysis)** - Kaplan-Meier estimate of how
   long shortages have been active, with censoring handled properly
4. **Recall overlay** - cross-reference CDER recall records against the
   shortage list by manufacturer to see whether recently-recalled firms are
   over-represented among current shortages

## Tools

Python, pandas, scikit-learn, lifelines (survival analysis), matplotlib,
pytest, Jupyter notebooks.
