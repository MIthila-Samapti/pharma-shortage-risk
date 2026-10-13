# Data

Source: openFDA (api.fda.gov), the FDA's public developer API. All data is a
US federal government work and is public domain.

- `raw/fda_shortages_2026-09-22.csv` - full snapshot of the `drug/shortages`
  endpoint (1,603 records) as of 2026-09-22. This first snapshot was taken
  from a public mirror of the openFDA feed
  (github.com/q3dresearch/wss-drug-scarcity), because the build environment
  could not reach api.fda.gov. It matches the raw openFDA capture for that
  day row for row.
- `raw/fda_shortages_YYYY-MM-DD.csv` - later snapshots, pulled straight from
  openFDA by `src/fetch_shortages.py` in the daily job. A new file is saved
  only on days when FDA's list changed. Old files are never overwritten: the
  feed keeps no history (FDA removes resolved shortages after 6 months and
  discontinuation notices after 1 year), so these files are the only record
  of what the list looked like on a given day.
- `raw/fetch_log.csv` - one line per day the feed was checked, whether or
  not it changed.
- `raw/fda_recalls_cder.csv` - CDER drug recall records (4,384 unique
  products), covering recall initiation dates from Jan 2024 through Dec 2025.

Processed/derived files live in `processed/` and are built by the scripts in
`src/`, not edited by hand.
