# Data

Source: openFDA (api.fda.gov), the FDA's public developer API. All data is a
US federal government work and is public domain.

- `raw/fda_shortages_2026-09-22.csv` - full snapshot of the `drug/shortages`
  endpoint (1,603 records) as of 2026-09-22. This endpoint only lists what is
  short *today* - it does not keep a history of shortages that have already
  ended, so a single snapshot is what's available; captured directly from
  the live API.
- `raw/fda_recalls_cder.csv` - CDER drug recall records (4,384 unique
  products), covering recall initiation dates from Jan 2024 through Dec 2025.

Processed/derived files live in `processed/` and are built by the scripts in
`src/`, not edited by hand.
