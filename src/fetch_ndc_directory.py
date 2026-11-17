"""Pulls the full FDA NDC (National Drug Code) Directory from openFDA.

This is the dataset `data_prep.add_supplier_concentration()` explicitly
flags as missing: "not true sole-source status (would need the full NDC
directory) but close enough to be a useful fragility signal." That
comment, and the site's own Limitations section ("it does not predict
which drugs go into shortage next"), describe exactly the gap this
script and `concentration_map.py` fill.

Unlike `fetch_shortages.py`, this does NOT keep a dated history of every
snapshot. The shortage list is ~1,600 rows and small enough to keep
forever; the NDC directory is at minimum one order of magnitude bigger
(openFDA's own pagination cap of 26,000 "skip" hits is routinely
exceeded by this endpoint, which is why search_after is required below -
see MIN_RECORDS), so keeping a full dated copy every day would bloat the
repo for comparatively little benefit - we mainly need today's count of
active labelers per drug, not a day-by-day history of the raw directory.
So this script overwrites a single file each run:
`data/raw/fda_ndc_directory_latest.csv.gz`.

openFDA's `/drug/ndc.json` endpoint has far more rows than
`/drug/shortages.json`, and openFDA's own docs say `skip`-based paging
only reaches 26,000 hits in a result set - past that you have to use
`search_after` ("deep paging") instead. See
https://open.fda.gov/apis/paging/. That's the paging strategy used
here; `fetch_shortages.py`'s simpler skip-based paging would silently
truncate this dataset.

Standard library only (matches the rest of this project's fetch
scripts), except gzip which is stdlib too.
"""

import csv
import gzip
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API_URL = "https://api.fda.gov/drug/ndc.json"
PAGE_SIZE = 1000  # the most openFDA returns per request
# openFDA doesn't publish a total record count for this endpoint (unlike
# /drug/shortages.json, which is small enough to count by hand), so this
# is a conservative floor, not a documented total - well above the
# 26,000-hit point where skip-paging would have failed anyway, which is
# the actual thing worth checking for. Revisit once a real pull confirms
# the real count.
MIN_RECORDS = 50000
RETRIES = 4
SLEEP_BETWEEN_PAGES = 0.3  # be a polite citizen of a free public API

RAW_DIR = "data/raw"
OUT_NAME = "fda_ndc_directory_latest.csv.gz"
LOG_NAME = "ndc_fetch_log.csv"
LOG_COLUMNS = ["checked_on", "records", "finished_records", "fetch_seconds"]

# Only fields concentration_map.py actually needs. openFDA's basic API
# doesn't support a server-side field projection, so we keep the
# response small by filtering query-side (finished:true) and trimming
# columns client-side instead.
COLUMNS = [
    "product_ndc", "generic_name", "brand_name", "labeler_name",
    "dosage_form", "route", "product_type", "marketing_category",
    "finished", "marketing_start_date", "marketing_end_date",
    "active_ingredients",
]


def _first(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _joined(value):
    if isinstance(value, list):
        return "; ".join(sorted(str(v) for v in value))
    return "" if value is None else str(value)


def _ingredients(value):
    # "name strength; name strength" - sorted so the same drug always
    # produces the same string regardless of FDA's field ordering
    if not isinstance(value, list):
        return ""
    parts = []
    for ing in value:
        if not isinstance(ing, dict):
            continue
        name = str(ing.get("name", "")).strip()
        strength = str(ing.get("strength", "")).strip()
        if name:
            parts.append(f"{name} {strength}".strip())
    return "; ".join(sorted(parts))


def record_to_row(rec):
    return {
        "product_ndc": _first(rec.get("product_ndc")),
        "generic_name": _first(rec.get("generic_name")),
        "brand_name": _first(rec.get("brand_name")),
        "labeler_name": _first(rec.get("labeler_name")),
        "dosage_form": _first(rec.get("dosage_form")),
        "route": _joined(rec.get("route")),
        "product_type": _first(rec.get("product_type")),
        "marketing_category": _first(rec.get("marketing_category")),
        "finished": _first(rec.get("finished")),
        "marketing_start_date": _first(rec.get("marketing_start_date")),
        "marketing_end_date": _first(rec.get("marketing_end_date")),
        "active_ingredients": _ingredients(rec.get("active_ingredients")),
    }


def _get_json(url):
    last_error = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rxcontinuity-concentration-map"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # a 404 from openFDA means "no more results" for search_after
            # paging, not an error - treat it like an empty page
            if e.code == 404:
                return {"results": [], "meta": {}}
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last_error = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"openFDA request failed after {RETRIES} tries: {last_error}")


def fetch_all_records(search="finished:true", page_size=PAGE_SIZE, max_pages=None):
    """Deep-pages through openFDA using search_after, per openFDA's own
    pagination docs (skip-based paging only reaches 26,000 hits and
    would silently truncate this dataset)."""
    records = []
    search_after = None
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    page = 0
    while True:
        params = {"limit": page_size, "sort": "product_ndc:asc", "search": search}
        if search_after:
            params["search_after"] = search_after
        if api_key:
            params["api_key"] = api_key
        data = _get_json(API_URL + "?" + urllib.parse.urlencode(params))
        batch = data.get("results") or []
        if not batch:
            break
        records.extend(batch)
        search_after = batch[-1].get("product_ndc")
        page += 1
        if len(batch) < page_size:
            break  # last page
        if max_pages is not None and page >= max_pages:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return records


def rows_to_csv_gz(rows, path):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(buf.getvalue())


def check_rows(rows):
    if len(rows) < MIN_RECORDS:
        raise RuntimeError(
            f"only {len(rows)} NDC records (expected well over {MIN_RECORDS}) - not saving"
        )
    missing = [r for r in rows if not r["product_ndc"]]
    if missing:
        raise RuntimeError(f"{len(missing)} records have no product_ndc - not saving")


def main():
    today = datetime.now(timezone.utc).date().isoformat()
    started = time.time()
    records = fetch_all_records()
    rows = [record_to_row(r) for r in records]
    check_rows(rows)

    os.makedirs(RAW_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DIR, OUT_NAME)
    rows_to_csv_gz(rows, out_path)

    finished_count = sum(1 for r in rows if r["finished"].lower() == "true")
    elapsed = round(time.time() - started, 1)

    log_path = os.path.join(RAW_DIR, LOG_NAME)
    line = {
        "checked_on": today, "records": len(rows),
        "finished_records": finished_count, "fetch_seconds": elapsed,
    }
    lines = []
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            lines = [l for l in csv.DictReader(f) if l.get("checked_on") != today]
    lines.append(line)
    lines.sort(key=lambda l: l["checked_on"])
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)

    print(f"pulled {len(rows)} NDC records ({finished_count} finished) in {elapsed}s -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # fail loudly so the job shows a red X
        print(f"fetch_ndc_directory failed: {e}", file=sys.stderr)
        sys.exit(1)
