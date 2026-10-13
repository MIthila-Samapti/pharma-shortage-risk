"""Download today's FDA drug shortage list and keep a dated copy.

Before this script existed, the daily job only re-scored the one
snapshot from 2026-09-22 and re-stamped the date, so the site said
"updated daily" while showing the same data every day. This script is
the missing step: it pulls the live `drug/shortages` feed from openFDA
and saves it under `data/raw/`.

Two things get written:

- `data/raw/fda_shortages_YYYY-MM-DD.csv` - a new dated snapshot, but
  only on days when FDA's list actually differs from the newest
  snapshot already saved. Old snapshots are never overwritten or
  deleted. The feed itself keeps no history (a shortage that ends just
  drops off), so these files are the only record of what the list
  looked like on a given day.
- `data/raw/fetch_log.csv` - one line per day the feed was checked,
  whether or not it changed. On a day with no new snapshot file, the
  list was identical to the snapshot named in that line.

If the download fails or looks wrong (too few records, count doesn't
match what FDA says the total is), the script exits with an error and
writes nothing, so a bad day can never replace good data or make the
site claim a refresh that didn't happen.

Standard library only, so it adds nothing to requirements.txt.
"""

import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API_URL = "https://api.fda.gov/drug/shortages.json"
PAGE_SIZE = 1000  # the most openFDA returns per request
MIN_RECORDS = 500  # the list has ~1,600 rows; far fewer means a bad pull
RETRIES = 4

RAW_DIR = "data/raw"
LOG_NAME = "fetch_log.csv"
SNAPSHOT_NAME = re.compile(r"^fda_shortages_(\d{4}-\d{2}-\d{2})\.csv$")

# The same 17 columns, in the same order, as the 2026-09-22 snapshot, so
# everything that already reads a snapshot keeps working unchanged.
BASE_COLUMNS = [
    "generic_name", "brand_name", "manufacturer_name", "company_name",
    "product_ndc", "package_ndc", "route", "dosage_form",
    "therapeutic_category", "product_type", "status", "update_type",
    "initial_posting_date", "update_date", "discontinued_date",
    "related_info", "presentation",
]

LOG_COLUMNS = [
    "checked_on", "records", "current", "to_be_discontinued", "resolved",
    "data_changed", "snapshot_file", "fda_last_updated",
]


def _first(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _joined(value):
    """All values of a list field, sorted so the order FDA happens to
    send them in doesn't make two identical days look different."""
    if isinstance(value, list):
        return "; ".join(sorted(str(v) for v in value))
    return "" if value is None else str(value)


def record_to_row(rec):
    """Flatten one openFDA shortage record into a CSV row.

    The 17 base columns follow the 2026-09-22 snapshot exactly:
    brand/manufacturer/route/product type come from the record's
    `openfda` block (first value), manufacturer falls back to the
    company name when FDA attached no `openfda` block, and
    `therapeutic_category` is the first category listed.

    FDA lists several therapeutic categories for many drugs and does not
    keep them in a consistent order, so "the first one" is partly
    arbitrary. The full list is kept in `therapeutic_category_all`.

    Every other field FDA sends (availability, contact_info,
    change_date, resolved_note, and anything added later) is kept as an
    extra column instead of being thrown away.
    """
    ofda = rec.get("openfda") or {}
    package = _first(rec.get("package_ndc"))
    products = ofda.get("product_ndc") or []
    product = next((p for p in products if package.startswith(str(p) + "-")),
                   _first(products))

    row = {
        "generic_name": _first(rec.get("generic_name")),
        "brand_name": _first(ofda.get("brand_name")),
        "manufacturer_name": _first(ofda.get("manufacturer_name"))
                             or _first(rec.get("company_name")),
        "company_name": _first(rec.get("company_name")),
        "product_ndc": product,
        "package_ndc": package,
        "route": _first(ofda.get("route")),
        "dosage_form": _first(rec.get("dosage_form")),
        "therapeutic_category": _first(rec.get("therapeutic_category")),
        "product_type": _first(ofda.get("product_type")),
        "status": _first(rec.get("status")),
        "update_type": _first(rec.get("update_type")),
        "initial_posting_date": _first(rec.get("initial_posting_date")),
        "update_date": _first(rec.get("update_date")),
        "discontinued_date": _first(rec.get("discontinued_date")),
        "related_info": _first(rec.get("related_info")),
        "presentation": _first(rec.get("presentation")),
    }

    row["therapeutic_category_all"] = _joined(rec.get("therapeutic_category"))
    row["route_all"] = _joined(ofda.get("route"))
    row["application_number"] = _joined(ofda.get("application_number"))
    row["substance_name"] = _joined(ofda.get("substance_name"))
    for key, value in rec.items():
        if key == "openfda" or key in row:
            continue
        row[key] = _joined(value)
    return row


def rows_to_csv(rows):
    """One CSV string for a list of rows: base columns first, then any
    extra columns in alphabetical order, rows in a fixed sort order.
    The fixed order is what makes "did the list change since the last
    snapshot" a plain text comparison."""
    extras = sorted({k for r in rows for k in r} - set(BASE_COLUMNS))
    columns = BASE_COLUMNS + extras
    ordered = sorted(rows, key=lambda r: [r.get(c, "") for c in
                                          ["package_ndc", "generic_name", "company_name"] + columns])
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for r in ordered:
        writer.writerow([r.get(c, "") for c in columns])
    return out.getvalue()


def _get_json(url):
    last_error = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rxcontinuity-daily-refresh"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # 4xx other than "slow down" won't fix itself by retrying
            if e.code not in (429, 500, 502, 503, 504):
                raise
            last_error = e
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last_error = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"openFDA request failed after {RETRIES} tries: {last_error}")


def fetch_all_records():
    """Every record on the shortage feed, plus FDA's own last_updated date.

    openFDA caps a request at 1,000 records, so this pages through with
    `skip` and then checks the count against the total FDA reports.
    An optional free API key in OPENFDA_API_KEY raises the rate limit;
    two requests a day don't need one."""
    records, total, last_updated = [], None, ""
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    while total is None or len(records) < total:
        params = {"limit": PAGE_SIZE, "skip": len(records)}
        if api_key:
            params["api_key"] = api_key
        data = _get_json(API_URL + "?" + urllib.parse.urlencode(params))
        page = data.get("results") or []
        meta = data.get("meta") or {}
        total = int((meta.get("results") or {}).get("total", 0))
        last_updated = meta.get("last_updated", "") or last_updated
        if not page:
            break
        records.extend(page)
    if total is None or len(records) != total:
        raise RuntimeError(f"got {len(records)} records but FDA reports {total} - not saving a partial list")
    return records, last_updated


def check_rows(rows):
    if len(rows) < MIN_RECORDS:
        raise RuntimeError(f"only {len(rows)} records (expected well over {MIN_RECORDS}) - not saving")
    missing = [r for r in rows if not r["generic_name"] or not r["status"]]
    if missing:
        raise RuntimeError(f"{len(missing)} records have no generic name or status - not saving")


def latest_snapshot(raw_dir=RAW_DIR):
    """(date string, path) of the newest dated snapshot, or (None, None)."""
    found = []
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            m = SNAPSHOT_NAME.match(name)
            if m:
                found.append((m.group(1), os.path.join(raw_dir, name)))
    return max(found) if found else (None, None)


def _read_text(path):
    with open(path, newline="", encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def save_snapshot(rows, today, raw_dir=RAW_DIR, fda_last_updated=""):
    """Write today's snapshot if the list changed, and log the check.
    Returns the log line that was written."""
    check_rows(rows)
    os.makedirs(raw_dir, exist_ok=True)
    text = rows_to_csv(rows)

    latest_date, latest_path = latest_snapshot(raw_dir)
    if latest_date is not None and latest_date > today:
        raise RuntimeError(f"newest snapshot is dated {latest_date}, after today ({today}) - check the clock")
    changed = latest_path is None or _read_text(latest_path) != text
    if changed:
        snapshot_path = os.path.join(raw_dir, f"fda_shortages_{today}.csv")
        with open(snapshot_path, "w", newline="", encoding="utf-8") as f:
            f.write(text)
    else:
        snapshot_path = latest_path

    statuses = [r["status"] for r in rows]
    line = {
        "checked_on": today,
        "records": len(rows),
        "current": statuses.count("Current"),
        "to_be_discontinued": statuses.count("To Be Discontinued"),
        "resolved": statuses.count("Resolved"),
        "data_changed": "yes" if changed else "no",
        "snapshot_file": os.path.basename(snapshot_path),
        "fda_last_updated": fda_last_updated,
    }

    # one line per day: a second run on the same day replaces that day's line
    log_path = os.path.join(raw_dir, LOG_NAME)
    lines = []
    if os.path.exists(log_path):
        with open(log_path, newline="", encoding="utf-8") as f:
            lines = [l for l in csv.DictReader(f) if l.get("checked_on") != today]
    lines.append(line)
    lines.sort(key=lambda l: l["checked_on"])
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)
    return line


def main():
    today = datetime.now(timezone.utc).date().isoformat()
    records, fda_last_updated = fetch_all_records()
    rows = [record_to_row(r) for r in records]
    line = save_snapshot(rows, today, fda_last_updated=fda_last_updated)
    print(f"checked openFDA on {today}: {line['records']} records "
          f"({line['current']} current, {line['to_be_discontinued']} to be discontinued, "
          f"{line['resolved']} resolved)")
    if line["data_changed"] == "yes":
        print(f"list changed - saved {line['snapshot_file']}")
    else:
        print(f"list unchanged since {line['snapshot_file']} - no new snapshot file")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # fail loudly so the daily job shows a red X
        print(f"fetch_shortages failed: {e}", file=sys.stderr)
        sys.exit(1)
