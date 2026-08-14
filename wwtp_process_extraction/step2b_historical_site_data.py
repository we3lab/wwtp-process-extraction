# Retrospective permit selection: which order was in force at a past date.
#
# step2 searches CIWQS with inStatus="Active" and dedupes to one row per facility, so
# superseded orders are never captured and "active as of 2021" cannot be reconstructed from
# site_data_all.csv. CIWQS's inStatus filter accepts "Any" (Active + Historical/Terminated),
# which returns the full order history with Effective and Termination dates. That is one extra
# scrape, not one per year -- every target date is then selected offline from the same table.
#
# PDFs and LLM outputs stay in the shared folders: a given document's extraction does not
# change year to year, so only orders whose PDFs we do not already hold need downloading.

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

import step2_scrape_npdes as s2

OUT = s2.OUT
HISTORY_DIR = os.path.join(OUT, "site_data", "_history")
RAW_TSV = os.path.join(HISTORY_DIR, "ciwqs_regulatory_measures_any.tsv")
# step2's Chrome export lands here and is already an unfiltered CIWQS dump covering every
# status (Active + Historical + Terminated), so it doubles as the history source -- no extra
# scrape is needed unless it is stale. --refresh forces a fresh pull.
EXPORT_FALLBACK = os.path.join(OUT, "other_pdfs", "Regualted_Facility_Report_Detail.xls")

# Permit type preference when several orders are in force on the same date (from step2).
TYPE_RANK = s2.TYPE_RANK
KEY = ["WDID", "Facility Name"]


def fetch_history_export(status="Any"):
    """Submit the CIWQS regulated-facility search at `status` and return the raw export.

    Mirrors run_ciwqs_search()'s request flow but keeps every row: no status filter and no
    dedupe to one order per facility, which is the whole point of the historical table.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    r = s2.retry_request(sess, "GET", s2.RFR_URL)
    soup0 = BeautifulSoup(r.text, "html.parser")
    hidden0 = {i["name"]: i.get("value", "") for i in soup0.find_all("input", type="hidden") if i.get("name")}
    csrf = hidden0.get("OWASP_CSRFTOKEN", "")
    in_status = s2.select_value(soup0, "inStatus", status, required_label="Related Permit Status")
    print(f"[requests] inStatus={status!r} -> form value {in_status!r}")

    post = s2.ciwqs_post_data(hidden0, soup0, list(s2.PROGRAMS),
                             facility_type=s2.CIWQS_FACILITY_TYPE,
                             waste_type=s2.CIWQS_WASTE_TYPE, status=in_status)
    resp = s2.retry_request(sess, "POST", f"{s2.CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}", data=post)
    total_url = s2.extract_drilldown_url(BeautifulSoup(resp.text, "html.parser"), allow_program_scope=False)
    if not total_url:
        raise RuntimeError("CIWQS: no Total drilldown URL in search response")
    print(f"[requests] export drilldown: {total_url[:110]}...")

    before = set(glob.glob(os.path.join(s2.pdfs_path, "*.xls*")) + glob.glob(os.path.join(OUT, "*.xls*")))
    driver = s2.new_chrome_driver(s2.pdfs_path)
    driver.set_page_load_timeout(s2.WAIT_TIME)
    try:
        s2._load_ciwqs_table(driver, total_url, "Facility page")
        s2._wait_ciwqs_grid(driver)
        time.sleep(5)
        parts = urlparse(total_url)
        pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "exportToExcel"]
        pairs.append(("exportToExcel", "Y"))
        driver.get(urlunparse(parts._replace(query=urlencode(pairs))))

        deadline = time.time() + s2.WAIT_TIME
        found = None
        while time.time() < deadline:
            new = [f for f in (set(glob.glob(os.path.join(s2.pdfs_path, "*.xls*")))
                               | set(glob.glob(os.path.join(OUT, "*.xls*")))) - before
                   if not f.lower().endswith(".crdownload")]
            if new and all(s2._file_stable(f) for f in new):
                found = max(new, key=os.path.getctime)
                break
            time.sleep(0.5)
        if not found:
            raise RuntimeError(f"no Excel export appeared within {s2.WAIT_TIME}s")
    finally:
        driver.quit()

    os.replace(found, RAW_TSV)
    df = pd.read_csv(RAW_TSV, sep="\t", encoding="latin-1", on_bad_lines="warn", dtype=str).fillna("")
    print(f"[export] {len(df)} raw rows -> {RAW_TSV}")
    return df


def load_history(refresh=False, status="Any", export_path=None):
    src = None
    if not refresh:
        for candidate in (export_path, RAW_TSV, EXPORT_FALLBACK):
            if candidate and os.path.exists(candidate):
                src = candidate
                break
    if src is None:
        return fetch_history_export(status=status)
    df = pd.read_csv(src, sep="\t", encoding="latin-1", on_bad_lines="warn", dtype=str).fillna("")
    print(f"[cache] {len(df)} rows from {src} (pass --refresh to re-scrape)")
    return df


def prepare(df):
    """Apply step2's form-aligned scope filter, keeping every status and every order."""
    n0 = len(df)
    df = df[
        df["Program"].str.upper().str.contains("|".join(s2.ACCEPTED_PROGRAMS), na=False, regex=True)
        & df["Place/Project Type"].str.upper().str.contains(s2.CIWQS_FACILITY_TYPE.upper(), na=False)
    ].copy()
    for col in ("Adoption Date", "Effective Date", "Termination Date", "Expiration/Review Date"):
        df[col + "_dt"] = pd.to_datetime(df[col], errors="coerce")
    df["type_rank"] = df["Regulatory Measure Type"].str.upper().map(
        {k.upper(): v for k, v in TYPE_RANK.items()}).fillna(99)
    print(f"[scope] {n0} -> {len(df)} rows in scope; statuses: "
          f"{df['Regulatory Measure Status'].value_counts().to_dict()}")
    return df


def select_active_at(df, as_of):
    """One order per facility: the one in force on `as_of`.

    In force = took effect on or before as_of, and not terminated on or before as_of.
    Expiration/Review Date is deliberately ignored -- CIWQS carries many Active orders past
    their review date (administrative extension), so expiry does not mean superseded.
    Ties broken by latest effective date, then step2's permit-type preference.
    """
    as_of = pd.Timestamp(as_of)
    eff = df["Effective Date_dt"]
    term = df["Termination Date_dt"]
    live = df[eff.notna() & (eff <= as_of) & (term.isna() | (term > as_of))].copy()
    live = live.sort_values(["Effective Date_dt", "type_rank"], ascending=[False, True])
    return live.drop_duplicates(subset=KEY, keep="first")


def norm_order(order_no):
    """Order numbers appear as 'R5-2020-0004', 'r5_2020_0004', '2014-0153-DWQ' etc."""
    return re.sub(r"[^a-z0-9]", "", str(order_no or "").lower())


def held_orders(facilities):
    """(placeID, normalized order no) -> pdf filenames, from facilities.json.

    facilities.json is the authoritative order-to-PDF mapping written by step2. Matching order
    numbers against PDF filenames is unreliable in both directions: permits are routinely filed
    under names that never mention the order ('Adopted DCTWRP NPDES CI-5695_01242023.pdf'), and
    a substring match on a short order number can hit an unrelated file.
    """
    held = {}
    for pid, v in facilities.items():
        pdfs = list(v.get("pdfs") or [])
        if not pdfs:
            continue
        order = norm_order(v.get("order_no"))
        if order:
            held[(str(pid), order)] = pdfs
    return held

# --- Per-facility permit history -------------------------------------------------------
# The bulk CIWQS export only carries each facility's CURRENT order: 1044 of 1140 in-scope
# facilities have exactly one row, and large plants like San Jose/Santa Clara WPCP show only
# their 2026 permit. The facility-at-a-glance page, by contrast, has a "Regulatory Measures"
# table listing every order with Effective Date, Expiration Date and Status. That page is the
# only workable source for "which permit was in force on date D".

REG_MEASURES_CSV = os.path.join(HISTORY_DIR, "reg_measures_by_facility.csv")
REG_COLS = ["Reg Measure ID", "Reg Measure Type", "Region", "Program", "Order No.",
            "WDID", "Effective Date", "Expiration Date", "Status", "Amended?"]


def fetch_facility_html(place_id, session):
    """Facility-at-a-glance HTML, parsed in memory. The pages are ~120 KB each and only the
    Regulatory Measures table matters, so reg_measures_by_facility.csv is the cache -- there is
    no reason to keep 1100 HTML files on disk."""
    r = session.get(s2.facility_url(place_id), timeout=180)
    r.raise_for_status()
    return r.text


def parse_reg_measures(html):
    """Pull the Regulatory Measures table out of a cached facility page.

    Uses BeautifulSoup rather than pd.read_html: lxml is not in the environment, and bs4 is
    already a step2 dependency. The table is a plain 10-column grid whose first row is the
    header, so locating the header text and reading its enclosing <table> is enough.
    """
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find(string=lambda s: s and "Reg Measure ID" in s)
    if node is None:
        return pd.DataFrame(columns=REG_COLS)
    table = node.find_parent("table")
    if table is None:
        return pd.DataFrame(columns=REG_COLS)
    rows = []
    header = None
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        if header is None:
            if "Reg Measure ID" in cells:
                header = cells
            continue
        if len(cells) == len(header):
            rows.append(cells)
    if header is None:
        return pd.DataFrame(columns=REG_COLS)
    df = pd.DataFrame(rows, columns=header)
    return df[[c for c in REG_COLS if c in df.columns]].copy()


def build_reg_measure_history(place_ids, workers=4):
    """Fetch + parse every facility page into one long history table."""
    sess = requests.Session()
    sess.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    from concurrent.futures import ThreadPoolExecutor, as_completed
    frames, failures = [], []

    def one(pid):
        df = parse_reg_measures(fetch_facility_html(pid, sess))
        df["placeID"] = pid
        return df

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, pid): pid for pid in place_ids}
        for n, fut in enumerate(as_completed(futs), 1):
            pid = futs[fut]
            try:
                frames.append(fut.result())
            except Exception as exc:
                failures.append((pid, str(exc)[:80]))
            if n % 100 == 0:
                print(f"    ...{n}/{len(place_ids)} pages ({len(failures)} failed)")

    hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REG_COLS + ["placeID"])
    print(f"[history] fetched {len(hist)} regulatory measures across "
          f"{hist['placeID'].nunique() if len(hist) else 0} facilities ({len(failures)} page failures)")
    return hist, failures


def prepare_reg_measures(hist):
    """Type the history table for as-of selection."""
    hist = hist.copy()
    for col in ("Effective Date", "Expiration Date"):
        hist[col + "_dt"] = pd.to_datetime(hist[col], errors="coerce")
    hist["Status"] = hist["Status"].astype(str).str.strip()
    hist["type_rank"] = hist["Reg Measure Type"].astype(str).str.upper().map(
        {k.upper(): v for k, v in TYPE_RANK.items()}).fillna(99)
    hist = hist[hist["Program"].astype(str).str.upper().str.contains(
        "|".join(s2.ACCEPTED_PROGRAMS), na=False, regex=True)]
    return hist[hist["Status"].ne("Never Active")]


def select_order_at(hist, as_of):
    """The order governing each facility on `as_of`, from the reg-measure history.

    Candidates are orders effective on or before as_of. Expiration is NOT used as a cutoff:
    CIWQS routinely shows a permit expiring before its replacement takes effect (San Jose's
    R2-2020-0001 expired 03/2025, R2-2025-0027 began 02/2026) because the old permit is
    administratively extended through the gap. Filtering on expiration would leave the
    facility with only an incidental Co-Permitee order, or nothing.

    Precedence: step2's permit-type rank first (NPDES Permit before Co-Permitee before WDR),
    then the most recent effective date within that type. That yields the governing permit
    document rather than whichever order happens to be newest.
    """
    as_of = pd.Timestamp(as_of)
    eff = hist["Effective Date_dt"]
    live = hist[eff.notna() & (eff <= as_of)].copy()
    live = live.sort_values(["type_rank", "Effective Date_dt"], ascending=[True, False])
    return live.drop_duplicates(subset="placeID", keep="first")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # The primary dataset was scraped 2026-06-01, so retrospective snapshots are anchored
    # there rather than "today" -- every folder is 06-01 of its year and the base-year folder
    # lines up with the data already in output/.
    ap.add_argument("--base-date", default="2026-06-01",
                    help="anchor date; snapshots are this date minus 0..N years")
    ap.add_argument("--years-back", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="concurrent facility-page fetches")
    ap.add_argument("--force-refetch", action="store_true", help="re-fetch all facility pages")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N facilities (smoke test)")
    args = ap.parse_args()

    with open(os.path.join(OUT, "facilities.json")) as f:
        facilities = json.load(f)
    place_ids = sorted(facilities)[: args.limit]
    print(f"[facilities] {len(place_ids)} place IDs from facilities.json")

    empty_path = os.path.join(HISTORY_DIR, "no_reg_measures.json")
    known_empty = set(json.load(open(empty_path))) if os.path.exists(empty_path) else set()
    if os.path.exists(REG_MEASURES_CSV) and not args.force_refetch:
        hist = pd.read_csv(REG_MEASURES_CSV, dtype=str).fillna("")
        missing = sorted(set(place_ids) - set(hist["placeID"]) - known_empty)
        if missing:
            print(f"[history] {len(hist)} cached rows; fetching {len(missing)} new facilities")
            extra, _ = build_reg_measure_history(missing, workers=args.workers)
            got = set(extra["placeID"]) if len(extra) else set()
            known_empty |= (set(missing) - got)
            hist = pd.concat([hist, extra], ignore_index=True)
            hist.to_csv(REG_MEASURES_CSV, index=False)
            os.makedirs(HISTORY_DIR, exist_ok=True)
            json.dump(sorted(known_empty), open(empty_path, "w"), indent=2)
        else:
            print(f"[history] {len(hist)} cached regulatory measures (--force-refetch to redo)")
    else:
        hist, _ = build_reg_measure_history(place_ids, workers=args.workers)
        os.makedirs(HISTORY_DIR, exist_ok=True)
        hist.to_csv(REG_MEASURES_CSV, index=False)

    prepared = prepare_reg_measures(hist)
    print(f"[history] {len(prepared)} in-scope measures across {prepared['placeID'].nunique()} facilities")

    name_of = {pid: v.get("Facility Name", "") for pid, v in facilities.items()}
    wdid_of = {pid: v.get("WDID", "") for pid, v in facilities.items()}
    held = held_orders(facilities)
    print(f"[pdfs] facilities.json maps {len(held)} (facility, order) pairs to downloaded PDFs\n")

    base = pd.Timestamp(args.base_date)
    summary, prev_orders = [], None
    for years in range(args.years_back + 1):
        as_of = (base - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
        sel = select_order_at(prepared, as_of).copy()
        sel["Facility Name"] = sel["placeID"].map(name_of)
        sel["WDID_facilities_json"] = sel["placeID"].map(wdid_of)

        out_dir = os.path.join(OUT, "site_data", as_of)
        os.makedirs(out_dir, exist_ok=True)

        # Same shape as the top-level facilities.json so downstream steps can consume a dated
        # snapshot unchanged: the order in force at as_of, plus any PDFs we already hold for it.
        snap = {}
        for rec in sel.to_dict("records"):
            pid = str(rec["placeID"])
            order_no = str(rec.get("Order No.", "")).strip()
            pdfs = held.get((pid, norm_order(order_no)), [])
            meta = facilities.get(pid, {})
            snap[pid] = {
                "Facility Name": meta.get("Facility Name", ""),
                "WDID": meta.get("WDID", ""),
                "order_no": order_no,
                "reg_measure_id": str(rec.get("Reg Measure ID", "")).strip(),
                "reg_measure_type": str(rec.get("Reg Measure Type", "")).strip(),
                "effective_date": str(rec.get("Effective Date", "")).strip(),
                "expiration_date": str(rec.get("Expiration Date", "")).strip(),
                "status": str(rec.get("Status", "")).strip(),
                "pdfs": pdfs,
                "pdfs_needed": not pdfs,
            }
        with open(os.path.join(out_dir, "facilities.json"), "w") as f:
            json.dump(snap, f, indent=2)

        pairs = {(str(pid), norm_order(o))
                 for pid, o in zip(sel["placeID"], sel["Order No."]) if str(o).strip()}
        have = sum(1 for k in pairs if k in held)
        orders = {o for _, o in pairs}
        churn = len(orders - prev_orders) if prev_orders is not None else None
        info = {"as_of": as_of, "years_back": years, "facilities": len(sel),
                "distinct_orders": len(orders), "facility_order_pairs": len(pairs),
                "pdf_already_held": have, "pdf_to_download": len(pairs) - have,
                "orders_changed_vs_next_newer": churn}
        summary.append(info)
        prev_orders = orders

    print(pd.DataFrame(summary).to_string(index=False))
    print("\npdf_already_held counts (facility, order) pairs that facilities.json maps to at least "
          "one downloaded PDF. facilities.json records only each facility's CURRENT order, so a "
          "historical order reads as not-held until step2 fetches it and records it per order -- "
          "see the note in the module docstring. PDFs and LLM runs stay in the shared folders.")


if __name__ == "__main__":
    main()
