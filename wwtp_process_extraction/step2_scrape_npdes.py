from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
import shutil
import threading
import time
import glob
import requests
import pandas as pd
import uuid
import tempfile
import fitz
import pdfplumber
import unicodedata
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode, urljoin, urlunparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import TimeoutException
from helpers.utils import normalize_text
# Modified using Claude 4.5

# Link to Interactive Regulated Facilities Report
CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"
RFR_URL = (
    "https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/"
    "CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility"
)

PROGRAMS = {"NPDES": {"NPDESWW", "NPDMUNI"}, "WDR": {"WDRMUNILRG", "WDRMUNIOTH"}}
ACCEPTED_PROGRAMS = set().union(*PROGRAMS.values())
TYPE_RANK = {
    "NPDES PERMIT": 0,
    "CO-PERMITTEE": 1,
    "ENROLLEE - NPDES": 2,
    "WDR": 3,
    "ENROLLEE - WDR": 4,
    "Individual Monitoring Requirem": 5
    } 

CIWQS_FACILITY_TYPE = "Wastewater Treatment Facility"
CIWQS_WASTE_TYPE = "Domestic wastewater"
CIWQS_RELATED_PERMIT_STATUS = "Active"
CIWQS_DRILLDOWN_QUERY_DROP = ("enrollee",) # drop enrollee=Y filter

WAIT_TIME = 300  # For CIWQS/grid/export - large number of results
CIWQS_OVERLAY_WAIT = 180  # loading spinner / overlay after changing page size
FACILITY_CIWS_COLUMNS = ["WDID", "Facility Name", "NPDES No."]
XP_GRID = "//table[contains(@class,'ciwqsReportDataTable')]"

OUT = "wwtp_process_extraction/output"
pdfs_path = os.path.join(OUT, "other_pdfs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(pdfs_path, exist_ok=True)

# PDF filenames matching this regex are skipped on the order page.
SEP = r"[ ._-]"  # - must be last to avoid range interpretation
SKIP_BASE_KW = ["rpts|rowd|memo|nov|map|rwd|gwmp|mgo"]  # match only with separators (e.g. "_memo_")
SKIP_PHRASE = (
    "report|financial|response to|rate study|ratestudy|study|"
    "addendum|registration|adoption|"
    "letter|covltr|cover_l|cover l|volumetric|"
    "form200|form 200|management zone|management_zone|management plan"
)  # skip if anywhere in filename
SKIP_RE = re.compile(rf"^(?:{'|'.join(SKIP_BASE_KW)}){SEP}|{SEP}(?:{'|'.join(SKIP_BASE_KW)}){SEP}|{SKIP_PHRASE}", re.IGNORECASE)

# skip these UNLESS keep_re is present too
CONTINGENT_SKIP_PHRASE = "amendment|mrp"
CONTINGENT_SKIP_RE = re.compile(CONTINGENT_SKIP_PHRASE, re.IGNORECASE)

# keep these, overriding contingent skip
KEEP_RE = re.compile(r"(?<![a-zA-Z])(noa|wdrs?|order|npdes)(?![a-zA-Z])", re.IGNORECASE)  # always keep NOA/WDR/NPDES files

def abs_url(href):
    return urljoin(f"{CIWQS_ROOT}/ciwqs/readOnly/", href) if href else href


def facility_url(place_id):
    """Reconstruct the CIWQS facility-at-a-glance URL from a place ID."""
    return f"{CIWQS_SERVLET}?reportName=facilityAtAGlance&placeID={place_id}"


def select_value(soup, name, visible_text, *, required_label=None):
    sel = soup.find("select", {"name": name})
    if not sel:
        if required_label:
            raise RuntimeError(f"CIWQS form missing <select name={name!r}> ({required_label}).")
        return visible_text
    opts = sel.find_all("option")
    match = next((o for o in opts if o.get_text(strip=True) == visible_text), None)
    if match:
        return match.get("value", visible_text)
    if required_label:
        choices = [o.get_text(strip=True) for o in opts]
        raise RuntimeError(
            f"CIWQS {required_label}: no {visible_text!r} in <select name={name!r}>; choices={choices!r}"
        )
    return visible_text


def retry_request(session, method, url, *, data=None, max_attempts=4, timeout=120):
    """Retry only on Timeout; other RequestException subclasses propagate immediately."""
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.request(method, url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            print(f"[requests] {method.upper()} timed out ({attempt}/{max_attempts}): {url}")
            if attempt == max_attempts:
                raise


def new_chrome_driver(pdfs_path):
    """Chrome for facility report (after requests submits the CIWQS search)."""
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.path.abspath(pdfs_path),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    options.add_experimental_option("prefs", prefs)
    # "normal" waits for load event; CIWQS often omits programDrop until then (eager returns too early).
    options.page_load_strategy = "normal"
    options.add_argument("--blink-settings=imagesEnabled=false")
    user_data_dir = os.path.join(tempfile.gettempdir(), f"chrome_user_data_{uuid.uuid4().hex}")
    options.add_argument(f"--user-data-dir={user_data_dir}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")  # for server/SSH
    options.binary_location = "/home/daly/bin/chrome/chrome-linux64/chrome"
    service = Service("/home/daly/bin/chrome/chromedriver-linux64/chromedriver")
    return webdriver.Chrome(service=service, options=options)


def ciwqs_post_data(hidden, soup, programs, *, facility_type, waste_type, status):
    """Build the CIWQS form POST body for one or more programs."""
    return (
        list(hidden.items())
        + [("programDrop", select_value(soup, "programDrop", p)) for p in programs]
        + [
            ("typeDrop", select_value(soup, "typeDrop", facility_type)),
            ("wasteTypeDrop", select_value(soup, "wasteTypeDrop", waste_type)),
            ("inStatus", status),
            ("enpRepButton", ""),
        ]
    )


def extract_drilldown_url(soup, *, allow_program_scope=True):
    """Pick RegulatedFacilityDetail drilldown from CIWQS search HTML."""
    candidates = [
        abs_url(a["href"])
        for a in soup.find_all("a", href=True)
        if "RegulatedFacilityDetail" in a["href"] and "drilldown" in a["href"]
    ]
    excluded = ["place=", "majorminor="] + ([] if allow_program_scope else ["program="])
    filtered = [c for c in candidates if not any(k in c.lower() for k in excluded)]
    chosen = filtered[0] if filtered else (candidates[0] if candidates else None)
    if not chosen:
        return None
    parts = urlparse(chosen)
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in CIWQS_DRILLDOWN_QUERY_DROP
    ]
    return urlunparse(parts._replace(query=urlencode(kept)))


@contextmanager
def _open_in_new_tab(driver, url, main_window, *, post_switch_sleep=0):
    """Open `url` in a new tab, yield, then close the tab and switch back to main."""
    driver.execute_script(f"window.open('{url}', '_blank');")
    time.sleep(1)
    new_window = next(h for h in driver.window_handles if h != main_window)
    driver.switch_to.window(new_window)
    if post_switch_sleep:
        time.sleep(post_switch_sleep)
    try:
        yield
    finally:
        try:
            driver.close()
        except Exception:
            pass
        driver.switch_to.window(main_window)


def _resolve_download_url(href, soup):
    reg_id_val = parse_qs(urlparse(href).query).get("regMeasID", [None])[0]
    attach_tag = soup.find(
        "a", href=lambda h, r=reg_id_val: h and "rmAttachmentPopup" in h
        and (not r or f"regMeasID={r}" in h)
    )
    return abs_url(attach_tag["href"]) if attach_tag else href


def _download_and_move(driver, url, worker_dir, main_window, check_dirs, pdfs_path):
    pdfs, missed, total = download_pdfs_for_order(driver, url, worker_dir, main_window, check_dirs=check_dirs)
    for fname in pdfs:
        src, dst = os.path.join(worker_dir, fname), os.path.join(pdfs_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    return pdfs, missed, total


def find_best_order(driver, fac_url, main_window):
    """Navigate to facility page, parse HTML, and return best active NPDES order.

    Returns: (order_url, reg_measure_type, wdid, eff, addtl_orders, order_no)
      order_url may be None if best order has no clickable link.
      addtl_orders is a list of (url, wdid, eff) for additional NPDES PERMIT orders with valid links.
      order_no is the Order No. text from the CIWQS table for the selected regulatory measure.
    """
    # Navigate to facility page
    with _open_in_new_tab(driver, fac_url, main_window):
        WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(1)
        page_html = driver.page_source

    # Parse HTML for best order
    soup = BeautifulSoup(page_html, "html.parser")
    for table in soup.find_all("table"):
        all_rows = table.find_all("tr")

        # Find header row
        for hdr_idx, row in enumerate(all_rows[:4]):
            texts = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]

            # Validate header row
            if len(texts) < 5 or any(len(t) > 80 for t in texts):
                continue
            if not all(any(req in t for t in texts) for req in ("Reg Measure Type", "Order No")):
                continue

            col_index = {t: i for i, t in enumerate(texts)}
            dcells = []  # rebound each data row; gc closes over it by name

            def gc(col_name):
                i = col_index.get(col_name, -1)
                return dcells[i].get_text(strip=True) if 0 <= i < len(dcells) else ""

            # Process data rows: collect (rank, -eff, href, rm_type, eff, wdid) tuples; min() picks the best.
            # href may be None for rows without a clickable link — still included so we can store order metadata.
            candidates = []
            for data_row in all_rows[hdr_idx + 1:]:
                dcells = data_row.find_all("td")
                if not dcells or gc("Status").lower() != "active":
                    continue

                # Apply validation checks
                rm_type = gc("Reg Measure Type").upper()
                if rm_type not in TYPE_RANK:
                    continue
                if rm_type == "WDR" and gc("Program").upper() not in PROGRAMS["WDR"]:
                    continue

                order_idx = col_index.get("Order No.", -1)
                order_cell = dcells[order_idx] if 0 <= order_idx < len(dcells) else None
                a_tag = order_cell.find("a", href=True) if order_cell else None
                href = abs_url(a_tag["href"]) if a_tag else None
                order_no = order_cell.get_text(strip=True) if order_cell else ""
                eff = pd.to_datetime(gc("Effective Date"), errors="coerce")
                if pd.isna(eff):
                    continue

                candidates.append((TYPE_RANK[rm_type], -eff.value, href, rm_type, eff, gc("WDID"), order_no))

            if candidates:
                rank, _, href, rm_type, eff, wdid, order_no = min(candidates, key=lambda c: (c[0], c[1], 0 if c[2] else 1))
                print(f"  Best order: {rm_type}, rank={rank}, effective={eff.date()}, order={order_no}, link={'yes' if href else 'no'}")

                download_url = _resolve_download_url(href, soup) if href else None

                # Collect additional NPDES PERMIT orders (rank=0) with valid links, excluding primary
                addtl_orders = []
                for _, _, c_href, c_rm_type, c_eff, c_wdid, _ in candidates:
                    if TYPE_RANK.get(c_rm_type, 99) != 0 or not c_href or c_href == href:
                        continue
                    addtl_orders.append((_resolve_download_url(c_href, soup), c_wdid, c_eff))

                return download_url, rm_type, wdid, eff, addtl_orders, order_no

    return None, None, None, None, [], ""


def download_pdfs_for_order(driver, order_url, output_dir, main_window, check_dirs=None):
    """Download PDFs from an order attachment page. Returns (downloaded_pdfs, missed_pdfs, total_on_page)."""
    if check_dirs is None:
        check_dirs = (output_dir, os.path.join(OUT, "permits"))
    downloaded_pdfs = []
    missed_pdfs = []
    PDF_XPATH = "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
    with _open_in_new_tab(driver, order_url, main_window, post_switch_sleep=0):
        try:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, PDF_XPATH)))
        except TimeoutException:
            pass
        pdf_documents = driver.find_elements(By.XPATH, PDF_XPATH)
        total_on_page = len(pdf_documents)
        print(f"  Found {total_on_page} PDFs on page")

        for pdf_element in pdf_documents:
            try:
                pdf_name = pdf_element.text
                if total_on_page > 1:
                    if SKIP_RE.search(pdf_name):
                        continue
                    if CONTINGENT_SKIP_RE.search(pdf_name) and not KEEP_RE.search(pdf_name):
                        continue


                # TODO: can remove if not re-running old dates
                if any(os.path.exists(os.path.join(d, pdf_name)) for d in check_dirs):
                    print(f"        Already exists, skipping: {pdf_name}")
                    downloaded_pdfs.append(pdf_name)
                    continue

                print(f"        Downloading: {pdf_name}")
                before = {f for f in os.listdir(output_dir) if f.lower().endswith(".pdf")}
                pdf_element.click()

                end = time.time() + 90
                new_file = None
                while time.time() < end:
                    current = {
                        f
                        for f in os.listdir(output_dir)
                        if f.lower().endswith(".pdf") and not f.endswith(".crdownload")
                    }
                    new = current - before
                    if new:
                        newest = max(new, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
                        if _file_stable(os.path.join(output_dir, newest)):
                            new_file = newest
                            break
                    time.sleep(0.5)

                if not new_file:
                    print(f"        X Timed out waiting for: {pdf_name}")
                    missed_pdfs.append(pdf_name)
                    continue
                downloaded_pdfs.append(new_file)
            except Exception as e:
                print(f"        X Download failed: {pdf_name} — {e}")
                missed_pdfs.append(pdf_name)

    return downloaded_pdfs, missed_pdfs, total_on_page


def _file_stable(path):
    try:
        s = os.path.getsize(path)
        time.sleep(0.5)
        return s > 0 and os.path.getsize(path) == s
    except OSError:
        return False


def _wait_ciwqs_grid(driver):
    WebDriverWait(driver, WAIT_TIME).until(
        EC.presence_of_element_located((By.XPATH, XP_GRID))
    )


def _load_ciwqs_table(driver, url, label="url"):
    wait = WebDriverWait(driver, WAIT_TIME)
    for attempt in range(1, 4):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.XPATH, XP_GRID)))
            return
        except TimeoutException:
            print(f"[selenium] {label} slow ({attempt}/3)…")
            if attempt == 3:
                raise
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass


def _set_page_all(driver):
    long_wait = WebDriverWait(driver, WAIT_TIME)
    overlay_wait = WebDriverWait(driver, CIWQS_OVERLAY_WAIT)
    for attempt in range(1, 4):
        try:
            if driver.find_elements(By.NAME, "pagesizeselect"):
                sel_el = WebDriverWait(driver, 120).until(
                    EC.element_to_be_clickable((By.NAME, "pagesizeselect"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sel_el)
                Select(sel_el).select_by_visible_text("ALL")
                time.sleep(2)
            try:
                overlay_wait.until(
                    EC.invisibility_of_element_located((By.CLASS_NAME, "loading"))
                )
            except TimeoutException:
                pass
            return long_wait.until(
                EC.presence_of_element_located((By.XPATH, XP_GRID))
            )
        except Exception as e:
            print(f"[selenium] pagesizeselect ALL did not stabilize ({attempt}/3): {e}")
            if attempt == 3:
                raise
            time.sleep(3)


def run_ciwqs_search():
    ciwqs = requests.Session()
    ciwqs.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    r = retry_request(ciwqs, 'GET', RFR_URL)
    soup0 = BeautifulSoup(r.text, "html.parser")
    hidden0 ={
            i["name"]: i.get("value", "")
            for i in soup0.find_all("input", type="hidden")
            if i.get("name")
        }
    csrf = hidden0.get("OWASP_CSRFTOKEN", "")
    in_status = select_value(
        soup0, "inStatus", CIWQS_RELATED_PERMIT_STATUS, required_label="Related Permit Status"
    )
    # Multi-select post → total drilldown URL (used for Excel export only).
    # Multi-select yields a grouped-by-agency view; re-submit per-program for the flat table.
    _kwargs = dict(facility_type=CIWQS_FACILITY_TYPE, waste_type=CIWQS_WASTE_TYPE, status=in_status)
    print("[requests] Submitting filters")
    resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
                        data=ciwqs_post_data(hidden0, soup0, list(PROGRAMS), **_kwargs))
    total_url = extract_drilldown_url(
        BeautifulSoup(resp.text, "html.parser"), allow_program_scope=False
    )
    if not total_url:
        raise RuntimeError("CIWQS: no Total drilldown URL in search response")
    print(f"[requests] Excel export URL: {total_url}")

    # Re-submit once per program to get per-facility flat-table drilldown URLs.
    program_urls = []
    for prog in list(PROGRAMS):
        prog_resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
                                data=ciwqs_post_data(hidden0, soup0, [prog], **_kwargs))
        prog_url = extract_drilldown_url(BeautifulSoup(prog_resp.text, "html.parser"))
        if prog_url:
            program_urls.append((prog, prog_url))
            print(f"[requests] {prog} facility URL: {prog_url}")
        else:
            print(f"[requests] Warning: no drilldown URL found for {prog}")

    # Fresh Chrome profile (empty cookies) — only touches the Excel summary drilldown URL (total_url).
    driver = new_chrome_driver(pdfs_path)
    driver.set_page_load_timeout(WAIT_TIME)
    _load_ciwqs_table(driver, total_url, "Facility page")
    print("Detail page loaded for Excel export")
    _wait_ciwqs_grid(driver)
    time.sleep(5)
    parts = urlparse(total_url)
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "exportToExcel"]
    pairs.append(("exportToExcel", "Y"))
    excel_export_url = urlunparse(parts._replace(query=urlencode(pairs)))
    
    for attempt in range(1, 3):
        try:
            driver.get(excel_export_url)
            break
        except TimeoutException:
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass
            if attempt == 2:
                raise
    time.sleep(2)

    poll_start = time.monotonic()
    end_time = time.time() + WAIT_TIME
    excel_files = []
    had_download_activity = False
    while time.time() < end_time:
        candidates = [f for d in (pdfs_path, OUT)
                    for f in glob.glob(os.path.join(d, "*.xls*"))
                    if not f.lower().endswith(".crdownload")]
        if candidates or any(
            glob.glob(os.path.join(root, "*.crdownload")) for root in (pdfs_path, OUT)
        ):
            had_download_activity = True

        stable_flags = [_file_stable(path) if os.path.isfile(path) else False for path in candidates]
        elapsed = time.monotonic() - poll_start
        if not had_download_activity and elapsed >= 90:
            break

        if candidates and all(stable_flags):
            excel_files = candidates
            break
        time.sleep(0.5)

    if not excel_files:
        print(f"No Excel file found after {WAIT_TIME}s")
        driver.quit()
        exit()

    excel_file = max(excel_files, key=os.path.getctime)
    df = pd.read_csv(excel_file, sep='\t', encoding='latin-1', on_bad_lines='warn', dtype=str)

    # Filtering matching original CIWQS form logic
    df = df[
        df["Program"].fillna("").str.upper().str.contains("|".join(ACCEPTED_PROGRAMS), na=False, regex=True) &
        (df["Regulatory Measure Status"].fillna("").str.upper() == CIWQS_RELATED_PERMIT_STATUS.upper()) &
        df["Place/Project Type"].fillna("").str.upper().str.contains(CIWQS_FACILITY_TYPE.upper(), na=False)
    ]
    print(f"After explicit form-aligned filtering: {len(df)} rows")

    df["Expiration/Review Date"] = pd.to_datetime(df["Expiration/Review Date"], errors='coerce')
    df_sorted = df.sort_values(["WDID", "Facility Name", "Expiration/Review Date"], ascending=[True, True, False])
    df_deduplicated = df_sorted.drop_duplicates(subset=["WDID", "Facility Name"], keep="first")
    duplicates_removed = df_sorted[df_sorted.duplicated(subset=["WDID", "Facility Name"], keep="first")]
    print(f"After deduplication and filtering: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")
    if len(duplicates_removed) > 0:
        cols = [c for c in ["Facility Name", "WDID", "NPDES No."] if c in duplicates_removed.columns]
        print("Duplicates removed (Facility Name, WDID, NPDES No.):")
        print(duplicates_removed[cols].to_string(index=False))

    df_deduplicated.to_csv(os.path.join(OUT, "site_data_all.csv"), index=False)
    print(f"Saved {len(df_deduplicated)} rows to site_data_all.csv")

    driver.quit()
    return program_urls


def collect_facility_page_urls(program_urls):
    print("\n STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")

    facilities_by_place = {}  # place_id -> {"facilities": [dict keyed by FACILITY_CIWS_COLUMNS, ...]}
    name_to_place_id = {}  # facility name -> place_id, collected pre-filter for reconciliation

    def _cell(cells, i):
        return cells[i].get_text(strip=True) if i is not None and 0 <= i < len(cells) else ""

    def _cell_href(cells, i):
        if i is None or not (0 <= i < len(cells)):
            return ""
        a = cells[i].find("a", href=True)
        return abs_url(a["href"]) if a else ""

    # Navigate per-program RegulatedFacilityDetail URLs. New Chrome profile per URL so no stale CIWQS cookies.
    col = {}  # CIWQS header text -> column index (detected on first program)

    for prog, prog_url in program_urls:
        print(f"\n--- {prog}: {prog_url}")
        # Try a new driver every time, since cookies from original search make tables too slow to load
        driver = new_chrome_driver(pdfs_path)
        driver.set_page_load_timeout(WAIT_TIME)
        _load_ciwqs_table(driver, prog_url, prog)
        _wait_ciwqs_grid(driver)
        _set_page_all(driver)

        page_soup = BeautifulSoup(driver.page_source, "html.parser")
        data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
        bs_rows_prog = data_table.find_all("tr") if data_table else []
        print(f"{prog}: {len(bs_rows_prog)} <tr> after ALL page size")

        if not col:
            for _tr in bs_rows_prog:
                _texts = [_td.get_text(strip=True) for _td in _tr.find_all("td")]
                if "Order No." in _texts or "Facility Name" in _texts:
                    col.clear()
                    col.update({t: i for i, t in enumerate(_texts) if t})
                    break
            missing = [c for c in FACILITY_CIWS_COLUMNS if c not in col]
            if missing:
                raise RuntimeError(
                    f"Missing columns in {prog} table: {missing}. "
                    f"Found: {list(col.keys())[:10]}"
                )

        for tr in bs_rows_prog:
            try:
                if tr.find("td", class_="ciwqsReportColumnName"):
                    continue
                cells = tr.find_all("td")
                if not cells:
                    continue

                status = _cell(cells, col.get("Regulatory Measure Status")).upper()
                plc_type = _cell(cells, col.get("Place/Project Type")).upper()

                if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
                    continue
                if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
                    continue

                # Collect facility name -> place_id unconditionally for reconciliation below
                place_id = parse_qs(
                    urlparse(_cell_href(cells, col.get("Facility Name"))).query
                ).get("placeID", [None])[0]
                raw_name = _cell(cells, col.get("Facility Name"))
                if place_id and raw_name:
                    name_to_place_id[raw_name] = place_id

                if not place_id:
                    continue

                facility = {name: _cell(cells, col.get(name)) for name in FACILITY_CIWS_COLUMNS}
                entry = facilities_by_place.setdefault(place_id, {"facilities": []})
                if not any(f["Facility Name"] == facility["Facility Name"] for f in entry["facilities"]):
                    entry["facilities"].append(facility)
            except Exception as e:
                print(f"Row parse error: {e}")
                continue

        driver.quit()

    # Reconcile against site_data_all.csv to catch any facilities missed by per-program scrapes
    site_data_all_path = os.path.join(OUT, "site_data_all.csv")
    if os.path.exists(site_data_all_path):
        npdes_df = pd.read_csv(site_data_all_path, dtype=str).fillna("")
        scraped_names = {
            f["Facility Name"]
            for entry in facilities_by_place.values()
            for f in entry.get("facilities", [])
        }
        added = 0
        for _, row in npdes_df.iterrows():
            fac_name = row.get("Facility Name", "").strip()
            if not fac_name or fac_name in scraped_names:
                continue
            place_id = name_to_place_id.get(fac_name)
            if place_id and place_id not in facilities_by_place:
                facilities_by_place[place_id] = {
                    "facilities": [{
                        "WDID": row.get("WDID", "").strip(),
                        "Facility Name": fac_name,
                        "NPDES No.": row.get("NPDES No.", "").strip(),
                    }]
                }
                print(f"  + Reconciled from site_data_all.csv: {fac_name} (placeID={place_id})")
                added += 1
            elif not place_id:
                print(f"  ! {fac_name} in site_data_all.csv but not found in any CIWQS table")
        if added:
            print(f"  Reconciliation added {added} missing facilities")

    print(f"\n✓ Found {len(facilities_by_place)} unique facilities (placeIDs)")

    with open(os.path.join(OUT, 'facilities.json'), 'w') as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: {len(facilities_by_place)} facilities → facilities.json")

    return facilities_by_place

def download_facility_page_pdfs(facilities_by_place, max_workers=12):
    # UPDATE max_workers to be higher if running on server
    print("\n STEP 2: Visiting facility pages and downloading PDFs")

    reg_id_to_info = {}
    lock = threading.Lock()
    permit_path = os.path.join(OUT, "permits")
    check_dirs = (pdfs_path, permit_path)

    items = list(facilities_by_place.items())
    total = len(items)

    def process_facility(args):
        idx, (place_id, entry) = args
        worker_dir = tempfile.mkdtemp(prefix="npdes_dl_")
        driver = new_chrome_driver(worker_dir)
        driver.set_page_load_timeout(WAIT_TIME)
        main_window = driver.window_handles[0]
        fac_url = facility_url(place_id)
        fac_name = entry["facilities"][0]["Facility Name"] if "facilities" in entry else entry.get("Facility Name", place_id)
        print(f"\n[{idx}/{total}] {fac_name}")
        try:
            order_url, rm_type, wdid, eff, addtl_orders, order_no = find_best_order(driver, fac_url, main_window)
            if rm_type is None:
                print("  X No suitable active NPDES order found")
                entry.update(
                    {"Facility Name": fac_name,
                     "WDID": wdid,
                     "pdfs": [],
                     "total_pdfs": 0,
                     "reg_measure_id": None,
                     "reg_measure_type": None,
                     "order_no": ""}
                )
                entry.pop("facilities", None)
                return

            reg_id = parse_qs(urlparse(order_url).query).get("regMeasID", [None])[0] if order_url else None

            # Store order metadata but skip PDFs if no link or pre-2004
            if not order_url or (eff and eff.year < 2004):
                reason = "pre-2004" if (eff and eff.year < 2004) else "no link"
                print(f"  Skipping PDFs ({reason}): {rm_type}, eff={eff.date() if eff else 'unknown'}")
                entry.update({"Facility Name": fac_name, "WDID": wdid, "pdfs": [],
                              "total_pdfs": 0, "reg_measure_id": reg_id,
                              "reg_measure_type": rm_type, "order_no": order_no,
                              "pdf_skip_reason": reason})
                entry.pop("facilities", None)
                return

            with lock:
                if reg_id and reg_id in reg_id_to_info:
                    print(f"  Dedup: reusing already-processed order {reg_id}")
                    entry.update(reg_id_to_info[reg_id])
                    entry["Facility Name"] = fac_name
                    entry["WDID"] = wdid
                    entry.pop("facilities", None)
                    return

            downloaded_pdfs, missed_pdfs, total_pdfs = _download_and_move(
                driver, order_url, worker_dir, main_window, check_dirs, pdfs_path
            )

            info = {
                "Facility Name": fac_name,
                "WDID": wdid,
                "pdfs": downloaded_pdfs,
                "missed_pdfs": missed_pdfs,
                "total_pdfs": total_pdfs,
                "reg_measure_id": reg_id,
                "reg_measure_type": rm_type,
                "order_no": order_no,
            }

            # Download PDFs for additional NPDES PERMIT orders (effective 2004+)
            addtl_reg_ids, addtl_wdids = [], []
            for addtl_url, addtl_wdid, addtl_eff in addtl_orders:
                if addtl_eff.year < 2004:
                    continue
                addtl_reg_id = parse_qs(urlparse(addtl_url).query).get("regMeasID", [None])[0]
                with lock:
                    cached = addtl_reg_id and addtl_reg_id in reg_id_to_info
                if cached:
                    extra_pdfs = reg_id_to_info[addtl_reg_id].get("pdfs", [])
                else:
                    extra_pdfs, extra_missed, _ = _download_and_move(
                        driver, addtl_url, worker_dir, main_window, check_dirs, pdfs_path
                    )
                    info["missed_pdfs"].extend(extra_missed)
                    with lock:
                        if addtl_reg_id:
                            reg_id_to_info[addtl_reg_id] = {"pdfs": extra_pdfs}
                info["pdfs"].extend(extra_pdfs)
                addtl_reg_ids.append(addtl_reg_id or "")
                addtl_wdids.append(addtl_wdid or "")
            if addtl_reg_ids:
                info["addtl_reg_measure_id"] = ",".join(addtl_reg_ids)
                info["addtl_WDID"] = ",".join(addtl_wdids)

            entry.update(info)
            entry.pop("facilities", None)
            with lock:
                if reg_id:
                    reg_id_to_info[reg_id] = info

        except Exception as e:
            print(f"  X {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
            shutil.rmtree(worker_dir, ignore_errors=True)

    # loop mutates each entry in facilities_by_place in-place to add 'pdfs', 'total_pdfs', 'reg_measure_id', 'reg_measure_type'
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_facility, enumerate(items, 1))

    def _needs_retry(entry):
        # Unprocessed (exception during process_facility left "facilities" key intact)
        if "facilities" in entry:
            return True
        # Deliberately skipped (no link or pre-2004) — not a failure, never retry
        if entry.get("pdf_skip_reason"):
            return False
        # A reg measure was clicked and 0 PDFs came back — should be impossible; retry
        if entry.get("reg_measure_type") and not entry.get("pdfs") and not entry.get("total_pdfs"):
            return True
        return False

    retry_count = 0
    try:
        while True:
            retry_items = [
                (place_id, entry)
                for place_id, entry in facilities_by_place.items()
                if _needs_retry(entry)
            ]
            if not retry_items:
                break
            retry_count += 1
            print(f"\nRetry pass {retry_count}: {len(retry_items)} facilities with failed downloads")
            with lock:
                for _, entry in retry_items:
                    reg_id_to_info.pop(entry.get("reg_measure_id"), None)
                    entry.pop("missed_pdfs", None)
            total = len(retry_items)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor.map(process_facility, [(i + 1, item) for i, item in enumerate(retry_items)])
    except KeyboardInterrupt:
        still_failing = [
            {"place_id": place_id}
            for place_id, entry in facilities_by_place.items()
            if _needs_retry(entry)
        ]
        if still_failing:
            pd.DataFrame(still_failing).to_csv(
                os.path.join(OUT, "failed_facilities.csv"), index=False
            )
            print(f"\nInterrupted. Wrote {len(still_failing)} unfinished facilities to failed_facilities.csv")
        raise

    with open(os.path.join(OUT, "facilities.json"), "w") as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: facilities.json (with order info)")

    return facilities_by_place


RULES = {
    "NPDES": {
        "patterns": ["Table 1. Discharger Information"],
        "detect_npdes_pattern": True,
        "max_pages": 5,
    },
    "NOA": {
        "patterns": ["notice of applicability"],
        "patterns_case_sensitive": ["NOA"],
        "max_pages": 5,
    },
    "WDR": {
        "patterns": ["waste discharge requirements", "wdrs", "water recycling requirements", "information sheet"],
        "detect_npdes_pattern": True,
        "max_pages": 5,
    },
}


def extract_pdf_text(pdf_path: str, max_pages=5, lowercase=True) -> str:
    parts = []
    doc = None
    doc = fitz.open(pdf_path)
    cat_xref = doc.pdf_catalog()
    is_portfolio = doc.xref_get_key(cat_xref, "Collection")[0] != "null"
    if is_portfolio:
        for i in range(doc.embfile_count()):
            info = doc.embfile_info(i)
            if info.get("filename", "").lower().endswith(".pdf"):
                buf = doc.embfile_get(i)
                sub = fitz.open("pdf", buf)
                for j in range(min(len(sub), max_pages)):
                    parts.append(sub[j].get_text())
                sub.close()
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text() or ""
                if not text.strip():
                    fpage = doc[i]
                    tp = fpage.get_textpage_ocr()
                    text = fpage.get_text(textpage=tp)
                parts.append(text)
    if doc:
        doc.close()

    raw = " ".join(parts)
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[­​‌‍﻿]", "", raw)
    raw = re.sub(r"[  ᠎ -   　]", "", raw)
    return raw.lower() if lowercase else raw


def detect_text_from_pdf(pdf_path: str, text_searched: str, max_pages=5):
    """Detect if 'text_searched' is in the first 'max_pages' of the PDF at 'pdf_path'."""
    # Use the combined normalized text from the first pages for more robust matching
    combined = extract_pdf_text(pdf_path, max_pages)
    if not combined:
        return False
    text_searched_normalized = normalize_text(text_searched)
    if text_searched_normalized in combined:
        return True
    # try spaceless fallback
    combined_nospace = re.sub(r"\s+", "", combined)
    k_nospace = re.sub(r"\s+", "", text_searched_normalized)
    if k_nospace and k_nospace in combined_nospace:
        return True
    return False


def detect_npdes_pattern(pdf_path: str, max_pages=5) -> bool:
    """Detect flexible NPDES-like sentences in a PDF.
    Matches patterns like:
      "the following <...> subject to <...> set forth in this <...> order"
    """
    txt = extract_pdf_text(pdf_path, max_pages)
    if not txt:
        return False

    # tolerant regex: allow spaces, lines changes and some special chars between letters
    inner_sep = r"(?:[\s­​\-])*"

    def fuzzy(word: str) -> str:
        """Return a regex that matches `word` even if the extractor inserted
        whitespace, soft-hyphens, zero-width spaces or hyphens between letters.
        """
        parts = []
        for ch in word:
            # escape regex metacharacters
            parts.append(re.escape(ch) + inner_sep)
        return "".join(parts)
    # fuzzy tokens for the fixed keywords
    f_the = fuzzy("the")
    f_following = fuzzy("following")
    f_subject = fuzzy("subject")
    f_to = fuzzy("to")
    f_set = fuzzy("set")
    f_forth = fuzzy("forth")
    f_in = fuzzy("in")
    f_this = fuzzy("this")
    f_order = fuzzy("order")

    # allow up to 600 chars in captures (non-greedy), DOTALL so dot matches newlines
    pattern = re.compile(
        rf"{f_the}{f_following}(.{{1,600}}?){f_subject}{f_to}(.{{1,600}}?){f_set}{f_forth}{f_in}{f_this}(.{{1,600}}?){f_order}",
        flags=re.I | re.DOTALL,
    )
    return bool(pattern.search(txt))


def length_of_pdf(pdf_path: str) -> int:
    try:
        doc = fitz.open(pdf_path)
        cat_xref = doc.pdf_catalog()
        is_portfolio = doc.xref_get_key(cat_xref, "Collection")[0] != "null"
        if is_portfolio:
            total = 0
            for i in range(doc.embfile_count()):
                info = doc.embfile_info(i)
                if info.get("filename", "").lower().endswith(".pdf"):
                    sub = fitz.open("pdf", doc.embfile_get(i))
                    total += len(sub)
                    sub.close()
            doc.close()
            return total
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


_CAG_PERMIT_RE = re.compile(r"\bca\s*g\d+", re.IGNORECASE)


def _rule_matches(pdf_file, rule, max_pages_default):
    mp = rule.get("max_pages", max_pages_default)
    text = extract_pdf_text(pdf_file, mp)
    text_nospace = re.sub(r"\s+", "", text)
    pattern_hit = any(
        (normalize_text(p) in text) or (re.sub(r"\s+", "", normalize_text(p)) in text_nospace)
        for p in rule.get("patterns", [])
    )
    if not pattern_hit and rule.get("patterns_case_sensitive"):
        raw = extract_pdf_text(pdf_file, mp, lowercase=False)
        pattern_hit = any(p in raw for p in rule["patterns_case_sensitive"])
    fuzzy_hit = bool(rule.get("detect_npdes_pattern")) and detect_npdes_pattern(pdf_file, mp)
    return pattern_hit or fuzzy_hit


def detect_npdes(pdf_file: str, max_pages=5, min_length=10) -> str | None:
    """Return matched doc type ("NPDES", "NOA", "WDR") or None by applying RULES."""
    if length_of_pdf(pdf_file) < min_length:
        return None

    # Check NOA first — needed to gate the CAG short-circuit
    noa_text = extract_pdf_text(pdf_file, RULES["NOA"].get("max_pages", max_pages))
    has_noa = _rule_matches(pdf_file, RULES["NOA"], max_pages)
    has_cag = bool(_CAG_PERMIT_RE.search(noa_text))

    # Generic CAG order (no NOA): not facility-specific, skip
    if has_cag and not has_noa:
        return None

    if has_noa:
        return "NOA"

    for doc_type in ("NPDES", "WDR"):
        if _rule_matches(pdf_file, RULES[doc_type], max_pages):
            return doc_type

    return None


def detect_and_move_npdes_pdfs(facilities_by_place):
    print("\n STEP 3: Detecting and moving NPDES PDFs")
    permit_path = os.path.join(OUT, "permits")
    os.makedirs(permit_path, exist_ok=True)

    # Include PDFs already moved to npdes/ in previous runs
    npdes_pdfs = {f for f in os.listdir(permit_path) if f.endswith(".pdf")}
    non_npdes_pdfs = set()
    # PDF to Reg MeasureGrouping
    pdf_to_groups, group_to_pdfs = {}, {}
    group_to_rm_type = {}
    for place_id, entry in facilities_by_place.items():
        g = entry.get("reg_measure_id") or place_id
        group_to_rm_type[g] = entry.get("reg_measure_type", "")
        for pdf in entry.get("pdfs", []):
            pdf_to_groups.setdefault(pdf, set()).add(g)
            group_to_pdfs.setdefault(g, set()).add(pdf)
    pdf_signals = {}

    # Detect NPDES signals for every PDF in pdfs/, then move NPDES-positive files to npdes/.
    pdf_files = [f for f in os.listdir(pdfs_path) if f.endswith(".pdf")]

    single_pdf_files = {
        pdf
        for entry in facilities_by_place.values()
        for pdf in entry.get("pdfs", [])
        if len(entry.get("pdfs", [])) == 1
    }

    for filename in pdf_files:
        pdf_signals[filename] = detect_npdes(os.path.join(pdfs_path, filename))

    for filename in pdf_files:
        matched_type = pdf_signals[filename]
        src = os.path.join(pdfs_path, filename)
        stem = os.path.splitext(filename)[0]
        if matched_type and SKIP_RE.search(stem) and not KEEP_RE.search(stem):
            matched_type = None  # general-order/non-permit filename pattern
        if matched_type in ("WDR", "NPDES"):
            assoc_types = {group_to_rm_type.get(g, "") for g in pdf_to_groups.get(filename, set())}
            if assoc_types and all(t.startswith("ENROLLEE") for t in assoc_types):
                matched_type = None  # general order for enrolled facilities, not facility-specific
        elif matched_type == "NOA":
            groups = pdf_to_groups.get(filename, set())
            assoc_types = {group_to_rm_type.get(g, "") for g in groups}
            if len(groups) > 1 and assoc_types and all(t.startswith("ENROLLEE") for t in assoc_types):
                matched_type = None  # shared general order contains NOA language but not facility-specific
        if matched_type:
            os.rename(src, os.path.join(permit_path, filename))
            print(f"{matched_type} detected: {filename}")
            npdes_pdfs.add(filename)
        elif filename in single_pdf_files and length_of_pdf(src) >= 3:
            os.rename(src, os.path.join(permit_path, filename))
            print(f"single PDF, kept: {filename}")
            npdes_pdfs.add(filename)
        else:
            non_npdes_pdfs.add(filename)

    print(f"\nNPDES/NOA/WDR PDFs moved: {len(npdes_pdfs)}")
    print(f"Non-NPDES/NOA/WDR PDFs kept in pdfs folder: {len(non_npdes_pdfs)}")

    return npdes_pdfs, non_npdes_pdfs, pdf_signals


def create_site_data_csv(facilities_by_place, npdes_pdfs, pdf_signals):
    print("\n STEP 4: Creating site_data_relevant with relevant NPDES/WDR/NOA documents only")

    # Build (WDID, Facility Name) → {Agency, Region, Major/Minor, Order_No, NPDES No.} from
    # site_data_all.csv. Order_No is overridden with the scraped regulatory measure value when available.
    csv_path = os.path.join(OUT, "site_data_all.csv")
    xls_path = os.path.join(OUT, "other_pdfs", "Regualted_Facility_Report_Detail.xls")
    meta_keys = ("Agency", "Region", "Major/Minor", "Order_No", "NPDES No.")
    enrich = {}
    for enrich_path, sep in [(csv_path, ","), (xls_path, "\t")]:
        if not os.path.exists(enrich_path):
            continue
        df = pd.read_csv(enrich_path, sep=sep, dtype=str, encoding="latin-1", on_bad_lines="warn").fillna("")
        if "WDID" not in df.columns or "Facility Name" not in df.columns:
            continue
        for _, row in df.iterrows():
            key = (str(row["WDID"]).strip(), str(row["Facility Name"]).strip())
            if key not in enrich:
                cell = {col: row.get(col, "") for col in ("Agency", "Region", "Major/Minor")}
                cell["Order_No"] = row.get("Order No.", "")
                cell["NPDES No."] = row.get("NPDES No.", "")
                enrich[key] = cell
        break
    print(f"  Enrichment lookup: {len(enrich)} entries from site_data_all.csv")

    # Count distinct place_ids mapping to each NPDES PDF (for Shared_PDF flag)
    pdf_to_n_facilities = {}
    for entry in facilities_by_place.values():
        for pdf in entry.get("pdfs", []):
            if pdf in npdes_pdfs:
                pdf_to_n_facilities[pdf] = pdf_to_n_facilities.get(pdf, 0) + 1

    rows = []
    for place_id, entry in facilities_by_place.items():
        fac_url = facility_url(place_id)
        rm_type = entry.get("reg_measure_type")
        fac_name = entry.get("Facility Name", "")
        wdid = entry.get("WDID", "")
        meta = enrich.get((wdid, fac_name), {})
        facility_npdes_pdfs = [p for p in entry.get("pdfs", []) if p in npdes_pdfs]
        meta_dict = {key: meta.get(key, "") for key in meta_keys}
        if entry.get("order_no"):
            meta_dict["Order_No"] = entry["order_no"]
        for pdf in (facility_npdes_pdfs or [""]):
            rows.append(
                {
                    "Place ID": place_id.strip(),
                    "WDID": wdid.strip() if wdid else "",
                    "Facility Name": fac_name.strip(),
                    **meta_dict,
                    "Facility_URL": fac_url,
                    "Reg_Measure_ID": entry.get("reg_measure_id"),
                    "Reg_Measure_Type": rm_type,
                    "Addtl_Reg_Measure_ID": entry.get("addtl_reg_measure_id", ""),
                    "Addtl_WDID": entry.get("addtl_WDID", ""),
                    "PDF_File": pdf,
                    "Shared_PDF": ("Yes" if pdf_to_n_facilities.get(pdf, 0) > 1 else "No"),
                    "Total_PDFs_Available": entry.get("total_pdfs")
                }
            )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "site_data_relevant.csv"), index=False)
    print(f"Wrote {len(rows)} rows to site_data_relevant.csv")

    # Breakdown by Reg_Measure_Type (one row per unique Place ID)
    df_fac = df_out.drop_duplicates(subset="Place ID")
    print("\n  Reg_Measure_Type breakdown (all facilities):")
    for rmt, count in df_fac["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count}")

    # Same breakdown but only facilities with at least one non-empty PDF
    has_pdf = df_out[df_out["PDF_File"].notna() & (df_out["PDF_File"] != "")]["Place ID"].unique()
    df_pdf = df_fac[df_fac["Place ID"].isin(has_pdf)]
    print(f"\n  Reg_Measure_Type breakdown (facilities with ≥1 PDF, n={len(df_pdf)}):")
    for rmt, count in df_pdf["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count}")

    total_pdfs = df_fac["Total_PDFs_Available"].apply(pd.to_numeric, errors="coerce").sum()
    print(f"\n  Total_PDFs_Available (sum across facilities): {int(total_pdfs)}")

if __name__ == "__main__":

    # Sometimes it takes a few tries to get the facility results page to load / Excel to download
    # Try a few times or wait a couple of hours to let CIWQS stabilize if needed

    # sometimes helpful to clear chrome temp files
    # find /tmp -maxdepth 1 -name "chrome_user_data_*" -user daly -print
    # find /tmp -maxdepth 1 -name "chrome_user_data_*" -user daly -exec rm -rf {} +

    # program_urls = run_ciwqs_search()
    # facilities = collect_facility_page_urls(program_urls)
    # To restart from a checkpoint, replace the step(s) above with:
    with open(os.path.join(OUT, "facilities.json")) as f:
        facilities = json.load(f)
    # facilities = download_facility_page_pdfs(facilities)
    npdes_pdfs, _, pdf_signals = detect_and_move_npdes_pdfs(facilities)
    create_site_data_csv(facilities, npdes_pdfs, pdf_signals)
