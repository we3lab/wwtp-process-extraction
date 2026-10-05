from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
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
from helpers.utils import normalize_text, is_general_order, COLLECTIVE_AGENCY_RE, OUTPUT_DIR, SITE_DATA_RELEVANT_CSV
# Modified using Claude 4.5

# Link to Interactive Regulated Facilities Report
CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"
REGULATED_FACILITY_REPORT_URL = f"{CIWQS_SERVLET}?inCommand=reset&reportName=RegulatedFacility"

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
FACILITY_CIWQS_COLUMNS = ["WDID", "Facility Name", "NPDES No."]
XP_GRID = "//table[contains(@class,'ciwqsReportDataTable')]"
PDF_XPATH = "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
# Older orders are occasionally filed as .doc
ATTACHMENT_XPATH = "//a[contains(@href, 'PublicAttachmentRetriever')]"

OTHER_PDFS_DIR = OUTPUT_DIR / "other_pdfs"
PERMITS_DIR = OUTPUT_DIR / "permits"
# Cache on size+mtime so a re-downloaded or edited file is still re-scanned.
SIGNAL_CACHE_PATH = OUTPUT_DIR / "pdf_signal_cache.json"

# Each scrape also lands a dated copy under output/site_data/<RUN_DATE>/
RUN_DATE = os.environ.get("RUN_DATE") or datetime.now().strftime("%Y-%m-%d")

# Unset (default) = today's active permit.
# Set AS_OF=2021-06-01 to pick the order in force on that date instead
AS_OF = pd.Timestamp(os.environ["AS_OF"]) if os.environ.get("AS_OF") else None
# A retrospective run belongs in its as-of folder, not today's.
SNAPSHOT_DATE = os.environ.get("AS_OF") or RUN_DATE
SNAPSHOT_DIR = OUTPUT_DIR / "site_data" / SNAPSHOT_DATE

# Each AS_OF run writes just that year's permits, but step3 and step5 --all_facilities both read the
# one top-level site_data_relevant.csv, so after writing its dated snapshot the top-level file is
# rewritten as the union of all snapshots: every document appears exactly once, and text/LLM
# extraction run once per document rather than once per document per year.
# A facility can hold several orders across snapshots, and one order can carry several PDFs;
# general orders (e.g. 2014-0153-DWQ) are shared by hundreds of facilities. Reg_Measure_ID is
# the stable per-order id, with Order_No as fallback where it is blank.
SNAPSHOT_UNION_KEY = ["Place ID", "order_key", "PDF_File"]

# The snapshot the top-level output/facilities.json should represent between runs
BASE_SNAPSHOT_DATE = os.environ.get("BASE_SNAPSHOT_DATE", "2026-06-01")

# Standalone Chrome + matching driver, under the user's home rather than a hardcoded username.
# Override with CHROME_BIN / CHROMEDRIVER_BIN if they live elsewhere.
CHROME_BIN = Path(os.environ.get("CHROME_BIN") or Path.home() / "bin/chrome/chrome-linux64/chrome")
CHROMEDRIVER_BIN = Path(os.environ.get("CHROMEDRIVER_BIN")
                        or Path.home() / "bin/chrome/chromedriver-linux64/chromedriver")

# PDF filenames matching this regex are skipped on the order page.
FILENAME_SEP = r"[ ._-]"  # - must be last to avoid range interpretation
SKIP_BASE_KW = "rpts|rowd|memo|nov|map|rwd|gwmp|mgo"  # match only with separators (e.g. "_memo_")
SKIP_PHRASE = (
    "report|financial|response to|rate study|ratestudy|study|"
    "addendum|registration|adoption|"
    "letter|covltr|cover_l|cover l|volumetric|"
    "form200|form 200|management zone|management_zone|management plan"
)  # skip if anywhere in filename
SKIP_RE = re.compile(rf"^(?:{SKIP_BASE_KW}){FILENAME_SEP}|{FILENAME_SEP}(?:{SKIP_BASE_KW}){FILENAME_SEP}|{SKIP_PHRASE}", re.IGNORECASE)

# skip these UNLESS keep_re is present too
CONTINGENT_SKIP_PHRASE = "amendment|mrp"
CONTINGENT_SKIP_RE = re.compile(CONTINGENT_SKIP_PHRASE, re.IGNORECASE)

# keep these, overriding contingent skip
KEEP_RE = re.compile(r"(?<![a-zA-Z])(noa|wdrs?|order|npdes)(?![a-zA-Z])", re.IGNORECASE)  # always keep NOA/WDR/NPDES files

RULES = {
    "NPDES": {
        "patterns": ["Table 1. Discharger Information"],
        "detect_npdes_pattern": True,
    },
    "NOA": {
        "patterns": ["notice of applicability"],
        "patterns_case_sensitive": ["NOA"],
    },
    "WDR": {
        "patterns": ["waste discharge requirements", "wdrs", "water recycling requirements", "information sheet"],
        "detect_npdes_pattern": True,
    },
}
MAX_SCAN_PAGES = 5
MIN_PDF_PAGES = 10

# tolerant regex: allow spaces, lines changes and some special chars between letters
FUZZY_INNER_SEP = r"(?:[\s­​\-])*"
# "the following <...> subject to <...> set forth in this <...> order", each letter fuzzy;
# allow up to 600 chars in captures (non-greedy), DOTALL so dot matches newlines
NPDES_SENTENCE_RE = re.compile(
    r"(.{1,600}?)".join(
        "".join(re.escape(ch) + FUZZY_INNER_SEP for ch in phrase)
        for phrase in ("thefollowing", "subjectto", "setforthinthis", "order")
    ),
    flags=re.I | re.DOTALL,
)
CAG_PERMIT_RE = re.compile(r"\bca\s*g\d+", re.IGNORECASE)

# Per-facility logging. Each worker buffers
# its lines in thread-local storage; facility_log flushes them in one locked write.
worker_log = threading.local()


def snapshot(filename):
    """Copy a just-written top-level output into the dated batch folder.

    No sidecar metadata: the folder name carries the date and the files carry their own row
    counts, so a run_info.json would only be derived state that can drift out of sync.
    """
    src = OUTPUT_DIR / filename
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    shutil.copy2(src, os.path.join(SNAPSHOT_DIR, filename))
    print(f"  snapshot -> site_data/{SNAPSHOT_DATE}/{filename}")


def restore_base_facilities():
    """Leave the top-level facilities.json holding the base snapshot, not this run's.

    A retrospective run overwrites output/facilities.json with the orders in force at its
    as-of date, so whichever year ran last would otherwise masquerade as current -- and this
    script reads that same file as its starting facility list. The dated copies under
    site_data/ are the durable per-year record; the top level always means "now".
    """
    src = OUTPUT_DIR / "site_data" / BASE_SNAPSHOT_DATE / "facilities.json"
    dst = OUTPUT_DIR / "facilities.json"
    if SNAPSHOT_DATE == BASE_SNAPSHOT_DATE or not os.path.exists(src):
        return
    shutil.copy2(src, dst)
    print(f"Top-level facilities.json restored to the {BASE_SNAPSHOT_DATE} snapshot")


def say(msg):
    """Buffer a line for the current worker, or print directly outside a worker."""
    buf = getattr(worker_log, "lines", None)
    if buf is None:
        print(msg)
    else:
        buf.append(msg)


@contextmanager
def facility_log(lock):
    worker_log.lines = []
    try:
        yield
    finally:
        lines = worker_log.lines
        worker_log.lines = None
        if lines:
            with lock:
                print("\n".join(lines), flush=True)


def repair_href(href):
    """Undo BeautifulSoup's entity decoding of '&regMeasID='.

    '&reg' is a valid named entity, so html.parser turns '...&regMeasID=436745' into
    '...\u00aeMeasID=436745'. Two things then break silently: requesting that URL returns
    HTTP 200 with an empty body, and parse_qs finds no regMeasID, which makes
    resolve_download_url fall back to the first rmAttachmentPopup link on the page --
    potentially a different order's attachments.
    """
    return href.replace("\u00aeMeasID", "&regMeasID") if href else href


def abs_url(href):
    return urljoin(f"{CIWQS_ROOT}/ciwqs/readOnly/", repair_href(href)) if href else href


def cell_text(cells, i):
    return cells[i].get_text(strip=True) if i is not None and 0 <= i < len(cells) else ""


def cell_href(cells, i):
    if i is None or not (0 <= i < len(cells)):
        return ""
    a = cells[i].find("a", href=True)
    return abs_url(a["href"]) if a else ""


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
    """Retry on transient transport errors; HTTPError and the rest propagate immediately.

    CIWQS drops the connection mid-response on larger payloads (SSLEOFError / connection
    reset), which is retryable and distinct from a real HTTP failure -- retrying a 404 would
    just waste four round trips.
    """
    transient = (requests.exceptions.Timeout,
                 requests.exceptions.SSLError,
                 requests.exceptions.ConnectionError)
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.request(method, url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except transient as exc:
            print(f"[requests] {method.upper()} {type(exc).__name__} ({attempt}/{max_attempts}): {url[:90]}")
            if attempt == max_attempts:
                raise
            time.sleep(3 * attempt)


def new_chrome_driver(download_dir):
    """Chrome for facility report (after requests submits the CIWQS search)."""
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": os.path.abspath(download_dir),
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
    options.binary_location = str(CHROME_BIN)
    service = Service(str(CHROMEDRIVER_BIN))
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
def open_in_new_tab(driver, url, main_window):
    """Open `url` in a new tab, yield, then close the tab and switch back to main."""
    driver.execute_script(f"window.open('{url}', '_blank');")
    time.sleep(1)
    new_window = next(h for h in driver.window_handles if h != main_window)
    driver.switch_to.window(new_window)
    try:
        yield
    finally:
        try:
            driver.close()
        except Exception:
            pass
        driver.switch_to.window(main_window)


def resolve_download_url(href, soup):
    """Map an order link to its attachment page, matching on regMeasID.

    Both sides must be run through repair_href: `href` arrives already repaired via abs_url,
    but the candidates come straight off the soup and still carry the '\u00aeMeasID=' form, so
    comparing a repaired id against an unrepaired href never matches. Getting that wrong
    silently returns the order page instead of the attachments page, which reads as
    "Found 0 PDFs on page".
    """
    reg_id_val = parse_qs(urlparse(repair_href(href)).query).get("regMeasID", [None])[0]

    def is_attachment_link(candidate):
        if not candidate:
            return False
        fixed = repair_href(candidate)
        return "rmAttachmentPopup" in fixed and (
            not reg_id_val or f"regMeasID={reg_id_val}" in fixed)

    attach_tag = soup.find("a", href=is_attachment_link)
    return abs_url(attach_tag["href"]) if attach_tag else href


def find_best_order(driver, fac_url, main_window):
    """Navigate to facility page, parse HTML, and return the governing NPDES order.

    Governing = active today, or in force at AS_OF when that env var is set.

    Returns: (order_url, reg_measure_type, wdid, eff, addtl_orders, order_no)
      order_url may be None if best order has no clickable link.
      addtl_orders is a list of (url, wdid, eff) for additional NPDES PERMIT orders with valid links.
      order_no is the Order No. text from the CIWQS table for the selected regulatory measure.
    """
    # Navigate to facility page
    with open_in_new_tab(driver, fac_url, main_window):
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

            # Process data rows: collect (rank, -eff, href, rm_type, eff, wdid) tuples; min() picks the best.
            # href may be None for rows without a clickable link — still included so we can store order metadata.
            candidates = []
            for data_row in all_rows[hdr_idx + 1:]:
                cells = data_row.find_all("td")
                if not cells:
                    continue
                status = cell_text(cells, col_index.get("Status")).lower()
                eff = pd.to_datetime(cell_text(cells, col_index.get("Effective Date")), errors="coerce")
                if AS_OF is None:
                    # default: today's governing permit, as before
                    if status != "active":
                        continue
                else:
                    # retrospective: accept Historical/Terminated rows too and let the
                    # effective-date cut decide. Ranking below already prefers permit type
                    # then newest effective date, which is the governing order at AS_OF.
                    if status == "never active":
                        continue
                    if pd.isna(eff) or eff > AS_OF:
                        continue

                # Apply validation checks
                rm_type = cell_text(cells, col_index.get("Reg Measure Type")).upper()
                if rm_type not in TYPE_RANK:
                    continue
                if rm_type == "WDR" and cell_text(cells, col_index.get("Program")).upper() not in PROGRAMS["WDR"]:
                    continue

                order_idx = col_index.get("Order No.", -1)
                order_cell = cells[order_idx] if 0 <= order_idx < len(cells) else None
                a_tag = order_cell.find("a", href=True) if order_cell else None
                href = abs_url(a_tag["href"]) if a_tag else None
                order_no = order_cell.get_text(strip=True) if order_cell else ""
                if pd.isna(eff):
                    continue

                candidates.append((TYPE_RANK[rm_type], -eff.value, href, rm_type, eff,
                                   cell_text(cells, col_index.get("WDID")), order_no))

            if candidates:
                rank, _, href, rm_type, eff, wdid, order_no = min(candidates, key=lambda c: (c[0], c[1], 0 if c[2] else 1))
                say(f"  Best order: {rm_type}, rank={rank}, effective={eff.date()}, order={order_no}, link={'yes' if href else 'no'}")

                download_url = resolve_download_url(href, soup) if href else None

                # Collect additional NPDES PERMIT orders (rank=0) with valid links, excluding primary
                addtl_orders = []
                for c_rank, _, c_href, _, c_eff, c_wdid, _ in candidates:
                    if c_rank != 0 or not c_href or c_href == href:
                        continue
                    addtl_orders.append((resolve_download_url(c_href, soup), c_wdid, c_eff))

                return download_url, rm_type, wdid, eff, addtl_orders, order_no

    return None, None, None, None, [], ""


def download_order_pdfs(driver, order_url, worker_dir, main_window):
    """Download PDFs from an order attachment page into worker_dir, then move them to other_pdfs/.

    Returns (downloaded_pdfs, missed_pdfs, total_on_page).
    """
    downloaded_pdfs = []
    missed_pdfs = []
    with open_in_new_tab(driver, order_url, main_window):
        try:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, ATTACHMENT_XPATH)))
        except TimeoutException:
            pass
        pdf_documents = driver.find_elements(By.XPATH, PDF_XPATH)
        all_attachments = driver.find_elements(By.XPATH, ATTACHMENT_XPATH)
        # total_on_page means "documents present", so an unusable format still counts as found
        total_on_page = max(len(all_attachments), len(pdf_documents))
        say(f"  Found {len(pdf_documents)} PDFs on page"
            + (f" ({len(all_attachments)} attachments total)"
               if len(all_attachments) != len(pdf_documents) else ""))
        if all_attachments and not pdf_documents:
            others = [a.text for a in all_attachments if a.text]
            say(f"  ! attachments present but none are PDF, skipping: {others}")

        for pdf_element in pdf_documents:
            try:
                pdf_name = pdf_element.text
                if total_on_page > 1:
                    if SKIP_RE.search(pdf_name):
                        continue
                    if CONTINGENT_SKIP_RE.search(pdf_name) and not KEEP_RE.search(pdf_name):
                        continue


                # TODO: can remove if not re-running old dates
                if any(os.path.exists(os.path.join(d, pdf_name)) for d in (OTHER_PDFS_DIR, PERMITS_DIR)):
                    say(f"        Already exists, skipping: {pdf_name}")
                    downloaded_pdfs.append(pdf_name)
                    continue

                say(f"        Downloading: {pdf_name}")
                before = {f for f in os.listdir(worker_dir) if f.lower().endswith(".pdf")}
                pdf_element.click()

                end = time.time() + 90
                new_file = None
                while time.time() < end:
                    current = {f for f in os.listdir(worker_dir) if f.lower().endswith(".pdf")}
                    new = current - before
                    if new:
                        newest = max(new, key=lambda f: os.path.getctime(os.path.join(worker_dir, f)))
                        if file_stable(os.path.join(worker_dir, newest)):
                            new_file = newest
                            break
                    time.sleep(0.5)

                if not new_file:
                    say(f"        X Timed out waiting for: {pdf_name}")
                    missed_pdfs.append(pdf_name)
                    continue
                downloaded_pdfs.append(new_file)
            except Exception as e:
                say(f"        X Download failed: {pdf_name} — {e}")
                missed_pdfs.append(pdf_name)

    for fname in downloaded_pdfs:
        src, dst = os.path.join(worker_dir, fname), os.path.join(OTHER_PDFS_DIR, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    return downloaded_pdfs, missed_pdfs, total_on_page


def file_stable(path):
    try:
        s = os.path.getsize(path)
        time.sleep(0.5)
        return s > 0 and os.path.getsize(path) == s
    except OSError:
        return False


def load_ciwqs_table(driver, url, label="url"):
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


def set_page_all(driver):
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
    r = retry_request(ciwqs, 'GET', REGULATED_FACILITY_REPORT_URL)
    search_soup = BeautifulSoup(r.text, "html.parser")
    hidden_inputs = {
            i["name"]: i.get("value", "")
            for i in search_soup.find_all("input", type="hidden")
            if i.get("name")
        }
    csrf = hidden_inputs.get("OWASP_CSRFTOKEN", "")
    in_status = select_value(
        search_soup, "inStatus", CIWQS_RELATED_PERMIT_STATUS, required_label="Related Permit Status"
    )
    # Multi-select post → total drilldown URL (used for Excel export only).
    # Multi-select yields a grouped-by-agency view; re-submit per-program for the flat table.
    filter_kwargs = dict(facility_type=CIWQS_FACILITY_TYPE, waste_type=CIWQS_WASTE_TYPE, status=in_status)
    print("[requests] Submitting filters")
    resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
                        data=ciwqs_post_data(hidden_inputs, search_soup, list(PROGRAMS), **filter_kwargs))
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
                                data=ciwqs_post_data(hidden_inputs, search_soup, [prog], **filter_kwargs))
        prog_url = extract_drilldown_url(BeautifulSoup(prog_resp.text, "html.parser"))
        if prog_url:
            program_urls.append((prog, prog_url))
            print(f"[requests] {prog} facility URL: {prog_url}")
        else:
            print(f"[requests] Warning: no drilldown URL found for {prog}")

    # Fresh Chrome profile (empty cookies) — only touches the Excel summary drilldown URL (total_url).
    driver = new_chrome_driver(OTHER_PDFS_DIR)
    driver.set_page_load_timeout(WAIT_TIME)
    load_ciwqs_table(driver, total_url, "Facility page")
    print("Detail page loaded for Excel export")
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
        candidates = [f for d in (OTHER_PDFS_DIR, OUTPUT_DIR)
                    for f in glob.glob(os.path.join(d, "*.xls*"))
                    if not f.lower().endswith(".crdownload")]
        if candidates or any(
            glob.glob(os.path.join(root, "*.crdownload")) for root in (OTHER_PDFS_DIR, OUTPUT_DIR)
        ):
            had_download_activity = True

        stable_flags = [file_stable(path) for path in candidates]
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
        raise SystemExit

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

    df_deduplicated.to_csv(OUTPUT_DIR / "site_data_all.csv", index=False)
    print(f"Saved {len(df_deduplicated)} rows to site_data_all.csv")
    snapshot("site_data_all.csv")

    driver.quit()
    return program_urls


def collect_facility_page_urls(program_urls):
    print("\n STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")

    facilities_by_place = {}  # place_id -> {"facilities": [dict keyed by FACILITY_CIWQS_COLUMNS, ...]}
    name_to_place_id = {}  # facility name -> place_id, collected pre-filter for reconciliation

    # Navigate per-program RegulatedFacilityDetail URLs. New Chrome profile per URL so no stale CIWQS cookies.
    col = {}  # CIWQS header text -> column index (detected on first program)

    for prog, prog_url in program_urls:
        print(f"\n--- {prog}: {prog_url}")
        # Try a new driver every time, since cookies from original search make tables too slow to load
        driver = new_chrome_driver(OTHER_PDFS_DIR)
        driver.set_page_load_timeout(WAIT_TIME)
        load_ciwqs_table(driver, prog_url, prog)
        set_page_all(driver)

        page_soup = BeautifulSoup(driver.page_source, "html.parser")
        data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
        bs_rows_prog = data_table.find_all("tr") if data_table else []
        print(f"{prog}: {len(bs_rows_prog)} <tr> after ALL page size")

        if not col:
            for header_row in bs_rows_prog:
                texts = [td.get_text(strip=True) for td in header_row.find_all("td")]
                if "Order No." in texts or "Facility Name" in texts:
                    col.update({t: i for i, t in enumerate(texts) if t})
                    break
            missing = [c for c in FACILITY_CIWQS_COLUMNS if c not in col]
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

                status = cell_text(cells, col.get("Regulatory Measure Status")).upper()
                plc_type = cell_text(cells, col.get("Place/Project Type")).upper()

                if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
                    continue
                if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
                    continue

                # Collect facility name -> place_id unconditionally for reconciliation below
                place_id = parse_qs(
                    urlparse(cell_href(cells, col.get("Facility Name"))).query
                ).get("placeID", [None])[0]
                raw_name = cell_text(cells, col.get("Facility Name"))
                if place_id and raw_name:
                    name_to_place_id[raw_name] = place_id

                if not place_id:
                    continue

                facility = {name: cell_text(cells, col.get(name)) for name in FACILITY_CIWQS_COLUMNS}
                entry = facilities_by_place.setdefault(place_id, {"facilities": []})
                if not any(f["Facility Name"] == facility["Facility Name"] for f in entry["facilities"]):
                    entry["facilities"].append(facility)
            except Exception as e:
                print(f"Row parse error: {e}")
                continue

        driver.quit()

    # Reconcile against site_data_all.csv to catch any facilities missed by per-program scrapes
    npdes_df = pd.read_csv(OUTPUT_DIR / "site_data_all.csv", dtype=str).fillna("")
    scraped_names = {
        f["Facility Name"]
        for entry in facilities_by_place.values()
        for f in entry["facilities"]
    }
    added = 0
    for _, row in npdes_df.iterrows():
        fac_name = row["Facility Name"].strip()
        if not fac_name or fac_name in scraped_names:
            continue
        place_id = name_to_place_id.get(fac_name)
        if place_id and place_id not in facilities_by_place:
            facilities_by_place[place_id] = {
                "facilities": [{
                    "WDID": row["WDID"].strip(),
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

    with open(OUTPUT_DIR / 'facilities.json', 'w') as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: {len(facilities_by_place)} facilities → facilities.json")
    snapshot("facilities.json")

    return facilities_by_place

def needs_retry(entry):
    # Unprocessed (exception during process_facility left "facilities" key intact)
    if "facilities" in entry:
        return True
    # Deliberately skipped (no link or pre-2004) — not a failure, never retry
    if entry.get("pdf_skip_reason"):
        return False
    # A reg measure was clicked and 0 PDFs came back. Usually transient (page slow to
    # render), but some attachment pages really are empty, so this is retried a bounded
    # number of times and then marked pdf_skip_reason above.
    if entry.get("reg_measure_type") and not entry.get("pdfs") and not entry.get("total_pdfs"):
        return True
    return False


def download_facility_page_pdfs(facilities_by_place, max_workers=24):
    # UPDATE max_workers to be higher if running on server
    print("\n STEP 2: Visiting facility pages and downloading PDFs")

    reg_id_to_info = {}
    addtl_reg_id_to_pdfs = {}
    lock = threading.Lock()

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
        say(f"[{idx}/{total}] {fac_name}")
        try:
            order_url, rm_type, wdid, eff, addtl_orders, order_no = find_best_order(driver, fac_url, main_window)
            if rm_type is None:
                say("  X No suitable active NPDES order found")
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
            if not order_url or eff.year < 2004:
                reason = "pre-2004" if eff.year < 2004 else "no link"
                say(f"  Skipping PDFs ({reason}): {rm_type}, eff={eff.date()}")
                entry.update({"Facility Name": fac_name, "WDID": wdid, "pdfs": [],
                              "total_pdfs": 0, "reg_measure_id": reg_id,
                              "reg_measure_type": rm_type, "order_no": order_no,
                              "pdf_skip_reason": reason})
                entry.pop("facilities", None)
                return

            with lock:
                if reg_id and reg_id in reg_id_to_info:
                    say(f"  Dedup: reusing already-processed order {reg_id}")
                    entry.update(reg_id_to_info[reg_id])
                    entry["Facility Name"] = fac_name
                    entry["WDID"] = wdid
                    entry.pop("facilities", None)
                    return

            downloaded_pdfs, missed_pdfs, total_pdfs = download_order_pdfs(
                driver, order_url, worker_dir, main_window
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
                    if addtl_reg_id in addtl_reg_id_to_pdfs:
                        extra_pdfs = addtl_reg_id_to_pdfs[addtl_reg_id]
                    elif addtl_reg_id in reg_id_to_info:
                        extra_pdfs = reg_id_to_info[addtl_reg_id].get("pdfs", [])
                    else:
                        extra_pdfs = None
                if extra_pdfs is None:
                    extra_pdfs, extra_missed, _ = download_order_pdfs(
                        driver, addtl_url, worker_dir, main_window
                    )
                    info["missed_pdfs"].extend(extra_missed)
                    with lock:
                        if addtl_reg_id:
                            addtl_reg_id_to_pdfs[addtl_reg_id] = extra_pdfs
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
            say(f"  X {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
            shutil.rmtree(worker_dir, ignore_errors=True)

    def logged_facility(args):
        """Buffer this worker's lines and flush them as one block, so a facility's output
        is contiguous instead of interleaved with the other workers."""
        with facility_log(lock):
            return process_facility(args)

    # loop mutates each entry in facilities_by_place in-place to add 'pdfs', 'total_pdfs', 'reg_measure_id', 'reg_measure_type'
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(logged_facility, enumerate(items, 1))

    retry_count = 0
    try:
        while True:
            retry_items = [
                (place_id, entry)
                for place_id, entry in facilities_by_place.items()
                if needs_retry(entry)
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
                executor.map(logged_facility, [(i + 1, item) for i, item in enumerate(retry_items)])
    except KeyboardInterrupt:
        still_failing = [
            {"place_id": place_id}
            for place_id, entry in facilities_by_place.items()
            if needs_retry(entry)
        ]
        if still_failing:
            pd.DataFrame(still_failing).to_csv(
                OUTPUT_DIR / "failed_facilities.csv", index=False
            )
            print(f"\nInterrupted. Wrote {len(still_failing)} unfinished facilities to failed_facilities.csv")
        raise

    with open(OUTPUT_DIR / "facilities.json", "w") as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print("Checkpoint saved: facilities.json (with order info)")
    snapshot("facilities.json")

    return facilities_by_place


def extract_pdf_text(pdf_path: str) -> str:
    """Text of the first MAX_SCAN_PAGES pages (per sub-file for a PDF Portfolio), original case."""
    parts = []
    doc = fitz.open(pdf_path)
    cat_xref = doc.pdf_catalog()
    is_portfolio = doc.xref_get_key(cat_xref, "Collection")[0] != "null"
    if is_portfolio:
        for i in range(doc.embfile_count()):
            info = doc.embfile_info(i)
            if info.get("filename", "").lower().endswith(".pdf"):
                buf = doc.embfile_get(i)
                sub = fitz.open("pdf", buf)
                for j in range(min(len(sub), MAX_SCAN_PAGES)):
                    parts.append(sub[j].get_text())
                sub.close()
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:MAX_SCAN_PAGES]):
                text = page.extract_text() or ""
                if not text.strip():
                    fitz_page = doc[i]
                    ocr_textpage = fitz_page.get_textpage_ocr()
                    text = fitz_page.get_text(textpage=ocr_textpage)
                parts.append(text)
    doc.close()

    raw = " ".join(parts)
    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[­​‌‍﻿]", "", raw)
    raw = re.sub(r"[  ᠎ -   　]", "", raw)
    return raw


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


def rule_matches(rule, text, raw_text):
    """True if a RULES entry matches. text is the lowercased raw_text."""
    text_nospace = re.sub(r"\s+", "", text)
    pattern_hit = any(
        (normalize_text(p) in text) or (re.sub(r"\s+", "", normalize_text(p)) in text_nospace)
        for p in rule.get("patterns", [])
    )
    if not pattern_hit and rule.get("patterns_case_sensitive"):
        pattern_hit = any(p in raw_text for p in rule["patterns_case_sensitive"])
    # Flexible NPDES-like sentence: "the following <...> subject to <...> set forth in this <...> order"
    fuzzy_hit = bool(rule.get("detect_npdes_pattern")) and bool(NPDES_SENTENCE_RE.search(text))
    return pattern_hit or fuzzy_hit


def detect_npdes(pdf_file: str) -> str | None:
    """Return matched doc type ("NPDES", "NOA", "WDR") or None by applying RULES."""
    if length_of_pdf(pdf_file) < MIN_PDF_PAGES:
        return None

    raw_text = extract_pdf_text(pdf_file)
    text = raw_text.lower()

    # Check NOA first — needed to gate the CAG short-circuit
    has_noa = rule_matches(RULES["NOA"], text, raw_text)
    has_cag = bool(CAG_PERMIT_RE.search(text))

    # Statewide general order: CAG_PERMIT_RE only catches CAG-numbered NPDES general permits,
    # and a WDR general order has no CAG number. Test before the NOA check — the order describes
    # the NOA process in its body prose, so has_noa is True and would otherwise rescue it.
    if is_general_order(text):
        return None

    # Generic CAG order (no NOA): not facility-specific, skip
    if has_cag and not has_noa:
        return None

    if has_noa:
        return "NOA"

    for doc_type in ("NPDES", "WDR"):
        if rule_matches(RULES[doc_type], text, raw_text):
            return doc_type

    return None


def save_signal_cache(cache):
    with open(SIGNAL_CACHE_PATH, "w") as f:
        json.dump(cache, f, sort_keys=True, indent=0)


def signal_cache_entry(cache, filename, path):
    """Entry for this file, reset if the file changed since it was cached."""
    st = os.stat(path)
    fingerprint = f"{st.st_size}:{st.st_mtime_ns}"
    entry = cache.get(filename)
    if not entry or entry.get("fp") != fingerprint:
        entry = {"fp": fingerprint}
        cache[filename] = entry
    return entry


def cached_length_of_pdf(cache, filename, path):
    entry = signal_cache_entry(cache, filename, path)
    if "pages" not in entry:
        entry["pages"] = length_of_pdf(path)
    return entry["pages"]


def detect_and_move_npdes_pdfs(facilities_by_place):
    print("\n STEP 3: Detecting and moving NPDES PDFs")

    # Include PDFs already moved to permits/ in previous runs
    npdes_pdfs = {f for f in os.listdir(PERMITS_DIR) if f.endswith(".pdf")}
    non_npdes_pdfs = set()
    # PDF -> reg measure groups (reg_measure_id, or place_id when there is none)
    pdf_to_groups = {}
    group_to_rm_type = {}
    pdf_to_places = {}
    for place_id, entry in facilities_by_place.items():
        group = entry.get("reg_measure_id") or place_id
        group_to_rm_type[group] = entry.get("reg_measure_type", "")
        for pdf in entry.get("pdfs", []):
            pdf_to_groups.setdefault(pdf, set()).add(group)
            pdf_to_places.setdefault(pdf, set()).add(place_id)
    pdf_signals = {}

    # Detect NPDES signals for every PDF in other_pdfs/, then move NPDES-positive files to permits/.
    pdf_files = [f for f in os.listdir(OTHER_PDFS_DIR) if f.endswith(".pdf")]

    single_pdf_files = {
        pdf
        for entry in facilities_by_place.values()
        for pdf in entry.get("pdfs", [])
        if len(entry.get("pdfs", [])) == 1
    }

    with open(SIGNAL_CACHE_PATH) as f:
        signal_cache = json.load(f)
    reused = 0

    for filename in pdf_files:
        path = os.path.join(OTHER_PDFS_DIR, filename)
        # a surviving "signal" is a cache hit
        entry = signal_cache_entry(signal_cache, filename, path)
        if "signal" in entry:
            reused += 1
        else:
            entry["signal"] = detect_npdes(path)
        pdf_signals[filename] = entry["signal"]
    save_signal_cache(signal_cache)
    print(f"  Scanned {len(pdf_files)} PDFs ({reused} reused from cache, "
          f"{len(pdf_files) - reused} newly scanned)")

    for filename in pdf_files:
        matched_type = pdf_signals[filename]
        src = os.path.join(OTHER_PDFS_DIR, filename)
        stem = os.path.splitext(filename)[0]
        if matched_type and SKIP_RE.search(stem) and not KEEP_RE.search(stem):
            matched_type = None  # general-order/non-permit filename pattern
        if matched_type in ("WDR", "NPDES"):
            assoc_types = {group_to_rm_type.get(group, "") for group in pdf_to_groups.get(filename, set())}
            if assoc_types and all(t.startswith("ENROLLEE") for t in assoc_types):
                matched_type = None  # general order for enrolled facilities, not facility-specific
        elif matched_type == "NOA":
            # Count facilities, not reg-measure groups: enrollees under one general order share a
            # single reg_measure_id, so the group count is 1 no matter how many plants point at it.
            places = pdf_to_places.get(filename, set())
            assoc_types = {group_to_rm_type.get(group, "") for group in pdf_to_groups.get(filename, set())}
            if len(places) > 1 and assoc_types and all(t.startswith("ENROLLEE") for t in assoc_types):
                matched_type = None  # shared general order contains NOA language but not facility-specific
        if matched_type:
            os.rename(src, os.path.join(PERMITS_DIR, filename))
            print(f"{matched_type} detected: {filename}")
            npdes_pdfs.add(filename)
        elif filename in single_pdf_files and cached_length_of_pdf(signal_cache, filename, src) >= 3:
            os.rename(src, os.path.join(PERMITS_DIR, filename))
            print(f"single PDF, kept: {filename}")
            npdes_pdfs.add(filename)
        else:
            non_npdes_pdfs.add(filename)

    save_signal_cache(signal_cache)
    print(f"\nNPDES/NOA/WDR PDFs moved: {len(npdes_pdfs)}")
    print(f"Non-NPDES/NOA/WDR PDFs kept in pdfs folder: {len(non_npdes_pdfs)}")

    return npdes_pdfs


def create_site_data_csv(facilities_by_place, npdes_pdfs):
    print("\n STEP 4: Creating site_data_relevant with relevant NPDES/WDR/NOA documents only")

    # Build (WDID, Facility Name) → {Agency, Region, Major/Minor, Order_No, NPDES No.} from
    # site_data_all.csv. Order_No is overridden with the scraped regulatory measure value when available.
    csv_path = OUTPUT_DIR / "site_data_all.csv"
    xls_path = OTHER_PDFS_DIR / "Regualted_Facility_Report_Detail.xls"
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
    skipped_collective = []
    for place_id, entry in facilities_by_place.items():
        fac_url = facility_url(place_id)
        rm_type = entry.get("reg_measure_type")
        fac_name = entry.get("Facility Name", "")
        wdid = entry.get("WDID", "")
        meta = enrich.get((wdid, fac_name), {})
        # Drop watershed permit "agencies"
        if COLLECTIVE_AGENCY_RE.search(meta.get("Agency", "")):
            skipped_collective.append(f"{place_id} {fac_name}")
            continue
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

    if skipped_collective:
        print(f"  Collective-permittee places dropped (not facilities): {len(skipped_collective)}")
        for label in skipped_collective:
            print(f"    {label}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(SITE_DATA_RELEVANT_CSV, index=False)
    print(f"Wrote {len(rows)} rows to site_data_relevant.csv")
    snapshot("site_data_relevant.csv")

    # Breakdown by Reg_Measure_Type (one row per unique Place ID)
    df_fac = df_out.drop_duplicates(subset="Place ID")
    print("\n  Reg_Measure_Type breakdown (all facilities):")
    for rmt, count in df_fac["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count}")

    # Same breakdown but only facilities with at least one non-empty PDF
    has_pdf = df_out[df_out["PDF_File"] != ""]["Place ID"].unique()
    df_pdf = df_fac[df_fac["Place ID"].isin(has_pdf)]
    print(f"\n  Reg_Measure_Type breakdown (facilities with ≥1 PDF, n={len(df_pdf)}):")
    for rmt, count in df_pdf["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count}")

    total_pdfs = df_fac["Total_PDFs_Available"].apply(pd.to_numeric, errors="coerce").sum()
    print(f"\n  Total_PDFs_Available (sum across facilities): {int(total_pdfs)}")


def union_site_data_snapshots():
    """Rewrite the top-level site_data_relevant.csv as the union of every dated snapshot."""
    print("snapshots found:")
    pattern = os.path.join(OUTPUT_DIR, "site_data", "*", "site_data_relevant.csv")
    frames = []
    for path in sorted(glob.glob(pattern)):
        as_of = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, dtype=str).fillna("")
        df["as_of"] = as_of
        frames.append(df)
        print(f"  {as_of}: {len(df):5} rows")
    allrows = pd.concat(frames, ignore_index=True)
    allrows["order_key"] = allrows["Reg_Measure_ID"].where(
        allrows["Reg_Measure_ID"].str.strip().ne(""), allrows["Order_No"])

    # Older snapshots predate the collective-permittee filter, so drop those places here too
    collective = allrows["Agency"].str.contains(COLLECTIVE_AGENCY_RE)
    if collective.any():
        dropped = allrows.loc[collective, ["Place ID", "Facility Name"]].drop_duplicates()
        print(f"\ncollective-permittee places dropped (not facilities): {len(dropped)}")
        print(dropped.to_string(index=False))
        allrows = allrows[~collective]

    # provenance: which snapshots each document appears in, for the year-over-year join later
    provenance = (allrows.groupby(SNAPSHOT_UNION_KEY)["as_of"]
                  .agg(lambda s: ";".join(sorted(set(s))))
                  .reset_index().rename(columns={"as_of": "as_of_dates"}))
    provenance["n_snapshots"] = provenance["as_of_dates"].str.count(";") + 1

    # keep one row per document, preferring the newest snapshot's metadata
    union = (allrows.sort_values("as_of", ascending=False)
             .drop_duplicates(subset=SNAPSHOT_UNION_KEY, keep="first"))

    # provenance rides along as columns rather than a side file: the year-over-year figure
    # needs to know which snapshots a document appeared in, and a second file only drifts
    union = union.merge(provenance, on=SNAPSHOT_UNION_KEY, how="left")
    cols = ([c for c in allrows.columns if c not in ("as_of", "order_key")]
            + ["as_of_dates", "n_snapshots"])
    union = union[cols]

    current = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")
    print()
    print(f"union            : {len(union):5} rows  ({union['PDF_File'].nunique()} distinct PDFs,"
          f" {union['Place ID'].nunique()} facilities)")
    print(f"current top-level: {len(current):5} rows  ({current['PDF_File'].nunique()} distinct PDFs)")
    new_pdfs = set(union["PDF_File"]) - set(current["PDF_File"])
    print(f"  documents the union adds: {len(new_pdfs)}")
    print()
    print("documents by snapshot coverage:")
    print(provenance["n_snapshots"].value_counts().sort_index().rename("documents").to_string())

    missing = [f for f in union["PDF_File"].unique()
               if f and not os.path.exists(PERMITS_DIR / f)]
    if missing:
        print(f"\nWARNING: {len(missing)} referenced PDFs are not in output/permits/ "
              f"(step3 will skip them): {missing[:3]}")

    union.to_csv(SITE_DATA_RELEVANT_CSV, index=False)
    print(f"\nwrote site_data_relevant.csv ({len(union)} rows)")


if __name__ == "__main__":

    # Sometimes it takes a few tries to get the facility results page to load / Excel to download
    # Try a few times or wait a couple of hours to let CIWQS stabilize if needed

    # sometimes helpful to clear chrome temp files
    # find /tmp -maxdepth 1 -name "chrome_user_data_*" -user daly -print
    # find /tmp -maxdepth 1 -name "chrome_user_data_*" -user daly -exec rm -rf {} +

    # program_urls = run_ciwqs_search()
    # facilities = collect_facility_page_urls(program_urls)
    # To restart from a checkpoint, replace the step(s) above with:
    with open(OUTPUT_DIR / "facilities.json") as f:
        facilities = json.load(f)
    try:
        facilities = download_facility_page_pdfs(facilities)
        npdes_pdfs = detect_and_move_npdes_pdfs(facilities)
        create_site_data_csv(facilities, npdes_pdfs)
        union_site_data_snapshots()
    finally:
        restore_base_facilities()
