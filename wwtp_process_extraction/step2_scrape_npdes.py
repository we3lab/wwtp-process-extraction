from contextlib import contextmanager
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import os
import re
import shutil
import threading
import time
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
from helpers.paths import *
from helpers.utils import is_general_order, ZERO_WIDTH_CHARS
# Modified using Claude 4.5

# Run settings
# None = today's active permit; a date like "2021-06-01" picks the order in force on that date
AS_OF = None
MAX_WORKERS = 24  # parallel Chrome sessions; raise if running on a server

# Paths
OTHER_PDFS_DIR = OUTPUT_DIR / "other_pdfs"
# Every run writes its outputs to output/site_data/<date>/
SNAPSHOT_DATE = AS_OF or datetime.now().strftime("%Y-%m-%d")
SNAPSHOT_DIR = OUTPUT_DIR / "site_data" / SNAPSHOT_DATE
# PDF signal cache (keyed on size+mtime): read this run's copy if it exists, else the top-level one
SIGNAL_CACHE_PATH = SNAPSHOT_DIR / "pdf_signal_cache.json"
BASE_SIGNAL_CACHE_PATH = OUTPUT_DIR / "pdf_signal_cache.json"
# The top-level files steps 3-6 read are a frozen base (2026-06-01 + 2026-08-13 snapshots).
# True copies this run's facilities.json / site_data_all.csv / pdf_signal_cache.json to the top level and re-unions
# site_data_relevant.csv from every dated folder (the 2026-08-13 folder is gone, so its rows would drop).
UPDATE_TOP_LEVEL = False
CHROME_BIN = Path.home() / "bin/chrome/chrome-linux64/chrome"
CHROMEDRIVER_BIN = Path.home() / "bin/chrome/chromedriver-linux64/chromedriver"

# CIWQS Interactive Regulated Facilities Report and its search filters
CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"
REGULATED_FACILITY_REPORT_URL = f"{CIWQS_SERVLET}?inCommand=reset&reportName=RegulatedFacility"
PROGRAMS = {"NPDES": {"NPDESWW", "NPDMUNI"}, "WDR": {"WDRMUNILRG", "WDRMUNIOTH"}}
ACCEPTED_PROGRAMS = set().union(*PROGRAMS.values())
CIWQS_FACILITY_TYPE = "Wastewater Treatment Facility"
CIWQS_WASTE_TYPE = "Domestic wastewater"
CIWQS_RELATED_PERMIT_STATUS = "Active"
CIWQS_DRILLDOWN_QUERY_DROP = ("enrollee",) # drop enrollee=Y filter
FACILITY_CIWQS_COLUMNS = ["WDID", "Facility Name", "NPDES No."]
COLLECTIVE_AGENCY = "organizations under" # watershed permits list this as the "agency"

# Browser waits and page elements
WAIT_TIME = 300  # CIWQS grid/export pages are large and slow
CIWQS_OVERLAY_WAIT = 180  # loading overlay after changing page size
XP_GRID = "//table[contains(@class,'ciwqsReportDataTable')]"
PDF_XPATH = "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
# Older orders are occasionally filed as .doc
ATTACHMENT_XPATH = "//a[contains(@href, 'PublicAttachmentRetriever')]"

# Which order governs a facility: lowest rank wins, then newest effective date
TYPE_RANK = {
    "NPDES PERMIT": 0,
    "CO-PERMITTEE": 1,
    "ENROLLEE - NPDES": 2,
    "WDR": 3,
    "ENROLLEE - WDR": 4,
}

# Skip non-permit PDFs (reports, letters, maps...) by filename
# Short words count only between the separators " ._-" (e.g. "_map_"), not inside other words
SKIP_BASE_WORDS = ("rpts", "rowd", "memo", "nov", "map", "rwd", "gwmp", "mgo")
SKIP_PHRASES = (  # skip if anywhere in filename
    "report", "financial", "response to", "study", "addendum", "registration", "adoption",
    "letter", "covltr", "cover_l", "cover l", "volumetric",
    "form200", "form 200", "management zone", "management_zone", "management plan",
)
# skip these unless a KEEP_WORDS word is also in the name
CONTINGENT_SKIP_WORDS = ("amendment", "mrp")
# NOA/WDR/order/NPDES as a whole word overrides the contingent skip
KEEP_WORDS = {"noa", "wdr", "wdrs", "order", "npdes"}

# Classify a downloaded PDF as NPDES / NOA / WDR from the text of its first pages
MAX_SCAN_PAGES = 5
MIN_PDF_PAGES = 10
# WDR title phrases, matched against the text with all whitespace removed
WDR_PHRASES = ("wastedischargerequirements", "wdrs", "waterrecyclingrequirements", "informationsheet")
# Unicode spaces that PDFs put between letters; removed like ZERO_WIDTH_CHARS
INVISIBLE_SPACES = dict.fromkeys(map(ord, "\u00a0\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005"
                                          "\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"))
# PDF text is messy: allow spaces, line breaks and hyphens between letters (soft hyphens are already removed)
FUZZY_INNER_SEP = r"[\s\-]*"
# Matches "the following ... subject to ... set forth in this ... order", up to 600 chars per gap
NPDES_SENTENCE_RE = re.compile(
    r"(.{1,600}?)".join(
        "".join(ch + FUZZY_INNER_SEP for ch in phrase)
        for phrase in ("thefollowing", "subjectto", "setforthinthis", "order")
    ),
    flags=re.I | re.DOTALL,
)
CAG_PERMIT_RE = re.compile(r"\bca\s*g\d+", re.IGNORECASE)

# Steps 3-5 read one site_data_relevant.csv: the union of all dated snapshots, one row per
# facility + order (Reg_Measure_ID, else Order_No) + PDF
SNAPSHOT_UNION_KEY = ["Place ID", "order_key", "PDF_File"]

# Each worker buffers its log lines; logged_facility prints them together so they don't interleave
worker_log = threading.local()


def save_facilities(facilities_by_place):
    with open(SNAPSHOT_DIR / "facilities.json", "w") as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: {len(facilities_by_place)} facilities -> site_data/{SNAPSHOT_DATE}/facilities.json")


def say(msg):
    """Buffer a line for the current worker, or print directly outside a worker."""
    buf = getattr(worker_log, "lines", None)
    if buf is None:
        print(msg)
    else:
        buf.append(msg)


def repair_href(href):
    """Undo BeautifulSoup turning '&regMeasID=' into '\u00aeMeasID=' ('&reg' is the (R) symbol).
    Otherwise the URL silently returns an empty page, or another order's attachments."""
    return href.replace("\u00aeMeasID", "&regMeasID")


def abs_url(href):
    return urljoin(f"{CIWQS_ROOT}/ciwqs/readOnly/", repair_href(href))


def cell_text(cells, i):
    return cells[i].get_text(strip=True) if i is not None and 0 <= i < len(cells) else ""


def cell_href(cells, i):
    if i is None or not (0 <= i < len(cells)):
        return ""
    a = cells[i].find("a", href=True)
    return abs_url(a["href"]) if a else ""


def url_param(url, name):
    """Value of query parameter `name` in `url`, or None."""
    return parse_qs(urlparse(url).query).get(name, [None])[0]


def facility_url(place_id):
    """Reconstruct the CIWQS facility-at-a-glance URL from a place ID."""
    return f"{CIWQS_SERVLET}?reportName=facilityAtAGlance&placeID={place_id}"


def name_skipped(name):
    """True if a PDF filename looks like a report, letter, map or other non-permit."""
    lowered = name.lower()
    # The start of the name counts as a separator before a short word; the end doesn't count after it
    spaced = " " + lowered.replace(".", " ").replace("_", " ").replace("-", " ")
    return any(f" {w} " in spaced for w in SKIP_BASE_WORDS) or any(p in lowered for p in SKIP_PHRASES)


def name_kept(name):
    """True if a KEEP_WORDS word stands alone (between non-letters) in the filename."""
    words = "".join(c if "a" <= c <= "z" else " " for c in name.lower()).split()
    return any(w in KEEP_WORDS for w in words)


def select_value(soup, name, wanted):
    """Option value of the CIWQS search-form dropdown `name` whose text (or value) is `wanted`."""
    sel = soup.find("select", {"name": name})
    if not sel:
        raise RuntimeError(f"CIWQS form has no <select name={name!r}>")
    opts = sel.find_all("option")
    match = next((o for o in opts if wanted in (o.get_text(strip=True), o.get("value"))), None)
    if not match:
        choices = [o.get_text(strip=True) for o in opts]
        raise RuntimeError(f"CIWQS <select name={name!r}> has no {wanted!r}; choices={choices!r}")
    return match.get("value", wanted)


def retry_request(session, method, url, data=None):
    """Retry timeouts and dropped connections (CIWQS drops big responses); HTTP errors raise at once."""
    for attempt in range(1, 5):
        try:
            r = session.request(method, url, data=data, timeout=120)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            print(f"[requests] {method.upper()} {type(exc).__name__} ({attempt}/4): {url[:90]}")
            if attempt == 4:
                raise
            time.sleep(3 * attempt)


def new_chrome_driver(download_dir):
    """Headless Chrome that saves downloads to download_dir (every browser session uses this)."""
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
    driver = webdriver.Chrome(service=Service(str(CHROMEDRIVER_BIN)), options=options)
    driver.set_page_load_timeout(WAIT_TIME)
    return driver


def ciwqs_post_data(hidden, soup, programs):
    """Build the CIWQS form POST body for one or more programs."""
    return (
        list(hidden.items())
        + [("programDrop", select_value(soup, "programDrop", p)) for p in programs]
        + [
            ("typeDrop", select_value(soup, "typeDrop", CIWQS_FACILITY_TYPE)),
            ("wasteTypeDrop", select_value(soup, "wasteTypeDrop", CIWQS_WASTE_TYPE)),
            ("inStatus", select_value(soup, "inStatus", CIWQS_RELATED_PERMIT_STATUS)),
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
        driver.close()
        driver.switch_to.window(main_window)


def resolve_download_url(href, soup):
    """Map an order link to its attachment page, matching on regMeasID."""
    reg_id = url_param(href, "regMeasID")
    # Soup hrefs still have the broken form (see repair_href), so repair them before comparing
    for a in soup.find_all("a", href=True):
        fixed = repair_href(a["href"])
        if "rmAttachmentPopup" in fixed and (not reg_id or f"regMeasID={reg_id}" in fixed):
            return abs_url(a["href"])
    return href


def find_best_order(driver, fac_url, main_window):
    """(order_url, reg_measure_type, wdid, eff, addtl_orders, order_no) of the active order
    (or active at AS_OF)

    order_url is None without a link

    addtl_orders = other linked NPDES PERMITs."""
    with open_in_new_tab(driver, fac_url, main_window):
        WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        time.sleep(1)
        page_html = driver.page_source

    soup = BeautifulSoup(page_html, "html.parser")
    for table in soup.find_all("table"):
        all_rows = table.find_all("tr")
        for hdr_idx, row in enumerate(all_rows[:4]):
            texts = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            # Header row; layout tables wrap whole nested tables, so their long cells rule them out
            if len(texts) < 5 or any(len(t) > 80 for t in texts):
                continue
            if not all(any(req in t for t in texts) for req in ("Reg Measure Type", "Order No")):
                continue

            col_index = {t: i for i, t in enumerate(texts)}

            # min() picks the best candidate; rows without a link still record order metadata
            candidates = []
            for data_row in all_rows[hdr_idx + 1:]:
                cells = data_row.find_all("td")
                status = cell_text(cells, col_index.get("Status")).lower()
                eff = pd.to_datetime(cell_text(cells, col_index.get("Effective Date")), errors="coerce")
                if pd.isna(eff):
                    continue
                # Default: active orders only; with AS_OF, also Historical/Terminated ones effective by then
                if AS_OF is None and status != "active":
                    continue
                if AS_OF is not None and (status == "never active" or eff > pd.Timestamp(AS_OF)):
                    continue

                # Only permit types in TYPE_RANK; WDRs only from municipal programs
                rm_type = cell_text(cells, col_index.get("Reg Measure Type")).upper()
                if rm_type not in TYPE_RANK:
                    continue
                if rm_type == "WDR" and cell_text(cells, col_index.get("Program")).upper() not in PROGRAMS["WDR"]:
                    continue

                href = cell_href(cells, col_index.get("Order No.")) or None
                order_no = cell_text(cells, col_index.get("Order No."))
                candidates.append((TYPE_RANK[rm_type], -eff.value, href, rm_type, eff,
                                   cell_text(cells, col_index.get("WDID")), order_no))

            if candidates:
                best = min(candidates, key=lambda c: (c[0], c[1], 0 if c[2] else 1))
                rank, _, href, rm_type, eff, wdid, order_no = best
                say(f"  Best order: {rm_type}, rank={rank}, effective={eff.date()}, order={order_no}, "
                    f"link={'yes' if href else 'no'}")
                download_url = resolve_download_url(href, soup) if href else None
                # Other NPDES PERMIT orders (rank 0) with links
                addtl_orders = [(resolve_download_url(c_href, soup), c_wdid, c_eff)
                                for c_rank, _, c_href, _, c_eff, c_wdid, _ in candidates
                                if c_rank == 0 and c_href and c_href != href]
                return download_url, rm_type, wdid, eff, addtl_orders, order_no

    return None, None, None, None, [], ""


def download_order_pdfs(driver, order_url, worker_dir, main_window):
    """Download an order page's PDFs into worker_dir, then move them to other_pdfs/.
    Returns (downloaded_pdfs, missed_pdfs, total_on_page)."""
    downloaded_pdfs = []
    missed_pdfs = []
    with open_in_new_tab(driver, order_url, main_window):
        try:
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, ATTACHMENT_XPATH)))
        except TimeoutException:
            pass
        pdf_documents = driver.find_elements(By.XPATH, PDF_XPATH)
        all_attachments = driver.find_elements(By.XPATH, ATTACHMENT_XPATH)
        # Count every attachment, so a page with only .doc files still counts as found
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
                    if name_skipped(pdf_name):
                        continue
                    if any(w in pdf_name.lower() for w in CONTINGENT_SKIP_WORDS) and not name_kept(pdf_name):
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
            driver.execute_script("window.stop();")


def set_page_all(driver):
    for attempt in range(1, 4):
        try:
            if driver.find_elements(By.NAME, "pagesizeselect"):
                sel_el = WebDriverWait(driver, 120).until(
                    EC.element_to_be_clickable((By.NAME, "pagesizeselect"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sel_el)
                Select(sel_el).select_by_visible_text("ALL")
                time.sleep(2)
            WebDriverWait(driver, CIWQS_OVERLAY_WAIT).until(
                EC.invisibility_of_element_located((By.CLASS_NAME, "loading"))
            )
            return WebDriverWait(driver, WAIT_TIME).until(
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
    # The all-programs search gives the Excel export link; per-program searches give flat tables
    print("[requests] Submitting filters")
    resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
                        data=ciwqs_post_data(hidden_inputs, search_soup, list(PROGRAMS)))
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
                                data=ciwqs_post_data(hidden_inputs, search_soup, [prog]))
        prog_url = extract_drilldown_url(BeautifulSoup(prog_resp.text, "html.parser"))
        if prog_url:
            program_urls.append((prog, prog_url))
            print(f"[requests] {prog} facility URL: {prog_url}")
        else:
            print(f"[requests] Warning: no drilldown URL found for {prog}")

    # Fresh Chrome profile (no cookies), used only for the Excel export
    driver = new_chrome_driver(OTHER_PDFS_DIR)
    load_ciwqs_table(driver, total_url, "Facility page")
    print("Detail page loaded for Excel export")
    time.sleep(5)
    parts = urlparse(total_url)
    pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "exportToExcel"]
    pairs.append(("exportToExcel", "Y"))
    excel_export_url = urlunparse(parts._replace(query=urlencode(pairs)))

    # Only files that appear after the request count, so an old export left in other_pdfs/ is ignored
    files_before = set(os.listdir(OTHER_PDFS_DIR))
    requested_at = time.time()
    for attempt in range(1, 3):
        try:
            driver.get(excel_export_url)
            break
        except TimeoutException:
            if attempt == 2:
                raise
            driver.execute_script("window.stop();")
    time.sleep(2)

    # Wait for the download (.crdownload while in progress); give up if nothing starts in 90 s
    excel_files = []
    while time.time() < requested_at + WAIT_TIME:
        new_files = [OTHER_PDFS_DIR / f for f in set(os.listdir(OTHER_PDFS_DIR)) - files_before]
        if not new_files and time.time() > requested_at + 90:
            break
        finished = [f for f in new_files if ".xls" in f.name and not f.name.endswith(".crdownload")]
        if finished and len(finished) == len(new_files) and all(map(file_stable, finished)):
            excel_files = finished
            break
        time.sleep(0.5)

    if not excel_files:
        print(f"No Excel file found after {WAIT_TIME}s")
        driver.quit()
        raise SystemExit

    excel_file = max(excel_files, key=os.path.getctime)
    df = pd.read_csv(excel_file, sep='\t', encoding='latin-1', on_bad_lines='warn', dtype=str)

    # Re-apply the search form's filters to the export
    df = df[
        df["Program"].fillna("").str.upper().map(lambda p: any(a in p for a in ACCEPTED_PROGRAMS)) &
        (df["Regulatory Measure Status"].fillna("").str.upper() == CIWQS_RELATED_PERMIT_STATUS.upper()) &
        df["Place/Project Type"].fillna("").str.upper().str.contains(CIWQS_FACILITY_TYPE.upper(), regex=False)
    ]
    print(f"After explicit form-aligned filtering: {len(df)} rows")

    df["Expiration/Review Date"] = pd.to_datetime(df["Expiration/Review Date"], errors='coerce')
    df_sorted = df.sort_values(["WDID", "Facility Name", "Expiration/Review Date"], ascending=[True, True, False])
    df_deduplicated = df_sorted.drop_duplicates(subset=["WDID", "Facility Name"])
    duplicates_removed = df_sorted[df_sorted.duplicated(subset=["WDID", "Facility Name"])]
    print(f"After deduplication and filtering: {len(df_deduplicated)} rows "
          f"(removed {len(df) - len(df_deduplicated)} duplicates)")
    if len(duplicates_removed) > 0:
        print("Duplicates removed:")
        print(duplicates_removed[["Facility Name", "WDID", "NPDES No."]].to_string(index=False))

    df_deduplicated.to_csv(SNAPSHOT_DIR / "site_data_all.csv", index=False)
    print(f"Saved {len(df_deduplicated)} rows to site_data/{SNAPSHOT_DATE}/site_data_all.csv")

    driver.quit()
    return program_urls


def collect_facility_page_urls(program_urls):
    print("\n STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")

    facilities_by_place = {}  # place_id -> {"facilities": [dict keyed by FACILITY_CIWQS_COLUMNS, ...]}
    col = {}  # CIWQS header text -> column index (detected on first program)

    for prog, prog_url in program_urls:
        print(f"\n--- {prog}: {prog_url}")
        # New Chrome per program: cookies from the search make tables load too slowly
        driver = new_chrome_driver(OTHER_PDFS_DIR)
        load_ciwqs_table(driver, prog_url, prog)
        set_page_all(driver)

        page_soup = BeautifulSoup(driver.page_source, "html.parser")
        # set_page_all already waited for this grid table
        data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
        bs_rows_prog = data_table.find_all("tr")
        print(f"{prog}: {len(bs_rows_prog)} <tr> after ALL page size")

        if not col:
            for header_row in bs_rows_prog:
                texts = [td.get_text(strip=True) for td in header_row.find_all("td")]
                if "Order No." in texts or "Facility Name" in texts:
                    col.update({t: i for i, t in enumerate(texts) if t})
                    break
            missing = [c for c in FACILITY_CIWQS_COLUMNS if c not in col]
            if missing:
                raise RuntimeError(f"{prog} table is missing columns {missing}")

        for tr in bs_rows_prog:
            if tr.find("td", class_="ciwqsReportColumnName"):
                continue
            cells = tr.find_all("td")
            status = cell_text(cells, col.get("Regulatory Measure Status")).upper()
            plc_type = cell_text(cells, col.get("Place/Project Type")).upper()

            if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
                continue
            if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
                continue

            place_id = url_param(cell_href(cells, col["Facility Name"]), "placeID")
            if not place_id:
                continue

            facility = {name: cell_text(cells, col[name]) for name in FACILITY_CIWQS_COLUMNS}
            entry = facilities_by_place.setdefault(place_id, {"facilities": []})
            if not any(f["Facility Name"] == facility["Facility Name"] for f in entry["facilities"]):
                entry["facilities"].append(facility)

        driver.quit()

    # Warn about site_data_all.csv facilities that no per-program table listed
    site_all = pd.read_csv(SNAPSHOT_DIR / "site_data_all.csv", dtype=str).fillna("")
    scraped_names = {
        f["Facility Name"]
        for entry in facilities_by_place.values()
        for f in entry["facilities"]
    }
    for fac_name in site_all["Facility Name"].str.strip():
        if fac_name and fac_name not in scraped_names:
            print(f"  ! {fac_name} in site_data_all.csv but not found in any CIWQS table")

    print(f"\n✓ Found {len(facilities_by_place)} unique facilities (placeIDs)")

    save_facilities(facilities_by_place)

    return facilities_by_place


def needs_retry(entry):
    # Unprocessed (exception during process_facility left "facilities" key intact)
    if "facilities" in entry:
        return True
    # Deliberately skipped (no link or pre-2004) — not a failure, never retry
    if entry.get("pdf_skip_reason"):
        return False
    # Clicked an order but got 0 PDFs: usually a slow page, so retry
    if entry.get("reg_measure_type") and not entry["pdfs"] and not entry["total_pdfs"]:
        return True
    return False


def download_facility_page_pdfs(facilities_by_place):
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
        main_window = driver.window_handles[0]
        fac_url = facility_url(place_id)
        fac_name = entry["facilities"][0]["Facility Name"] if "facilities" in entry else entry.get("Facility Name", place_id)
        say(f"[{idx}/{total}] {fac_name}")
        try:
            order_url, rm_type, wdid, eff, addtl_orders, order_no = find_best_order(driver, fac_url, main_window)
            reg_id = url_param(order_url, "regMeasID") if order_url else None
            info = {
                "Facility Name": fac_name,
                "WDID": wdid,
                "pdfs": [],
                "total_pdfs": 0,
                "reg_measure_id": reg_id,
                "reg_measure_type": rm_type,
                "order_no": order_no,
            }
            with lock:
                already_done = reg_id_to_info.get(reg_id) if reg_id else None

            if rm_type is None:
                say("  X No suitable active NPDES order found")
            # Store order metadata but skip PDFs if no link or pre-2004
            elif not order_url or eff.year < 2004:
                info["pdf_skip_reason"] = "pre-2004" if eff.year < 2004 else "no link"
                say(f"  Skipping PDFs ({info['pdf_skip_reason']}): {rm_type}, eff={eff.date()}")
            elif already_done:
                say(f"  Dedup: reusing already-processed order {reg_id}")
                info = {**already_done, "Facility Name": fac_name, "WDID": wdid}
            else:
                downloaded_pdfs, missed_pdfs, total_pdfs = download_order_pdfs(
                    driver, order_url, worker_dir, main_window
                )
                info.update(pdfs=downloaded_pdfs, missed_pdfs=missed_pdfs, total_pdfs=total_pdfs)

                # Download PDFs for additional NPDES PERMIT orders (effective 2004+)
                addtl_reg_ids, addtl_wdids = [], []
                for addtl_url, addtl_wdid, addtl_eff in addtl_orders:
                    if addtl_eff.year < 2004:
                        continue
                    addtl_reg_id = url_param(addtl_url, "regMeasID")
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
                with lock:
                    if reg_id:
                        reg_id_to_info[reg_id] = info

            entry.update(info)
            entry.pop("facilities", None)
        except Exception as e:
            say(f"  X {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
            shutil.rmtree(worker_dir, ignore_errors=True)

    def logged_facility(args):
        """Print each facility's lines as one block, not mixed with other workers."""
        worker_log.lines = []
        try:
            process_facility(args)
        finally:
            lines, worker_log.lines = worker_log.lines, None
            if lines:
                with lock:
                    print("\n".join(lines), flush=True)

    # Fills in pdfs, total_pdfs, reg_measure_id/type on each entry in place
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(logged_facility, enumerate(items, 1))

    retry_count = 0
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
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            executor.map(logged_facility, enumerate(retry_items, 1))

    save_facilities(facilities_by_place)

    return facilities_by_place


def portfolio_subdocs(doc):
    """The PDFs embedded in a PDF Portfolio, or None for a normal PDF."""
    if doc.xref_get_key(doc.pdf_catalog(), "Collection")[0] == "null":
        return None
    return [fitz.open("pdf", doc.embfile_get(i)) for i in range(doc.embfile_count())
            if doc.embfile_info(i)["filename"].lower().endswith(".pdf")]


def extract_pdf_text(pdf_path: str) -> str:
    """Text of the first MAX_SCAN_PAGES pages (per sub-file for a PDF Portfolio), original case."""
    parts = []
    with fitz.open(pdf_path) as doc:
        subdocs = portfolio_subdocs(doc)
        if subdocs is not None:
            for sub in subdocs:
                parts += [sub[j].get_text() for j in range(min(len(sub), MAX_SCAN_PAGES))]
                sub.close()
        else:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages[:MAX_SCAN_PAGES]):
                    text = page.extract_text()
                    if not text.strip():
                        fitz_page = doc[i]
                        text = fitz_page.get_text(textpage=fitz_page.get_textpage_ocr())
                    parts.append(text)
    raw = unicodedata.normalize("NFKC", " ".join(parts))
    return raw.translate(ZERO_WIDTH_CHARS).translate(INVISIBLE_SPACES)


def length_of_pdf(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        subdocs = portfolio_subdocs(doc)
        return len(doc) if subdocs is None else sum(len(sub) for sub in subdocs)


def detect_npdes(pdf_file: str) -> str | None:
    """Doc type ("NPDES", "NOA", "WDR") of a PDF from the text of its first pages, or None."""
    if length_of_pdf(pdf_file) < MIN_PDF_PAGES:
        return None

    raw_text = extract_pdf_text(pdf_file)
    text = raw_text.lower()
    squashed = "".join(text.split())

    # Statewide general orders aren't facility permits; check before NOA, since they describe NOAs
    if is_general_order(text):
        return None
    if "NOA" in raw_text or "noticeofapplicability" in squashed:
        return "NOA"
    # Generic CAG order (no NOA): not facility-specific, skip
    if CAG_PERMIT_RE.search(text):
        return None
    # Flexible NPDES-like sentence: "the following <...> subject to <...> set forth in this <...> order"
    if "table1.dischargerinformation" in squashed or NPDES_SENTENCE_RE.search(text):
        return "NPDES"
    if any(phrase in squashed for phrase in WDR_PHRASES):
        return "WDR"
    return None


def detect_and_move_npdes_pdfs(facilities_by_place):
    print("\n STEP 3: Detecting and moving NPDES PDFs")

    # Include PDFs already moved to permits/ in previous runs
    npdes_pdfs = {f for f in os.listdir(PERMITS_DIR) if f.endswith(".pdf")}
    # PDF -> reg measure groups (reg_measure_id, or place_id when there is none)
    pdf_to_groups = {}
    group_to_rm_type = {}
    pdf_to_places = {}
    for place_id, entry in facilities_by_place.items():
        group = entry.get("reg_measure_id") or place_id
        group_to_rm_type[group] = entry.get("reg_measure_type", "")
        for pdf in entry["pdfs"]:
            pdf_to_groups.setdefault(pdf, set()).add(group)
            pdf_to_places.setdefault(pdf, set()).add(place_id)

    single_pdf_files = {entry["pdfs"][0] for entry in facilities_by_place.values()
                        if len(entry["pdfs"]) == 1}

    with open(SIGNAL_CACHE_PATH if SIGNAL_CACHE_PATH.exists() else BASE_SIGNAL_CACHE_PATH) as f:
        signal_cache = json.load(f)

    # Detect NPDES signals for every PDF in other_pdfs/, then move NPDES-positive files to permits/.
    pdf_files = [f for f in os.listdir(OTHER_PDFS_DIR) if f.endswith(".pdf")]
    reused = kept_out = 0
    for filename in pdf_files:
        src = os.path.join(OTHER_PDFS_DIR, filename)
        # Reset the cache entry if the file changed since it was cached; a surviving "signal" is a cache hit
        st = os.stat(src)
        fingerprint = f"{st.st_size}:{st.st_mtime_ns}"
        entry = signal_cache.get(filename)
        if not entry or entry["fp"] != fingerprint:
            entry = {"fp": fingerprint}
            signal_cache[filename] = entry
        if "signal" in entry:
            reused += 1
        else:
            entry["signal"] = detect_npdes(src)
        matched_type = entry["signal"]

        stem = os.path.splitext(filename)[0]
        assoc_types = {group_to_rm_type[group] for group in pdf_to_groups.get(filename, set())}
        enrollee_only = bool(assoc_types) and all(t.startswith("ENROLLEE") for t in assoc_types)
        if matched_type and name_skipped(stem) and not name_kept(stem):
            matched_type = None  # general-order/non-permit filename pattern
        if matched_type in ("WDR", "NPDES") and enrollee_only:
            matched_type = None  # general order for enrolled facilities, not facility-specific
        # Count facilities, not orders: many enrollees share one general order's reg_measure_id
        if matched_type == "NOA" and len(pdf_to_places.get(filename, set())) > 1 and enrollee_only:
            matched_type = None  # shared general order contains NOA language but not facility-specific

        if not matched_type and filename in single_pdf_files and "pages" not in entry:
            entry["pages"] = length_of_pdf(src)
        if matched_type or (filename in single_pdf_files and entry["pages"] >= 3):
            os.rename(src, os.path.join(PERMITS_DIR, filename))
            print(f"{matched_type} detected: {filename}" if matched_type else f"single PDF, kept: {filename}")
            npdes_pdfs.add(filename)
        else:
            kept_out += 1

    with open(SIGNAL_CACHE_PATH, "w") as f:
        json.dump(signal_cache, f, sort_keys=True, indent=0)
    print(f"  Scanned {len(pdf_files)} PDFs ({reused} reused from cache, "
          f"{len(pdf_files) - reused} newly scanned)")
    print(f"\nNPDES/NOA/WDR PDFs moved: {len(npdes_pdfs)}")
    print(f"Non-NPDES/NOA/WDR PDFs kept in pdfs folder: {kept_out}")

    return npdes_pdfs


def create_site_data_csv(facilities_by_place, npdes_pdfs):
    print("\n STEP 4: Creating site_data_relevant with relevant NPDES/WDR/NOA documents only")

    # (WDID, Facility Name) -> metadata from site_data_all.csv; scraped order_no overrides Order_No.
    # This run's export if it ran the CIWQS search, else the base one.
    meta_keys = ["Agency", "Region", "Major/Minor", "Order_No", "NPDES No."]
    site_all_path = SNAPSHOT_DIR / "site_data_all.csv"
    site_all = pd.read_csv(site_all_path if site_all_path.exists() else SITE_DATA_ALL_CSV, dtype=str)
    site_all = site_all.fillna("").rename(columns={"Order No.": "Order_No"})
    site_all["WDID"] = site_all["WDID"].str.strip()
    site_all["Facility Name"] = site_all["Facility Name"].str.strip()
    enrich = (site_all.drop_duplicates(["WDID", "Facility Name"])
              .set_index(["WDID", "Facility Name"])[meta_keys].to_dict("index"))

    # Times each NPDES PDF appears across facilities' PDF lists (for Shared_PDF flag)
    pdf_to_n_facilities = Counter(
        pdf for entry in facilities_by_place.values() for pdf in entry["pdfs"] if pdf in npdes_pdfs
    )

    rows = []
    skipped_collective = []
    for place_id, entry in facilities_by_place.items():
        fac_name = entry["Facility Name"]
        wdid = entry["WDID"]
        meta = enrich.get((wdid, fac_name), {})
        if COLLECTIVE_AGENCY in meta.get("Agency", "").lower():
            skipped_collective.append(f"{place_id} {fac_name}")
            continue
        facility_npdes_pdfs = [p for p in entry["pdfs"] if p in npdes_pdfs]
        meta_dict = {key: meta.get(key, "") for key in meta_keys}
        meta_dict["Order_No"] = entry.get("order_no") or meta_dict["Order_No"]
        for pdf in (facility_npdes_pdfs or [""]):
            rows.append(
                {
                    "Place ID": place_id.strip(),
                    "WDID": wdid.strip() if wdid else "",
                    "Facility Name": fac_name.strip(),
                    **meta_dict,
                    "Facility_URL": facility_url(place_id),
                    "Reg_Measure_ID": entry.get("reg_measure_id"),
                    "Reg_Measure_Type": entry.get("reg_measure_type"),
                    "Addtl_Reg_Measure_ID": entry.get("addtl_reg_measure_id", ""),
                    "Addtl_WDID": entry.get("addtl_WDID", ""),
                    "PDF_File": pdf,
                    "Shared_PDF": "Yes" if pdf_to_n_facilities[pdf] > 1 else "No",
                    "Total_PDFs_Available": entry.get("total_pdfs")
                }
            )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(SNAPSHOT_DIR / "site_data_relevant.csv", index=False)

    # Reg_Measure_Type counts, one row per facility
    df_fac = df_out.drop_duplicates(subset="Place ID")
    has_pdf = df_fac["Place ID"].isin(df_out.loc[df_out["PDF_File"] != "", "Place ID"])
    breakdown = pd.DataFrame({
        "all": df_fac["Reg_Measure_Type"].value_counts(dropna=False),
        "with PDF": df_fac.loc[has_pdf, "Reg_Measure_Type"].value_counts(dropna=False),
    }).fillna(0).astype(int)
    total_pdfs = pd.to_numeric(df_fac["Total_PDFs_Available"], errors="coerce").sum()

    print(f"  Enrichment lookup: {len(enrich)} entries from site_data_all.csv")
    if skipped_collective:
        print(f"  Collective-permittee places dropped: {', '.join(skipped_collective)}")
    print(f"Wrote {len(rows)} rows to site_data/{SNAPSHOT_DATE}/site_data_relevant.csv")
    print(f"\n  Reg_Measure_Type by facility (n={len(df_fac)}, {has_pdf.sum()} with a PDF):")
    print(breakdown.to_string())
    print(f"  Total_PDFs_Available (sum across facilities): {int(total_pdfs)}")


def union_site_data_snapshots():
    """Rewrite the top-level site_data_relevant.csv as the union of every dated snapshot."""
    frames = []
    for path in sorted((OUTPUT_DIR / "site_data").glob("*/site_data_relevant.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        df["as_of"] = path.parent.name
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    snapshot_rows = allrows["as_of"].value_counts().sort_index()
    allrows["order_key"] = allrows["Reg_Measure_ID"].where(
        allrows["Reg_Measure_ID"].str.strip().ne(""), allrows["Order_No"])

    # Older snapshots predate the collective-permittee filter, so drop those places here too
    collective = allrows["Agency"].str.lower().str.contains(COLLECTIVE_AGENCY, regex=False)
    dropped = allrows.loc[collective, ["Place ID", "Facility Name"]].drop_duplicates()
    allrows = allrows[~collective]

    # provenance: which snapshots each document appears in, for the year-over-year join later
    provenance = (allrows.groupby(SNAPSHOT_UNION_KEY)["as_of"]
                  .agg(lambda s: ";".join(sorted(set(s))))
                  .reset_index().rename(columns={"as_of": "as_of_dates"}))
    provenance["n_snapshots"] = provenance["as_of_dates"].str.count(";") + 1

    # keep one row per document, preferring the newest snapshot's metadata
    union = (allrows.sort_values("as_of", ascending=False)
             .drop_duplicates(subset=SNAPSHOT_UNION_KEY))

    # Provenance is stored as columns, not a side file, so it can't drift out of sync
    union = union.merge(provenance, on=SNAPSHOT_UNION_KEY, how="left")
    cols = ([c for c in allrows.columns if c not in ("as_of", "order_key")]
            + ["as_of_dates", "n_snapshots"])
    union = union[cols]

    current = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")
    new_pdfs = set(union["PDF_File"]) - set(current["PDF_File"])
    missing = [f for f in union["PDF_File"].unique()
               if f and not os.path.exists(PERMITS_DIR / f)]
    union.to_csv(SITE_DATA_RELEVANT_CSV, index=False)

    print("\nsnapshot rows:")
    print(snapshot_rows.to_string())
    if len(dropped):
        print(f"collective-permittee places dropped: {len(dropped)}")
        print(dropped.to_string(index=False))
    print(f"wrote site_data_relevant.csv: {len(union)} rows, {union['PDF_File'].nunique()} distinct PDFs, "
          f"{union['Place ID'].nunique()} facilities ({len(new_pdfs)} PDFs new vs the previous file)")
    print("documents by number of snapshots they appear in:")
    print(provenance["n_snapshots"].value_counts().sort_index().rename("documents").to_string())
    if missing:
        print(f"WARNING: {len(missing)} referenced PDFs are not in output/permits/ "
              f"(step3 will skip them): {missing[:3]}")


if __name__ == "__main__":
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # CIWQS is flaky: retry, or wait a few hours. Clearing old Chrome temp files can help:
    # find /tmp -maxdepth 1 -name "chrome_user_data_*" -user $USER -exec rm -rf {} +

    # Full run (uncomment):
    # program_urls = run_ciwqs_search()
    # facilities = collect_facility_page_urls(program_urls)
    # To restart from a checkpoint, replace the step(s) above with:
    with open(OUTPUT_DIR / "facilities.json") as f:
        facilities = json.load(f)
    facilities = download_facility_page_pdfs(facilities)
    npdes_pdfs = detect_and_move_npdes_pdfs(facilities)
    create_site_data_csv(facilities, npdes_pdfs)
    if UPDATE_TOP_LEVEL:
        for filename in ("facilities.json", "site_data_all.csv", "pdf_signal_cache.json"):
            if (SNAPSHOT_DIR / filename).exists():
                shutil.copy2(SNAPSHOT_DIR / filename, OUTPUT_DIR / filename)
                print(f"top-level {filename} updated from site_data/{SNAPSHOT_DATE}/")
        union_site_data_snapshots()
    else:
        print(f"\nTop-level files unchanged (UPDATE_TOP_LEVEL = False); this run is in site_data/{SNAPSHOT_DATE}/")
