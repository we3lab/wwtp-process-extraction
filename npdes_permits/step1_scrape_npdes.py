from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import uuid
import tempfile
import os
import time
import csv
import glob
from urllib.parse import urlparse, parse_qs
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from helpers.npdes_detection import detect_npdes

CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"


def _abs_url(href):
    if not href or href.startswith("http"):
        return href
    if href.startswith("/"):
        return CIWQS_ROOT + href
    return f"{CIWQS_ROOT}/ciwqs/readOnly/{href}"


def _hidden_fields(soup):
    return {i["name"]: i.get("value", "")
            for i in soup.find_all("input", type="hidden") if i.get("name")}


def _select_value(soup, name, visible_text, *, required_label=None):
    sel = soup.find("select", {"name": name})
    if not sel:
        if required_label:
            raise RuntimeError(f"CIWQS form missing <select name={name!r}> ({required_label}).")
        return visible_text
    for opt in sel.find_all("option"):
        if opt.get_text(strip=True) == visible_text:
            return opt.get("value", visible_text)
    if required_label:
        choices = [opt.get_text(strip=True) for opt in sel.find_all("option")]
        raise RuntimeError(
            f"CIWQS {required_label}: no {visible_text!r} in <select name={name!r}>; choices={choices!r}"
        )
    return visible_text


def _retry_request(session, method, url, *, data=None, max_attempts=4, timeout=120):
    for attempt in range(1, max_attempts + 1):
        try:
            r = session.request(method, url, data=data, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.Timeout:
            print(f"[requests] {method.upper()} timed out ({attempt}/{max_attempts}): {url}")
            if attempt == max_attempts:
                raise
        except requests.exceptions.RequestException:
            raise


def _new_chrome_driver():
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


# Function that selects options from filters
def selection(name, text):
    select_elements = driver.find_elements(By.NAME, name)
    if not select_elements:
        return False
    select = Select(select_elements[0])
    select.select_by_visible_text(text)
    return True

# Link to Interactive Regulated Facilities Report
rfr_url = 'https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility'

# Visible labels for CIWQS form (must stay aligned with post_data and STEP 1 row checks).
CIWQS_PROGRAM = "NPDES"
CIWQS_FACILITY_TYPE = "Wastewater Treatment Facility"
CIWQS_WASTE_TYPE = "Domestic wastewater"
CIWQS_RELATED_PERMIT_STATUS = "Active"
# Optional test scope: set CIWQS_TEST_REGION=1 (or 2..9) to use region-scoped drilldown URL.
# Leave unset for normal full-statewide run.
CIWQS_TEST_REGION = os.getenv("CIWQS_TEST_REGION", "").strip()
# Optional: extra per-page debug logging (very noisy during downloads).
CIWQS_VERBOSE = os.getenv("CIWQS_VERBOSE", "").strip().lower() in ("1", "true", "yes")

# Creates folder to store downloaded PDFs
path = 'npdes_permits/output'
now = datetime.now()
pdf_folder = f'{now.year}-{now.month}-{now.day}'
full_path = os.path.join(path, pdf_folder)
pdfs_path = os.path.join(full_path, 'pdfs')
os.makedirs(full_path, exist_ok=True)
os.makedirs(pdfs_path, exist_ok=True)

ciwqs = requests.Session()
ciwqs.headers["User-Agent"] = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
print("[requests] CIWQS reset form…")
r = _retry_request(ciwqs, 'GET', rfr_url)
soup0 = BeautifulSoup(r.text, "html.parser")
hidden0 = _hidden_fields(soup0)
csrf = hidden0.get("OWASP_CSRFTOKEN", "")
in_status = _select_value(
    soup0, "inStatus", CIWQS_RELATED_PERMIT_STATUS, required_label="Related Permit Status"
)
post_data = {
    **hidden0,
    "programDrop": _select_value(soup0, "programDrop", CIWQS_PROGRAM),
    "typeDrop": _select_value(soup0, "typeDrop", CIWQS_FACILITY_TYPE),
    "wasteTypeDrop": _select_value(soup0, "wasteTypeDrop", CIWQS_WASTE_TYPE),
    "inStatus": in_status,
    "enpRepButton": "",
}
print("[requests] Submitting filters…")
resp = _retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}", data=post_data)
soup1 = BeautifulSoup(resp.text, "html.parser")
full_results_url = None
drilldown_candidates = []
for a in soup1.find_all("a", href=True):
    h = a["href"]
    if "RegulatedFacilityDetail" in h and "drilldown" in h:
        drilldown_candidates.append(_abs_url(h))

# Prefer the unscoped facility-results link (all regions/majors),
# not a summary-row link that includes place=... or majorminor=....
if CIWQS_TEST_REGION:
    for candidate in drilldown_candidates:
        if f"place={CIWQS_TEST_REGION}" in candidate.lower():
            full_results_url = candidate
            break
for candidate in drilldown_candidates:
    lower = candidate.lower()
    if full_results_url:
        break
    if "place=" not in lower and "majorminor=" not in lower:
        full_results_url = candidate
        break
if not full_results_url and drilldown_candidates:
    full_results_url = drilldown_candidates[0]
if not full_results_url:
    raise RuntimeError("CIWQS: no RegulatedFacilityDetail drilldown link in search response")
if CIWQS_TEST_REGION:
    print(f"[requests] Test mode region filter: place={CIWQS_TEST_REGION}")
print(f"[requests] Facility-results URL: {full_results_url}")

driver = _new_chrome_driver()
wait = WebDriverWait(driver, 120)
main_window = driver.current_window_handle

driver.set_page_load_timeout(120)
for attempt in range(1, 4):
    try:
        driver.get(full_results_url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ciwqsReportDataTable")))
        print("[selenium] Facility table ready")
        break
    except TimeoutException:
        print(f"[selenium] Facility page slow ({attempt}/3), same URL…")
        if attempt == 3:
            raise
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass
driver.save_screenshot('npdes_permits/output/web_page.png')
print("Detail page loaded, filters should be preserved")
time.sleep(5)

def _wait_for_results_table(timeout=90):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.XPATH, "//table[contains(@class,'ciwqsReportDataTable')]"))
    )


def _set_page_size_all_and_wait():
    page_size_set = selection('pagesizeselect', 'ALL')
    if page_size_set:
        print("Waiting for table to load with ALL rows...")
    else:
        print("pagesizeselect not present; continuing with current rows")
    try:
        wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, 'loading')))
        print("Loading indicator disappeared")
    except Exception:
        pass
    return _wait_for_results_table(timeout=90)


for attempt in range(1, 3):
    try:
        table_body = _set_page_size_all_and_wait()
        break
    except Exception as e:
        print(f"[selenium] pagesizeselect ALL did not stabilize ({attempt}/2): {e}")
        if attempt == 2:
            raise
        time.sleep(2)

table_rows = table_body.find_elements(By.TAG_NAME, 'tr')
print(f"Final table has {len(table_rows)} rows before export")

# Export Excel file and create mapping from it
export_link = wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
driver.execute_script("arguments[0].scrollIntoView(true);", export_link)
driver.execute_script("arguments[0].click();", export_link)
time.sleep(3)
try:
    export_link = wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
    driver.execute_script("arguments[0].click();", export_link)
except Exception:
    print("Second export click not needed or timed out; continuing")

def wait_for_downloads(directory, timeout=60):
    """Wait for an .xls/.xlsx file to appear and finish downloading."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        files = glob.glob(os.path.join(directory, '*.xlsx')) + glob.glob(os.path.join(directory, '*.xls'))
        files = [f for f in files if not f.lower().endswith('.crdownload')]
        if files:
            stable = True
            for f in files:
                try:
                    s1 = os.path.getsize(f)
                    time.sleep(0.5)
                    s2 = os.path.getsize(f)
                    if s1 != s2:
                        stable = False
                        break
                except OSError:
                    stable = False
                    break
            if stable:
                return files
        time.sleep(0.5)
    return []

excel_files = wait_for_downloads(pdfs_path, timeout=60)
if not excel_files:
    excel_files = wait_for_downloads(full_path, timeout=5)

if not excel_files:
    print("No Excel file found")
    exit()

excel_file = max(excel_files, key=os.path.getctime)
df = pd.read_csv(excel_file, sep='\t', encoding='latin-1')

print(f"Excel file has {len(df)} rows (table had {len(table_rows)} rows)")

# Enforce the same intent as the CIWQS form so downstream CSV/STEP 2 stay in sync.
if "Program" in df.columns:
    df = df[df["Program"].fillna("").str.upper().str.contains("NPD", na=False)]
if "Regulatory Measure Status" in df.columns:
    df = df[df["Regulatory Measure Status"].fillna("").str.upper() == CIWQS_RELATED_PERMIT_STATUS.upper()]
if "Place/Project Type" in df.columns:
    df = df[
        df["Place/Project Type"].fillna("").str.upper().str.contains(CIWQS_FACILITY_TYPE.upper(), na=False)
    ]
print(f"After explicit form-aligned filtering: {len(df)} rows")

df['Expiration/Review Date'] = pd.to_datetime(df['Expiration/Review Date'], errors='coerce')
df_sorted = df.sort_values(['WDID', 'Facility Name', 'Expiration/Review Date'], ascending=[True, True, False])
# Deduplicate by BOTH WDID and Facility Name to preserve different facilities with same WDID
df_deduplicated = df_sorted.drop_duplicates(subset=['WDID', 'Facility Name'], keep='first')
duplicates_removed = df_sorted[df_sorted.duplicated(subset=['WDID', 'Facility Name'], keep='first')]
print(f"After deduplication and filtering: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")
if len(duplicates_removed) > 0:
    cols = [c for c in ['Facility Name', 'WDID', 'NPDES No.'] if c in duplicates_removed.columns]
    print("Duplicates removed (Facility Name, WDID, NPDES No.):")
    print(duplicates_removed[cols].to_string(index=False))

# PDF keywords to skip downloading
# base_keywords get separator-bounded patterns (_kw_, kw_ at start) to avoid false positives.
# "per" (Performance Evaluation Reports) needs this treatment — bare "per" matches "permit".
# "ci" (Compliance Inspection) is removed entirely — it's a valid permit identifier (e.g. CI 0066).
base_keywords = ["noa", "noi", "rpts", "rowd", "per"]
separators = [" ", ".", "-", "_"]
patterns = set()

for keyword in base_keywords:
    for sep1 in separators:
        for sep2 in separators:
            patterns.add(f"{sep1}{keyword}{sep2}")
skip_keywords = list(patterns)

beginning_patterns = []
for keyword in base_keywords:
    for sep in separators:
        beginning_patterns.append(f"{keyword}{sep}")
skip_keywords.extend([
    "report", "financial", "notice of", "response to",
    "rate study", "ratestudy", "study", "letter",
])

def find_col_index_by_header(header_text: str):
    """Find the 1-based column index of a header by its visible text.
    Tries both <a> in header cell and plain text headers.
    """
    # CIWQS often uses td.ciwqsReportColumnName for header rows
    elems = driver.find_elements(
        By.XPATH,
        f"//table[contains(@class,'ciwqsReportDataTable')]//td[contains(@class,'ciwqsReportColumnName')][normalize-space(.)='{header_text}']",
    )
    for parent in elems:
        if parent.tag_name in ('td', 'th'):
            return driver.execute_script("return arguments[0].cellIndex + 1;", parent)

    # try link text header cell
    elems = driver.find_elements(By.XPATH, f"//table//*[self::th or self::td][.//a[normalize-space(text())='{header_text}']]//a[normalize-space(text())='{header_text}']")
    for el in elems:
        parent = el.find_element(By.XPATH, "./..")
        if parent.tag_name in ('td', 'th'):
            return driver.execute_script("return arguments[0].cellIndex + 1;", parent)
    # try plain text header cell
    elems = driver.find_elements(By.XPATH, f"//table//*[self::th or self::td][normalize-space(text())='{header_text}']")
    for parent in elems:
        if parent.tag_name in ('td', 'th'):
            return driver.execute_script("return arguments[0].cellIndex + 1;", parent)
    return None

# Find required column indices dynamically
order_no_col_index = find_col_index_by_header('Order No.')
agency_col_index = find_col_index_by_header('Agency')
facility_col_index = find_col_index_by_header('Facility Name')
npdes_col_index = find_col_index_by_header('NPDES No.')
region_col_index = find_col_index_by_header('Region')
major_minor_col_index = find_col_index_by_header('Major/Minor')
program_col_index = find_col_index_by_header('Program')
reg_measure_status_col_index = find_col_index_by_header('Regulatory Measure Status')
place_type_col_index = find_col_index_by_header('Place/Project Type')

if not order_no_col_index:
    # Print visible headers to help debug CIWQS table variants.
    header_cells = driver.find_elements(
        By.XPATH,
        "//table[contains(@class,'ciwqsReportDataTable')]//td[contains(@class,'ciwqsReportColumnName')]",
    )
    headers = []
    for cell in header_cells:
        try:
            links = cell.find_elements(By.TAG_NAME, "a")
            headers.append((links[0].text.strip() if links else cell.text.strip()) or "<blank>")
        except Exception:
            headers.append("<unreadable>")
    print("[debug] Could not locate 'Order No.' column.")
    print(f"[debug] Detected header cells ({len(headers)}): {headers}")
    raise RuntimeError("Could not locate 'Order No.' column in ciwqsReportDataTable")

# ============================================================================
# STEP 1: COLLECT FACILITY PAGE URLs — rows must match search filters
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Collecting facility page URLs for Active NPDES/WWTF rows")
print("="*80)

NPDES_PROGRAMS = {"NPDESWW", "NPDMUNIOTH", "NPDMUNILRG"}
facility_url_to_facilities = {}

# Parse the page source once with BeautifulSoup — far faster than per-row Selenium calls.
page_soup = BeautifulSoup(driver.page_source, "html.parser")
data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
bs_rows = data_table.find_all("tr") if data_table else []
print(f"Total table rows: {len(bs_rows)}")

# Column indices are 1-based from find_col_index_by_header; convert to 0-based for BS4 indexing.
def _col(one_based_idx):
    return (one_based_idx - 1) if one_based_idx else None

def _cell(cells, one_based_idx):
    i = _col(one_based_idx)
    if i is None or i >= len(cells):
        return ""
    return cells[i].get_text(strip=True)

def _cell_href(cells, one_based_idx):
    i = _col(one_based_idx)
    if i is None or i >= len(cells):
        return ""
    a = cells[i].find("a", href=True)
    return _abs_url(a["href"]) if a else ""

for tr in bs_rows:
    try:
        if tr.find("td", class_="ciwqsReportColumnName"):
            continue
        cells = tr.find_all("td")
        if not cells:
            continue

        status   = _cell(cells, reg_measure_status_col_index).upper()
        program  = _cell(cells, program_col_index).upper()
        plc_type = _cell(cells, place_type_col_index).upper()

        if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
            continue
        if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
            continue
        if program and not any(p in program for p in NPDES_PROGRAMS):
            continue

        facility_url = _cell_href(cells, facility_col_index)
        if not facility_url:
            continue

        facility = {
            'agency':        _cell(cells, agency_col_index),
            'facility_name': _cell(cells, facility_col_index),
            'npdes_no':      _cell(cells, npdes_col_index),
            'region':        _cell(cells, region_col_index),
            'major_minor':   _cell(cells, major_minor_col_index),
            'order_no':      _cell(cells, order_no_col_index),
            'facility_url':  facility_url,
        }
        facility_url_to_facilities.setdefault(facility_url, []).append(facility)
    except Exception as e:
        print(f"Row parse error: {e}")
        continue

print(f"✓ Found {len(facility_url_to_facilities)} unique facility URLs")

def extract_reg_measure_id(href: str):
    if not href:
        return None
    try:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        if 'regMeasID' in query and query['regMeasID']:
            return str(query['regMeasID'][0])
    except Exception:
        pass
    match = re.search(r"regMeasID=(\d+)", href, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

reg_measure_id_to_url = {}
failed_facilities = []

# ============================================================================
# STEP 2: VISIT FACILITY PAGES, PICK BEST ORDER NO., DOWNLOAD PDFs
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Visiting facility pages and downloading PDFs")
print("="*80)

TYPE_RANK = {"NPDES PERMIT": 0, "CO-PERMITTEE": 1, "ENROLLEE - NPDES": 2}
url_to_facilities = {}

# Track which PDFs belong to which facilities
pdf_to_facilities = {}  # Key: downloaded PDF filename, Value: list of facilities
url_to_pdfs = {}

def list_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def wait_for_new_pdf_file(directory, before_set, timeout=60):
    """Wait for a new .pdf file to appear in directory compared to before_set,
    and for its size to stabilize."""
    end = time.time() + timeout
    while time.time() < end:
        current = set(list_pdf_files(directory))
        new_files = [f for f in current - before_set if not f.lower().endswith('.crdownload')]
        if new_files:
            newest = max(new_files, key=lambda f: os.path.getctime(os.path.join(directory, f)))
            path = os.path.join(directory, newest)
            try:
                s1 = os.path.getsize(path)
                time.sleep(0.5)
                s2 = os.path.getsize(path)
                if s1 == s2 and s1 > 0:
                    return newest
            except OSError:
                pass
        time.sleep(0.5)
    return None

downloaded_count = 0
skipped_count = 0

def download_pdfs_for_url(url, facilities, allow_noa=False):
    global downloaded_count, skipped_count
    noa_set = {'noa', 'noi'} if allow_noa else set()
    local_skip = [k for k in skip_keywords if not any(n in k.lower() for n in noa_set)]
    local_begin = [p for p in beginning_patterns if not any(p.lower().startswith(n) for n in noa_set)]
    print(f"\nProcessing URL")
    print(f"  Facilities: {', '.join([f['facility_name'] for f in facilities])}")
    print(f"  URL: {url}")

    try:
        print(f"  Opening URL in new tab...")
        driver.execute_script(f"window.open('{url}', '_blank');")
        time.sleep(1)

        new_window = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_window)
        print(f"  Switched to new window")

        time.sleep(3)
        if CIWQS_VERBOSE:
            current_url = driver.current_url
            print(f"  Current URL: {current_url}")

            page_source_length = len(driver.page_source)
            print(f"  Page source length: {page_source_length} characters")

            if page_source_length < 500:
                print(f"  X Page appears to be blank/empty!")
                print(f"  Page content preview: {driver.page_source[:500]}")

        pdf_documents = driver.find_elements(By.XPATH, "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]")
        print(f"  Found {len(pdf_documents)} PDFs on page")

        if len(pdf_documents) == 0:
            print(f"  No PDFs found, trying href-based detection...")
            pdf_documents = driver.find_elements(By.XPATH, "//a[contains(translate(@href, 'PDF', 'pdf'), '.pdf')]")
            print(f"  Href method found {len(pdf_documents)} PDFs")

            if len(pdf_documents) == 0 and CIWQS_VERBOSE:
                all_links = driver.find_elements(By.TAG_NAME, 'a')
                print(f"  Debug: Found {len(all_links)} total links on page")
                for link in all_links[:10]:
                    print(f"    Link text: '{link.text}' | href: {link.get_attribute('href')}")

        for j, pdf_doc in enumerate(pdf_documents):
            try:
                pdf_name = pdf_doc.text

                should_skip = any(keyword.lower() in pdf_name.lower() for keyword in local_skip)
                if should_skip:
                    print(f"        Skipping: {pdf_name} (matched skip keyword)")
                    skipped_count += 1
                    continue

                should_skip = any(pdf_name.lower().startswith(pattern.lower()) for pattern in local_begin)
                if should_skip:
                    print(f"        Skipping: {pdf_name} (begins with skip pattern)")
                    skipped_count += 1
                    continue

                print(f"        Downloading: {pdf_name}")
                before = set(list_pdf_files(pdfs_path))
                pdf_doc.click()
                new_file = wait_for_new_pdf_file(pdfs_path, before, timeout=60)
                if not new_file:
                    print("        X Download did not complete or file not detected")
                    continue
                downloaded_count += 1

                if url not in url_to_pdfs:
                    url_to_pdfs[url] = []
                if new_file not in url_to_pdfs[url]:
                    url_to_pdfs[url].append(new_file)

                if new_file not in pdf_to_facilities:
                    pdf_to_facilities[new_file] = []
                pdf_to_facilities[new_file].extend(facilities)

            except Exception as e:
                print(f"        X Download {j} failed: {e}")

        driver.close()
        driver.switch_to.window(main_window)
        time.sleep(1)

    except Exception as e:
        print(f"  X Failed to process URL: {e}")
        try:
            if driver.current_window_handle != main_window:
                driver.close()
            driver.switch_to.window(main_window)
        except:
            pass

def parse_date_safe(date_str):
    s = str(date_str).strip()
    if not s or s.lower() in ('null', 'none', 'nan', ''):
        return pd.NaT
    return pd.to_datetime(s, errors='coerce')


def find_best_order_href(html):
    """Parse facility page HTML with BeautifulSoup and return the best active NPDES order.

    Returns (order_href, type_rank, effective_dt, rm_type) or (None, None, pd.NaT, None).
    Uses cell-length filtering (< 80 chars) to skip outer layout tables whose cells
    contain the entire page text rather than individual column header strings.
    """
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all('table'):
        all_rows = table.find_all('tr')
        for hdr_idx, row in enumerate(all_rows[:4]):
            cells = row.find_all(['td', 'th'])
            texts = [c.get_text(strip=True) for c in cells]
            if len(texts) < 5:
                continue
            if any(len(t) > 80 for t in texts):
                continue
            if not (any('Reg Measure Type' in t for t in texts)
                    and any('Order No' in t for t in texts)):
                continue
            col_index = {t: i for i, t in enumerate(texts)}
            best_href = None
            best_type_rank = 99
            best_effective = pd.NaT
            best_rm_type = None
            for data_row in all_rows[hdr_idx + 1:]:
                dcells = data_row.find_all('td')
                if not dcells:
                    continue
                def gc(col_name, _dc=dcells, _ci=col_index):
                    i = _ci.get(col_name, -1)
                    return _dc[i].get_text(strip=True) if 0 <= i < len(_dc) else ""
                if gc('Status').lower() != 'active':
                    continue
                rm_type = gc('Reg Measure Type').upper()
                if rm_type not in TYPE_RANK:
                    continue
                order_idx = col_index.get('Order No.', -1)
                if order_idx < 0 or order_idx >= len(dcells):
                    continue
                a_tag = dcells[order_idx].find('a', href=True)
                if not a_tag:
                    continue
                href = _abs_url(a_tag['href'])
                if not href:
                    continue
                type_rank = TYPE_RANK[rm_type]
                effective_dt = parse_date_safe(gc('Effective Date'))
                if pd.isna(effective_dt):
                    continue
                if (type_rank < best_type_rank or
                        (type_rank == best_type_rank and effective_dt > best_effective)):
                    best_href = href
                    best_type_rank = type_rank
                    best_effective = effective_dt
                    best_rm_type = rm_type
            if best_href:
                return best_href, best_type_rank, best_effective, best_rm_type
    return None, None, pd.NaT, None


for idx, (facility_url, facilities) in enumerate(facility_url_to_facilities.items(), 1):
    print(f"\n[{idx}/{len(facility_url_to_facilities)}] {facilities[0]['facility_name']}")
    try:
        driver.execute_script(f"window.open('{facility_url}', '_blank');")
        time.sleep(1)
        new_win = [h for h in driver.window_handles if h != main_window][0]
        driver.switch_to.window(new_win)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
        time.sleep(1)

        page_html = driver.page_source
        driver.close()
        driver.switch_to.window(main_window)

        order_href, best_type_rank, best_effective, best_rm_type = find_best_order_href(page_html)
        if not order_href:
            print("  X No suitable active NPDES order found")
            continue

        eff_str = best_effective.date() if not pd.isna(best_effective) else 'N/A'
        print(f"  Best order: {best_rm_type}, rank={best_type_rank}, effective={eff_str}")

        reg_id = extract_reg_measure_id(order_href)
        if reg_id and reg_id in reg_measure_id_to_url:
            existing_url = reg_measure_id_to_url[reg_id]
            url_to_facilities.setdefault(existing_url, []).extend(facilities)
            for pdf_name in url_to_pdfs.get(existing_url, []):
                pdf_to_facilities.setdefault(pdf_name, []).extend(facilities)
            print(f"  Dedup: already processed reg measure {reg_id}")
            continue

        url_to_facilities[order_href] = facilities
        if reg_id:
            reg_measure_id_to_url[reg_id] = order_href
        download_pdfs_for_url(order_href, facilities, allow_noa=(best_rm_type == "ENROLLEE - NPDES"))

    except Exception as e:
        print(f"  X {e}")
        for f in facilities:
            failed_facilities.append({
                'facility_name': f['facility_name'],
                'facility_url': facility_url,
                'error': str(e)[:200],
            })
        try:
            if driver.current_window_handle != main_window:
                driver.close()
            driver.switch_to.window(main_window)
        except Exception:
            pass

if failed_facilities:
    pd.DataFrame(failed_facilities).to_csv(os.path.join(full_path, 'failed_facilities.csv'), index=False)
    print(f"Wrote {len(failed_facilities)} failed facilities to failed_facilities.csv")

print(f"\nDownloaded {downloaded_count} PDFs")
print(f"Skipped {skipped_count} PDFs (keyword filters)")

# ============================================================================
# STEP 3: PDF SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 3: PDF summary")
print("="*80)
print(f"Unique PDFs: {len(pdf_to_facilities)}")
print(f"Shared PDFs: {sum(1 for f in pdf_to_facilities.values() if len(f) > 1)}")

# ============================================================================
# STEP 4: DETECT AND MOVE NPDES PDFs TO SEPARATE FOLDER
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Detecting and moving NPDES PDFs")
print("="*80)

npdes_path = os.path.join(full_path, 'npdes')
os.makedirs(npdes_path, exist_ok=True)

npdes_pdfs = set()
non_npdes_pdfs = set()

for filename in os.listdir(pdfs_path):
    if not filename.endswith('.pdf'):
        continue

    file_path = os.path.join(pdfs_path, filename)

    try:
        if detect_npdes(file_path):
            os.rename(file_path, os.path.join(npdes_path, filename))
            print(f"NPDES detected: {filename}")
            npdes_pdfs.add(filename)
        else:
            print(f"Non-NPDES content: {filename}")
            non_npdes_pdfs.add(filename)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        non_npdes_pdfs.add(filename)

print(f"\nNPDES PDFs moved: {len(npdes_pdfs)}")
print(f"Non-NPDES PDFs kept in pdfs folder: {len(non_npdes_pdfs)}")

# ============================================================================
# STEP 5: CREATE CSV WITH ONLY NPDES PDFs
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Creating site_data.csv with NPDES permits only")
print("="*80)

pdfs_csv = open(os.path.join(full_path, 'site_data.csv'), 'w', newline='')
rename_data = csv.writer(pdfs_csv)

rename_data.writerow(['Agency', 'Facility_Name', 'NPDES_No', 'Region', 'Major/Minor', 'PDF_File', 'Shared_PDF'])

csv_row_count = 0
skipped_non_npdes_count = 0

for pdf_name, facilities in pdf_to_facilities.items():
    if pdf_name in npdes_pdfs:
        is_shared = "Yes" if len(facilities) > 1 else "No"
        seen = set()
        for facility in facilities:
            key = (facility['agency'], facility['facility_name'], facility['npdes_no'])
            if key in seen:
                continue
            seen.add(key)
            rename_data.writerow([
                facility['agency'],
                facility['facility_name'],
                facility['npdes_no'],
                facility['region'],
                facility['major_minor'],
                pdf_name,
                is_shared
            ])
            csv_row_count += 1
    elif pdf_name in non_npdes_pdfs:
        skipped_non_npdes_count += len(facilities)
        print(f"  Skipping non-NPDES PDF: {pdf_name} ({len(facilities)} facilities)")

pdfs_csv.close()

print(f"\nWrote {csv_row_count} CSV rows for NPDES permits")
print(f"Skipped {skipped_non_npdes_count} facility entries (non-NPDES PDFs)")

# ============================================================================
# SAVE TABLE DATA TO CSV (using Excel export)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving table data to CSV from Excel export")
print("="*80)

df_deduplicated.to_csv(os.path.join(full_path, 'all_ca_npdes.csv'), index=False)

print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

# Quit driver
driver.quit()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Unique facility page URLs collected: {len(facility_url_to_facilities)}")
print(f"Unique Order No. URLs visited: {len(url_to_facilities)}")
print(f"Shared Order No. URLs (multiple facilities): {sum(1 for facs in url_to_facilities.values() if len(facs) > 1)}")
print(f"PDFs downloaded: {downloaded_count}")
print(f"PDFs skipped (keywords): {skipped_count}")
print(f"NPDES PDFs detected: {len(npdes_pdfs)}")
print(f"Non-NPDES PDFs: {len(non_npdes_pdfs)}")
print(f"CSV rows written (NPDES only): {csv_row_count}")
print(f"Facility entries skipped (non-NPDES): {skipped_non_npdes_count}")
print(f"{'='*80}")