from contextlib import contextmanager
from datetime import datetime
import json
import os
import re
import time
import glob
import requests
import pandas as pd
import uuid
import tempfile
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode, urljoin, urlunparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.select import Select
from selenium.common.exceptions import TimeoutException
from helpers.npdes_detection import detect_npdes

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
# CIWQS_DRILLDOWN_QUERY_DROP = ()

WAIT_TIME = 300  # For CIWQS/grid/export - large number of results
CIWQS_OVERLAY_WAIT = 180  # loading spinner / overlay after changing page size
FACILITY_CIWS_COLUMNS = ["WDID", "Facility Name", "NPDES No."]
XP_GRID = "//table[contains(@class,'ciwqsReportDataTable')]"

DATE_FOLDER = None  # set to e.g. '2026-4-26' to re-run steps 3/4 against an existing folder
OUT = "npdes_permits/output"
full_path = os.path.join(OUT, DATE_FOLDER or f"{datetime.now().year}-{datetime.now().month}-{datetime.now().day}")
pdfs_path = os.path.join(full_path, "pdfs")
os.makedirs(full_path, exist_ok=True)
os.makedirs(pdfs_path, exist_ok=True)

# PDF filenames matching this regex are skipped on the order page.
SEP = " .-_"
BASE_KW = ["rpts|rowd|memo|nov"]  # match only with separators (e.g. "_memo_")
PHRASE_KW = "report|financial|notice of|response to|rate study|ratestudy|study|letter" # match anywhere in filename
SKIP_RE = re.compile(rf"^(?:{'|'.join(BASE_KW)})[{SEP}]|[{SEP}](?:{'|'.join(BASE_KW)})[{SEP}]|{PHRASE_KW}", re.IGNORECASE)

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


def find_best_order(driver, fac_url, main_window):
    """Navigate to facility page, parse HTML, and return best active NPDES order.

    Returns: (order_url, reg_measure_type) or (None, None)
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

            # Process data rows: collect (rank, -eff, href, rm_type, eff) tuples; min() picks the best.
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

                # Require a clickable Order No.
                order_idx = col_index.get("Order No.", -1)
                a_tag = dcells[order_idx].find("a", href=True) if 0 <= order_idx < len(dcells) else None
                href = abs_url(a_tag["href"]) if a_tag else None
                eff = pd.to_datetime(gc("Effective Date"), errors="coerce")
                if not href or pd.isna(eff):
                    continue

                # Calculate priority
                candidates.append((TYPE_RANK[rm_type], -eff.value, href, rm_type, eff))

            # Update best if higher priority (lowest tuple wins: smaller rank, then newer eff date)
            if candidates:
                rank, _, href, rm_type, eff = min(candidates)
                print(f"  Best order: {rm_type}, rank={rank}, effective={eff.date()}")
                return href, rm_type

    return None, None


def download_pdfs_for_order(driver, order_url, output_dir, main_window):
    """Download PDFs from an order page. Returns (downloaded_pdfs, total_on_page)."""
    downloaded_pdfs = []
    with _open_in_new_tab(driver, order_url, main_window, post_switch_sleep=3):
        pdf_documents = driver.find_elements(
            By.XPATH, "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
        )
        total_on_page = len(pdf_documents)
        print(f"  Found {total_on_page} PDFs on page")

        for pdf_element in pdf_documents:
            try:
                pdf_name = pdf_element.text
                if SKIP_RE.search(pdf_name):
                    continue

                print(f"        Downloading: {pdf_name}")
                before = {f for f in os.listdir(output_dir) if f.lower().endswith(".pdf")}
                pdf_element.click()

                end = time.time() + 60
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
                    continue
                downloaded_pdfs.append(new_file)
            except Exception as e:
                print(f"        X Download failed: {e}")

    return downloaded_pdfs, total_on_page


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
        candidates = [f for d in (pdfs_path, full_path)
                    for f in glob.glob(os.path.join(d, "*.xls*"))
                    if not f.lower().endswith(".crdownload")]
        if candidates or any(
            glob.glob(os.path.join(root, "*.crdownload")) for root in (pdfs_path, full_path)
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
    df = pd.read_csv(excel_file, sep='\t', encoding='latin-1', on_bad_lines='warn', low_memory=False)

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

    df_deduplicated.to_csv(os.path.join(full_path, "all_ca_npdes.csv"), index=False)
    print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

    driver.quit()
    return program_urls


def collect_facility_page_urls(program_urls):
    print("\n STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")

    facilities_by_place = {}  # place_id -> {"facilities": [dict keyed by FACILITY_CIWS_COLUMNS, ...]}

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
                program_cell = _cell(cells, col.get("Program")).upper()
                plc_type = _cell(cells, col.get("Place/Project Type")).upper()

                if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
                    continue
                if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
                    continue
                if program_cell and not any(p in program_cell for p in ACCEPTED_PROGRAMS):
                    continue

                place_id = parse_qs(
                    urlparse(_cell_href(cells, col.get("Facility Name"))).query
                ).get("placeID", [None])[0]
                if not place_id:
                    continue

                facility = {name: _cell(cells, col.get(name)) for name in FACILITY_CIWS_COLUMNS}
                entry = facilities_by_place.setdefault(place_id, {"facilities": []})
                if facility not in entry["facilities"]:
                    entry["facilities"].append(facility)
            except Exception as e:
                print(f"Row parse error: {e}")
                continue

        driver.quit()

    print(f"\n✓ Found {len(facilities_by_place)} unique facilities (placeIDs)")

    with open(os.path.join(full_path, 'facilities.json'), 'w') as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: {len(facilities_by_place)} facilities → facilities.json")

    return facilities_by_place

def download_facility_page_pdfs(facilities_by_place):
    driver = new_chrome_driver(pdfs_path)
    driver.set_page_load_timeout(WAIT_TIME)
    print("\n STEP 2: Visiting facility pages and downloading PDFs")

    failed_facilities = []
    reg_id_to_info = {}
    main_window = driver.window_handles[0]
    # loop mutates each entry in facilities_by_place in-place to add 'pdfs', 'total_pdfs', 'reg_measure_id', 'reg_measure_type'
    for idx, (place_id, entry) in enumerate(facilities_by_place.items(), 1):
        fac_url = facility_url(place_id)
        print(f"\n[{idx}/{len(facilities_by_place)}] {entry['facilities'][0]['Facility Name']}")
        try:
            order_url, rm_type = find_best_order(driver, fac_url, main_window)
            if not order_url:
                print("  X No suitable active NPDES order found")
                entry.update(pdfs=[], total_pdfs=0, reg_measure_id=None, reg_measure_type=None)
                continue

            reg_id = parse_qs(urlparse(order_url).query).get("regMeasID", [None])[0]

            if reg_id and reg_id in reg_id_to_info:
                print(f"  Dedup: reusing already-processed order {reg_id}")
                entry.update(reg_id_to_info[reg_id])
                continue

            downloaded_pdfs, total_pdfs = download_pdfs_for_order(
                driver, order_url, pdfs_path, main_window
            )
            info = {
                "pdfs": downloaded_pdfs,
                "total_pdfs": total_pdfs,
                "reg_measure_id": reg_id,
                "reg_measure_type": rm_type,
            }
            entry.update(info)
            if reg_id:
                reg_id_to_info[reg_id] = info

        except Exception as e:
            print(f"  X {e}")
            for f in entry["facilities"]:
                failed_facilities.append({**f, "place_id": place_id, "error": str(e)[:200]})

    if failed_facilities:
        pd.DataFrame(failed_facilities).to_csv(
            os.path.join(full_path, "failed_facilities.csv"), index=False
        )
        print(f"Wrote {len(failed_facilities)} failed facilities to failed_facilities.csv")

    with open(os.path.join(full_path, "facilities.json"), "w") as f:
        json.dump(facilities_by_place, f, indent=2, default=str)
    print(f"Checkpoint saved: facilities.json (with order info)")

    driver.quit()
    return facilities_by_place


def detect_and_move_npdes_pdfs(facilities_by_place):
    print("\n STEP 3: Detecting and moving NPDES PDFs")
    npdes_path = os.path.join(full_path, "npdes")
    os.makedirs(npdes_path, exist_ok=True)

    npdes_pdfs = set()
    non_npdes_pdfs = set()
    # PDF to Reg MeasureGrouping
    pdf_to_groups, group_to_pdfs = {}, {}
    for place_id, entry in facilities_by_place.items():
        g = entry.get("reg_measure_id") or place_id
        for pdf in entry.get("pdfs", []):
            pdf_to_groups.setdefault(pdf, set()).add(g)
            group_to_pdfs.setdefault(g, set()).add(pdf)
    pdf_signals = {}

    # Detect NPDES signals for every PDF in pdfs/, then move NPDES-positive files to npdes/.
    pdf_files = [f for f in os.listdir(pdfs_path) if f.endswith(".pdf")]

    for filename in pdf_files:
        pdf_signals[filename] = detect_npdes(os.path.join(pdfs_path, filename))

    group_has_noa = {
        g: any(pdf_signals[p]["has_noa"] for p in ps if p in pdf_signals)
        for g, ps in group_to_pdfs.items()
    }

    for filename in pdf_files:
        signals = pdf_signals[filename]
        src = os.path.join(pdfs_path, filename)
        if signals["is_generic_cag"] and any(
            group_has_noa.get(g, False) for g in pdf_to_groups.get(filename, ())
        ):
            print(f"Skipped generic CAG (NOA in same order): {filename}")
            non_npdes_pdfs.add(filename)
        elif signals["is_npdes"]:
            os.rename(src, os.path.join(npdes_path, filename))
            print(f"NPDES detected: {filename}")
            npdes_pdfs.add(filename)
        else:
            non_npdes_pdfs.add(filename)

    print(f"\nNPDES PDFs moved: {len(npdes_pdfs)}")
    print(f"Non-NPDES PDFs kept in pdfs folder: {len(non_npdes_pdfs)}")

    return npdes_pdfs, non_npdes_pdfs, pdf_signals


def create_site_data_csv(facilities_by_place, npdes_pdfs, pdf_signals):
    print("\n STEP 4: Creating site_data.csv with NPDES permits only")

    # Build (WDID, Facility Name) → {Agency, Region, Major/Minor, Order_No} from
    # all_ca_npdes.csv (same key as excel dedupe; consistent with WDID+FACILITY in step2).
    csv_path = os.path.join(full_path, "all_ca_npdes.csv")
    xls_path = os.path.join(full_path, "pdfs", "Regualted_Facility_Report_Detail.xls")
    meta_keys = ("Agency", "Region", "Major/Minor", "Order_No", "WDID")
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
                cell = {col: row.get(col, "") for col in ("Agency", "Region", "Major/Minor", "WDID")}
                cell["Order_No"] = row.get("Order No.", "")
                enrich[key] = cell
        break
    print(f"  Enrichment lookup: {len(enrich)} entries from all_ca_npdes.csv")

    # Count distinct facilities mapping to each NPDES PDF (for Shared_PDF flag)
    pdf_to_n_facilities = {}
    for entry in facilities_by_place.values():
        n = len(entry.get("facilities", []))
        for pdf in entry.get("pdfs", []):
            if pdf in npdes_pdfs:
                pdf_to_n_facilities[pdf] = pdf_to_n_facilities.get(pdf, 0) + n

    def _derive_RegMeasureType(pdf_name, reg_measure_type):
        signals = pdf_signals.get(pdf_name, {})
        if signals.get("has_noa"):
            return "NOA"
        return "WDR" if "WDR" in str(reg_measure_type or "").upper() else "NPDES"

    rows = []
    for place_id, entry in facilities_by_place.items():
        fac_url = facility_url(place_id)
        rm_type = entry.get("reg_measure_type")
        for facility in entry.get("facilities", []):
            facility_values = {key: facility[key].strip() for key in FACILITY_CIWS_COLUMNS}
            meta = enrich.get(
                (facility_values["WDID"], facility_values["Facility Name"]), {}
            )
            for pdf in entry.get("pdfs", []):
                if pdf not in npdes_pdfs:
                    continue
                meta_dict = {key: meta.get(key, "") for key in meta_keys}
                rows.append(
                    {
                        **facility_values,
                        **meta_dict,
                        "Facility_URL": fac_url,
                        "Reg_Measure_ID": entry.get("reg_measure_id"),
                        "Reg_Measure_Type": rm_type, "PDF_File": pdf,
                        "RegMeasureType": _derive_RegMeasureType(pdf, rm_type),
                        "Shared_PDF": ("Yes" if pdf_to_n_facilities.get(pdf, 0) > 1 else "No"),
                        "Total_PDFs_Available": entry.get("total_pdfs")
                    }
                )

    pd.DataFrame(rows).to_csv(os.path.join(full_path, "site_data.csv"), index=False)
    print(f"Wrote {len(rows)} rows to site_data.csv")

if __name__ == "__main__":
    # To restart from a checkpoint, replace the step(s) above with:
    #   with open(os.path.join(full_path, "facilities.json")) as f:
    #       facilities = json.load(f)

    # Sometimes it takes a few tries to get the facility results page to load / Excel to download
    # Try a few times or wait a couple of hours to let CIWQS stabilize if needed
    program_urls = run_ciwqs_search()
    facilities = collect_facility_page_urls(program_urls)
    facilities = download_facility_page_pdfs(facilities)
    npdes_pdfs, _, pdf_signals = detect_and_move_npdes_pdfs(facilities)
    create_site_data_csv(facilities, npdes_pdfs, pdf_signals)
