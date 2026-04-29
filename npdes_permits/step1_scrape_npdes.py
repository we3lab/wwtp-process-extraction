from datetime import datetime
import json
import os
import time
import glob
import requests
import pandas as pd
import uuid
import tempfile
from urllib.parse import urlparse, parse_qs
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

CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"

# Link to Interactive Regulated Facilities Report
rfr_url = "https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility"

# Set INCLUDE_WDR=False to restrict collection to NPDES-only facilities (exclude WDR/WDRMUNIL).
INCLUDE_WDR = True

# Visible labels for CIWQS form (must stay aligned with STEP 1 row checks).
CIWQS_PROGRAMS = ["NPDES", "WDR"] if INCLUDE_WDR else ["NPDES"]
ACCEPTED_PROGRAMS = {"NPDESWW", "NPDMUNI"} | ({"WDRMUNIL"} if INCLUDE_WDR else set())
CIWQS_FACILITY_TYPE = "Wastewater Treatment Facility"
CIWQS_WASTE_TYPE = "Domestic wastewater"
CIWQS_RELATED_PERMIT_STATUS = "Active"

# Maps output CSV key -> CIWQS table column header.
# Used both to build facility dicts and to validate header detection.
FACILITY_FIELDS = [
    ("Agency", "Agency"),
    ("Facility_Name", "Facility Name"),
    ("NPDES_No", "NPDES No."),
    ("Region", "Region"),
    ("Major/Minor", "Major/Minor"),
    ("Order_No", "Order No."),
]

# Output path — set DATE_FOLDER to re-run steps 3/4 against an existing run's folder.
DATE_FOLDER = "2026-4-26"  # set to e.g. '2026-4-26' to re-run steps 3/4 against an existing folder
path = "npdes_permits/output"
if DATE_FOLDER:
    full_path = os.path.join(path, DATE_FOLDER)
else:
    now = datetime.now()
    full_path = os.path.join(path, f"{now.year}-{now.month}-{now.day}")
pdfs_path = os.path.join(full_path, "pdfs")
os.makedirs(full_path, exist_ok=True)
os.makedirs(pdfs_path, exist_ok=True)

_SEP = " .-_"
_BASE_KW = ["noa", "noi", "rpts", "rowd", "per"]
_PHRASE_KW = [
    "report",
    "financial",
    "notice of",
    "response to",
    "rate study",
    "ratestudy",
    "study",
    "letter",
]
SKIP_CONFIG = {
    "embedded": [f"{s1}{kw}{s2}" for kw in _BASE_KW for s1 in _SEP for s2 in _SEP] + _PHRASE_KW,
    "beginning": [f"{kw}{s}" for kw in _BASE_KW for s in _SEP],
}
NOA_SET = {"noa", "noi"}


def abs_url(href):
    if not href or href.startswith("http"):
        return href
    if href.startswith("/"):
        return CIWQS_ROOT + href
    return f"{CIWQS_ROOT}/ciwqs/readOnly/{href}"


def hidden_fields(soup):
    return {
        i["name"]: i.get("value", "")
        for i in soup.find_all("input", type="hidden")
        if i.get("name")
    }


def select_value(soup, name, visible_text, *, required_label=None):
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


def retry_request(session, method, url, *, data=None, max_attempts=4, timeout=120):
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
    """Extract the best drilldown URL from a CIWQS search results page."""
    candidates = [
        abs_url(a["href"])
        for a in soup.find_all("a", href=True)
        if "RegulatedFacilityDetail" in a["href"] and "drilldown" in a["href"]
    ]
    excluded = ["place=", "majorminor="] + ([] if allow_program_scope else ["program="])
    filtered = [c for c in candidates if not any(k in c.lower() for k in excluded)]
    return filtered[0] if filtered else (candidates[0] if candidates else None)


def process_all_facilities(driver, facility_url_to_facilities, output_dir):
    """Main processing loop.

    Returns: (failed_facilities, facility_order_info)
      facility_order_info: {facility_url: {'pdfs', 'total_pdfs', 'reg_measure_id', 'reg_measure_type'}}
    """
    failed_facilities = []
    facility_order_info = {}
    reg_id_to_info = {}
    main_window = driver.window_handles[0]

    for idx, (facility_url, facilities) in enumerate(facility_url_to_facilities.items(), 1):
        print(f"\n[{idx}/{len(facility_url_to_facilities)}] {facilities[0]['Facility_Name']}")
        try:
            order_url, rm_type = find_best_order(driver, facility_url, main_window)
            if not order_url:
                print("  X No suitable active NPDES order found")
                facility_order_info[facility_url] = {
                    "pdfs": [],
                    "total_pdfs": 0,
                    "reg_measure_id": None,
                    "reg_measure_type": None,
                }
                continue

            reg_id = parse_qs(urlparse(order_url).query).get("regMeasID", [None])[0]

            if reg_id and reg_id in reg_id_to_info:
                print(f"  Dedup: reusing already-processed order {reg_id}")
                facility_order_info[facility_url] = reg_id_to_info[reg_id]
                continue

            downloaded_pdfs, total_pdfs = download_pdfs_for_order(
                driver,
                order_url,
                output_dir,
                allow_noa=(rm_type == "ENROLLEE - NPDES"),
                main_window=main_window,
            )
            info = {
                "pdfs": downloaded_pdfs,
                "total_pdfs": total_pdfs,
                "reg_measure_id": reg_id,
                "reg_measure_type": rm_type,
            }
            facility_order_info[facility_url] = info
            if reg_id:
                reg_id_to_info[reg_id] = info

        except Exception as e:
            print(f"  X {e}")
            for f in facilities:
                failed_facilities.append({**f, "error": str(e)[:200]})
            try:
                if driver.current_window_handle != main_window:
                    driver.close()
                driver.switch_to.window(main_window)
            except Exception:
                pass

    return failed_facilities, facility_order_info


def build_pdf_group_context(facility_order_info):
    """PDF filename -> reg_measure groups; group key -> PDFs (for NOA companion checks)."""
    pdf_to_groups, group_to_pdfs = {}, {}
    for url, info in facility_order_info.items():
        g = info.get("reg_measure_id") or url
        for pdf in info.get("pdfs", []):
            pdf_to_groups.setdefault(pdf, set()).add(g)
            group_to_pdfs.setdefault(g, set()).add(pdf)
    return pdf_to_groups, group_to_pdfs


def find_best_order(driver, facility_url, main_window):
    """Navigate to facility page, parse HTML, and return best active NPDES order.

    Returns: (order_url, reg_measure_type) or (None, None)
    """
    TYPE_RANK = {"NPDES PERMIT": 0, "CO-PERMITTEE": 1, "ENROLLEE - NPDES": 2}
    REQUIRED_HEADERS = ["Reg Measure Type", "Order No"]
    COLUMN_CHECKS = [
        ("Status", lambda v: v.lower() == "active"),
        ("Reg Measure Type", lambda v: v.upper() in TYPE_RANK),
    ]

    # Navigate to facility page
    driver.execute_script(f"window.open('{facility_url}', '_blank');")
    time.sleep(1)
    new_win = [h for h in driver.window_handles if h != main_window][0]
    driver.switch_to.window(new_win)

    WebDriverWait(driver, 120).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    time.sleep(1)

    page_html = driver.page_source
    driver.close()
    driver.switch_to.window(main_window)

    # Parse HTML for best order
    soup = BeautifulSoup(page_html, "html.parser")
    for table in soup.find_all("table"):
        all_rows = table.find_all("tr")

        # Find header row
        for hdr_idx, row in enumerate(all_rows[:4]):
            cells = row.find_all(["td", "th"])
            texts = [c.get_text(strip=True) for c in cells]

            # Validate header row
            if len(texts) < 5 or any(len(t) > 80 for t in texts):
                continue
            if not all(any(req in t for t in texts) for req in REQUIRED_HEADERS):
                continue

            col_index = {t: i for i, t in enumerate(texts)}
            best_href, best_type_rank = None, 99
            best_effective, best_rm_type = pd.NaT, None
            dcells = []  # rebound each data row; gc closes over it by name

            def gc(col_name):
                i = col_index.get(col_name, -1)
                return dcells[i].get_text(strip=True) if 0 <= i < len(dcells) else ""

            # Process data rows
            for data_row in all_rows[hdr_idx + 1 :]:
                dcells = data_row.find_all("td")
                if not dcells:
                    continue

                # Apply validation checks
                if not all(check_fn(gc(col_name)) for col_name, check_fn in COLUMN_CHECKS):
                    continue

                # Extract order URL
                order_idx = col_index.get("Order No.", -1)
                if order_idx < 0 or order_idx >= len(dcells):
                    continue

                a_tag = dcells[order_idx].find("a", href=True)
                if not a_tag:
                    continue

                href = abs_url(a_tag["href"])
                if not href:
                    continue

                # Calculate priority
                rm_type = gc("Reg Measure Type").upper()
                type_rank = TYPE_RANK[rm_type]
                effective_dt = pd.to_datetime(gc("Effective Date"), errors="coerce")

                if pd.isna(effective_dt):
                    continue

                # Update best if higher priority
                if type_rank < best_type_rank or (
                    type_rank == best_type_rank and effective_dt > best_effective
                ):
                    best_href = href
                    best_type_rank = type_rank
                    best_effective = effective_dt
                    best_rm_type = rm_type

            if best_href:
                eff_str = best_effective.date() if not pd.isna(best_effective) else "N/A"
                print(f"  Best order: {best_rm_type}, rank={best_type_rank}, effective={eff_str}")
                return best_href, best_rm_type

    return None, None


def download_pdfs_for_order(driver, order_url, output_dir, allow_noa=False, main_window=None):
    """Download PDFs from an order page. Returns (downloaded_pdfs, total_on_page)."""
    if main_window is None:
        main_window = driver.window_handles[0]

    active_skip = (
        {
            k: [p for p in v if not any(n in p.lower() for n in NOA_SET)]
            for k, v in SKIP_CONFIG.items()
        }
        if allow_noa
        else SKIP_CONFIG
    )

    driver.execute_script(f"window.open('{order_url}', '_blank');")
    time.sleep(1)
    new_window = [h for h in driver.window_handles if h != main_window][0]
    driver.switch_to.window(new_window)
    time.sleep(3)

    pdf_documents = driver.find_elements(
        By.XPATH, "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]"
    )
    total_on_page = len(pdf_documents)
    print(f"  Found {total_on_page} PDFs on page")

    downloaded_pdfs = []
    for pdf_element in pdf_documents:
        try:
            pdf_name = pdf_element.text
            pdf_lower = pdf_name.lower()
            if any(kw.lower() in pdf_lower for kw in active_skip["embedded"]) or any(
                pdf_lower.startswith(p.lower()) for p in active_skip["beginning"]
            ):
                print(f"        Skipping: {pdf_name}")
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
                print("        X Download did not complete or file not detected")
                continue
            downloaded_pdfs.append(new_file)
        except Exception as e:
            print(f"        X Download failed: {e}")

    driver.close()
    driver.switch_to.window(main_window)
    return downloaded_pdfs, total_on_page


def _file_stable(path):
    try:
        s = os.path.getsize(path)
        time.sleep(0.5)
        return s > 0 and os.path.getsize(path) == s
    except OSError:
        return False


def wait_for_downloads(directory, timeout=60):
    """Wait for an .xls/.xlsx file to appear and finish downloading."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        files = [
            f
            for f in glob.glob(os.path.join(directory, "*.xlsx"))
            + glob.glob(os.path.join(directory, "*.xls"))
            if not f.lower().endswith(".crdownload")
        ]
        if files and all(_file_stable(f) for f in files):
            return files
        time.sleep(0.5)
    return []


def _load_ciwqs_table(url, label="url"):
    for attempt in range(1, 4):
        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ciwqsReportDataTable")))
            return
        except TimeoutException:
            print(f"[selenium] {label} slow ({attempt}/3)…")
            if attempt == 3:
                raise
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass


def selection(driver, name, text):
    select_elements = driver.find_elements(By.NAME, name)
    if not select_elements:
        return False
    select = Select(select_elements[0])
    select.select_by_visible_text(text)
    return True


def _set_page_all():
    for attempt in range(1, 3):
        try:
            selection(driver, "pagesizeselect", "ALL")
            try:
                wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, "loading")))
            except Exception:
                pass
            return WebDriverWait(driver, 90).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//table[contains(@class,'ciwqsReportDataTable')]")
                )
            )
        except Exception as e:
            print(f"[selenium] pagesizeselect ALL did not stabilize ({attempt}/2): {e}")
            if attempt == 2:
                raise
            time.sleep(2)


# ciwqs = requests.Session()
# ciwqs.headers["User-Agent"] = (
#     "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
#     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# )
# print("[requests] CIWQS reset form…")
# r = retry_request(ciwqs, 'GET', rfr_url)
# soup0 = BeautifulSoup(r.text, "html.parser")
# hidden0 = hidden_fields(soup0)
# csrf = hidden0.get("OWASP_CSRFTOKEN", "")
# in_status = select_value(
#     soup0, "inStatus", CIWQS_RELATED_PERMIT_STATUS, required_label="Related Permit Status"
# )
# # Multi-select post → total drilldown URL (used for Excel export only).
# # Multi-select yields a grouped-by-agency view; re-submit per-program for the flat table.
# _kwargs = dict(facility_type=CIWQS_FACILITY_TYPE, waste_type=CIWQS_WASTE_TYPE, status=in_status)
# print("[requests] Submitting filters…")
# resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
#                      data=ciwqs_post_data(hidden0, soup0, CIWQS_PROGRAMS, **_kwargs))
# total_url = extract_drilldown_url(BeautifulSoup(resp.text, "html.parser"), allow_program_scope=False)
# if not total_url:
#     raise RuntimeError("CIWQS: no Total drilldown URL in search response")
# print(f"[requests] Excel export URL: {total_url}")

# # Re-submit once per program to get per-facility flat-table drilldown URLs.
# program_urls = []
# for prog in CIWQS_PROGRAMS:
#     prog_resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}",
#                               data=ciwqs_post_data(hidden0, soup0, [prog], **_kwargs))
#     prog_url = extract_drilldown_url(BeautifulSoup(prog_resp.text, "html.parser"))
#     if prog_url:
#         program_urls.append((prog, prog_url))
#         print(f"[requests] {prog} facility URL: {prog_url}")
#     else:
#         print(f"[requests] Warning: no drilldown URL found for {prog}")

# # Create driver fresh after requests has all URLs — matches old bf548c5 pattern.
# driver = new_chrome_driver(pdfs_path)
# wait = WebDriverWait(driver, 120)
# main_window = driver.current_window_handle
# driver.set_page_load_timeout(120)

# _load_ciwqs_table(total_url, "Facility page")
# print("Detail page loaded, filters should be preserved")
# time.sleep(5)
# table_body = _set_page_all()

# table_rows = table_body.find_elements(By.TAG_NAME, 'tr')
# print(f"Final table has {len(table_rows)} rows before export")

# # Export Excel file and create mapping from it
# export_link = wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
# driver.execute_script("arguments[0].scrollIntoView(true);", export_link)
# driver.execute_script("arguments[0].click();", export_link)
# time.sleep(3)
# try:
#     export_link = wait.until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
#     driver.execute_script("arguments[0].click();", export_link)
# except Exception:
#     print("Second export click not needed or timed out; continuing")

# excel_files = wait_for_downloads(pdfs_path, timeout=60)
# if not excel_files:
#     excel_files = wait_for_downloads(full_path, timeout=5)

# if not excel_files:
#     print("No Excel file found")
#     exit()

# excel_file = max(excel_files, key=os.path.getctime)
# df = pd.read_csv(excel_file, sep='\t', encoding='latin-1', on_bad_lines='warn')

# print(f"Excel file has {len(df)} rows (table had {len(table_rows)} rows)")

# # Enforce the same intent as the CIWQS form so downstream CSV/STEP 2 stay in sync.
# if "Program" in df.columns:
#     df = df[df["Program"].fillna("").str.upper().str.contains(r"NPD|WDRMUNIL", na=False, regex=True)]
# if "Regulatory Measure Status" in df.columns:
#     df = df[df["Regulatory Measure Status"].fillna("").str.upper() == CIWQS_RELATED_PERMIT_STATUS.upper()]
# if "Place/Project Type" in df.columns:
#     df = df[
#         df["Place/Project Type"].fillna("").str.upper().str.contains(CIWQS_FACILITY_TYPE.upper(), na=False)
#     ]
# print(f"After explicit form-aligned filtering: {len(df)} rows")

# df['Expiration/Review Date'] = pd.to_datetime(df['Expiration/Review Date'], errors='coerce')
# df_sorted = df.sort_values(['WDID', 'Facility Name', 'Expiration/Review Date'], ascending=[True, True, False])
# df_deduplicated = df_sorted.drop_duplicates(subset=['WDID', 'Facility Name'], keep='first')
# duplicates_removed = df_sorted[df_sorted.duplicated(subset=['WDID', 'Facility Name'], keep='first')]
# print(f"After deduplication and filtering: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")
# if len(duplicates_removed) > 0:
#     cols = [c for c in ['Facility Name', 'WDID', 'NPDES No.'] if c in duplicates_removed.columns]
#     print("Duplicates removed (Facility Name, WDID, NPDES No.):")
#     print(duplicates_removed[cols].to_string(index=False))

# df_deduplicated.to_csv(os.path.join(full_path, 'all_ca_npdes.csv'), index=False)
# print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

# # ============================================================================
# # STEP 1: COLLECT FACILITY PAGE URLs — navigate per-program for clean results
# # ===========================================================================
# print("\n STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")

# facility_url_to_facilities = {}

# # Column indices are 1-based; convert to 0-based for BS4 indexing.
# def _col(one_based_idx):
#     return (one_based_idx - 1) if one_based_idx else None

# def _cell(cells, one_based_idx):
#     i = _col(one_based_idx)
#     if i is None or i >= len(cells):
#         return ""
#     return cells[i].get_text(strip=True)

# def _cell_href(cells, one_based_idx):
#     i = _col(one_based_idx)
#     if i is None or i >= len(cells):
#         return ""
#     a = cells[i].find("a", href=True)
#     return abs_url(a["href"]) if a else ""

# # Navigate per-program URLs (NPDES, WDR) to avoid loading the full statewide dataset.
# # Column indices detected from the BS4 header row — avoids Selenium cellIndex
# # issues on CIWQS grouped tables that lack the ciwqsReportColumnName class.
# col = {}  # CIWQS header text -> 1-based column index (detected on first program)

# for prog, prog_url in program_urls:
#     print(f"\n--- {prog}: {prog_url}")
#     _load_ciwqs_table(prog_url, prog)
#     _set_page_all()

#     page_soup = BeautifulSoup(driver.page_source, "html.parser")
#     data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
#     bs_rows_prog = data_table.find_all("tr") if data_table else []
#     print(f"{prog}: {len(bs_rows_prog)} table rows")

#     if not col:
#         for _tr in bs_rows_prog:
#             _tds = _tr.find_all("td")
#             _texts = [_td.get_text(strip=True) for _td in _tds]
#             if 'Order No.' in _texts or 'Facility Name' in _texts:
#                 col = {t: i + 1 for i, t in enumerate(_texts) if t}
#                 break
#         missing = [ciwqs_col for _, ciwqs_col in FACILITY_FIELDS if not col.get(ciwqs_col)]
#         if missing:
#             raise RuntimeError(
#                 f"Missing columns in {prog} table: {missing}. "
#                 f"Found: {list(col.keys())[:10]}"
#             )
#         print(f"[debug] col indices: {col}")

#     for tr in bs_rows_prog:
#         try:
#             if tr.find("td", class_="ciwqsReportColumnName"):
#                 continue
#             cells = tr.find_all("td")
#             if not cells:
#                 continue

#             status   = _cell(cells, col.get('Regulatory Measure Status')).upper()
#             program  = _cell(cells, col.get('Program')).upper()
#             plc_type = _cell(cells, col.get('Place/Project Type')).upper()

#             if status and status != CIWQS_RELATED_PERMIT_STATUS.upper():
#                 continue
#             if plc_type and CIWQS_FACILITY_TYPE.upper() not in plc_type:
#                 continue
#             if program and not any(p in program for p in ACCEPTED_PROGRAMS):
#                 continue

#             facility_url = _cell_href(cells, col.get('Facility Name'))
#             if not facility_url:
#                 continue

#             facility = {key: _cell(cells, col.get(ciwqs_col)) for key, ciwqs_col in FACILITY_FIELDS}
#             facility['Facility_URL'] = facility_url
#             facility_url_to_facilities.setdefault(facility_url, []).append(facility)
#         except Exception as e:
#             print(f"Row parse error: {e}")
#             continue

# print(f"\n✓ Found {len(facility_url_to_facilities)} unique facility URLs")

# with open(os.path.join(full_path, 'facility_urls.json'), 'w') as f:
#     json.dump(facility_url_to_facilities, f, indent=2, default=str)
# print(f"Checkpoint saved: {len(facility_url_to_facilities)} facility URLs → facility_urls.json")

# ============================================================================
# STEP 2: VISIT FACILITY PAGES, PICK BEST ORDER NO., DOWNLOAD PDFs
# To restart from here (skip Step 1 URL scraping), comment out everything above
# through the checkpoint save and instead load the checkpoint:
#
with open(os.path.join(full_path, "facility_urls.json")) as f:
    facility_url_to_facilities = json.load(f)
driver = new_chrome_driver(pdfs_path)
wait = WebDriverWait(driver, 120)
main_window = driver.current_window_handle
driver.set_page_load_timeout(120)

# ============================================================================
print("\n STEP 2: Visiting facility pages and downloading PDFs")

failed_facilities, facility_order_info = process_all_facilities(
    driver=driver,
    facility_url_to_facilities=facility_url_to_facilities,
    output_dir=pdfs_path,
)

if failed_facilities:
    pd.DataFrame(failed_facilities).to_csv(
        os.path.join(full_path, "failed_facilities.csv"), index=False
    )
    print(f"Wrote {len(failed_facilities)} failed facilities to failed_facilities.csv")

order_info_path = os.path.join(full_path, "facility_order_info.json")
with open(order_info_path, "w") as f:
    json.dump(facility_order_info, f, indent=2, default=str)
print(f"Checkpoint saved: facility_order_info.json")

# Quit driver
driver.quit()

print(f"Unique facility page URLs collected: {len(facility_url_to_facilities)}")

# ============================================================================
# To re-run from STEP 3 (PDFs already in pdfs/), comment out STEP 2 above and uncomment:
# with open(os.path.join(full_path, 'facility_urls.json')) as f:
#     facility_url_to_facilities = json.load(f)
# with open(os.path.join(full_path, 'facility_order_info.json')) as f:
#     facility_order_info = json.load(f)
# ============================================================================

# ============================================================================
# STEP 3: DETECT AND MOVE NPDES PDFs TO SEPARATE FOLDER
# ============================================================================
print("\n STEP 3: Detecting and moving NPDES PDFs")

npdes_path = os.path.join(full_path, "npdes")
os.makedirs(npdes_path, exist_ok=True)

npdes_pdfs = set()
non_npdes_pdfs = set()
pdf_to_groups, group_to_pdfs = build_pdf_group_context(facility_order_info)
pdf_signals = {}

# classify everything in pdfs/ and move NPDES files to npdes/
scan_dirs = [(pdfs_path, True)]  # (dir, move_to_npdes_if_pass)

for scan_dir, _ in scan_dirs:
    for filename in os.listdir(scan_dir):
        if not filename.endswith(".pdf"):
            continue
        path = os.path.join(scan_dir, filename)
        try:
            pdf_signals[filename] = detect_npdes(path)
        except Exception as e:
            print(f"Error reading signals for {filename}: {e}")
            non_npdes_pdfs.add(filename)

group_has_noa = {
    g: any(pdf_signals[p]["has_noa"] for p in ps if p in pdf_signals)
    for g, ps in group_to_pdfs.items()
}

for scan_dir, move_if_pass in scan_dirs:
    for filename in os.listdir(scan_dir):
        if not filename.endswith(".pdf"):
            continue
        signals = pdf_signals.get(filename)
        if signals is None:
            continue
        path = os.path.join(scan_dir, filename)
        try:
            if signals["is_generic_cag"] and any(group_has_noa.get(g, False) for g in pdf_to_groups.get(filename, ())):
                print(f"Skipped generic CAG (NOA in same order): {filename}")
                non_npdes_pdfs.add(filename)
                continue
            if signals["is_npdes"]:
                if move_if_pass:
                    os.rename(path, os.path.join(npdes_path, filename))
                print(f"NPDES detected: {filename}")
                npdes_pdfs.add(filename)
            else:
                non_npdes_pdfs.add(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            non_npdes_pdfs.add(filename)

print(f"\nNPDES PDFs moved: {len(npdes_pdfs)}")
print(f"Non-NPDES PDFs kept in pdfs folder: {len(non_npdes_pdfs)}")

# ============================================================================
# STEP 4: CREATE CSV WITH ONLY NPDES PDFs
# ============================================================================
print("\n STEP 4: Creating site_data.csv with NPDES permits only")

site_data_path = os.path.join(full_path, "site_data.csv")


def _build_npdes_to_wdid(full_path):
    """Build NPDES_No → WDID lookup from all_ca_npdes.csv or the raw Excel export."""
    csv_path = os.path.join(full_path, "all_ca_npdes.csv")
    xls_path = os.path.join(full_path, "pdfs", "Regualted_Facility_Report_Detail.xls")
    for path, sep in [(csv_path, ","), (xls_path, "\t")]:
        if os.path.exists(path):
            df = pd.read_csv(
                path, sep=sep, dtype=str, encoding="latin-1", on_bad_lines="warn"
            ).fillna("")
            permit_col = next((c for c in ["NPDES No.", "NPDES_No"] if c in df.columns), None)
            if permit_col and "WDID" in df.columns:
                return df.set_index(df[permit_col].str.strip().str.upper())["WDID"].to_dict()
    return {}


npdes_to_wdid = _build_npdes_to_wdid(full_path)
print(f"  WDID lookup: {len(npdes_to_wdid)} entries")

# Count distinct facilities mapping to each NPDES PDF (for Shared_PDF flag)
pdf_to_n_facilities = {}
for facility_url, info in facility_order_info.items():
    n = len(facility_url_to_facilities.get(facility_url, []))
    for pdf in info["pdfs"]:
        if pdf in npdes_pdfs:
            pdf_to_n_facilities[pdf] = pdf_to_n_facilities.get(pdf, 0) + n

rows = []
WDR_TOKEN = "WDR"


def _derive_doc_mode(pdf_name, reg_measure_type):
    signals = pdf_signals.get(pdf_name, {})
    if signals.get("has_noa"):
        return "noa"
    return "wdr" if WDR_TOKEN in str(reg_measure_type or "").upper() else "npdes"


for facility_url, info in facility_order_info.items():
    for facility in facility_url_to_facilities.get(facility_url, []):
        for pdf in info["pdfs"]:
            if pdf not in npdes_pdfs:
                continue
            npdes_key = facility.get("NPDES_No", "").strip().upper()
            rows.append(
                {
                    "WDID": npdes_to_wdid.get(npdes_key, ""),
                    **facility,
                    "Reg_Measure_ID": info["reg_measure_id"],
                    "Reg_Measure_Type": info["reg_measure_type"],
                    "PDF_File": pdf,
                    "Doc_Mode": _derive_doc_mode(pdf, info["reg_measure_type"]),
                    "Shared_PDF": ("Yes" if pdf_to_n_facilities.get(pdf, 0) > 1 else "No"),
                    "Total_PDFs_Available": info["total_pdfs"],
                }
            )

pd.DataFrame(rows).to_csv(site_data_path, index=False)
print(f"Wrote {len(rows)} rows to site_data.csv")
print(f"Non-NPDES PDFs: {len(non_npdes_pdfs)}")
