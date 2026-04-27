from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import json
import os
import time
import csv
import glob
import requests
from bs4 import BeautifulSoup
import pandas as pd
from helpers.npdes_detection import detect_npdes
from npdes_permits.helpers.scraping import process_all_facilities, abs_url, hidden_fields, select_value, retry_request, new_chrome_driver, selection

CIWQS_ROOT = "https://ciwqs.waterboards.ca.gov"
CIWQS_SERVLET = f"{CIWQS_ROOT}/ciwqs/readOnly/CiwqsReportServlet"

# Link to Interactive Regulated Facilities Report
rfr_url = 'https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility'

# Visible labels for CIWQS form (must stay aligned with post_data and STEP 1 row checks).
CIWQS_PROGRAMS = ["NPDES", "WDR"]   # both selected; drives form submission and row filtering
CIWQS_FACILITY_TYPE = "Wastewater Treatment Facility"
CIWQS_WASTE_TYPE = "Domestic wastewater"
CIWQS_RELATED_PERMIT_STATUS = "Active"
CIWQS_TEST_REGION = ""       # set to "1"–"9" to scope to a single region; leave "" for statewide
# CIWQS_TEST_REGION = "5"       # set to "1"–"9" to scope to a single region; leave "" for statewide
# RETRY_FAILED_DIR  = None
RETRY_FAILED_DIR  = "npdes_permits/output/2026-4-26"  # set to a previous run's output dir to retry only failed facilities. Otherwise set to None

# Output path — retry mode reuses the previous run's directory.
path = 'npdes_permits/output'
now = datetime.now()
pdf_folder = f'{now.year}-{now.month}-{now.day}'
full_path = RETRY_FAILED_DIR if RETRY_FAILED_DIR else os.path.join(path, pdf_folder)
pdfs_path = os.path.join(full_path, 'pdfs')
os.makedirs(full_path, exist_ok=True)
os.makedirs(pdfs_path, exist_ok=True)

driver = new_chrome_driver(pdfs_path)
wait = WebDriverWait(driver, 120)
main_window = driver.current_window_handle
driver.set_page_load_timeout(120)

if not RETRY_FAILED_DIR:
    ciwqs = requests.Session()
    ciwqs.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    print("[requests] CIWQS reset form…")
    r = retry_request(ciwqs, 'GET', rfr_url)
    soup0 = BeautifulSoup(r.text, "html.parser")
    hidden0 = hidden_fields(soup0)
    csrf = hidden0.get("OWASP_CSRFTOKEN", "")
    in_status = select_value(
        soup0, "inStatus", CIWQS_RELATED_PERMIT_STATUS, required_label="Related Permit Status"
    )
    # Multi-select: send one programDrop value per program (list-of-tuples supports repeated keys)
    post_data = (
        list(hidden0.items())
        + [("programDrop", select_value(soup0, "programDrop", prog)) for prog in CIWQS_PROGRAMS]
        + [
            ("typeDrop",      select_value(soup0, "typeDrop",      CIWQS_FACILITY_TYPE)),
            ("wasteTypeDrop", select_value(soup0, "wasteTypeDrop", CIWQS_WASTE_TYPE)),
            ("inStatus",      in_status),
            ("enpRepButton",  ""),
        ]
    )
    print("[requests] Submitting filters…")
    resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}", data=post_data)
    soup1 = BeautifulSoup(resp.text, "html.parser")
    drilldown_candidates = [
        abs_url(a["href"]) for a in soup1.find_all("a", href=True)
        if "RegulatedFacilityDetail" in a["href"] and "drilldown" in a["href"]
    ]

    # Total URL (no place/majorminor/program scope) — used for Excel export only.
    # Multi-select (NPDES+WDR) submission yields a grouped-by-agency view on drilldown;
    # for STEP 1 we re-submit once per program to get the per-facility flat table.
    total_url = next(
        (c for c in drilldown_candidates
         if "place=" not in c.lower() and "majorminor=" not in c.lower() and "program=" not in c.lower()),
        drilldown_candidates[0] if drilldown_candidates else None,
    )
    if not total_url:
        raise RuntimeError("CIWQS: no Total drilldown URL in search response")
    print(f"[requests] Excel export URL: {total_url}")

    # Re-submit once per program (single-program post) to get per-facility drilldown URLs.
    # Single-program submissions return the non-grouped flat table that column detection works on.
    program_urls = []
    for prog in CIWQS_PROGRAMS:
        single_post = (
            list(hidden0.items())
            + [("programDrop", select_value(soup0, "programDrop", prog))]
            + [
                ("typeDrop",      select_value(soup0, "typeDrop",      CIWQS_FACILITY_TYPE)),
                ("wasteTypeDrop", select_value(soup0, "wasteTypeDrop", CIWQS_WASTE_TYPE)),
                ("inStatus",      in_status),
                ("enpRepButton",  ""),
            ]
        )
        prog_resp = retry_request(ciwqs, 'POST', f"{CIWQS_SERVLET}?OWASP_CSRFTOKEN={csrf}", data=single_post)
        prog_soup = BeautifulSoup(prog_resp.text, "html.parser")
        prog_candidates = [
            abs_url(a["href"]) for a in prog_soup.find_all("a", href=True)
            if "RegulatedFacilityDetail" in a["href"] and "drilldown" in a["href"]
        ]
        if CIWQS_TEST_REGION:
            prog_url = next(
                (c for c in prog_candidates if f"place={CIWQS_TEST_REGION}" in c.lower()),
                None,
            )
        else:
            prog_url = next(
                (c for c in prog_candidates
                 if "place=" not in c.lower() and "majorminor=" not in c.lower()),
                prog_candidates[0] if prog_candidates else None,
            )
        if prog_url:
            program_urls.append((prog, prog_url))
            print(f"[requests] {prog} facility URL: {prog_url}")
        else:
            print(f"[requests] Warning: no drilldown URL found for {prog}")

    for attempt in range(1, 4):
        try:
            driver.get(total_url)
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
        page_size_set = selection(driver, 'pagesizeselect', 'ALL')
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
    df = pd.read_csv(excel_file, sep='\t', encoding='latin-1', on_bad_lines='warn')

    print(f"Excel file has {len(df)} rows (table had {len(table_rows)} rows)")

    # Enforce the same intent as the CIWQS form so downstream CSV/STEP 2 stay in sync.
    if "Program" in df.columns:
        df = df[df["Program"].fillna("").str.upper().str.contains(r"NPD|WDRMUNIL", na=False, regex=True)]
    if "Regulatory Measure Status" in df.columns:
        df = df[df["Regulatory Measure Status"].fillna("").str.upper() == CIWQS_RELATED_PERMIT_STATUS.upper()]
    if "Place/Project Type" in df.columns:
        df = df[
            df["Place/Project Type"].fillna("").str.upper().str.contains(CIWQS_FACILITY_TYPE.upper(), na=False)
        ]
    print(f"After explicit form-aligned filtering: {len(df)} rows")

    df['Expiration/Review Date'] = pd.to_datetime(df['Expiration/Review Date'], errors='coerce')
    df_sorted = df.sort_values(['WDID', 'Facility Name', 'Expiration/Review Date'], ascending=[True, True, False])
    df_deduplicated = df_sorted.drop_duplicates(subset=['WDID', 'Facility Name'], keep='first')
    duplicates_removed = df_sorted[df_sorted.duplicated(subset=['WDID', 'Facility Name'], keep='first')]
    print(f"After deduplication and filtering: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")
    if len(duplicates_removed) > 0:
        cols = [c for c in ['Facility Name', 'WDID', 'NPDES No.'] if c in duplicates_removed.columns]
        print("Duplicates removed (Facility Name, WDID, NPDES No.):")
        print(duplicates_removed[cols].to_string(index=False))

    # ============================================================================
    # STEP 1: COLLECT FACILITY PAGE URLs — navigate per-program for clean results
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: Collecting facility page URLs for Active NPDES+WDR/WWTF rows")
    print("="*80)

    ACCEPTED_PROGRAMS = {"NPDESWW", "NPDMUNI", "WDRMUNIL"}
    facility_url_to_facilities = {}

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
        return abs_url(a["href"]) if a else ""

    # Navigate per-program URLs (NPDES, WDR) to avoid loading the full statewide dataset.
    # Column indices are detected from the BS4 header row — avoids Selenium cellIndex
    # issues on CIWQS grouped tables that lack the ciwqsReportColumnName class.
    col_indices_found = False
    order_no_col_index = agency_col_index = facility_col_index = npdes_col_index = None
    region_col_index = major_minor_col_index = program_col_index = None
    reg_measure_status_col_index = place_type_col_index = None

    for prog, prog_url in program_urls:
        print(f"\n--- {prog}: {prog_url}")
        for attempt in range(1, 4):
            try:
                driver.get(prog_url)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ciwqsReportDataTable")))
                break
            except TimeoutException:
                print(f"[selenium] {prog} page slow ({attempt}/3)...")
                if attempt == 3:
                    raise
                try:
                    driver.execute_script("window.stop();")
                except Exception:
                    pass

        for attempt in range(1, 3):
            try:
                _set_page_size_all_and_wait()
                break
            except Exception as e:
                print(f"[selenium] pagesizeselect ALL did not stabilize ({attempt}/2): {e}")
                if attempt == 2:
                    raise
                time.sleep(2)

        page_soup = BeautifulSoup(driver.page_source, "html.parser")
        data_table = page_soup.find("table", class_=lambda c: c and "ciwqsReportDataTable" in c)
        bs_rows_prog = data_table.find_all("tr") if data_table else []
        print(f"{prog}: {len(bs_rows_prog)} table rows")

        if not col_indices_found:
            # Find the first row that looks like a column header (contains "Order No.").
            _col_idx_map = {}
            for _tr in bs_rows_prog:
                _tds = _tr.find_all("td")
                _texts = [_td.get_text(strip=True) for _td in _tds]
                if 'Order No.' in _texts or 'Facility Name' in _texts:
                    _col_idx_map = {t: i + 1 for i, t in enumerate(_texts) if t}
                    break
            order_no_col_index           = _col_idx_map.get('Order No.')
            agency_col_index             = _col_idx_map.get('Agency')
            facility_col_index           = _col_idx_map.get('Facility Name')
            npdes_col_index              = _col_idx_map.get('NPDES No.')
            region_col_index             = _col_idx_map.get('Region')
            major_minor_col_index        = _col_idx_map.get('Major/Minor')
            program_col_index            = _col_idx_map.get('Program')
            reg_measure_status_col_index = _col_idx_map.get('Regulatory Measure Status')
            place_type_col_index         = _col_idx_map.get('Place/Project Type')
            print(f"[debug] col indices: order_no={order_no_col_index} facility={facility_col_index} "
                  f"program={program_col_index} status={reg_measure_status_col_index} "
                  f"plc_type={place_type_col_index} region={region_col_index}")
            if not order_no_col_index:
                raise RuntimeError(
                    f"Could not find 'Order No.' header in {prog} table. "
                    f"Found cols: {list(_col_idx_map.keys())[:10]}"
                )
            col_indices_found = True

        for tr in bs_rows_prog:
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
                if program and not any(p in program for p in ACCEPTED_PROGRAMS):
                    continue
                if CIWQS_TEST_REGION and region_col_index:
                    row_region = _cell(cells, region_col_index).upper()
                    if row_region and not row_region.startswith(CIWQS_TEST_REGION):
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

    print(f"\n✓ Found {len(facility_url_to_facilities)} unique facility URLs")

else:
    # ============================================================================
    # RETRY MODE: load failed facilities from previous run
    # ============================================================================
    print("\n" + "="*80)
    print(f"RETRY MODE: loading failed facilities from {full_path}")
    print("="*80)
    failed_csv = os.path.join(full_path, 'failed_facilities.csv')
    if not os.path.exists(failed_csv):
        raise RuntimeError(f"No failed_facilities.csv found in {full_path}")
    failed_df = pd.read_csv(failed_csv, dtype=str).fillna('')
    facility_url_to_facilities = {}
    for _, row in failed_df.iterrows():
        fac = {k: row.get(k, '') for k in
               ['agency', 'facility_name', 'npdes_no', 'region', 'major_minor', 'order_no', 'facility_url']}
        facility_url_to_facilities.setdefault(row['facility_url'], []).append(fac)
    df_deduplicated = pd.read_csv(os.path.join(full_path, 'site_data.csv'), dtype=str)
    print(f"✓ {len(facility_url_to_facilities)} facility URLs to retry")

    # ============================================================================
    # SAVE TABLE DATA TO CSV (using Excel export)
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: Saving table data to CSV from Excel export")
    print("="*80)

    df_deduplicated.to_csv(os.path.join(full_path, 'all_ca_npdes.csv'), index=False)

    print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

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

reg_measure_id_to_url = {}
failed_facilities = []

# ============================================================================
# STEP 2: VISIT FACILITY PAGES, PICK BEST ORDER NO., DOWNLOAD PDFs
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Visiting facility pages and downloading PDFs")
print("="*80)

# Prepare skip keywords structure for process_all_facilities
skip_keywords_config = {
    'embedded': skip_keywords,
    'beginning': beginning_patterns
}

# Call the helper function
failed_facilities = process_all_facilities(
    driver=driver,
    facility_url_to_facilities=facility_url_to_facilities,
    output_dir=pdfs_path,
    skip_keywords=skip_keywords_config
)

# Save failed facilities
if failed_facilities:
    pd.DataFrame(failed_facilities).to_csv(
        os.path.join(full_path, 'failed_facilities.csv'), 
        index=False
    )
    print(f"Wrote {len(failed_facilities)} failed facilities to failed_facilities.csv")


# ============================================================================
# STEP 3: PDF SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 3: PDF summary")
print("="*80)

# Quit driver
driver.quit()

print(f"Unique facility page URLs collected: {len(facility_url_to_facilities)}")

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

# classify everything in pdfs/ and move NPDES files to npdes/
scan_dirs = [(pdfs_path, True)]   # (dir, move_to_npdes_if_pass)

for scan_dir, move_if_pass in scan_dirs:
    for filename in os.listdir(scan_dir):
        if not filename.endswith('.pdf'):
            continue

        file_path = os.path.join(scan_dir, filename)

        try:
            if detect_npdes(file_path):
                if move_if_pass:
                    os.rename(file_path, os.path.join(npdes_path, filename))
                print(f"NPDES detected: {filename}")
                npdes_pdfs.add(filename)
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

site_data_path = os.path.join(full_path, 'site_data.csv')

# Append when retrying a facility
open_mode = 'a' if (RETRY_FAILED_DIR and os.path.exists(site_data_path)) else 'w'
pdfs_csv = open(site_data_path, open_mode, newline='')

csv_row_count = 0
skipped_non_npdes_count = 0

pdfs_csv.close()

print(f"\nWrote {csv_row_count} CSV rows for NPDES permits")
print(f"Skipped {skipped_non_npdes_count} facility entries (non-NPDES PDFs)")

print(f"Non-NPDES PDFs: {len(non_npdes_pdfs)}")
print(f"CSV rows written (NPDES only): {csv_row_count}")
print(f"Facility entries skipped (non-NPDES): {skipped_non_npdes_count}")
print(f"{'='*80}")