from selenium import webdriver
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
import pandas as pd
from helpers.npdes_detection import detect_npdes

# Function that selects options from filters
def selection(name, text):
    select_element = driver.find_element(By.NAME, name)
    select = Select(select_element)
    select.select_by_visible_text(text)

# Link to Interactive Regulated Facilities Report
rfr_url = 'https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility'

# Creates folder to store downloaded PDFs
path = 'npdes_permits/output'
now = datetime.now()
pdf_folder = f'{now.year}-{now.month}-{now.day}'
full_path = os.path.join(path, pdf_folder)
pdfs_path = os.path.join(full_path, 'pdfs')
if not os.path.exists(full_path):
    os.mkdir(full_path)
if not os.path.exists(pdfs_path):
    os.mkdir(pdfs_path)

# Sets up Chrome and folder for downloads
options = webdriver.ChromeOptions()
prefs = {
    'download.default_directory': os.path.abspath(pdfs_path),
    'download.prompt_for_download': False,
    'download.directory_upgrade': True,
    'safebrowsing.enabled': True,
    'profile.default_content_settings.popups': 0,
    'profile.default_content_setting_values.automatic_downloads': 1
}
options.add_experimental_option('prefs', prefs)
options.page_load_strategy = "eager"
options.add_argument('--blink-settings=imagesEnabled=false')
user_data_dir = os.path.join(tempfile.gettempdir(), f"chrome_user_data_{uuid.uuid4().hex}")
options.add_argument(f"--user-data-dir={user_data_dir}")
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--headless')  # for server/SSH

# Set Chrome binary and ChromeDriver paths
options.binary_location = '/home/constance/chrome/chrome-linux64/chrome'
service = Service('/home/constance/chrome/chromedriver-linux64/chromedriver')
driver = webdriver.Chrome(service=service, options=options)

# Software gets url
driver.get(rfr_url)

# Wait for the page to load and form elements to be available
wait = WebDriverWait(driver, 60)
wait.until(EC.presence_of_element_located((By.NAME, 'programDrop')))

# Keep a stable handle to the main table page (window #1)
main_window = driver.current_window_handle

# Selection clicks desired filters
selection('programDrop', 'NPDES')
selection('typeDrop', 'Wastewater Treatment Facility')
selection('wasteTypeDrop', 'Domestic wastewater')
selection('inStatus', 'Active')
time.sleep(5)
driver.find_element(By.NAME, 'enpRepButton').click()
time.sleep(5)

table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
rows = table_body.find_elements(By.TAG_NAME, 'tr')
last_row = rows[-1] if rows else None
cells = last_row.find_elements(By.TAG_NAME, 'td')

total_cell = driver.find_element(By.XPATH, "//table//tr[last()]/td[6]")
initial_url = driver.current_url
print(f"Clicking cell w/ text '{total_cell.text}'")
total_cell.find_element(By.TAG_NAME, "a").click()

# Wait for the new page to load (but DON'T quit and restart the driver!)
# This preserves the session and filter state
print("Waiting for detail page to load...")
wait.until(EC.url_changes(initial_url))
current_url = driver.current_url
print(f"Navigated to: {current_url}")

# Wait for the table to appear on the new page
wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'ciwqsReportDataTable')))
driver.save_screenshot('npdes_permits/output/web_page.png')
print("Detail page loaded, filters should be preserved")
time.sleep(5)

time.sleep(5)
selection('pagesizeselect', 'ALL')

# Wait for the table to fully load after changing page size
print("Waiting for table to load with ALL rows...")
time.sleep(10)  # Initial wait for page size change to take effect

# Wait until the loading indicator disappears (if present)
try:
    wait.until(EC.invisibility_of_element_located((By.CLASS_NAME, 'loading')))
    print("Loading indicator disappeared")
except:
    pass  # No loading indicator found, continue

table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
table_rows = table_body.find_elements(By.TAG_NAME, 'tr')
print(f"Final table has {len(table_rows)} rows before export")

# Export Excel file and create mapping from it
export_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
export_link.click()
export_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
export_link.click()

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

def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).lower() == 'null':
        return pd.NaT
    return pd.to_datetime(date_str, errors='coerce')

df['Expiration/Review Date'] = df['Expiration/Review Date'].apply(parse_date)
df_sorted = df.sort_values(['WDID', 'Facility Name', 'Expiration/Review Date'], ascending=[True, True, False])
# Deduplicate by BOTH WDID and Facility Name to preserve different facilities with same WDID
df_deduplicated = df_sorted.drop_duplicates(subset=['WDID', 'Facility Name'], keep='first')
print(f"After deduplication and filtering: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")

order_to_data = {}
for _, row in df_deduplicated.iterrows():
    order_no = str(row['Order No.']).strip()
    if order_no and order_no != 'nan':
        order_to_data[order_no] = {}
        # Store all required keys to prevent KeyError later
        for key in ['Agency', 'Facility Name', 'NPDES No.', 'Program', 'Region', 'Major/Minor']:
            order_to_data[order_no][key] = str(row[key]).strip()

# PDF keywords to skip downloading
base_keywords = ["noa", "noi", "rpts", "rowd"]
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
    "report", "reports", "financial", "notice of", "response to",
    "rate study", "ratestudy", "study", "studies", "letter",
    "PER", "NOA", "NOI", "CI"
])

def find_col_index_by_header(header_text: str):
    """Find the 1-based column index of a header by its visible text.
    Tries both <a> in header cell and plain text headers.
    """
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

# ============================================================================
# STEP 1: EXTRACT URLs FROM ALL TABLE ROWS (NO FILTERING)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Extracting URLs from ALL table rows by reading href attributes in Order No. column (no filtering)")
print("="*80)

# Dictionary structure:
# Key: URL (from href attribute)
# Value: list of facility dicts with {agency, facility_name, npdes_no, region, major_minor, order_no, row_index}
url_to_facilities = {}
non_clickable_row_indices = []

wait = WebDriverWait(driver, 5)


# Iterate all data rows in the table (skip header if present)
# Using XPath to avoid stale references and ensure 1-based indices.
total_rows = len(driver.find_elements(By.XPATH, "//table[contains(@class,'ciwqsReportDataTable')]//tr"))
print(f"Total <tr> elements found: {total_rows}")


for row_index in range(2, total_rows + 1):  # skip row 1 (header)
    try:
        # Order No. link element from this row
        order_no_link_elements = driver.find_elements(
            By.XPATH,
            f"//table[contains(@class,'ciwqsReportDataTable')]//tr[{row_index}]/td[{order_no_col_index}]/a"
        )
        if not order_no_link_elements:
            non_clickable_row_indices.append(row_index)
            print(f"Row {row_index}: Order No. not clickable")
            continue

        # Extract the URL from href attribute WITHOUT clicking
        url = order_no_link_elements[0].get_attribute('href')
        order_no = order_no_link_elements[0].text.strip()

        # Read facility info for this row
        def cell_text(col_idx):
            try:
                return driver.find_element(
                    By.XPATH,
                    f"//table[contains(@class,'ciwqsReportDataTable')]//tr[{row_index}]/td[{col_idx}]"
                ).text.strip()
            except Exception:
                return ""

        agency_name = cell_text(agency_col_index) if agency_col_index else ""
        facility_name = cell_text(facility_col_index) if facility_col_index else ""
        npdes_no = cell_text(npdes_col_index) if npdes_col_index else ""
        region = cell_text(region_col_index) if region_col_index else ""
        major_minor = cell_text(major_minor_col_index) if major_minor_col_index else ""

        # Add to dictionary grouped by URL
        if url not in url_to_facilities:
            url_to_facilities[url] = []

        url_to_facilities[url].append({
            'row_index': row_index,
            'agency': agency_name,
            'facility_name': facility_name,
            'npdes_no': npdes_no,
            'region': region,
            'major_minor': major_minor,
            'order_no': order_no
        })

        print(f'Row {row_index}: {order_no} - {agency_name} - {facility_name}')
        print(f'  URL: {url}')
    except Exception as e:
        print(f"Row {row_index}: Could not read data: {e}")
        continue

print(f"\n✓ Scanned all rows")
print(f"✓ Found {len(url_to_facilities)} unique URLs")
print(f"✓ Non-clickable Order No. rows: {len(non_clickable_row_indices)}")

def extract_reg_measure_id(href: str):
    if not href:
        return None
    try:
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        for key in ['regMeasID']:
            if key in query and query[key]:
                return str(query[key][0])
    except Exception:
        pass
    match = re.search(r"regMeasID=(\d+)", href, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

reg_measure_id_to_url = {}
for url in list(url_to_facilities.keys()):
    reg_id = extract_reg_measure_id(url)
    if reg_id:
        reg_measure_id_to_url[reg_id] = url

# ============================================================================
# STEP 2: DOWNLOAD PDFs BY VISITING EACH UNIQUE URL
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Visiting each unique URL and downloading PDFs")
print("="*80)

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

def download_pdfs_for_url(url, facilities):
    global downloaded_count, skipped_count
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

            if len(pdf_documents) == 0:
                all_links = driver.find_elements(By.TAG_NAME, 'a')
                print(f"  Debug: Found {len(all_links)} total links on page")
                for link in all_links[:10]:
                    print(f"    Link text: '{link.text}' | href: {link.get_attribute('href')}")

        for j, pdf_doc in enumerate(pdf_documents):
            try:
                pdf_name = pdf_doc.text

                should_skip = any(keyword.lower() in pdf_name.lower() for keyword in skip_keywords)
                if should_skip:
                    print(f"        Skipping: {pdf_name} (matched skip keyword)")
                    skipped_count += 1
                    continue

                should_skip = any(pdf_name.lower().startswith(pattern.lower()) for pattern in beginning_patterns)
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

# Process each unique URL (only once per URL!)
for url_index, (url, facilities) in enumerate(url_to_facilities.items(), 1):
    print(f"\n[{url_index}/{len(url_to_facilities)}] Processing URL")
    download_pdfs_for_url(url, facilities)

print(f"\nDownloaded {downloaded_count} PDFs")
print(f"Skipped {skipped_count} PDFs (keyword filters)")

# ==========================================================================
# STEP 2B: PROCESS NON-CLICKABLE ORDER NO. ROWS VIA FACILITY PAGE
# ==========================================================================
print("\n" + "="*80)
print("STEP 2B: Processing non-clickable Order No. rows")
print("="*80)

def get_table_headers(table):
    header_cells = table.find_elements(By.CSS_SELECTOR, 'td.ciwqsReportColumnName')
    if not header_cells:
        return []
    headers = []
    for cell in header_cells:
        links = cell.find_elements(By.TAG_NAME, 'a')
        headers.append(links[0].text.strip() if links else cell.text.strip())
    return headers

def find_regulatory_measures_table():
    tables = driver.find_elements(By.TAG_NAME, 'table')
    required_cols = {
        'Reg Measure ID', 'Reg Measure Type', 'Region', 'Program', 'Order No.',
        'WDID', 'Effective Date', 'Expiration Date', 'Status'
    }
    for table in tables:
        headers = get_table_headers(table)
        if headers and required_cols.issubset(set(headers)):
            return table, headers
    return None, []

def parse_date_safe(text):
    try:
        return pd.to_datetime(text, errors='coerce')
    except Exception:
        return pd.NaT

new_urls_added = 0

for idx, row_index in enumerate(non_clickable_row_indices, 1):
    try:
        def cell_text(col_idx):
            try:
                return driver.find_element(
                    By.XPATH,
                    f"//table[contains(@class,'ciwqsReportDataTable')]//tr[{row_index}]/td[{col_idx}]"
                ).text.strip()
            except Exception:
                return ""

        base_facility = {
            'row_index': row_index,
            'agency': cell_text(agency_col_index) if agency_col_index else "",
            'facility_name': cell_text(facility_col_index) if facility_col_index else "",
            'npdes_no': cell_text(npdes_col_index) if npdes_col_index else "",
            'region': cell_text(region_col_index) if region_col_index else "",
            'major_minor': cell_text(major_minor_col_index) if major_minor_col_index else "",
            'order_no': ''
        }

        facility_link = None
        try:
            facility_link = driver.find_element(
                By.XPATH,
                f"//table[contains(@class,'ciwqsReportDataTable')]//tr[{row_index}]/td[{facility_col_index}]/a"
            )
        except Exception:
            facility_link = driver.find_element(
                By.XPATH,
                f"//table[contains(@class,'ciwqsReportDataTable')]//tr[{row_index}]/td[{facility_col_index}]//span/a"
            )

        if not facility_link:
            print(f"Row {row_index}: No facility link found")
            continue

        facility_url = facility_link.get_attribute('href')
        print(f"\n[{idx}/{len(non_clickable_row_indices)}] Row {row_index}: opening facility page")
        handles_before = set(driver.window_handles)
        driver.execute_script(f"window.open('{facility_url}', '_blank');")
        time.sleep(1)

        handles_after = set(driver.window_handles)
        new_handles = list(handles_after - handles_before)
        if new_handles:
            driver.switch_to.window(new_handles[0])
        else:
            driver.switch_to.window(main_window)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'table')))
        time.sleep(1)

        reg_table, headers = find_regulatory_measures_table()
        if not reg_table:
            print(f"  X Regulatory Measures table not found")
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)
            continue

        col_index = {name: headers.index(name) for name in headers}
        max_required_idx = max(col_index.values()) if col_index else -1
        data_rows = reg_table.find_elements(By.TAG_NAME, 'tr')

        best_row = None
        best_effective = pd.NaT
        for data_row in data_rows:
            cells = data_row.find_elements(By.TAG_NAME, 'td')
            if not cells or data_row.find_elements(By.CSS_SELECTOR, 'td.ciwqsReportColumnName'):
                continue
            if len(cells) <= max_required_idx:
                continue

            status_text = cells[col_index['Status']].text.strip()
            if status_text.lower() != 'active':
                continue

            order_cell = cells[col_index['Order No.']]
            order_links = order_cell.find_elements(By.TAG_NAME, 'a')
            if not order_links:
                continue

            effective_text = cells[col_index['Effective Date']].text.strip()
            effective_dt = parse_date_safe(effective_text)
            if pd.isna(effective_dt):
                continue

            if best_row is None or effective_dt > best_effective:
                best_row = cells
                best_effective = effective_dt

        if not best_row:
            print(f"  X No active clickable Order No. found")
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)
            continue

        order_cell = best_row[col_index['Order No.']]
        order_link = order_cell.find_element(By.TAG_NAME, 'a')
        order_href = order_link.get_attribute('href')
        order_no_text = order_link.text.strip()
        reg_region = best_row[col_index['Region']].text.strip()

        base_facility['region'] = reg_region or base_facility['region']
        base_facility['order_no'] = order_no_text

        reg_id = extract_reg_measure_id(order_href)
        existing_url = reg_measure_id_to_url.get(reg_id)
        if existing_url:
            url_to_facilities.setdefault(existing_url, []).append(base_facility)
            for pdf_name in url_to_pdfs.get(existing_url, []):
                if pdf_name not in pdf_to_facilities:
                    pdf_to_facilities[pdf_name] = []
                pdf_to_facilities[pdf_name].append(base_facility)
            print(f"  ✓ Found existing regMeasID {reg_id}; added facility to existing URL")
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)
            continue

        url_to_facilities.setdefault(order_href, []).append(base_facility)
        if reg_id:
            reg_measure_id_to_url[reg_id] = order_href
        new_urls_added += 1
        print(f"  ✓ New regMeasID {reg_id}; downloading PDFs")

        driver.close()
        driver.switch_to.window(main_window)
        time.sleep(1)

        download_pdfs_for_url(order_href, url_to_facilities[order_href])

    except Exception as e:
        print(f"Row {row_index}: Failed to process non-clickable row: {e}")
        try:
            if driver.current_window_handle != main_window:
                driver.close()
            driver.switch_to.window(main_window)
        except Exception:
            pass

print(f"\nNew URLs added from non-clickable rows: {new_urls_added}")

# ============================================================================
# STEP 3: CREATE CSV WITH SHARED_PDF COLUMN
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Creating site_data.csv with facility-PDF mapping")
print("="*80)

csv_rows = []
for pdf_name, facilities in pdf_to_facilities.items():
    # Deduplicate facilities by (agency, facility_name, npdes_no) to avoid duplicate rows
    seen = set()
    is_shared = "Yes" if len(facilities) > 1 else "No"
    for facility in facilities:
        key = (facility['agency'], facility['facility_name'], facility['npdes_no'])
        if key in seen:
            continue
        seen.add(key)
        csv_rows.append([
            facility['agency'],
            facility['facility_name'],
            facility['npdes_no'],
            facility['region'],
            facility['major_minor'],
            pdf_name,
            is_shared
        ])

# Sort by PDF name to group shared PDFs together
csv_rows.sort(key=lambda x: (x[2], x[0]))

print(f"Created {len(csv_rows)} CSV rows")
print(f"Unique PDFs: {len(pdf_to_facilities)}")
print(f"Shared PDFs: {sum(1 for pdf, fac in pdf_to_facilities.items() if len(fac) > 1)}")

# ============================================================================
# STEP 4: DETECT AND MOVE NPDES PDFs TO SEPARATE FOLDER
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Detecting and moving NPDES PDFs")
print("="*80)

npdes_path = os.path.join(full_path, 'npdes')
if not os.path.exists(npdes_path):
    os.mkdir(npdes_path)

npdes_pdfs = set()
non_npdes_pdfs = set()

npdes_count = 0
non_npdes_count = 0

for filename in os.listdir(pdfs_path):
    if not filename.endswith('.pdf'):
        continue

    file_path = os.path.join(pdfs_path, filename)

    try:
        if detect_npdes(file_path):
            new_path = os.path.join(npdes_path, filename)
            os.rename(file_path, new_path)
            print(f"NPDES detected: {filename}")
            npdes_pdfs.add(filename)
            npdes_count += 1
        else:
            print(f"Non-NPDES content: {filename}")
            non_npdes_pdfs.add(filename)
            non_npdes_count += 1
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        non_npdes_pdfs.add(filename)
        non_npdes_count += 1

print(f"\nNPDES PDFs moved: {npdes_count}")
print(f"Non-NPDES PDFs kept in pdfs folder: {non_npdes_count}")

# ============================================================================
# STEP 5: CREATE CSV WITH ONLY NPDES PDFs
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Creating site_data.csv with NPDES permits only")
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
print("STEP 5: Saving table data to CSV from Excel export")
print("="*80)

file = open(os.path.join(full_path, 'all_ca_npdes.csv'), 'w', newline='')
df_deduplicated.to_csv(file, index=False)
file.close()

print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

# Quit driver
driver.quit()

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total rows scanned: {total_rows}")
print(f"Unique URLs found: {len(url_to_facilities)}")
print(f"Shared URLs (multiple facilities): {sum(1 for facs in url_to_facilities.values() if len(facs) > 1)}")
print(f"PDFs downloaded: {downloaded_count}")
print(f"PDFs skipped (keywords): {skipped_count}")
print(f"NPDES PDFs detected: {npdes_count}")
print(f"Non-NPDES PDFs: {non_npdes_count}")
print(f"CSV rows written (NPDES only): {csv_row_count}")
print(f"Facility entries skipped (non-NPDES): {skipped_non_npdes_count}")
print(f"{'='*80}")