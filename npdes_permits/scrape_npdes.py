from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import uuid
import tempfile
import os
import time
import csv
import glob
import pandas as pd
from npdes_detection import detect_npdes

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

# Process each row for PDF downloads
# Build target Order Nos directly from dataframe to avoid any dict overwrites
try:
    target_order_nos = set(
        df_deduplicated.loc[
            df_deduplicated['Program'].astype(str).str.contains('NPDMUNILRG|NPDMUNIOTH', na=False, regex=True),
            'Order No.'
        ].astype(str).str.strip()
    )
except Exception:
    # Fallback to whatever is in order_to_data if dataframe structure changes
    target_order_nos = set()
    for order_no, data in order_to_data.items():
        if 'NPDMUNILRG' in data.get('Program', '') or 'NPDMUNIOTH' in data.get('Program', ''):
            target_order_nos.add(order_no)
print(f'{len(target_order_nos)} Order Nos matching criteria')

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

# Find row indices that contain our target Order Nos using XPath
order_nos_xpath = " or ".join([f"td[{order_no_col_index}]/a[text()='{order_no}']" for order_no in target_order_nos])
xpath_query = f"//tr[{order_nos_xpath}]"

target_rows = driver.find_elements(By.XPATH, xpath_query)
target_row_indices = []
for row in target_rows:
    row_index = driver.execute_script("return arguments[0].rowIndex + 1;", row)
    target_row_indices.append(row_index)

print(f'Found {len(target_row_indices)} rows with matching Order Nos using XPath')

# ============================================================================
# STEP 1: EXTRACT URLs FROM TABLE (NO CLICKING YET)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Extracting URLs from table by reading href attributes")
print("="*80)

# Dictionary structure:
# Key: URL (from href attribute)
# Value: list of facility dicts with {agency, facility_name, npdes_no, region, major_minor, order_no, row_index}
url_to_facilities = {}

wait = WebDriverWait(driver, 5)
for row_index in target_row_indices:
    try:
        # Get Order No. link element from this row
        order_no_link_elements = driver.find_elements(By.XPATH, f"//tr[{row_index}]/td[{order_no_col_index}]/a")
        if order_no_link_elements:
            # Extract the URL from href attribute WITHOUT clicking
            url = order_no_link_elements[0].get_attribute('href')
            order_no = order_no_link_elements[0].text.strip()
            
            # Read facility info for this row
            def cell_text(col_idx):
                try:
                    return driver.find_element(By.XPATH, f"//tr[{row_index}]/td[{col_idx}]").text.strip()
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

print(f"\n✓ Scanned {len(target_row_indices)} facilities")
print(f"✓ Found {len(url_to_facilities)} unique URLs")
print(f"✓ Duplicate URLs detected: {len(target_row_indices) - len(url_to_facilities)}")

# Show some statistics about shared URLs
shared_urls = {url: facilities for url, facilities in url_to_facilities.items() if len(facilities) > 1}
if shared_urls:
    print(f"\n{len(shared_urls)} URLs shared by multiple facilities:")
    for url, facilities in list(shared_urls.items())[:5]:  # Show first 5
        facility_names = [f"{f['agency']} ({f['npdes_no']})" for f in facilities]
        print(f"  • {len(facilities)} facilities: {', '.join(facility_names)}")
        print(f"    URL: {url}")
    if len(shared_urls) > 5:
        print(f"  ... and {len(shared_urls) - 5} more")

# ============================================================================
# STEP 2: DOWNLOAD PDFs BY VISITING EACH UNIQUE URL
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Visiting each unique URL and downloading PDFs")
print("="*80)

# Track which PDFs belong to which facilities
pdf_to_facilities = {}  # Key: downloaded PDF filename, Value: list of facilities

def list_pdf_files(directory):
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

def wait_for_new_pdf_file(directory, before_set, timeout=60):
    """Wait for a new .pdf file to appear in directory compared to before_set,
    and for its size to stabilize."""
    end = time.time() + timeout
    last_new = None
    while time.time() < end:
        current = set(list_pdf_files(directory))
        new_files = [f for f in current - before_set if not f.lower().endswith('.crdownload')]
        if new_files:
            # pick the newest by ctime
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

# Process each unique URL (only once per URL!)
for url_index, (url, facilities) in enumerate(url_to_facilities.items(), 1):
    print(f"\n[{url_index}/{len(url_to_facilities)}] Processing URL")
    print(f"  Facilities: {', '.join([f['facility_name'] for f in facilities])}")
    print(f"  URL: {url}")
    
    try:
        # Open URL in new tab
        print(f"  Opening URL in new tab...")
        driver.execute_script(f"window.open('{url}', '_blank');")
        time.sleep(1)
        
        # Wait for new window to open
        wait_for_window = WebDriverWait(driver, 10)
        wait_for_window.until(lambda d: len(d.window_handles) > 1)
        
        # Switch to the new window
        driver.switch_to.window(driver.window_handles[1])
        print(f"  Switched to new window")
        
        # Wait for page to load
        time.sleep(3)
        
        # Verify we're on the correct page
        current_url = driver.current_url
        print(f"  Current URL: {current_url}")
        
        # Check if page has content
        page_source_length = len(driver.page_source)
        print(f"  Page source length: {page_source_length} characters")
        
        if page_source_length < 500:
            print(f"  X Page appears to be blank/empty!")
            print(f"  Page content preview: {driver.page_source[:500]}")
        
        # Find all PDF links on the page (by link text containing .pdf)
        # This works better than checking href attribute
        pdf_documents = driver.find_elements(By.XPATH, "//a[contains(text(), '.pdf') or contains(text(), '.PDF')]")
        print(f"  Found {len(pdf_documents)} PDFs on page")
        
        # If still nothing, try alternative methods for debugging
        if len(pdf_documents) == 0:
            print(f"  No PDFs found, trying href-based detection...")
            pdf_documents = driver.find_elements(By.XPATH, "//a[contains(translate(@href, 'PDF', 'pdf'), '.pdf')]")
            print(f"  Href method found {len(pdf_documents)} PDFs")
            
            # If still nothing, print all links for debugging
            if len(pdf_documents) == 0:
                all_links = driver.find_elements(By.TAG_NAME, 'a')
                print(f"  Debug: Found {len(all_links)} total links on page")
                for link in all_links[:10]:  # Show first 10
                    print(f"    Link text: '{link.text}' | href: {link.get_attribute('href')}")
        
        for j, pdf_doc in enumerate(pdf_documents):
            try:
                pdf_name = pdf_doc.text
                
                # Check if the PDF name contains any skip keyword
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
                # Wait for the actual file to appear and stabilize
                new_file = wait_for_new_pdf_file(pdfs_path, before, timeout=60)
                if not new_file:
                    print("        X Download did not complete or file not detected")
                    continue
                downloaded_count += 1

                # Track this actual downloaded PDF for all associated facilities
                if new_file not in pdf_to_facilities:
                    pdf_to_facilities[new_file] = []
                pdf_to_facilities[new_file].extend(facilities)
                
            except Exception as e:
                print(f"        X Download {j} failed: {e}")
        
        # Close tab and return to main window
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(1)
    
    except Exception as e:
        print(f"  X Failed to process Order No. {order_no}: {e}")
        # Try to close any open tabs and return to main window
        try:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
        except:
            pass

print(f"\nDownloaded {downloaded_count} PDFs")
print(f"Skipped {skipped_count} PDFs (keyword filters)")

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
# STEP 3: DETECT AND MOVE NPDES PDFs TO SEPARATE FOLDER
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Detecting and moving NPDES PDFs")
print("="*80)

npdes_path = os.path.join(full_path, 'npdes')
if not os.path.exists(npdes_path):
    os.mkdir(npdes_path)

# Track which PDFs are actual NPDES permits
npdes_pdfs = set()  # Set of PDF filenames that passed NPDES detection
non_npdes_pdfs = set()  # Set of PDF filenames that failed NPDES detection

npdes_count = 0
non_npdes_count = 0

for filename in os.listdir(pdfs_path):
    if not filename.endswith('.pdf'):
        continue
        
    file_path = os.path.join(pdfs_path, filename)
    
    try:
        if detect_npdes(file_path):
            # Move to NPDES folder
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
# STEP 4: CREATE CSV WITH ONLY NPDES PDFs
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Creating site_data.csv with NPDES permits only")
print("="*80)

# Open CSV file now (after NPDES detection)
pdfs_csv = open(os.path.join(full_path, 'site_data.csv'), 'w', newline='')
rename_data = csv.writer(pdfs_csv)

# Write header (fixed missing comma)
rename_data.writerow(['Agency', 'Facility_Name', 'NPDES_No', 'Region', 'Major/Minor', 'PDF_File', 'Shared_PDF'])

csv_row_count = 0
skipped_non_npdes_count = 0

# Only write rows for PDFs that passed NPDES detection
for pdf_name, facilities in pdf_to_facilities.items():
    if pdf_name in npdes_pdfs:
        is_shared = "Yes" if len(facilities) > 1 else "No"
        # Deduplicate facility rows per PDF
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
        # Just count, don't try to write
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

# Reopen the file we closed earlier to write the Excel data
file = open(os.path.join(full_path, 'all_ca_npdes.csv'), 'w', newline='')
df_deduplicated.to_csv(file, index=False)
file.close()

print(f"Saved {len(df_deduplicated)} rows to all_ca_npdes.csv")

# Quit driver
driver.quit()  # Use quit() instead of close() to properly clean up

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total facilities scanned: {len(target_row_indices)}")
print(f"Unique URLs found: {len(url_to_facilities)}")
print(f"Shared URLs (multiple facilities): {sum(1 for facs in url_to_facilities.values() if len(facs) > 1)}")
print(f"PDFs downloaded: {downloaded_count}")
print(f"PDFs skipped (keywords): {skipped_count}")
print(f"NPDES PDFs detected: {npdes_count}")
print(f"Non-NPDES PDFs: {non_npdes_count}")
print(f"CSV rows written (NPDES only): {csv_row_count}")
print(f"Facility entries skipped (non-NPDES): {skipped_non_npdes_count}")
print(f"{'='*80}")