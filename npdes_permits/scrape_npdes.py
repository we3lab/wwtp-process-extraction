from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import uuid
import os
import time
import csv
import glob
import pandas as pd

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

# Opens all necessary CSV files
file = open(os.path.join(full_path, 'all_ca_npdes.csv'), 'w', newline = '')
npdes_permits = csv.writer(file)
pdfs = open(os.path.join(full_path, 'site_data.csv'), 'w', newline = '')
rename_data = csv.writer(pdfs)

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
options.add_argument(f'--user-data-dir=/tmp/chrome_user_data_{uuid.uuid4().hex}')
options.add_argument('--headless')  # for server/SSH

# Set Chrome binary and ChromeDriver paths
options.binary_location = '/home/daly/bin/chrome/chrome-linux64/chrome'
service = Service('/home/daly/bin/chrome/chromedriver-linux64/chromedriver')
driver = webdriver.Chrome(service=service, options=options)

# Software gets url 
driver.get(rfr_url)

# Wait for the page to load and form elements to be available
wait = WebDriverWait(driver, 60)
wait.until(EC.presence_of_element_located((By.NAME, 'programDrop')))

# Selection clicks desired filters
selection('programDrop','NPDES')
selection('typeDrop','Wastewater Treatment Facility')
selection('wasteTypeDrop','Domestic wastewater')
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

# Monitor loading progress every 30 seconds
start_time = time.time()
while True:
    if driver.current_url != initial_url:
        current_url = driver.current_url
        print(current_url)
        driver.quit() 
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 60)  # Re-initialize wait with new driver
        driver.get(current_url)
        # screenshot for debugging
        driver.save_screenshot('npdes_permits/output/web_page.png')
        time.sleep(30)
        break

time.sleep(5)
selection('pagesizeselect', 'ALL')  # Sort to "ALL" results on page
table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
table_rows = table_body.find_elements(By.TAG_NAME, 'tr')

# Export Excel file and create mapping from it
export_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
export_link.click()
time.sleep(10)
excel_files = glob.glob(f'{full_path}/*.xlsx') + glob.glob(f'{full_path}/*.xls')
if not excel_files:
    print("No Excel file found")
    exit()

# Read Excel and create mapping of Order No. to Agency, NPDES No., and Program
excel_file = max(excel_files, key=os.path.getctime)
df = pd.read_csv(excel_file, sep='\t', encoding='latin-1')
print(f"Excel file length {len(df)}")

# Handle duplicate WDIDs by keeping the most recent Expiration/Review Date
def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).lower() == 'null':
        return pd.NaT
    return pd.to_datetime(date_str, errors='coerce')

df['Expiration/Review Date'] = df['Expiration/Review Date'].apply(parse_date)
df_sorted = df.sort_values(['WDID', 'Expiration/Review Date'], ascending=[True, False])
df_deduplicated = df_sorted.drop_duplicates(subset=['WDID'], keep='first')
print(f"After deduplication: {len(df_deduplicated)} rows (removed {len(df) - len(df_deduplicated)} duplicates)")

order_to_data = {}
for _, row in df_deduplicated.iterrows():
    order_no = str(row['Order No.']).strip()
    if order_no and order_no != 'nan':
        order_to_data[order_no] = {}
        for key in ['Agency', 'NPDES No.', 'Program']:
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

# Patterns that should only match at beginning of filename
beginning_patterns = []
for keyword in base_keywords:
    for sep in separators:
        beginning_patterns.append(f"{keyword}{sep}")
skip_keywords.extend([
    "report", "reports", "financial", "notice of", "response to", 
    "rate study", "ratestudy", "study", "studies"
])

# Process each row for PDF downloads
target_order_nos = set()
for order_no, data in order_to_data.items():
    if 'NPDMUNILRG' in data['Program'] or 'NPDMUNIOTH' in data['Program']:
        target_order_nos.add(order_no)
print(f'{len(target_order_nos)} Order Nos matching criteria')

# Find Order No. column dynamically
order_no_any = driver.find_elements(By.XPATH, "//*[contains(text(), 'Order No.')]")
for element in order_no_any:
    if element.tag_name == 'a' and element.text.strip() == 'Order No.':
        # find parent td cell for header
        parent_td = element.find_element(By.XPATH, "./..")
        if parent_td.tag_name == 'td':
            order_no_col_index = driver.execute_script("return arguments[0].cellIndex + 1;", parent_td)
            break

# Find row indices that contain our target Order Nos using XPath
target_row_indices = []
print('Finding rows with target Order Nos using XPath...')

# Create XPath to find rows where Order No. is in our target list
order_nos_xpath = " or ".join([f"td[{order_no_col_index}]/a[text()='{order_no}']" for order_no in target_order_nos])
xpath_query = f"//tr[{order_nos_xpath}]"

target_rows = driver.find_elements(By.XPATH, xpath_query)
for row in target_rows:
    # Get the row index
    row_index = driver.execute_script("return arguments[0].rowIndex + 1;", row)
    target_row_indices.append(row_index)

print(f'Found {len(target_row_indices)} rows with matching Order Nos using XPath')

# Process the target rows
wait = WebDriverWait(driver, 5)
for row_index in target_row_indices:
    try:  # Refresh table elements
        table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
        table_rows = table_body.find_elements(By.TAG_NAME, 'tr')
    except:
        print(f"Row {row_index}: Could not refresh table elements, continuing...")
        continue
    
    # Get Order No. from the specific row
    order_no_link_elements = driver.find_elements(By.XPATH, f"//tr[{row_index}]/td[{order_no_col_index}]/a")
    if order_no_link_elements:
        order_no = order_no_link_elements[0].text.strip()
        program = order_to_data[order_no]['Program']
        print(f'Row {row_index}: Processing Order No. {order_no} with Program: {program}')
        order_no_link_elements[0].click()
        time.sleep(1)
        driver.switch_to.window(driver.window_handles[1])
        time.sleep(1)

        # Find all PDF links on the Order No. page
        pdf_documents = driver.find_elements(By.PARTIAL_LINK_TEXT, '.pdf')
        print(f'Row {row_index}: Found {len(pdf_documents)} PDFs for Order No. {order_no}')
        for j, pdf_doc in enumerate(pdf_documents):
            try:
                pdf_name = pdf_doc.text
                # Skipped non-NPDES PDFs
                should_skip = any(keyword.lower() in pdf_name.lower() for keyword in skip_keywords)
                if not should_skip:
                    should_skip = any(pdf_name.lower().startswith(pattern.lower()) for pattern in beginning_patterns)
                if should_skip:
                    print(f'Row {row_index}: skipping {pdf_name} (contains skip keyword)')
                    continue
                
                print(f'Row {row_index}: downloading {pdf_name}')
                pdf_doc.click()
                time.sleep(2)
                
                agency_name = order_to_data[order_no]['Agency']
                npdes_no = order_to_data[order_no]['NPDES No.']
                rename_data.writerow([agency_name, npdes_no, pdf_name])
                print(f'Row {row_index}: processed {agency_name}, {npdes_no}, {pdf_name} pdf {j}')
            except:
                print(f'Row {row_index}: download {j} failed')
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(2)

# Save all table data to CSV
for row in table_rows:
    table_data = row.find_elements(By.TAG_NAME, 'td')
    row_data = [data.text for data in table_data]
    npdes_permits.writerow(row_data)

# Close CSV file and quit driver
file.close()
pdfs.close()
driver.close()