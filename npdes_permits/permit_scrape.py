from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.client_config import ClientConfig
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime
import uuid
import os
import time
import csv
import glob
import pandas as pd
import requests

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
if not os.path.exists(full_path):
    os.mkdir(full_path)

# Opens all necessary CSV files
file = open('npdes_permits/output/all_ca_npdes.csv', 'w', newline = '')
npdes_permits = csv.writer(file)
pdfs = open('npdes_permits/output/site_data.csv', 'w', newline = '')
rename_data = csv.writer(pdfs)

# Write headers for OTHER_CSV.csv
# OTHER_CSV_NAME.writerow(['Agency Name', 'NPDES No.'])

# Sets up Chrome and folder for downloads
options = webdriver.ChromeOptions()
prefs = {
    'download.default_directory': os.path.abspath(full_path),
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
driver.save_screenshot('npdes_permits/output/img1.png')

# Wait for the page to load and form elements to be available
wait = WebDriverWait(driver, 60)
wait.until(EC.presence_of_element_located((By.NAME, 'programDrop')))

# Selection clicks desired filters
selection('programDrop','NPDES')
selection('typeDrop','Wastewater Treatment Facility')
selection('wasteTypeDrop','Domestic wastewater')
# selection('regDrop', region)

driver.find_element(By.NAME, 'enpRepButton').click()
time.sleep(5)
driver.save_screenshot('npdes_permits/output/img2.png')

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
        driver.save_screenshot('npdes_permits/output/img3a.png')
        print('restarting driver with produced url')
        current_url = driver.current_url
        driver.quit() 
        driver = webdriver.Chrome(service=service, options=options)
        wait = WebDriverWait(driver, 60)  # Re-initialize wait with new driver
        driver.get(current_url)
        driver.save_screenshot('npdes_permits/output/img3b.png')
        time.sleep(30)
        break

time.sleep(5)
selection('pagesizeselect', 'ALL')
table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
table_rows = table_body.find_elements(By.TAG_NAME, 'tr')

# Download the Excel file for all facilities
export_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'EXPORT THIS REPORT TO EXCEL')))
export_link.click()
time.sleep(10)  # Wait for download to complete
excel_files = glob.glob(f'{full_path}/*.xlsx') + glob.glob(f'{full_path}/*.xls')
if not excel_files:
    print("No Excel file found")
    exit()

excel_file = max(excel_files, key=os.path.getctime)  # Get the most recent file
df = pd.read_csv(excel_file, sep='\t', encoding='latin-1')
print(f"Excel file length {len(df)}")

# Create a mapping of Order No. to Agency and NPDES No.
order_to_data = {}
for _, row in df.iterrows():
    order_no = str(row['Order No.']).strip()
    agency = str(row['Agency']).strip()
    npdes_no = str(row['NPDES No.']).strip()
    if order_no and order_no != 'nan':
        order_to_data[order_no] = {'agency': agency, 'npdes_no': npdes_no}

wait = WebDriverWait(driver, 5)
for i in range(2, len(table_rows) + 1):
    # order_no_header = driver.find_element(By.XPATH, "//th[contains(text(), 'Order No.')]")
    # order_no_col_index = driver.execute_script("return arguments[0].cellIndex + 1;", order_no_header)    
    # pdf_link_elements = driver.find_elements(By.XPATH, f'/html/body/table/tbody/tr[3]/td/table/tbody/tr[1]/td/table[2]/tbody/tr[7]/td[2]/table[1]/tbody/tr[{i}]/td[{order_no_col_index}]/a')
    # Look for "Order No." header (link to sort column)
    order_no_any = driver.find_elements(By.XPATH, "//*[contains(text(), 'Order No.')]")
    for element in order_no_any:
        if element.tag_name == 'a' and element.text.strip() == 'Order No.':
            # find parent td cell for header
            parent_td = element.find_element(By.XPATH, "./..")
            if parent_td.tag_name == 'td':
                order_no_col_index = driver.execute_script("return arguments[0].cellIndex + 1;", parent_td)
                pdf_link_elements = driver.find_elements(By.XPATH, f"//tr[{i}]/td[{order_no_col_index}]/a")
                break
    # # hardcoded column 14
    # pdf_link_elements = driver.find_elements(By.XPATH, f"//tr[{i}]/td[14]/a")
    
    if pdf_link_elements:
        pdf_link = pdf_link_elements[0]
        order_no = pdf_link.text.strip()
        pdf_link.click()
        time.sleep(1)
        driver.switch_to.window(driver.window_handles[1])
        # pdf_documents = driver.find_elements(By.PARTIAL_LINK_TEXT, '.pdf')
        # document_download = pdf_documents[0]
        document_download = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, '.pdf')))
        pdf_name = document_download.text
        print(f'downloading {pdf_name}')
        document_download.click()
        time.sleep(5)
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
            
        agency_name = order_to_data[order_no]['agency']
        npdes_no = order_to_data[order_no]['npdes_no']
        rename_data.writerow([agency_name, npdes_no, pdf_name])
        print(f'processed {agency_name}, {npdes_no}, {pdf_name}, {order_no}')
        driver.execute_script("document.body.style.zoom='0.5'")
        driver.save_screenshot('npdes_permits/output/img4a.png')
    else:
        print(f'No PDF for row {i}')
        driver.execute_script("document.body.style.zoom='0.5'")
        driver.save_screenshot('npdes_permits/output/img4b.png')
    i += 1

for row in table_rows:
    table_data = row.find_elements(By.TAG_NAME, 'td')
    row_data = []
    for data in table_data:
        row_data.append(data.text)
    npdes_permits.writerow(row_data)

# Close CSV file and quit driver
file.close()
pdfs.close()
driver.close()