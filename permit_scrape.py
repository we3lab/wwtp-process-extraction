from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.client_config import ClientConfig
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import os
import time
import csv

# Function that selects options from filters
def selection(name, text):
  select_element = driver.find_element(By.NAME, name)
  select = Select(select_element)
  select.select_by_visible_text(text)

# Link to Interactive Regulated Facilities Report
rfr_url = 'https://ciwqs.waterboards.ca.gov/ciwqs/readOnly/CiwqsReportServlet?inCommand=reset&reportName=RegulatedFacility'

# Creates folder to store downloaded PDFs
path = '/Users/ashleyramirez/Documents/WE3_Lab'
now = datetime.now()
pdf_folder = f'{now.year}-{now.month}-{now.day}'
full_path = os.path.join(path, pdf_folder)
os.mkdir(full_path)

# Opens all necessary CSV files
file = open('all_ca_npdes.csv', 'w', newline = '')
npdes_permits = csv.writer(file)
pdfs = open('site_data.csv', 'w', newline = '')
rename_data = csv.writer(pdfs)

# Write headers for OTHER_CSV.csv
# OTHER_CSV_NAME.writerow(['Agency Name', 'NPDES No.'])

# Sets up Chrome and folder for downloads
options = webdriver.ChromeOptions()
prefs = {'download.default_directory': full_path}
options.add_experimental_option('prefs', prefs)
options.page_load_strategy = "eager"
options.add_argument('--blink-settings=imagesEnabled=false')
driver = webdriver.Chrome(options = options)
# Wait time for driver
wait = WebDriverWait(driver, 10)

# Software gets url
driver.get(rfr_url)

# Selection clicks desired filters
selection('programDrop','NPDES')
selection('typeDrop','Wastewater Treatment Facility')
selection('wasteTypeDrop','Domestic wastewater')

# Software hits submit button
driver.find_element(By.NAME, 'enpRepButton').click()

driver.command_executor.client_config.timeout = 1000 #Timeout to avoid error.

# Software hits the desired data set. FIND A BETTER METHOD TO ACCOUNT FOR FUTURE UPDATES*
driver.find_element(By.LINK_TEXT, '37').click()

driver.command_executor.client_config.timeout = 1000 #Timeout to avoid error.

# Software reformats page to show all reports
selection('pagesizeselect', 'ALL')

# Variables to store table & rows
table_body = driver.find_element(By.CLASS_NAME, 'ciwqsReportDataTable')
table_rows = table_body.find_elements(By.TAG_NAME, 'tr')

 #Downloads all available PDFS
for i in range(2, len(table_rows) + 1):
  try:
      pdf_link = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/table/tbody/tr[3]/td/table/tbody/tr[1]/td/table[2]/tbody/tr[7]/td[2]/table[1]/tbody/tr['+ str(i) + ']/td[14]/a')))
      pdf_link.click()
      time.sleep(1)
      driver.switch_to.window(driver.window_handles[1])
      document_download = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, '.pdf')))
      pdf_name = document_download.text
      document_download.click()
      time.sleep(5)
      driver.close()
      driver.switch_to.window(driver.window_handles[0])
      # Extracts relevant data for the CSV data collection
      agency_name = driver.find_element(By.XPATH, '/html/body/table/tbody/tr[3]/td/table/tbody/tr[1]/td/table[2]/tbody/tr[7]/td[2]/table[1]/tbody/tr['+ str(i) + ']/td[1]/span/a').text
      npdes_no = driver.find_element(By.XPATH, '/html/body/table/tbody/tr[3]/td/table/tbody/tr[1]/td/table[2]/tbody/tr[7]/td[2]/table[1]/tbody/tr['+ str(i) +']/td[16]/span').text
      rename_data.writerow([agency_name, npdes_no, pdf_name])
      i += 1
  except:
      print('PDF unavailable')
      i += 1 # Accounts for the skipped rows

# Extracts data from table and writes to CSV
for row in table_rows:
  table_data = row.find_elements(By.TAG_NAME, 'td')
  row_data = []
  for data in table_data:
      row_data.append(data.text)
  npdes_permits.writerow(row_data)

# Closes CSV file and quits driver
file.close()
pdfs.close()
driver.close()