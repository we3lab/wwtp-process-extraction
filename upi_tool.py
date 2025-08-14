import PyPDF2
from PyPDF2 import PdfReader
from pathlib import Path
import re
import os
import csv
import glob
import json

# Gets all keys from dictonary for CSV headers and key[value] = #  use
def get_all_keys(processes_dict):
    processes = []
    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            if 'alt_names' in details and details['alt_names']:
                processes.append(process_name)
            else:  # Parent category, recursively get children
                processes.extend(get_all_keys(details))
    return processes

# Function that finds all unit process keywords
def search_processes_in_text(text, processes_dict, results, parent_name=None):
    sub_category_found = False
    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            # Check for alt_names
            if 'alt_names' in details:
                if process_name not in results:
                    results[process_name] = 0
                for i, alt_name in enumerate(details['alt_names']):
                    case_sensitive = details['alt_names_case_sensitive'][i] if 'alt_names_case_sensitive' in details and i < len(details['alt_names_case_sensitive']) else "N"
                    if case_sensitive == "Y":
                        found = alt_name in text
                    else:
                        found = alt_name.lower() in text.lower()
                    if found:
                        results[process_name] = 1
                        sub_category_found = True
                        break
            # Search sub-categories (if they exist)
            else:  # This is a parent category. Recursively search children’s keywords
                sub_found = search_processes_in_text(text, details, results, process_name)
                if sub_found:
                    sub_category_found = True
    # If this is a sub-category call and something was found, mark the parent
    if parent_name and sub_category_found:
        results[parent_name] = 1

    return sub_category_found

# Method that writes the given pdf pathname into a string
def find_pages(pdf, text_section):
   reader = PdfReader(pdf)
   for pg in range(2, len(reader.pages)):
    text = reader.pages[pg].extract_text()
    res_search = re.search(text_section, text)
    res_search_2 = re.search(text_section.upper(), text)
    if res_search or res_search_2:
        return pg + 1
    
def pdf_text(path, text_selection):
   pdf = PdfReader(path)
   text = ''
   start = find_pages(path, text_selection)
   for i in range(start, len(pdf.pages)):
     text += pdf.pages[i].extract_text()   
   text = text.replace('\n', ' ').replace('\r', ' ')
   text = text.replace('  ', ' ')
   return text

# CSV names
file = 'unit_processes.csv'
rfr_data = 'site_data.csv'

csv_file = open(file, 'w', newline = '') # File that will store results
ps_file = open(rfr_data, 'r', newline = '') # File that contains agency names, NPDES numbers, and PDF file names
upi = csv.writer(csv_file)

# Opens JSON file with keywords
with open('unitprocess_keywords.json', 'r') as f:
    keywords = json.load(f)

# Directory where PDFS are stored
directory = '/Users/ashleyramirez/Documents/WE3_Lab/2025-7-30'

# Lists to store headers & first/second column data
agency_name = []
npdesNO = []
pdfs = []

# Reads rfr_data file and extracts agency names, NPDES numbers, and PDF file names
with open(rfr_data, 'r') as f:
  for line in csv.reader(f):
      agency_name.append(line[0])
with open(rfr_data, 'r') as f:
   for line in csv.reader(f):
       npdesNO.append(line[1])
with open(rfr_data, 'r') as f:
   for line in csv.reader(f):
       pdfs.append(line[2]) 
 
# This segment creates the formatting of the CSV file
headers = []
all_keys = (get_all_keys(keywords))
headers.append('AGENCY_NAME')  # Add agency name column
headers.append('PERMIT_NUMBER')  # Add permit number column
for i in range(len(all_keys)):
    headers.append(all_keys[i])
upi.writerow(headers)

# Main code that executes keyword search 
for i in range(len(pdfs)):
    path = directory + '/' + pdfs[i]
    solid_text = pdf_text(path, 'Facility Description')
    data_row = []
    results = {}
    data_row.append(agency_name[i])  # Adds agency name
    data_row.append(npdesNO[i]) # Adds NPDES No.
    for category, processes in keywords.items():
        if isinstance(processes, dict):
            search_processes_in_text(solid_text, processes, results, None)
    for keys in all_keys:
        data_row.append(results[keys]) # For Loop appends all detection of unit process keywords
    upi.writerow(data_row)

csv_file.close()
ps_file.close()