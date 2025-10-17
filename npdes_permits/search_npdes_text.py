import PyPDF2
from PyPDF2 import PdfReader
from pathlib import Path
import re
import os
import csv
import glob
import json
from datetime import datetime
from text_extraction import extract_permit_sections

DATE_FOLDER = '2025-10-8'

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


# CSV names
file = f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv'
rfr_data = f'npdes_permits/output/{DATE_FOLDER}/site_data.csv'

csv_file = open(file, 'w', newline='')
ps_file = open(rfr_data, 'r', newline='')
upi = csv.writer(csv_file)

# Opens JSON file with keywords
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    keywords = json.load(f)

# Directory where PDFs are stored
directory = f'npdes_permits/output/{DATE_FOLDER}/npdes'

# Lists to store headers & data
agency_name = []
npdesNO = []
pdfs = []

# Read site data
with open(rfr_data, 'r') as f:
    for line in csv.reader(f):
        agency_name.append(line[0])
with open(rfr_data, 'r') as f:
    for line in csv.reader(f):
        npdesNO.append(line[1])
with open(rfr_data, 'r') as f:
    for line in csv.reader(f):
        pdfs.append(line[2])

# Create CSV headers with status suffixes
headers = ['AGENCY_NAME', 'PERMIT_NUMBER']
all_keys = get_all_keys(keywords)

# Add columns for each treatment: TREATMENT_NAME_status and TREATMENT_NAME_binary
for key in all_keys:
    headers.append(f'{key}_status')  # "present", "future", or "not_found"
    headers.append(f'{key}_binary')  # 1 if found anywhere, 0 if not

upi.writerow(headers)

# Main processing loop
for i in range(len(pdfs)):
    path = os.path.join(directory, pdfs[i])
    
    print(f"\n{'='*80}")
    print(f"Processing {pdfs[i]} ({i+1}/{len(pdfs)})")
    print(f"{'='*80}")
    
    # Extract sections from PDF
    extraction_result = extract_permit_sections(path)
    
    if extraction_result is None:
        print(f"Skipping {pdfs[i]} - extraction failed")
        continue
    
    # Search for treatments in both sections
    present_results = {}
    future_results = {}
    
    # Search Facility Description section (present treatments)
    for category, processes in keywords.items():
        if isinstance(processes, dict):
            search_processes_in_text(
                extraction_result['txt_section'], 
                processes, 
                present_results, 
                None
            )
    
    # Search Planned Changes section (future treatments)
    if extraction_result['txt_changes']:
        for category, processes in keywords.items():
            if isinstance(processes, dict):
                search_processes_in_text(
                    extraction_result['txt_changes'], 
                    processes, 
                    future_results, 
                    None
                )
    
    # Build data row
    data_row = [agency_name[i], npdesNO[i]]
    for key in all_keys:
        is_present = present_results.get(key, 0) == 1
        is_future = future_results.get(key, 0) == 1
        if is_present and is_future:
            status = "present_and_future"
        elif is_present:
            status = "present"
        elif is_future:
            status = "future"
        else:
            status = "not_found"
        binary = 1 if (is_present or is_future) else 0
        data_row.append(status)
        data_row.append(binary)
    upi.writerow(data_row)
    print(f"✓ Processed {pdfs[i]}")

csv_file.close()
ps_file.close()

print(f"\n{'='*80}")
print(f"Processing complete! Results saved to: {file}")
print(f"{'='*80}")