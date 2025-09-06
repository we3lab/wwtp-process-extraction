import PyPDF2
from PyPDF2 import PdfReader
import os
import csv
import glob
import json
import re
import pandas as pd

from utils import *

# Returns string of all text in PDF
def pdf_string(pdf):
    reader = PdfReader(pdf)
    text = ''
    for page in reader.pages:
        text += page.extract_text()

    # Remove line breaks in PDF text and replace with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('  ', ' ')
    
    # Find the 2nd instance of "FACILITY DESCRIPTION"
    first_occurrence = text.find("FACILITY DESCRIPTION")
    if first_occurrence != -1:
        second_occurrence = text.find("FACILITY DESCRIPTION", first_occurrence + 50)
        if second_occurrence != -1:
            start_pos = second_occurrence + 50
            extracted_text = text[start_pos:-1].strip()
            return extracted_text
        else:
            start_pos = first_occurrence + 50
            extracted_text = text[start_pos:-1].strip()
    else:
        return text

# Method that returns T/F if agency name is found
def track_class(pdf, class_code):
   reader = PdfReader(pdf)
   text = reader.pages[0].extract_text()
   res_search = re.search(class_code, text)
   if res_search:
       return True
   else:
       return False

def search_processes_in_text(processes_dict, results, parent_name=None):
    """    
    This function performs a hierarchical search through the unit process keywords json,
    checking for matches in the text. It handles parent categories and their nested
    sub-categories, automatically updating parent categories when sub-categories are found.
    
    Args:
        processes_dict (dict): Dictionary containing process definitions with 'alt_names' 
        results (dict): Dictionary to store search results, where keys are process names 
                       and values are 0 (not found) or 1 (found)
        parent_name (str, optional): Name of the parent category being processed. 
                                   Used for updating parent categories when sub-categories are found.
    
    Returns:
        bool: True if any sub-category was found in the current level, False otherwise.
              This return value is used to update parent categories in recursive calls.
    
    Note:
        - The function modifies the 'results' dictionary in-place
        - Case sensitivity is determined by the 'alt_names_case_sensitive' field in the process definition
        - When a sub-category is found, the parent category is automatically marked as present
    """
    sub_category_found = False  # Initialize as false

    for process_name, details in processes_dict.items():  # Loop through items in json
        if isinstance(details, dict):
            # Check if this is a process with alt_names (lowest level)
            if 'alt_names' in details:
                # Initialize unit process key to 0
                if process_name not in results:
                    results[process_name] = 0  # Initialize unit process to zero
                
                # Check each alt_name for each process
                for i, alt_name in enumerate(details['alt_names']):
                    case_sensitive = details['alt_names_case_sensitive'][i] if i < len(details['alt_names_case_sensitive']) else "N"
                    
                    if case_sensitive == "Y":
                        found = alt_name in text
                    else:
                        found = alt_name.lower() in text.lower()
                    
                    if found:
                        # print(f' found {process_name}')
                        results[process_name] = 1
                        sub_category_found = True
                        break  # Found one alt_name, don't need to check others
            else:  # This is a parent category. Recursively search children's keywords
                sub_found = search_processes_in_text(details, results, process_name)
                if sub_found:
                    sub_category_found = True
    
    if parent_name and sub_category_found:  # if this is parent category and any sub-category found
        results[parent_name] = 1
    
    return sub_category_found

# Opens JSON file with keywords
with open('data/unitprocess_keywords.json', 'r') as f:
    keywords = json.load(f)

# Creates mock data for later use
file = open('output/mock_data.csv', 'w', newline = '')
mock_csv = csv.writer(file)
mock_csv.writerow(['Agency', 'Email', 'TEueXT', 'Numbers'])
mock_csv.writerow(['Benicia', 'NONE', 'dfhjdskf', '10'])
mock_csv.writerow(['County', 'NONE', 'dfhjdskf', '0'])
file.close()

# Name of the CSV to be used
csv_name = 'output/test_results.csv'
# Reopens CSV to write data
csv_file = open(csv_name, 'w', newline ='')
wd_csv = csv.writer(csv_file)

# Directory where PDFS are stored
directory = 'data/pdfs'

# Get first column of a previously created CSV
data = []
with open('output/mock_data.csv', 'r') as f:
   for line in csv.reader(f):
       data.append(line[0])

# Get all process names
all_keys = get_all_keys(keywords)

# Creates header for CSV
headers = []
headers.append(data[0])
headers.append('PERMIT_NUMBER')  # Add permit number column
for i in range(len(all_keys)):
    headers.append(all_keys[i])

wd_csv.writerow(headers)


# print(data)
# MAIN CODE
file_list = glob.glob(directory + '/*.pdf')
# print(pdf_string(file_list[1]))
for pdf in file_list: # Loops through all PDFs in the diretory
    text = pdf_string(pdf)
    for i in range(1, len(data)): # Checks if PDF and left column match
       if track_class(pdf, data[i]): 
        trial = [data[i]]# If match is found 
        print('Match was found for ' + data[i] + ' in ' + pdf)
        permit_no = pdf.split("/")[2] # assuming filename is NPDES #
        trial.append(permit_no.split(".")[0])

        results = {}
        for category, processes in keywords.items():
            if isinstance(processes, dict):
                search_processes_in_text(processes, results, None)
        for process_name in all_keys: # Add results in the same order as headers
            trial.append(results.get(process_name, 0))
        
        wd_csv.writerow(trial)




"""
results = {}
file_list = glob.glob(directory + '/*.pdf')
for pdf in file_list:
    for i in range(1, len(data)):
        if track_class(pdf, data[i]):
            word_finder2(pdf)
""" 

"""

for x in range(len(file_list)):
   for i in range(1, len(data)):
      if track_class(file_list[x], data[i]):
         print(file_list[x] + " matched with " + data[i])
         pdf_status[x] = 1
         results = []
         results.append(data[i])
         for word in word_list:
            results.append(word_finder(file_list[x], word))
         wd_csv.writerow(results)
"""
#####################
"""
file_list = glob.glob(directory + '/*.pdf')
for pdf in file_list:
   for i in range(1, len(data)): # Setting this range allows for header to be ignored
       if track_class(pdf, data[i]):
           print(pdf + " matched with " + data[i])
           results = []
           for word in word_list:
               
               results.write row(word_finder(pdf, word))
           wd_csv.writerow(results)
   print(str(pdf) + ' was scanned.')
"""