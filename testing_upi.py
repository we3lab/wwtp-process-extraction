import PyPDF2
from PyPDF2 import PdfReader
import os
import csv
import glob
import json
import re
import pandas as pd

# Method that returns list of all keys in a dictionar, includding nested keys
def get_all_keys(dictionary):
    keys =[]
    for key, value in dictionary.items():
        if isinstance(value, dict):
            for nested_key in value.keys():
                if nested_key != 'alt_names' and nested_key != 'alt_names_case_sensitive' and nested_key != 'CWNS_2022_code' and nested_key != 'CWNS_2012_code':
                    keys.append(nested_key)
    return keys

# Returns string of all text in PDF
def pdf_string(pdf):
    reader = PdfReader(pdf)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
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

# Opens JSON file with keywords
with open('unitprocess_keywords.json', 'r') as f:
    keywords = json.load(f)
# Creates mock data for later use
file = open('mock_data.csv', 'w', newline = '')
mock_csv = csv.writer(file)
mock_csv.writerow(['Agency', 'Email', 'TEueXT', 'Numbers'])
mock_csv.writerow(['Benicia', 'NONE', 'dfhjdskf', '10'])
mock_csv.writerow(['County', 'NONE', 'dfhjdskf', '0'])
file.close()

# Name of the CSV to be used
csv_name = 'test_results.csv'
# Reopens CSV to write data
csv_file = open(csv_name, 'w', newline ='')
wd_csv = csv.writer(csv_file)

# Directory where PDFS are stored
directory = '/Users/ashleyramirez/Documents/TEST/PDFs'

# Get first column of a previously created CSV
data = []
with open('mock_data.csv', 'r') as f:
   for line in csv.reader(f):
       data.append(line[0])

# Variable for all UPI names
all_keys = (get_all_keys(keywords))
# Creates header for CSV
headers = []
headers.append(data[0])
for i in range(len(all_keys)):
    headers.append(all_keys[i])
wd_csv.writerow(headers)


print(data)
# MAIN CODE
file_list = glob.glob(directory + '/*.pdf')
print(pdf_string(file_list[1]))
for pdf in file_list: # Loops through all PDFs in the diretory
    text = pdf_string(pdf)
    for i in range(1, len(data)): # Checks if PDF and left column match
       if track_class(pdf, data[i]): 
        trial = [data[i]]# If match is found 
        print('Match was found for ' + data[i] + ' in ' + pdf)
        results = {}
        for category, processes in keywords.items():
            if isinstance(processes, dict):
                for process_name, details in processes.items():
                    if isinstance(details, dict) and 'alt_names' in details:
                        # Initialize unit process key to 0
                        if process_name not in results:
                            results[process_name] = 0
                        
                        # Check each alt_name for each process
                        for k, alt_name in enumerate(details['alt_names']):
                            case_sensitive = details['alt_names_case_sensitive'][k] if k < len(details['alt_names_case_sensitive']) else "N"
                            
                            if case_sensitive == "Y":
                                found = alt_name in text
                            else:
                                found = alt_name.lower() in text.lower()
                            if found:
                                results[process_name] = 1
                                break
                    print(process_name)
                                  # Found one alt_name, don't need to check others
        for process_name in results:
            trial.append(results[process_name])
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