from PyPDF2 import PdfReader
import re
import os
import csv
import json
from helpers.utils import extract_leaves
from helpers.npdes_text_extraction import (
    normalize_text, find_attachment_f, find_nth_occurrence, extract_permit_sections
)

DATE_FOLDER = '2026-2-18'


def search_processes_in_text(text, processes_dict, results, parent_name=None):
    """Search that supports the JSON structure.

    - `alt_names` is a list of alternative strings.
    - `alt_names_case_sensitive` is a list of alternative names that should be
      matched using case-sensitive matching. If empty, matching is case-insensitive.
    The function recurses through nested dicts and sets `results[process_name]=1`
    when any of its alt names are found. If a subtree contains any matches,
    the parent_name (if provided) is set to 1.
    """
    sub_category_found = False

    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            # Leaf node with alt_names
            if 'alt_names' in details:
                if process_name not in results:
                    results[process_name] = 0

                alt_names = details.get('alt_names', []) or []
                cs_list = details.get('alt_names_case_sensitive', []) or []

                for alt_name in alt_names:
                    # If the alt_name is listed in cs_list, match case-sensitively
                    if alt_name in cs_list:
                        found = alt_name in text
                    else:
                        found = alt_name.lower() in text.lower()

                    if found:
                        results[process_name] = 1
                        sub_category_found = True
                        break

            else:
                # Nested dictionary: recurse
                sub_found = search_processes_in_text(text, details, results, process_name)
                if sub_found:
                    sub_category_found = True

    if parent_name and sub_category_found:
        results[parent_name] = 1

    return sub_category_found


def main():
    """Main processing function with present/future treatment tracking"""
    
    # CSV names
    file = f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv'
    rfr_data = f'npdes_permits/output/{DATE_FOLDER}/site_data.csv'
    os.makedirs(os.path.dirname(file), exist_ok=True)
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
    facility_name=[]
    npdesNO = []
    pdfs = []
    shared_pdfs = []
    
    # Read site data
    with open(rfr_data, 'r') as f:
        for line in csv.reader(f):
            agency_name.append(line[0])
            facility_name.append(line[1])
            npdesNO.append(line[2])
            pdfs.append(line[5])
            shared_pdfs.append(line[6] if len(line) > 6 else '')
    
    # Create CSV headers - one column per process
    headers = ['AGENCY_NAME', 'FACILITY_NAME', 'PERMIT_NUMBER', 'PDF_File', 'Shared_PDF']
    leaves = extract_leaves(keywords, ignore_disposal=False)
    all_keys = [name for name, _, _ in leaves]

    # Add one column for each treatment process
    headers.extend(name for name, _, _ in leaves)
    
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
        data_row = [agency_name[i], facility_name[i], npdesNO[i], pdfs[i], shared_pdfs[i] if len(shared_pdfs) > i else '']
        
        for key in all_keys:
            is_present = present_results.get(key, 0) == 1
            is_future = future_results.get(key, 0) == 1
            
            # Determine status: 0, "present", "present_and_future", or "future"
            if is_present and is_future:
                status = "present_and_future"
            elif is_present:
                status = "present"
            elif is_future:
                status = "future"
            else:
                status = "0"
            
            data_row.append(status)
        
        upi.writerow(data_row)
        print(f"✓ Processed {pdfs[i]}")
    
    csv_file.close()
    ps_file.close()
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Results saved to: {file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()