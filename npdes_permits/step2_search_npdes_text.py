import os
import csv
import json
from helpers.utils import extract_leaves
from helpers.npdes_text_extraction import extract_permit_sections

DATE_FOLDER = '2026-4-26'


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
        for row in csv.DictReader(f):
            agency_name.append(row.get('Agency', ''))
            facility_name.append(row.get('Facility_Name', ''))
            npdesNO.append(row.get('NPDES_No', ''))
            pdfs.append(row.get('PDF_File', ''))
            shared_pdfs.append(row.get('Shared_PDF', ''))
    
    # Create CSV headers - one column per process
    headers = ['AGENCY_NAME', 'FACILITY_NAME', 'PERMIT_NUMBER', 'PDF_File', 'Shared_PDF']
    leaves = extract_leaves(keywords, ignore_disposal=False)
    all_keys = [name for name, _, _ in leaves]

    # Add one column for each treatment process
    headers.extend(name for name, _, _ in leaves)
    
    upi.writerow(headers)

    # Pre-compute unique PDFs so shared files are only extracted once
    unique_pdfs = list(dict.fromkeys(p for p in pdfs if p))
    pdf_cache = {}  # filename -> (present_results, future_results) or None

    for j, pdf_file in enumerate(unique_pdfs):
        path = os.path.join(directory, pdf_file)
        print(f"Processing {pdf_file} ({j+1}/{len(unique_pdfs)})")

        extraction_result = extract_permit_sections(path, regenerate_text_excerpts=True)
        if extraction_result is None:
            print(f"Skipping {pdf_file} - extraction failed")
            pdf_cache[pdf_file] = None
            continue

        present_results = {}
        future_results = {}

        for category, processes in keywords.items():
            if not isinstance(processes, dict):
                continue
            if 'alt_names' in processes:
                # Top-level leaf node — wrap so search_processes_in_text sees expected structure
                wrapped = {category: processes}
                search_processes_in_text(extraction_result['txt_section'], wrapped, present_results, None)
                if extraction_result['txt_changes']:
                    search_processes_in_text(extraction_result['txt_changes'], wrapped, future_results, None)
            else:
                # Category node containing sub-processes
                search_processes_in_text(extraction_result['txt_section'], processes, present_results, None)
                if extraction_result['txt_changes']:
                    search_processes_in_text(extraction_result['txt_changes'], processes, future_results, None)

        pdf_cache[pdf_file] = (present_results, future_results)

    # Write one output row per facility (shared PDFs reuse cached results)
    for i in range(len(pdfs)):
        pdf_file = pdfs[i]
        cached = pdf_cache.get(pdf_file)
        if cached is None:
            continue

        present_results, future_results = cached
        data_row = [agency_name[i], facility_name[i], npdesNO[i], pdf_file, shared_pdfs[i]]

        for key in all_keys:
            is_present = present_results.get(key, 0) == 1
            is_future = future_results.get(key, 0) == 1

            if is_present and is_future:
                status = "PRESENT_AND_FUTURE"
            elif is_present:
                status = "PRESENT"
            elif is_future:
                status = "FUTURE"
            else:
                status = "0"

            data_row.append(status)

        upi.writerow(data_row)
    
    csv_file.close()
    ps_file.close()
    
    print(f"\n{'='*80}")
    print(f"Processing complete! Results saved to: {file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()