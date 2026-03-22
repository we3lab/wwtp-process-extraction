from PyPDF2 import PdfReader
import re
import os
import csv
import json
from helpers.utils import extract_leaves

DATE_FOLDER = '2026-2-18'


def normalize_text(text):
    """Normalize text by replacing line breaks and multiple spaces"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def find_attachment_f(pdf_path, start_page=10):
    """
    Find the page containing 'ATTACHMENT F - FACT SHEET' after start_page.
    Handles different hyphen types (-, –, —, ‐).
    """
    reader = PdfReader(pdf_path)
    pattern = r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET"
    
    for pg in range(start_page, len(reader.pages)):
        try:
            text = reader.pages[pg].extract_text()
            if re.search(pattern, text, re.IGNORECASE):
                return pg, None
        except Exception as e:
            print(f"Error reading page {pg}: {e}")
            continue
    
    return None, None


def find_nth_occurrence(text, search_text, n=1, start_pos=0):
    """Find the nth occurrence of search_text in text (case-insensitive)."""
    pattern = re.escape(search_text)
    matches = list(re.finditer(pattern, text[start_pos:], re.IGNORECASE))
    
    if len(matches) >= n:
        return start_pos + matches[n - 1].start()
    
    return -1


def extract_permit_sections(pdf_path):
    """
    Extract Facility Description and Planned Changes sections from permit PDF.
    Returns dict with 'txt_section', 'txt_changes', and metadata.
    """
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return None
    
    try:
        reader = PdfReader(pdf_path)
        print(f"Total pages: {len(reader.pages)}")
        
        # Step 1: Find ATTACHMENT F - FACT SHEET after page 10
        attachment_page, _ = find_attachment_f(pdf_path, start_page=10)
        
        if attachment_page is None:
            print(f"'ATTACHMENT F - FACT SHEET' not found after page 10")
            return None
        
        print(f"Found ATTACHMENT F on page {attachment_page + 1}")
        
        # Step 2: Extract text from Attachment F onwards
        full_text = ''
        for i in range(attachment_page, len(reader.pages)):
            try:
                page_text = reader.pages[i].extract_text()
                full_text += page_text + ' '
            except Exception as e:
                print(f"Error extracting page {i}: {e}")
                continue
        
        full_text = normalize_text(full_text)
        
        if not full_text:
            print("No text extracted from Attachment F")
            return None
        
        # Step 3: Find "Facility Description" occurrences
        first_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=1)
        second_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=2)
        
        if second_fac_desc != -1:
            fac_desc_start = second_fac_desc
            print(f"Using 2nd 'Facility Description' at position {second_fac_desc}")
        elif first_fac_desc != -1:
            fac_desc_start = first_fac_desc
            print(f"Only 1 'Facility Description' found at position {first_fac_desc}")
        else:
            print("'Facility Description' not found")
            return None
        
        # Step 4: Find "Planned Changes" after Facility Description (with 200 char margin)
        search_start = fac_desc_start + 200
        planned_changes_pos = find_nth_occurrence(
            full_text, "Planned Changes", n=1, start_pos=search_start
        )
        
        if planned_changes_pos == -1:
            print(f"'Planned Changes' not found after position {search_start}")
            txt_section = full_text[fac_desc_start:].strip()
            return {
                'txt_section': txt_section,
                'txt_changes': '',
                'full_text': full_text,
                'metadata': {
                    'attachment_f_page': attachment_page + 1,
                    'facility_desc_pos': fac_desc_start,
                    'planned_changes_pos': None,
                    'applicable_plans_pos': None
                }
            }
        
        print(f"Found 'Planned Changes' at position {planned_changes_pos}")
        
        # Step 5: Find "applicable plans, policies and regulations" after Planned Changes
        applicable_plans_pos = find_nth_occurrence(
            full_text, "applicable plans, policies and regulations",
            n=1, start_pos=planned_changes_pos
        )
        
        if applicable_plans_pos == -1:
            alternatives = [
                "applicable plans and policies",
                "applicable plans",
                "plans, policies and regulations",
                "plans and regulations"
            ]
            
            for alt in alternatives:
                applicable_plans_pos = find_nth_occurrence(
                    full_text, alt, n=1, start_pos=planned_changes_pos
                )
                if applicable_plans_pos != -1:
                    print(f"Found alternative: '{alt}' at position {applicable_plans_pos}")
                    break
        
        if applicable_plans_pos == -1:
            print("'applicable plans' section not found")
            applicable_plans_pos = min(planned_changes_pos + 5000, len(full_text))
        else:
            print(f"Found 'applicable plans' at position {applicable_plans_pos}")
        
        txt_section = full_text[fac_desc_start:planned_changes_pos].strip()
        txt_changes = full_text[planned_changes_pos:applicable_plans_pos].strip()
        
        return {
            'txt_section': txt_section,
            'txt_changes': txt_changes,
            'full_text': full_text,
            'metadata': {
                'attachment_f_page': attachment_page + 1,
                'facility_desc_pos': fac_desc_start,
                'planned_changes_pos': planned_changes_pos,
                'applicable_plans_pos': applicable_plans_pos,
                'txt_section_length': len(txt_section),
                'txt_changes_length': len(txt_changes)
            }
        }
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


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
            
            # Determine status: 0, "present", "future", or "present_and_future"
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