import PyPDF2
from PyPDF2 import PdfReader
import re
import os


def normalize_text(text):
    """Normalize text by replacing line breaks and multiple spaces"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip()

def find_attachment_f(pdf_path, start_page=10):
    """
    Find the page containing 'ATTACHMENT <A.Z> - FACT SHEET' after start_page.
    Handles different hyphen types (-, –, —, ‐).
    
    Returns: (page_number, position_in_full_text) or (None, None)
    """
    reader = PdfReader(pdf_path)
    
    # Pattern that matches various hyphen types
    pattern = r"ATTACHMENT\s+[A-Z]\s*[-–—‐]\s*FACT\s+SHEET"

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
    pattern = re.escape(search_text)
    matches = list(re.finditer(pattern, text[start_pos:], re.IGNORECASE))
    
    if len(matches) >= n:
        return start_pos + matches[n - 1].start()
    
    return -1

def extract_permit_sections(pdf_path):
    """
    Extract Facility Description and Planned Changes sections from permit PDF.
    
    Returns: dict with keys:
        - 'txt_section': Text from 2nd "Facility Description" to "Planned Changes"
        - 'txt_changes': Text from "Planned Changes" to "applicable plans"
        - 'full_text': Full text from Attachment F onwards
        - 'metadata': Dictionary with positions and page numbers
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
        
        # Normalize text
        full_text = normalize_text(full_text)
        
        if not full_text:
            print("No text extracted from Attachment F")
            return None
        
        # Step 3: Find "Facility Description" occurrences
        first_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=1)
        second_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=2)
        
        # Use second occurrence if found, otherwise use first
        if second_fac_desc != -1:
            fac_desc_start = second_fac_desc
            print(f"Using 2nd 'Facility Description' at position {second_fac_desc}")
        elif first_fac_desc != -1:
            first_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=1)
            second_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=2)
            if second_disch_desc != -1:
                fac_desc_start = second_disch_desc
                print(f"Using 2nd 'Discharge Description' at position {second_disch_desc}")
            elif first_disch_desc != -1 and first_fac_desc > first_disch_desc:
                fac_desc_start = first_disch_desc
                print(f"Using 1st 'Discharge Description' at position {first_disch_desc} before 'Facility Description'")
            else:
                fac_desc_start = first_fac_desc
                print(f"Using 1st 'Facility Description' at position {first_fac_desc}")
        else:
            first_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=1)
            second_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=2)
            if second_disch_desc != -1:
                fac_desc_start = second_disch_desc
                print(f"Using 2nd 'Discharge Description' at position {second_disch_desc}")
            elif first_disch_desc != -1:
                fac_desc_start = first_disch_desc
                print(f"Using 1st 'Discharge Description' at position {first_disch_desc}")
            print("'Facility Description' not found")
            return None
        
        # Step 4: Find "Planned Changes" after Facility Description (with 200 char margin)
        search_start = fac_desc_start + 200 # making sure we did not take the occurence from the Table of Contents
        planned_changes_pos = find_nth_occurrence(full_text, "Planned Changes", n=1, start_pos=search_start)
        
        if planned_changes_pos == -1:
            planned_changes_pos = find_nth_occurrence(full_text, "Planned upgrades", n=1, start_pos=search_start)
        
        # Step 5: Find "applicable plans, policies and regulations" after Planned Changes
        applicable_plans_pos = find_nth_occurrence(
            full_text,
            "applicable plans, policies and regulations",
            n=1,
            start_pos=planned_changes_pos
        )
        
        if applicable_plans_pos == -1:
            # Try alternative phrasings
            alternatives = [
                "applicable plans and policies",
                "applicable plans",
                "plans, policies and regulations",
                "plans and regulations"
            ]
            
            for alt in alternatives:
                applicable_plans_pos = find_nth_occurrence(
                    full_text, alt, n=1, start_pos=max(planned_changes_pos, fac_desc_start)
                )
                if applicable_plans_pos != -1:
                    print(f"Found alternative: '{alt}' at position {applicable_plans_pos}")
                    break

        if planned_changes_pos == -1:
            print("'Planned Changes' section not found")
            if applicable_plans_pos != -1:
                print("Found 'applicable plans' section, using it as end of Facility Description")

                txt_section = full_text[fac_desc_start:applicable_plans_pos].strip()
                return {
                    'txt_section': txt_section,
                    'txt_changes': '',
                    'full_text': full_text,
                    'metadata': {
                        'attachment_f_page': attachment_page + 1,
                        'facility_desc_pos': fac_desc_start,
                        'planned_changes_pos': None,
                        'applicable_plans_pos': applicable_plans_pos
                    }
                }
            else:
                print("Neither 'Planned Changes' nor 'applicable plans' sections found, using end of text")
                txt_section = full_text[fac_desc_start:fac_desc_start+10000].strip()
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
        else:
            print(f"Found 'Planned Changes' at position {planned_changes_pos}")

            if applicable_plans_pos == -1:
                print("'applicable plans' section not found, using end of Planned Changes section")
                # Use a reasonable end (e.g., 5000 chars after Planned Changes)
                applicable_plans_pos = min(planned_changes_pos + 5000, len(full_text))
            else:
                print(f"Found 'applicable plans' at position {applicable_plans_pos}")
            
            # Extract the two sections
            txt_section = full_text[fac_desc_start:planned_changes_pos].strip()
            txt_changes = full_text[planned_changes_pos:applicable_plans_pos].strip()
            
            # Return structured results
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
