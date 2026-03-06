from PyPDF2 import PdfReader
import pandas as pd
from pathlib import Path
from langchain.schema.document import Document
import re
import os

def load_documents(pdf_file):
    """Load PDFs from a directory and return a list of Documents.

    For each PDF in `pdf_directory`, attempt to extract the relevant
    permit sections (Facility Description and Planned Changes) using
    `extract_permit_sections`. If extraction succeeds, create a
    `Document` with the extracted text (preferring `txt_section` and
    `txt_changes`) and metadata including the source path and the
    Attachment F start page.
    """
    document = []
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_file}")
        return document

    res = extract_permit_sections(str(pdf_path))
    if not res:
        # fallback: try to load full PDF text
        reader = PdfReader(str(pdf_path))
        full_text = ""
        for p in reader.pages:
            try:
                full_text += (p.extract_text() or "") + "\n"
            except Exception:
                continue
        page_content = normalize_text(full_text)
        metadata = {"source": str(pdf_path), "page": 0}
    else:
        # prefer the focused section + planned changes
        page_content = (res.get("txt_section", "") + "\n\n" + res.get("txt_changes", "")).strip()
        if not page_content:
            page_content = res.get("full_text", "")
        metadata = {"source": str(pdf_path), "page": res.get("metadata", {}).get("attachment_f_page", 0)}

    if page_content:
        document.append(Document(page_content=page_content, metadata=metadata))
    return document


def normalize_text(text):
    """Normalize text by replacing line breaks and multiple spaces"""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
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
            text = reader.pages[pg].extract_text() or ""
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
        # Step 1: Find ATTACHMENT F - FACT SHEET after page 10
        attachment_page, _ = find_attachment_f(pdf_path, start_page=10)

        if attachment_page is None:
            return None

        # Step 2: Extract text from Attachment F onwards
        full_text = ''
        for i in range(attachment_page, len(reader.pages)):
            try:
                page_text = reader.pages[i].extract_text() or ""
                full_text += page_text + ' '
            except Exception as e:
                continue

        # Normalize text
        full_text = normalize_text(full_text)

        if not full_text:
            return None

        # Step 3: Find "Facility Description" occurrences
        first_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=1)
        second_fac_desc = find_nth_occurrence(full_text, "Facility Description", n=2)

        # Use second occurrence if found, otherwise use first
        if second_fac_desc != -1:
            fac_desc_start = second_fac_desc
        elif first_fac_desc != -1:
            first_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=1)
            second_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=2)
            if second_disch_desc != -1:
                fac_desc_start = second_disch_desc
            elif first_disch_desc != -1 and first_fac_desc > first_disch_desc:
                fac_desc_start = first_disch_desc
            else:
                fac_desc_start = first_fac_desc
        else:
            first_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=1)
            second_disch_desc = find_nth_occurrence(full_text, "Discharge Description", n=2)
            if second_disch_desc != -1:
                fac_desc_start = second_disch_desc
            elif first_disch_desc != -1:
                fac_desc_start = first_disch_desc
            else:
                return None

        # Step 4: Find "applicable plans, policies and regulations" after Facility Description
        search_start = fac_desc_start + 800
        applicable_plans_pos = find_nth_occurrence(
            full_text,
            "applicable plans, policies and regulations",
            n=1,
            start_pos=search_start
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
                    full_text, alt, n=1, start_pos=search_start
                )
                if applicable_plans_pos != -1:
                    break

        # Step 5: Find "Planned Changes" between Facility Description and applicable plans
        if applicable_plans_pos != -1:
            planned_changes_pos = find_nth_occurrence(full_text, "Planned Changes", n=1, start_pos=search_start)

            if planned_changes_pos == -1 or planned_changes_pos > applicable_plans_pos:
                planned_changes_pos = find_nth_occurrence(full_text, "Planned upgrades", n=1, start_pos=search_start)

            # Make sure planned changes is before applicable plans
            if planned_changes_pos != -1 and planned_changes_pos > applicable_plans_pos:
                planned_changes_pos = -1
        else:
            # If no applicable plans found, search for planned changes anyway
            planned_changes_pos = find_nth_occurrence(full_text, "Planned Changes", n=1, start_pos=search_start)

            if planned_changes_pos == -1:
                planned_changes_pos = find_nth_occurrence(full_text, "Planned upgrades", n=1, start_pos=search_start)

        # Build outputs depending on what's found
        if planned_changes_pos == -1:
            if applicable_plans_pos != -1:
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
            if applicable_plans_pos == -1:
                applicable_plans_pos = min(planned_changes_pos + 5000, len(full_text))

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
        return None