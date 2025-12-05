# this code will import a newer version of sqlite3 if the version is less than 3.35.X
# which is necessary for chromadb >= v0.4
import sqlite3
import sys

if (sqlite3.sqlite_version_info[0] < 3) or (
    (sqlite3.sqlite_version_info[0] == 3) and (sqlite3.sqlite_version_info[1] < 35)
):
    print("Upgrading sqlite3 version from " + sqlite3.sqlite_version)
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    import sqlite3

    print("New sqlite3 version is " + sqlite3.sqlite_version)
    import chromadb  # noqa: F401

import os
import shutil
import warnings
import re
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.schema.document import Document
from langchain_core._api import LangChainDeprecationWarning
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Path where Chroma will persist its DB
CHROMA_PATH = "chroma"


def clear_database():
    """Remove the Chroma persistence directory if it exists."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


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

        # Step 4: Find "Planned Changes" after Facility Description (with 800 char margin)
        search_start = fac_desc_start + 800
        planned_changes_pos = find_nth_occurrence(full_text, "Planned Changes", n=1, start_pos=search_start)

        if planned_changes_pos == -1:
            planned_changes_pos = find_nth_occurrence(full_text, "Planned upgrades", n=1, start_pos=search_start)

        # Step 5: Find "applicable plans, policies and regulations" after Planned Changes
        applicable_plans_pos = find_nth_occurrence(
            full_text,
            "applicable plans, policies and regulations",
            n=1,
            start_pos=planned_changes_pos if planned_changes_pos != -1 else fac_desc_start
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
                    full_text, alt, n=1, start_pos=max(planned_changes_pos if planned_changes_pos!=-1 else 0, fac_desc_start)
                )
                if applicable_plans_pos != -1:
                    break

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


def split_documents(documents: list[Document]):
    """Split documents into text chunks and return the chunk list.

    Uses `RecursiveCharacterTextSplitter` with the same settings as the
    original single-file implementation.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    """Return the OllamaEmbeddings instance used for embedding text."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def calculate_chunk_ids(chunks):
    """Assign stable IDs to chunks based on source page and chunk index.

    IDs look like "data/monopoly.pdf:6:2" (source:page:chunk_index)
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    """Add new chunks to a Chroma DB located at CHROMA_PATH.

    Only documents with IDs that are not already present will be added.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Compute simple length stats on each document and add to metadata so
    # it is persisted with the vector entry. This helps reporting and
    # debugging (char length, word count).
    for doc in chunks:
        text = getattr(doc, "page_content", "") or ""
        doc.metadata["char_length"] = len(text)
        doc.metadata["word_count"] = len(text.split())

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        # Report length statistics for the batch we are adding
        total_chars = sum(c.metadata.get("char_length", 0) for c in new_chunks)
        total_words = sum(c.metadata.get("word_count", 0) for c in new_chunks)
        print(f"Total characters added: {total_chars}, total words: {total_words}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")
