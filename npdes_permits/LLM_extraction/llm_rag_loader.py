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
import sys
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

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from helpers.npdes_text_extraction import (
    normalize_text, extract_permit_sections
)

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
