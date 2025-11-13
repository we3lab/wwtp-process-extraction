#!/usr/bin/env python3
"""Batch index and query PDFs one-by-one using the existing RAG code.

For each PDF found under the `npdes_permits` folder this script will:
 1. Reset the Chroma DB (clear previous index).
 2. Extract the relevant text from the PDF (Attachment F section if present).
 3. Initialize the DB with the single document (no chunking).
 4. Run the RAG query asking about implemented processes.
 5. Save the parsed results per-PDF in CSV and JSON formats under `output/query_results/`.

Run from the repository root:
  python3 npdes_permits/LLM_extraction/batch_query.py

Adjust `ROOT_PDF_DIR` if you want to target a different directory.
"""

from pathlib import Path
import os
import json
from typing import List

import pandas as pd

import llm_rag_loader as loader
import llm_rag
from langchain.schema.document import Document

# Where to search for PDFs
ROOT_PDF_DIR = Path("npdes_permits")

# Output directory for per-pdf results
OUT_DIR = Path("output/query_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTION = "What processes are implemented in the wastewater treatment facility?"


def index_single_pdf_and_query(pdf_path: Path, question: str = QUESTION):
    print(f"\n=== Processing {pdf_path} ===")

    # 1) Reset DB
    loader.clear_database()

    # 2) Extract text (prefer focused Attachment F extraction)
    try:
        res = loader.extract_permit_sections(str(pdf_path))
    except Exception as e:
        print(f"Error extracting sections from {pdf_path}: {e}")
        res = None

    if res:
        text = (res.get("txt_section", "") + "\n\n" + res.get("txt_changes", "")).strip()
        page_meta = res.get("metadata", {}).get("attachment_f_page", 0)
    else:
        # Fallback: load whole PDF text
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        text = ""
        for p in reader.pages:
            try:
                text += (p.extract_text() or "") + "\n"
            except Exception:
                continue
        page_meta = 0

    if not text.strip():
        print(f"No text extracted from {pdf_path}, skipping.")
        return None

    # 3) Initialize DB with single document (no-split)
    doc = Document(page_content=text, metadata={"source": str(pdf_path), "page": page_meta})
    loader.add_to_chroma([doc])
    # create no_split flag so query uses whole-document context
    try:
        os.makedirs(loader.CHROMA_PATH, exist_ok=True)
        open(os.path.join(loader.CHROMA_PATH, "no_split"), "w").close()
    except Exception:
        pass

    # 4) Query
    try:
        analysis, sources = llm_rag.query_rag(question, k=5, verbose=False)
    except Exception as e:
        print(f"Error running query for {pdf_path}: {e}")
        return None

    if analysis is None:
        print(f"No analysis returned for {pdf_path}")
        return None

    # 5) Save results per-pdf
    rows: List[dict] = []
    facility = getattr(analysis, "facility_name", None)
    design_capacity = getattr(analysis, "design_capacity", None)

    for proc in getattr(analysis, "processes", []):
        rows.append(
            {
                "pdf": pdf_path.name,
                "pdf_path": str(pdf_path),
                "facility_name": facility,
                "design_capacity": design_capacity,
                "process_name": getattr(proc, "process_name", None),
                "category": getattr(proc, "category", None),
                "subcategory": getattr(proc, "subcategory", None),
                "confidence": getattr(proc, "confidence", None),
                "alternative_name_matched": getattr(proc, "alternative_name_matched", None),
                "sources": ",".join(s for s in (sources or []) if s),
            }
        )

    df = pd.DataFrame(rows)

    out_csv = OUT_DIR / (pdf_path.stem + ".csv")
    out_json = OUT_DIR / (pdf_path.stem + ".json")

    df.to_csv(out_csv, index=False)
    # Save full analysis as JSON
    try:
        with open(out_json, "w") as f:
            json.dump(analysis.model_dump(), f, indent=2, default=str)
    except Exception:
        # fallback: try to write as string
        with open(out_json, "w") as f:
            f.write(str(analysis))

    print(f"Saved CSV: {out_csv}  JSON: {out_json}")
    return df


def main():
    pdfs = list(ROOT_PDF_DIR.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {ROOT_PDF_DIR}")
        return

    summary_rows = []
    for pdf in pdfs:
        df = index_single_pdf_and_query(pdf)
        if df is None:
            continue
        # summary: count processes
        summary_rows.append({"pdf": pdf.name, "n_processes": len(df), "csv": str(OUT_DIR / (pdf.stem + ".csv"))})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "summary.csv", index=False)
        print("Wrote summary at", OUT_DIR / "summary.csv")


if __name__ == "__main__":
    main()
