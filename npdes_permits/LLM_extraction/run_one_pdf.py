#!/usr/bin/env python3
"""Simple runner to index one PDF (no-split), run the query, and save JSON output.

Usage examples (from repo root):
  python3 npdes_permits/LLM_extraction/run_one_pdf.py --pdf npdes_permits/7-18-22_MorroBay_finalorder.pdf

This script performs:
 1. Reset the Chroma DB
 2. Initialize the DB with the provided PDF (no chunking)
 3. Run the fixed query
 4. Save the returned WWTPAnalysis JSON under `output/llm_results/` named by the pdf stem
"""

import argparse
import os
from pathlib import Path
import json
import pandas as pd

import llm_test2
import llm_rag_loader as loader

QUESTION = "What treatments are implemented (or planned) in this facility based on the facility permit given?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="PDF path (absolute or relative to repo root)")
    parser.add_argument("--outdir", default="output/llm_results2", help="Output folder for JSON results")
    args = parser.parse_args()

    pdf_path = args.pdf
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Clearing database...")
    loader.clear_database()

    print(f"Loading and indexing PDF (no-split): {pdf_path}")
    documents = loader.load_documents(pdf_path)
    if not documents:
        print("No document produced by loader, aborting.")
        return

    # Add to chroma and mark as no_split
    loader.add_to_chroma(documents)
    try:
        os.makedirs(loader.CHROMA_PATH, exist_ok=True)
        open(os.path.join(loader.CHROMA_PATH, "no_split"), "w").close()
    except Exception:
        pass

    print("Running query...")
    analysis, sources = llm_test2.query_rag(QUESTION, k=5, verbose=True)

    if analysis is None:
        print("No analysis returned from query.")
        return

    # Save JSON
    pdf_stem = Path(pdf_path).stem
    out_file = outdir / (pdf_stem + ".json")
    try:
        # prefer model_dump_json if available
        if hasattr(analysis, "model_dump_json"):
            json_text = analysis.model_dump_json(indent=2)
            out_file.write_text(json_text)
        else:
            data = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.dict()
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
        print(f"Saved analysis to {out_file}")
    except Exception as e:
        print("Failed to save analysis:", e)
    """
    # Convert matched processes to a DataFrame and save as CSV
    try:
        rows = []
        for p in getattr(analysis, "processes", []):
            rows.append(
                {
                    "process": getattr(p, "process_name", None),
                    "category": getattr(p, "category", None),
                    "subcategory": getattr(p, "subcategory", None),
                    "confidence": float(getattr(p, "confidence", None)) if getattr(p, "confidence", None) is not None else None,
                    "match_sentence": getattr(p, "match_sentence", None),
                    "alt_name_used": getattr(p, "alternative_name_used", None),
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            csv_file = outdir / (pdf_stem + ".csv")
            df.to_csv(csv_file, index=False)
            print(f"Saved CSV to {csv_file}")
        else:
            print("No matched processes found; skipping CSV creation.")

    except Exception as e:
        print("Failed to create/save CSV using pandas:", e)
        # Fallback: write a simple CSV using the csv module
        try:
            import csv

            csv_file = outdir / (pdf_stem + ".csv")
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["process", "category", "subcategory", "confidence", "match_sentence", "alt_name_used"])
                for p in getattr(analysis, "processes", []):
                    writer.writerow([
                        getattr(p, "process_name", ""),
                        getattr(p, "category", ""),
                        getattr(p, "subcategory", "") or "",
                        getattr(p, "confidence", ""),
                        (getattr(p, "match_sentence", "") or "").replace("\n", " ").strip(),
                        getattr(p, "alternative_name_used", "") or "",
                    ])
            print(f"Saved fallback CSV to {csv_file}")
        except Exception as e2:
            print("Failed fallback CSV write:", e2)
    """

if __name__ == "__main__":
    main()
