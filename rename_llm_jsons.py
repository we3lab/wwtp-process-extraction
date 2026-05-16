"""Rename LLM extraction JSONs from {pdf_stem}_{facility_slug}.json
to {pdf_stem}_{place_id}.json using site_data.csv for the mapping.

Run from project root:
    conda run -n npdes-permits python3 rename_llm_jsons.py
"""
import re
import os
from pathlib import Path

import pandas as pd

DATE_FOLDER = "2026-5-15"
LLM_DIR = Path(f"npdes_permits/output/{DATE_FOLDER}/llm_extraction")
SITE_CSV = Path(f"npdes_permits/output/{DATE_FOLDER}/site_data.csv")


def _norm_pdf(s):
    return s.lower().replace(" ", "_")


def _norm_fac(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


site_df = pd.read_csv(SITE_CSV, dtype=str).fillna("")

# Build normalized-pdf-stem → list of {Place ID, Facility Name, pdf_stem_orig}
pdf_map = {}
for _, row in site_df.iterrows():
    orig_stem = Path(row["PDF_File"]).stem  # preserves original casing
    key = _norm_pdf(orig_stem)
    pdf_map.setdefault(key, []).append({
        "Place ID": row["Place ID"].strip(),
        "Facility Name": row["Facility Name"].strip(),
        "pdf_stem_orig": orig_stem,
    })

renamed = 0
skipped = 0
unmatched = []

for json_file in sorted(LLM_DIR.glob("*.json")):
    norm_stem = _norm_pdf(json_file.stem)

    # Find matching pdf_stem (longest match first to avoid prefix collisions)
    identity = None
    for pdf_stem in sorted(pdf_map, key=len, reverse=True):
        if norm_stem == pdf_stem or norm_stem.startswith(pdf_stem + "_"):
            rows = pdf_map[pdf_stem]
            if len(rows) == 1:
                identity = rows[0]
            else:
                # Multi-facility PDF: pick closest facility name match
                suffix = norm_stem[len(pdf_stem) + 1:]
                suffix_stripped = _norm_fac(suffix)
                identity = max(rows, key=lambda r: len(
                    os.path.commonprefix([suffix_stripped, _norm_fac(r["Facility Name"])])
                ))
            break

    if not identity or not identity["Place ID"]:
        unmatched.append(json_file.name)
        skipped += 1
        continue

    new_name = f"{identity['pdf_stem_orig']}_{identity['Place ID']}.json"
    new_path = json_file.parent / new_name

    if json_file.name == new_name:
        skipped += 1
        continue

    if new_path.exists():
        print(f"  COLLISION: {json_file.name} → {new_name} (target exists, skipping)")
        skipped += 1
        continue

    print(f"  {json_file.name}\n    → {new_name}")
    json_file.rename(new_path)
    renamed += 1

print(f"\nRenamed: {renamed}  Skipped: {skipped}  Unmatched: {len(unmatched)}")
if unmatched:
    print("Unmatched (no Place ID found):")
    for f in unmatched:
        print(f"  {f}")
