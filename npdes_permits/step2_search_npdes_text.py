import os
import csv
import json
import pandas as pd
from pathlib import Path
from helpers.utils import extract_leaves, collapse_facility_processes, PRESENT_STATUSES
from helpers.npdes_text_extraction import extract_permit_sections

DATE_FOLDER = "2026-4-26"

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
            if "alt_names" in details:
                if process_name not in results:
                    results[process_name] = 0

                alt_names = details.get("alt_names", []) or []
                cs_list = details.get("alt_names_case_sensitive", []) or []

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
    file = f"npdes_permits/output/{DATE_FOLDER}/kw_unit_processes_by_pdf.csv"
    rfr_data = f"npdes_permits/output/{DATE_FOLDER}/site_data.csv"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    csv_file = open(file, "w", newline="")
    ps_file = open(rfr_data, "r", newline="")
    upi = csv.writer(csv_file)

    # Opens JSON file with keywords
    with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
        keywords = json.load(f)

    # Directory where PDFs are stored
    directory = f"npdes_permits/output/{DATE_FOLDER}/npdes"

    # Lists to store headers & data
    wdids = []
    agency_name = []
    facility_name = []
    npdesNO = []
    pdfs = []
    shared_pdfs = []

    # Read site data
    with open(rfr_data, "r") as f:
        for row in csv.DictReader(f):
            wdids.append(row.get("WDID", ""))
            agency_name.append(row.get("Agency", ""))
            facility_name.append(row.get("Facility_Name", ""))
            npdesNO.append(row.get("NPDES_No", ""))
            pdfs.append(row.get("PDF_File", ""))
            shared_pdfs.append(row.get("Shared_PDF", ""))

    # Create CSV headers - one column per process
    headers = [
        "WDID",
        "AGENCY_NAME",
        "FACILITY_NAME",
        "PERMIT_NUMBER",
        "PDF_File",
        "Shared_PDF",
    ]
    leaves = extract_leaves(keywords, ignore_disposal=False)
    all_keys = [name for name, _, _ in leaves]
    group_to_columns = {}
    column_priority = {}
    for name, details, group_id in leaves:
        if group_id:
            group_to_columns.setdefault(group_id, []).append(name)
        column_priority[name] = details.get("priority", 1) if isinstance(details, dict) else 1

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
            if "alt_names" in processes:
                # Top-level leaf node — wrap so search_processes_in_text sees expected structure
                wrapped = {category: processes}
                search_processes_in_text(
                    extraction_result["txt_section"], wrapped, present_results, None
                )
                if extraction_result["txt_changes"]:
                    search_processes_in_text(
                        extraction_result["txt_changes"], wrapped, future_results, None
                    )
            else:
                # Category node containing sub-processes
                search_processes_in_text(
                    extraction_result["txt_section"], processes, present_results, None
                )
                if extraction_result["txt_changes"]:
                    search_processes_in_text(
                        extraction_result["txt_changes"],
                        processes,
                        future_results,
                        None,
                    )

        pdf_cache[pdf_file] = (present_results, future_results)

    # Write one output row per facility (shared PDFs reuse cached results)
    for i in range(len(pdfs)):
        pdf_file = pdfs[i]
        cached = pdf_cache.get(pdf_file)
        if cached is None:
            continue

        present_results, future_results = cached
        row_meta = [
            wdids[i],
            agency_name[i],
            facility_name[i],
            npdesNO[i],
            pdf_file,
            shared_pdfs[i],
        ]

        row_status = {}
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

            row_status[key] = status

        # Mirror step4 sibling-priority behavior: only one PRESENT-like sibling per group.
        for sibling_cols in group_to_columns.values():
            present_cols = [c for c in sibling_cols if row_status.get(c, "0") in PRESENT_STATUSES]
            if len(present_cols) <= 1:
                continue
            winner = min(present_cols, key=lambda c: (column_priority.get(c, 1), c))
            for col in present_cols:
                if col != winner:
                    row_status[col] = "0"

        data_row = row_meta + [row_status[key] for key in all_keys]

        upi.writerow(data_row)

    csv_file.close()
    ps_file.close()

    kw_by_fac_path = f"npdes_permits/output/{DATE_FOLDER}/kw_unit_processes_by_facility.csv"
    raw_df = pd.read_csv(file, dtype=str).fillna("")
    collapsed = collapse_facility_processes(
        raw_df,
        key_cols=["PERMIT_NUMBER", "FACILITY_NAME"],
        meta_cols=["WDID", "AGENCY_NAME", "PDF_File", "Shared_PDF"],
    )
    collapsed.to_csv(kw_by_fac_path, index=False)
    print(
        f"Collapsed {len(raw_df)} PDF rows → {len(collapsed)} facilities → kw_unit_processes_by_facility.csv"
    )

    # After Step 2: print top text cache files generated across the workspace    
    workspace_root = Path(__file__).resolve().parents[2]
    txt_files = list(workspace_root.glob("**/text/*.txt"))
    if txt_files:
        sizes = []
        for f in txt_files:
            try:
                sizes.append((f, f.stat().st_size))
            except Exception:
                continue
        sizes.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop text cache files under {workspace_root}:")
        for f, size in sizes[:10]:
            try:
                rel = f.relative_to(workspace_root)
            except Exception:
                rel = f
            print(f"  {rel}: {size} bytes")

    print(f"\n{'='*80}")
    print(f"Processing complete! Results saved to: {file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
