import os
import re
import csv
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from helpers.utils import extract_leaves, collapse_facility_processes, PRESENT_STATUSES, build_secondary_category_lookup, apply_secondary_category_backfill
from helpers.npdes_text_extraction import extract_permit_sections, WASTEWATER_VOCAB_RE, SEP

DATE_FOLDER = "2026-4-26"

def _extract_one(args):
    directory, pdf_file, j, total = args
    path = os.path.join(directory, pdf_file)
    # print(f"Processing {pdf_file} ({j+1}/{total})")
    return pdf_file, extract_permit_sections(path, regenerate_text_excerpts=True)

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
    upi = csv.writer(csv_file)

    # Opens JSON file with keywords
    with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
        keywords = json.load(f)

    # Directory where PDFs are stored
    directory = f"npdes_permits/output/{DATE_FOLDER}/npdes"

    pdf_cache = {}  # filename -> (present_results, future_results) or None

    site_df = pd.read_csv(rfr_data, dtype=str).fillna("")
    place_ids   = site_df["Place ID"].tolist()
    wdids       = site_df["WDID"].tolist()
    agency_name = site_df["Agency"].tolist()
    facility_name = site_df["Facility Name"].tolist()
    order_no     = site_df["Order_No"].tolist()
    npdes_no    = site_df["NPDES No."].tolist()
    pdfs        = site_df["PDF_File"].tolist()
    shared_pdfs = site_df["Shared_PDF"].tolist()

    # Create CSV headers - one column per process
    headers = [
        "Place ID", "WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File", "Shared_PDF"
    ]
    leaves = [(name, details, group) for name, details, group in extract_leaves(keywords, ignore_disposal=True)]
    all_keys = [name for name, _, _ in leaves]
    group_to_columns = {}
    column_priority = {}
    for name, details, group_id in leaves:
        if group_id:
            group_to_columns.setdefault(group_id, []).append(name)
        column_priority[name] = details.get("priority", 1) if isinstance(details, dict) else 1
    top_category_to_columns, column_secondary_categories, column_global_priority = \
        build_secondary_category_lookup(keywords)

    # Add one column for each treatment process
    headers.extend(name for name, _, _ in leaves)

    upi.writerow(headers)

    # Pre-compute unique PDFs so shared files are only extracted once
    unique_pdfs = list(dict.fromkeys(p for p in pdfs if p))
    total = len(unique_pdfs)

    extractions = {}
    args = [(directory, pdf_file, j, total) for j, pdf_file in enumerate(unique_pdfs)]
    with ProcessPoolExecutor(max_workers=20) as executor:
        for pdf_file, result in executor.map(_extract_one, args):
            extractions[pdf_file] = result

    _page_marker_re = re.compile(r"===PAGE \d+===")
    flag_counts = {"unreadable": 0, "septic": 0}

    def _flag(pdf_file, reason):
        cache = Path(directory) / "text" / f"{Path(pdf_file).stem}.txt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(SEP, encoding="utf-8")
        flag_counts[reason] += 1
        pdf_cache[pdf_file] = None

    for pdf_file, r in extractions.items():
        if r is None:
            pdf_cache[pdf_file] = None
            continue
        txt = r.get("txt_section", "")
        full_text = r.get("full_text", "")
        check_text = txt or full_text
        if not txt and len(_page_marker_re.sub("", full_text).strip()) < 100:
            _flag(pdf_file, "unreadable")
            continue
        if len(WASTEWATER_VOCAB_RE.findall(txt)) <= 1 and "septic" in check_text.lower():
            _flag(pdf_file, "septic")
            continue
        if not txt:
            pdf_cache[pdf_file] = None
            continue

        present_results = {}
        future_results = {}

        for category, processes in keywords.items():
            if not isinstance(processes, dict):
                continue
            proc_dict = {category: processes} if "alt_names" in processes else processes
            txt_sources = [(txt, present_results)]
            if r["txt_changes"]:
                txt_sources.append((r["txt_changes"], future_results))
            for t, results in txt_sources:
                search_processes_in_text(t, proc_dict, results, None)

        pdf_cache[pdf_file] = (present_results, future_results)

    print(f"\nNon-machine-readable PDFs: {flag_counts['unreadable']}")
    print(f"Septic/leachfield PDFs: {flag_counts['septic']}")

    # Write one output row per facility (shared PDFs reuse cached results)
    for i in range(len(pdfs)):
        pdf_file = pdfs[i]
        cached = pdf_cache.get(pdf_file)
        if cached is None:
            continue

        present_results, future_results = cached
        row_meta = [
            place_ids[i],
            wdids[i],
            agency_name[i],
            facility_name[i],
            order_no[i],
            npdes_no[i],
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

        apply_secondary_category_backfill(
            row_status, column_secondary_categories, top_category_to_columns,
            column_global_priority, column_priority,
        )

        data_row = row_meta + [row_status[key] for key in all_keys]

        upi.writerow(data_row)

    csv_file.close()

    kw_by_fac_path = f"npdes_permits/output/{DATE_FOLDER}/kw_unit_processes_by_facility.csv"
    raw_df = pd.read_csv(file, dtype=str).fillna("")
    collapsed = collapse_facility_processes(
        raw_df,
        key_cols=["Place ID"],
        meta_cols=["WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File", "Shared_PDF"]
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

    # Deduplicate by place ID and sum PDFs available on CIWQS.
    site_data = pd.read_csv(rfr_data, dtype=str).fillna("")
    site_data["Total_PDFs_Available"] = pd.to_numeric(
        site_data["Total_PDFs_Available"], errors="coerce"
    )
    reg_measures = (
        site_data.groupby("Reg_Measure_ID")["Total_PDFs_Available"].max().reset_index()
    )
    total_available = int(reg_measures["Total_PDFs_Available"].fillna(0).sum())
    print(f"Total permits: {len(reg_measures)}")
    print(f"Total PDFs available across permits: {total_available}")


if __name__ == "__main__":
    main()
