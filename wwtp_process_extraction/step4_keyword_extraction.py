import csv
import json
import re
from functools import lru_cache
import pandas as pd
from helpers.utils import (
    SEP,
    normalize_text,
    build_txt_jobs,
    extract_leaves,
    collapse_facility_processes,
    PRESENT_STATUSES,
    build_secondary_category_lookup,
    apply_secondary_category_backfill,
    add_county_and_sort,
    current_permit_mask,
)


@lru_cache(maxsize=None)
def _cs_pattern(term):
    # Bare acronyms match case-sensitively on token boundaries (optional plural s), so TF
    # doesn't hit WWTF, AD doesn't hit SCADA, and CAS doesn't hit permit number CAS000001.
    # Anything else (prefix stems like "Aerobic Digest", mixed case like FeCl3) stays a
    # plain substring so it still matches "Aerobic Digesters".
    core = term.strip()
    if core.isalnum() and core.isupper():
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(core) + r"s?(?![A-Za-z0-9])")
    return re.compile(re.escape(core))


def search_processes_in_text(text, processes_dict, results, parent_name=None, text_cs=None):
    # text is lowercased for alt_names; text_cs keeps original case for the acronym list.
    # Defaults to text so callers passing raw (unnormalized) text still work.
    if text_cs is None:
        text_cs = text
    text_lower = text.lower()
    sub_category_found = False
    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            if "alt_names" in details:
                if process_name not in results:
                    results[process_name] = 0
                alt_names = details.get("alt_names", []) or []
                cs_list = details.get("alt_names_case_sensitive", []) or []
                found = (any(a.lower() in text_lower for a in alt_names)
                         or any(_cs_pattern(c).search(text_cs) for c in cs_list))
                if found:
                    results[process_name] = 1
                    sub_category_found = True
            else:
                sub_found = search_processes_in_text(text, details, results, process_name, text_cs)
                if sub_found:
                    sub_category_found = True
    if parent_name and sub_category_found:
        results[parent_name] = 1
    return sub_category_found


def main():
    rfr_data = f"wwtp_process_extraction/output/site_data_relevant.csv"
    txt_folder = f"wwtp_process_extraction/output/permits/text"
    out_file = f"wwtp_process_extraction/output/unit_processes_by_pdf_kw.csv"

    with open("wwtp_process_extraction/data/unitprocess_keywords.json", "r") as f:
        keywords = json.load(f)

    site_df = pd.read_csv(rfr_data, dtype=str).fillna("")

    leaves = list(extract_leaves(keywords))
    all_keys = [name for name, _, _ in leaves]
    group_to_columns = {}
    column_priority = {}
    for name, details, group_id in leaves:
        if group_id:
            group_to_columns.setdefault(group_id, []).append(name)
        column_priority[name] = details.get("priority", 1) if isinstance(details, dict) else 1
    top_category_to_columns, column_secondary_categories, column_global_priority = \
        build_secondary_category_lookup(keywords)

    jobs = build_txt_jobs(txt_folder, rfr_data)

    # Extract keyword results per unique txt file
    txt_cache = {}  # txt_stem -> (present_results, future_results)
    seen_stems = set()
    for row_idx, txt_path, *_ in jobs:
        stem = txt_path.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        content = txt_path.read_text(encoding="utf-8")
        parts = content.split(SEP, 1)
        txt_section = normalize_text(parts[0]) if parts else ""
        txt_changes = normalize_text(parts[1]) if len(parts) > 1 else ""
        txt_section_cs = normalize_text(parts[0], lower=False) if parts else ""
        txt_changes_cs = normalize_text(parts[1], lower=False) if len(parts) > 1 else ""
        if not txt_section.strip():
            txt_cache[stem] = None
            continue
        present_results, future_results = {}, {}
        for category, processes in keywords.items():
            if not isinstance(processes, dict):
                continue
            proc_dict = {category: processes} if "alt_names" in processes else processes
            search_processes_in_text(txt_section, proc_dict, present_results, None, txt_section_cs)
            if txt_changes:
                search_processes_in_text(txt_changes, proc_dict, future_results, None, txt_changes_cs)
        txt_cache[stem] = (present_results, future_results)

    headers = ["Place ID", "WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File", "Shared_PDF",
               "document_order_no"]
    headers.extend(all_keys)

    with open(out_file, "w", newline="") as csv_file:
        upi = csv.writer(csv_file)
        upi.writerow(headers)

        for row_idx, txt_path, *_ in jobs:
            stem = txt_path.stem
            cached = txt_cache.get(stem)
            if cached is None:
                continue
            present_results, future_results = cached
            row = site_df.iloc[row_idx]
            row_meta = [
                row.get("Place ID", ""),
                row.get("WDID", ""),
                row.get("Agency", ""),
                row.get("Facility Name", ""),
                row.get("Order_No", ""),
                row.get("NPDES No.", ""),
                row.get("PDF_File", ""),
                row.get("Shared_PDF", ""),
                row.get("document_order_no", ""),
            ]
            row_status = {}
            for key in all_keys:
                is_present = present_results.get(key, 0) == 1
                is_future = future_results.get(key, 0) == 1
                if is_present and is_future:
                    row_status[key] = "PRESENT_AND_FUTURE"
                elif is_present:
                    row_status[key] = "PRESENT"
                elif is_future:
                    row_status[key] = "FUTURE"
                else:
                    row_status[key] = "0"

            for sibling_cols in group_to_columns.values():
                present_cols = [c for c in sibling_cols if row_status.get(c, "0") in PRESENT_STATUSES]
                if len(present_cols) <= 1:
                    continue
                best_priority = min(column_priority.get(c, 1) for c in present_cols)
                for col in present_cols:
                    if column_priority.get(col, 1) > best_priority:
                        row_status[col] = "0"

            apply_secondary_category_backfill(
                row_status, column_secondary_categories, top_category_to_columns,
                column_global_priority, column_priority,
            )
            upi.writerow(row_meta + [row_status[key] for key in all_keys])

    kw_by_fac_path = f"wwtp_process_extraction/output/unit_processes_by_facility_kw.csv"
    raw_df = pd.read_csv(out_file, dtype=str).fillna("")
    # Same current-permit restriction step6 applies, so the keyword and LLM facility tables
    # are built from the same documents.
    meta = {"Place ID", "WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File",
            "Shared_PDF", "document_order_no", "County"}
    proc_cols = [c for c in raw_df.columns if c not in meta]
    has_content = raw_df[proc_cols].isin(
        {"PRESENT", "PRESENT_AND_FUTURE", "FUTURE", "PAST", "OFFSITE"}).any(axis=1)
    current = current_permit_mask(raw_df, content=has_content)
    print(f"Current-permit documents: {int(current.sum())} of {len(raw_df)} "
          f"({int((~current).sum())} superseded rows excluded from the facility collapse)")
    collapsed = collapse_facility_processes(
        raw_df[current],
        key_cols=["Place ID"],
        meta_cols=["WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File", "Shared_PDF",
                   "document_order_no"],
    )
    collapsed = add_county_and_sort(collapsed, "Facility Name", place_id_col="Place ID", wdid_col="WDID")
    collapsed.to_csv(kw_by_fac_path, index=False)
    add_county_and_sort(raw_df, "Facility Name", place_id_col="Place ID", wdid_col="WDID").to_csv(out_file, index=False)
    print(f"Collapsed {int(current.sum())} PDF rows → {len(collapsed)} facilities → unit_processes_by_facility_kw.csv")


if __name__ == "__main__":
    main()
