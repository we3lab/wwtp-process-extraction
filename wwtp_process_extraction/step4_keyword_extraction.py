import csv
import re
from functools import lru_cache
import pandas as pd
from helpers.utils import (
    SEP,
    OUTPUT_DIR,
    SITE_DATA_RELEVANT_CSV,
    STATUS_TOKENS,
    unitprocess_keywords,
    normalize_text,
    build_txt_jobs,
    extract_leaves,
    collapse_facility_processes,
    build_secondary_category_lookup,
    apply_secondary_category_backfill,
    keep_best_priority,
    add_county_and_sort,
    current_permit_mask,
)

PDF_KW_CSV = OUTPUT_DIR / "unit_processes_by_pdf_kw.csv"
FACILITY_KW_CSV = OUTPUT_DIR / "unit_processes_by_facility_kw.csv"
METADATA_COLUMNS = ["Place ID", "WDID", "Agency", "Facility Name", "Order_No", "NPDES No.", "PDF_File", "Shared_PDF",
                    "document_order_no"]


@lru_cache(maxsize=None)
def case_sensitive_pattern(term):
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
        if "alt_names" in details:
            if process_name not in results:
                results[process_name] = 0
            case_sensitive_names = details.get("alt_names_case_sensitive", [])
            found = (any(a.lower() in text_lower for a in details["alt_names"])
                     or any(case_sensitive_pattern(c).search(text_cs) for c in case_sensitive_names))
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
    site_df = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")

    leaves = extract_leaves(unitprocess_keywords)
    all_keys = [name for name, _, _ in leaves]
    group_to_columns = {}
    column_priority = {}
    for name, details, group_id in leaves:
        if group_id:
            group_to_columns.setdefault(group_id, []).append(name)
        column_priority[name] = details.get("priority", 1)
    top_category_to_columns, column_secondary_categories, column_global_priority = \
        build_secondary_category_lookup(unitprocess_keywords)

    jobs = build_txt_jobs(SITE_DATA_RELEVANT_CSV)

    # Extract keyword results per unique txt file
    txt_cache = {}  # txt_stem -> (present_results, future_results)
    for _, txt_path, *_ in jobs:
        stem = txt_path.stem
        if stem in txt_cache:
            continue
        content = txt_path.read_text(encoding="utf-8")
        parts = content.split(SEP, 1)
        txt_section = normalize_text(parts[0])
        txt_changes = normalize_text(parts[1]) if len(parts) > 1 else ""
        txt_section_case_sensitive = normalize_text(parts[0], lower=False)
        txt_changes_case_sensitive = normalize_text(parts[1], lower=False) if len(parts) > 1 else ""
        if not txt_section:
            txt_cache[stem] = None
            continue
        present_results, future_results = {}, {}
        search_processes_in_text(txt_section, unitprocess_keywords, present_results, None, txt_section_case_sensitive)
        if txt_changes:
            search_processes_in_text(txt_changes, unitprocess_keywords, future_results, None, txt_changes_case_sensitive)
        txt_cache[stem] = (present_results, future_results)

    with open(PDF_KW_CSV, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(METADATA_COLUMNS + all_keys)

        for row_idx, txt_path, *_ in jobs:
            stem = txt_path.stem
            cached = txt_cache.get(stem)
            if cached is None:
                continue
            present_results, future_results = cached
            row_meta = site_df.iloc[row_idx][METADATA_COLUMNS].tolist()
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
                keep_best_priority(row_status, sibling_cols, column_priority, cleared="0")

            apply_secondary_category_backfill(
                row_status, column_secondary_categories, top_category_to_columns,
                column_global_priority, column_priority,
            )
            writer.writerow(row_meta + [row_status[key] for key in all_keys])

    raw_df = pd.read_csv(PDF_KW_CSV, dtype=str).fillna("")
    # Same current-permit restriction step6 applies, so the keyword and LLM facility tables
    # are built from the same documents.
    has_content = raw_df[all_keys].isin(STATUS_TOKENS).any(axis=1)
    current = current_permit_mask(raw_df, content=has_content)
    print(f"Current-permit documents: {int(current.sum())} of {len(raw_df)} "
          f"({int((~current).sum())} superseded rows excluded from the facility collapse)")
    collapsed = collapse_facility_processes(
        raw_df[current],
        key_cols=["Place ID"],
        meta_cols=[c for c in METADATA_COLUMNS if c != "Place ID"],
    )
    collapsed = add_county_and_sort(collapsed, "Facility Name", place_id_col="Place ID", wdid_col="WDID")
    collapsed.to_csv(FACILITY_KW_CSV, index=False)
    add_county_and_sort(raw_df, "Facility Name", place_id_col="Place ID", wdid_col="WDID").to_csv(PDF_KW_CSV, index=False)
    print(f"Collapsed {int(current.sum())} PDF rows → {len(collapsed)} facilities → unit_processes_by_facility_kw.csv")


if __name__ == "__main__":
    main()
