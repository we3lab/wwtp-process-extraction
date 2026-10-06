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
LEAVES = extract_leaves(unitprocess_keywords)
# (found in the description, found in planned changes) -> status
STATUS_BY_FLAGS = {(True, True): "PRESENT_AND_FUTURE", (True, False): "PRESENT", (False, True): "FUTURE", (False, False): "0"}


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


def search_processes_in_text(text):
    """Names of the unit processes mentioned in text.

    alt_names match case-insensitively as substrings; alt_names_case_sensitive (acronyms) match
    the original case via case_sensitive_pattern.
    """
    text_lower = text.lower()
    return {
        name for name, details, _ in LEAVES
        if any(alt.lower() in text_lower for alt in details["alt_names"])
        or any(case_sensitive_pattern(term).search(text) for term in details.get("alt_names_case_sensitive", []))
    }


def main():
    site_df = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")

    all_keys = [name for name, _, _ in LEAVES]
    group_to_columns = {}
    column_priority = {}
    for name, details, group_id in LEAVES:
        if group_id:
            group_to_columns.setdefault(group_id, []).append(name)
        column_priority[name] = details.get("priority", 1)
    top_category_to_columns, column_secondary_categories, column_global_priority = \
        build_secondary_category_lookup(unitprocess_keywords)

    jobs = build_txt_jobs(SITE_DATA_RELEVANT_CSV)

    # Keyword matches per unique txt file: (description matches, planned-changes matches)
    txt_cache = {}
    for _, txt_path, *_ in jobs:
        if txt_path.stem in txt_cache:
            continue
        section, _, changes = txt_path.read_text(encoding="utf-8").partition(SEP)
        section, changes = normalize_text(section, lower=False), normalize_text(changes, lower=False)
        txt_cache[txt_path.stem] = (
            (search_processes_in_text(section), search_processes_in_text(changes) if changes else set())
            if section else None
        )

    rows = []
    for row_idx, txt_path, *_ in jobs:
        if txt_cache[txt_path.stem] is None:
            continue
        present, future = txt_cache[txt_path.stem]
        row_status = {key: STATUS_BY_FLAGS[(key in present, key in future)] for key in all_keys}
        for sibling_cols in group_to_columns.values():
            keep_best_priority(row_status, sibling_cols, column_priority, cleared="0")
        apply_secondary_category_backfill(
            row_status, column_secondary_categories, top_category_to_columns,
            column_global_priority, column_priority,
        )
        rows.append(site_df.iloc[row_idx][METADATA_COLUMNS].tolist() + [row_status[key] for key in all_keys])

    raw_df = pd.DataFrame(rows, columns=METADATA_COLUMNS + all_keys)
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
