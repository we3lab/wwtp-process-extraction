import pandas as pd
import os
import re
import json
import io
import PyPDF2
from pathlib import Path
import unicodedata

# Canonical status vocabulary: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, '' (absent)
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"})

SEP = "\n\n===PLANNED CHANGES===\n\n"

mapping_df = pd.read_csv(
    "wwtp_process_extraction/data/ciwqs_to_cwns.csv", dtype=str, keep_default_na=False
).fillna("")

for c in mapping_df.columns:
    mapping_df[c] = mapping_df[c].str.strip()

mapping_df = mapping_df.sort_values(
    by="NPDES No.", key=lambda s: s.eq(""), ascending=True
).drop_duplicates(subset=["Place ID", "FACILITY_ID"], keep="first")

cwns_mapping = mapping_df[
    mapping_df["CWNS_ID"].ne("") & mapping_df["CWNS_ID"].str.upper().ne("NA")
].copy()

no_cwns_pids: set[str] = set(mapping_df.loc[mapping_df["CWNS_ID"].str.upper().eq("NA"), "Place ID"])

with open("wwtp_process_extraction/data/unitprocess_keywords.json", "r") as f:
    unitprocess_keywords = json.load(f)


def package_sub_readers(reader):
    """For a PDF Package/Portfolio, yield a PdfReader for each embedded PDF sub-file."""
    try:
        root = reader.trailer['/Root'].get_object()
        names_obj = root['/Names'].get_object()
        emb_node = names_obj.get('/EmbeddedFiles')
        if not emb_node:
            return
        emb_names = emb_node.get_object()['/Names']
        for i in range(0, len(emb_names), 2):
            try:
                fspec = emb_names[i + 1].get_object()
                ef = fspec.get('/EF', {}).get_object()
                fstream = ef.get('/F') or ef.get('/UF')
                if fstream:
                    yield PyPDF2.PdfReader(io.BytesIO(fstream.get_object().get_data()))
            except Exception:
                continue
    except Exception:
        return
    

def hasprocess_fragments(graph, cls, watr_ns, sh_ns):
    """Process fragments declared by a class's own SHACL hasProcess shape(s).

    The ontology declares "this equipment implies this process" two equivalent ways:
    `sh:hasValue watr:Process-X` or `sh:qualifiedValueShape [ sh:class watr:Process-X ]`.
    """
    fragments = set()
    for prop in graph.objects(cls, sh_ns.property):
        for path in graph.objects(prop, sh_ns.path):
            if path != watr_ns.hasProcess:
                continue
            for val in graph.objects(prop, sh_ns.hasValue):
                if val.fragment:
                    fragments.add(val.fragment)
            for qualified_shape in graph.objects(prop, sh_ns.qualifiedValueShape):
                for val in graph.objects(qualified_shape, sh_ns["class"]):
                    if val.fragment:
                        fragments.add(val.fragment)
    return fragments


def normalize_text(text):
    """Normalize for matching: NFKC, drop zero-width chars, collapse whitespace, lowercase."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[­​‌‍﻿]", "", text)  # zero-width chars
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def parse_status(val) -> str:
    """Normalize any status cell to a canonical token.

    Handles manual sheet values (messy text), LLM output (clean tokens), and CWNS values.
    Returns: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, or ''.
    """
    if val is None or (isinstance(val, float) and val != val):
        return ""
    s = str(val).strip()
    if not s or s in ("0", "0.0"):
        return ""
    t = s.upper().replace("-", "_")
    if t in ("NAN", "NONE"):
        return ""
    if "PRESENT" in t and "FUTURE" in t:
        return "PRESENT_AND_FUTURE"
    for keyword in ["PRESENT", "FUTURE", "PAST", "OFFSITE"]:
        if keyword in t:
            return keyword
    return ""


def is_present(val) -> bool:
    """True if val indicates the process is currently installed (PRESENT or PRESENT_AND_FUTURE).

    FUTURE is excluded — it means planned but not yet in service.
    """
    return parse_status(val) in PRESENT_STATUSES


def build_cwns_presence_mask(series):
    """Return boolean mask for CWNS presence values (any detectable status, including FUTURE/PAST)."""
    return series.map(parse_status).isin({"PRESENT", "PRESENT_AND_FUTURE", "FUTURE", "PAST"})


def extract_leaves(processes_dict, group_id=None, exclude_keys=()):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if name in exclude_keys:
            continue
        if not isinstance(details, dict):
            continue
        if "alt_names" in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name, exclude_keys=exclude_keys))
    return leaves


def get_leaf_names(cat_name, cat_val, exclude_categories=("Disposal",), exclude_unspecified=False):
    """Return leaf process names for a category from the keywords hierarchy.

    exclude_unspecified drops catch-all 'Unspecified X' leaves (priority 1000) — use for
    leaf-level comparisons where matching a catch-all exactly would be unfair.
    """
    if exclude_categories and cat_name in exclude_categories:
        return []
    if isinstance(cat_val, dict) and "alt_names" in cat_val:
        return [cat_name]
    leaves = extract_leaves(cat_val, exclude_keys=exclude_categories)
    if exclude_unspecified:
        # 'Unspecified X' catch-alls are nested leaves, never top-level categories
        leaves = [(n, d, g) for n, d, g in leaves
                  if not (str(n).lower().startswith("unspecified") and d.get("priority") == 1000)]
    return [name for name, _, _ in leaves]


def build_secondary_category_lookup(keywords_dict):
    """Return (top_category_to_columns, column_secondary_categories, column_global_priority)."""
    top_category_to_columns = {}
    column_secondary_categories = {}
    column_global_priority = {}
    for top_cat, cat_val in keywords_dict.items():
        for name, details, _ in extract_leaves({top_cat: cat_val}):
            top_category_to_columns.setdefault(top_cat, []).append(name)
            if isinstance(details, dict):
                column_global_priority[name] = details.get("global_priority", 1)
                sc = details.get("secondary_category", [])
                if sc and isinstance(sc, list):
                    column_secondary_categories[name] = sc
    return top_category_to_columns, column_secondary_categories, column_global_priority


def apply_secondary_category_backfill(
    status_dict,
    column_secondary_categories,
    top_category_to_columns,
    column_global_priority,
    column_priority,
    ontology_resolve_fn=None,
):
    """Backfill secondary categories: if a PRESENT process requests a secondary category
    that has no PRESENT process, mark the best fallback (unspecified-first) as PRESENT.

    ontology_resolve_fn(source_col, sec_cat, sec_cols) -> str | None: optional hook for
    ontology-based selection (used by step4). Returns the chosen column name, or None to
    fall back to unspecified-first heuristic.
    """
    present_cols = [c for c, v in status_dict.items() if v in PRESENT_STATUSES]
    for source_col in present_cols:
        for sec_cat in column_secondary_categories.get(source_col, []):
            sec_cols = top_category_to_columns.get(sec_cat, [])
            if not sec_cols or any(status_dict.get(c) in PRESENT_STATUSES for c in sec_cols):
                continue
            available = [c for c in sec_cols if c in status_dict]
            if not available:
                continue
            chosen = ontology_resolve_fn(source_col, sec_cat, sec_cols) if ontology_resolve_fn else None
            if chosen is None:
                # Prefer the category-level catch-all (e.g., "Unspecified Filtration")
                # before nested catch-alls (e.g., "Unspecified FFR").
                target_unspecified = f"Unspecified {sec_cat}".strip().lower()
                exact_unspecified = [
                    c for c in available
                    if str(c).strip().lower() == target_unspecified
                ]
                unspecified = [c for c in available if "Unspecified" in c]
                pool = exact_unspecified or unspecified or available
                chosen = min(pool, key=lambda c: (column_priority.get(c, 1), column_global_priority.get(c, 1), c))
            status_dict[chosen] = "PRESENT"


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes."""
    el_abbadi_dir = os.path.join(os.path.dirname(__file__), "data", "el_abbadi", "input")
    werf_codes_df = pd.read_csv(
        os.path.join(el_abbadi_dir, "UNIT_PROCESS_EI_CODES_WERF_modified.csv"), dtype=str
    )
    matching = werf_codes_df[werf_codes_df["FINAL_UNIT_PROCESS_NAME"] == cwns_process_name]
    return matching["WERF_CODE"].unique().tolist() if not matching.empty else []


def merge_column_statuses(column) -> str:
    """Highest-priority status across all values in column."""
    tokens = {parse_status(v) for v in column}
    if "PRESENT_AND_FUTURE" in tokens or ("PRESENT" in tokens and "FUTURE" in tokens):
        return "PRESENT_AND_FUTURE"
    for token in ("PRESENT", "FUTURE", "PAST", "OFFSITE"):
        if token in tokens:
            return token
    return ""


def collapse_facility_processes(
    df: pd.DataFrame, key_cols: list[str], meta_cols: list[str]
) -> pd.DataFrame:
    """One row per unique key_cols group; highest-priority status per process column.

    Process columns (everything not in key_cols or meta_cols) are merged via
    merge_column_statuses. Meta columns take the first non-empty value. Column order preserved.
    """
    all_fixed = set(key_cols) | set(meta_cols)
    proc_cols = [c for c in df.columns if c not in all_fixed]
    rows = []
    for _, grp in df.groupby(key_cols, dropna=False, sort=False):
        out = {
            col: next((v for v in grp[col] if pd.notna(v) and str(v).strip()), "")
            for col in (key_cols + meta_cols)
            if col in df.columns
        }
        for col in proc_cols:
            out[col] = merge_column_statuses(grp[col])
        rows.append(out)
    return pd.DataFrame(rows).reindex(columns=list(df.columns))


def build_cwns_facility_processes(ca_cwns_df, target_facilities=None):
    proc_cols = list({name for name, _, _ in extract_leaves(unitprocess_keywords)})
    left = cwns_mapping[["Place ID", "WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]]
    if target_facilities is not None:
        left = left[left["Place ID"].isin(target_facilities)]
    right = collapse_facility_processes(ca_cwns_df[["CWNS_ID"] + proc_cols], ["CWNS_ID"], [])
    merged = left.merge(right, on="CWNS_ID", how="inner", indicator="_cwns_merge")
    cwns_by_facility = collapse_facility_processes(
        merged, ["Place ID"], ["WDID", "Facility Name", "CWNS_ID", "FACILITY_ID", "_cwns_merge"]
    ).drop(columns=["CWNS_ID", "FACILITY_ID", "_cwns_merge"], errors="ignore").fillna("")
    return cwns_by_facility, merged

def build_txt_jobs(txt_folder: str, facilities_information: str):
    txt_folder_path = Path(txt_folder)
    facilities_path = Path(facilities_information)
    facilities_df = pd.read_csv(facilities_path, dtype=str).fillna("")
    required_columns = {"Facility Name", "PDF_File"}
    missing_columns = required_columns.difference(set(facilities_df.columns))
    if missing_columns:
        raise ValueError(
            "--facilities_information is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    jobs = []
    for row_idx, row in facilities_df.iterrows():
        facility_name = str(row["Facility Name"]).strip()
        pdf_file_value = str(row["PDF_File"]).strip()

        if not facility_name or facility_name.lower() == "nan":
            continue
        if not pdf_file_value or pdf_file_value.lower() == "nan":
            continue

        txt_path = Path(pdf_file_value)
        if not txt_path.is_absolute():
            path_value = Path(pdf_file_value)
            path_value = txt_folder_path / path_value
            txt_name = path_value.with_suffix(".txt").name
            txt_path = txt_folder_path / txt_name

        if not txt_path.exists() or not txt_path.is_file():
            print(f"No txt for '{facility_name}': {txt_path.name}, skipping.")
            continue

        jobs.append((row_idx, txt_path, txt_path.name, facility_name))

    return jobs
