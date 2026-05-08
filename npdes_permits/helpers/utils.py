import pandas as pd
import os
import json
# Canonical status vocabulary: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, '' (absent)
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"})

mapping_df = pd.read_csv(
    "npdes_permits/data/ciwqs_to_cwns.csv", dtype=str, keep_default_na=False
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

with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
    unitprocess_keywords = json.load(f)

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


def extract_leaves(processes_dict, group_id=None, ignore_disposal=True):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if not isinstance(details, dict):
            continue
        if ignore_disposal and name == "Disposal":
            continue
        if "alt_names" in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name, ignore_disposal=ignore_disposal))
    return leaves


def get_leaf_names(cat_name, cat_val):
    """Return leaf process names for a category from the keywords hierarchy."""
    if isinstance(cat_val, dict) and "alt_names" in cat_val:
        return [cat_name]
    return [name for name, _, _ in extract_leaves(cat_val)]


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes. Not directly used in this codebase"""
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
    proc_cols = list({name for name, _, _ in extract_leaves(unitprocess_keywords, ignore_disposal=False)})
    left = cwns_mapping[["Place ID", "WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]]
    if target_facilities is not None:
        left = left[left["Place ID"].isin(target_facilities)]
    right = collapse_facility_processes(ca_cwns_df[["CWNS_ID"] + proc_cols], ["CWNS_ID"], [])
    merged = left.merge(right, on="CWNS_ID", how="left", indicator="_cwns_merge")
    cwns_by_facility = collapse_facility_processes(
        merged, ["Place ID"], ["WDID", "Facility Name", "CWNS_ID", "FACILITY_ID", "_cwns_merge"]
    ).drop(columns=["CWNS_ID", "FACILITY_ID", "_cwns_merge"], errors="ignore").fillna("")
    return cwns_by_facility, merged