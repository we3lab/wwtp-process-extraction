import pandas as pd

# Canonical status vocabulary: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, '' (absent)
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"})

mapping_df = pd.read_csv(
    "npdes_permits/data/ciwqs_to_cwns.csv", dtype=str, keep_default_na=False
).fillna("")
for c in mapping_df.columns:
    mapping_df[c] = mapping_df[c].astype(str).str.strip()


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


def get_cwns_unit_process_names(process_name, process_details):
    """Get CWNS unit process names for a given process from keywords"""
    if isinstance(process_details, dict) and "cwns_processes" in process_details:
        names = process_details["cwns_processes"]
        return names if isinstance(names, list) else [names]
    return []


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes. Not directly used in this codebase"""
    el_abbadi_dir = os.path.join(os.path.dirname(__file__), "data", "el_abbadi", "input")
    werf_codes_df = pd.read_csv(
        os.path.join(el_abbadi_dir, "UNIT_PROCESS_EI_CODES_WERF_modified.csv")
    )
    matching = werf_codes_df[werf_codes_df["FINAL_UNIT_PROCESS_NAME"] == cwns_process_name]
    return matching["WERF_CODE"].unique().tolist() if not matching.empty else []


def _mapping_cw_join_key(cw_cell: str) -> str | None:
    s = str(cw_cell).strip()
    if not s or s.upper() == "NA":
        return None
    return s


def cwns_process_column_names(cwns_df: pd.DataFrame) -> list[str]:
    """Unit-process columns on the CWNS CA export (everything except facility meta)."""
    meta = {"CWNS_ID", "FACILITY_NAME", "PERMIT_NUMBER", "STATE_CODE", "NPDES_PERMIT"}
    return [c for c in cwns_df.columns if c not in meta]


def merge_mapping_with_cwns_processes(cwns_ca_df):
    """Left-join each mapping row to CA CWNS process data on (CWNS_ID, CWNS_Facility_Name)."""
    m = mapping_df.copy()
    c = cwns_ca_df.copy()

    proc_cols = cwns_process_column_names(c)  # compute before adding temp columns

    m["_cw_id"] = m["CWNS_ID"].apply(lambda x: _mapping_cw_join_key(str(x)))
    m["_cw_name"] = m["CWNS_Facility_Name"].astype(str).str.strip().str.upper()
    c["_cw_id"] = c["CWNS_ID"].astype(str).str.strip()
    c["_cw_name"] = c["FACILITY_NAME"].astype(str).str.strip().str.upper()
    right_cols = (
        ["_cw_id", "_cw_name"]
        + proc_cols
        + (["FACILITY_NAME"] if "FACILITY_NAME" in c.columns else [])
    )
    right = (
        c[right_cols]
        .drop_duplicates(subset=["_cw_id", "_cw_name"], keep="first")
        .rename(columns={"_cw_id": "_r_id", "_cw_name": "_r_name"})
    )

    out = m.merge(
        right,
        left_on=["_cw_id", "_cw_name"],
        right_on=["_r_id", "_r_name"],
        how="left",
        indicator="_cwns_merge",
    )
    return out.drop(columns=["_cw_id", "_cw_name", "_r_id", "_r_name"])


def mapping_facility_cwns_sets() -> tuple[set, set]:
    """Returns (with_cwns, no_cwns) — sets of (WDID, Facility_Name) tuples.

    with_cwns: rows declaring a non-empty, non-NA CWNS_ID.
    no_cwns:   rows explicitly marked CWNS_ID == 'NA'.
    """
    with_cwns, no_cwns = set(), set()
    for _, row in mapping_df.iterrows():
        fac = (
            str(row.get("WDID", "")).strip(),
            str(row.get("Facility_Name", "")).strip(),
        )
        cid = str(row.get("CWNS_ID", "")).strip().upper()
        if _mapping_cw_join_key(cid) is not None:
            with_cwns.add(fac)
        elif cid == "NA":
            no_cwns.add(fac)
    return with_cwns, no_cwns


name_to_fac: dict[str, tuple[str, str]] = {
    name: (wdid, name)
    for _, r in mapping_df.iterrows()
    for name in [str(r.get("Facility_Name", "")).strip()]
    for wdid in [str(r.get("WDID", "")).strip()]
    if name and wdid
}
facilities_with_cwns, facilities_without_cwns = mapping_facility_cwns_sets()


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


def union_cwns_processes(merged: pd.DataFrame, proc_cols: list[str]) -> pd.DataFrame:
    """One row per (WDID, Facility_Name), unioning process statuses across mapping edges.

    Carries the first non-blank NPDES_No per group for downstream permit-based bridges.
    """
    present_proc = [c for c in proc_cols if c in merged.columns]
    if not present_proc:
        return pd.DataFrame(columns=["WDID", "Facility_Name", "NPDES_No"])

    chunks = []
    for (wdid, fname), grp in merged.groupby(["WDID", "Facility_Name"], dropna=False, sort=False):
        if not str(wdid).strip() and not str(fname).strip():
            continue
        npdes_vals = [v for v in grp.get("NPDES_No", []) if str(v).strip()]
        d = {
            "WDID": wdid,
            "Facility_Name": fname,
            "NPDES_No": npdes_vals[0] if npdes_vals else "",
        }
        for pc in present_proc:
            d[pc] = merge_column_statuses(grp[pc])
        chunks.append(d)
    return (
        pd.DataFrame(chunks)
        if chunks
        else pd.DataFrame(columns=["WDID", "Facility_Name", "NPDES_No"] + present_proc)
    )


def map_facility_names_to_tuples(
    df: pd.DataFrame,
    name_col: str,
    name_to_fac: dict[str, tuple[str, str]],
) -> set[tuple[str, str]]:
    """Map a dataframe's facility-name column to canonical (WDID, Facility_Name) tuples."""
    out = set()
    if name_col not in df.columns:
        return out
    for name in df[name_col].astype(str).str.strip():
        if name in name_to_fac:
            out.add(name_to_fac[name])
    return out


def build_cwns_facility_processes(
    ca_cwns_df: pd.DataFrame,
    target_facilities: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one CWNS process row per mapped facility, optionally filtered to target facilities.

    Returns `(cwns_by_facility, merged_mapping)` where `merged_mapping` includes merge indicators.
    """
    merged_mapping = merge_mapping_with_cwns_processes(ca_cwns_df)
    declared_cw = merged_mapping["CWNS_ID"].apply(
        lambda x: bool(str(x).strip() and str(x).strip().upper() != "NA")
    )
    fac_col = merged_mapping.apply(lambda r: (r["WDID"], r["Facility_Name"]), axis=1)
    if target_facilities is not None:
        slice_map = merged_mapping.loc[declared_cw & fac_col.isin(target_facilities)]
    else:
        slice_map = merged_mapping.loc[declared_cw]
    proc_cols = [c for c in cwns_process_column_names(ca_cwns_df) if c in slice_map.columns]
    cwns_by_facility = union_cwns_processes(slice_map, proc_cols)
    return cwns_by_facility, merged_mapping
