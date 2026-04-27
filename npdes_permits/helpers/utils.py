import re
import pandas as pd
import os

# Canonical status vocabulary: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, '' (absent)
PRESENT_STATUSES = frozenset({'PRESENT', 'PRESENT_AND_FUTURE'})


def parse_status(val) -> str:
    """Normalize any status cell to a canonical token.

    Handles manual sheet values (messy text), LLM output (clean tokens), and CWNS values.
    Returns: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, or ''.
    """
    if val is None or (isinstance(val, float) and val != val):
        return ''
    s = str(val).strip()
    if not s or s in ('0', '0.0'):
        return ''
    t = s.upper().replace('-', '_')
    if t in ('NAN', 'NONE'):
        return ''
    if 'PRESENT' in t and 'FUTURE' in t:
        return 'PRESENT_AND_FUTURE'
    for keyword in ['PRESENT', 'FUTURE', 'PAST', 'OFFSITE']:
        if keyword in t:
            return keyword
    return ''


def is_present(val) -> bool:
    """True if val indicates the process is currently installed (PRESENT or PRESENT_AND_FUTURE).

    FUTURE is excluded — it means planned but not yet in service.
    """
    return parse_status(val) in PRESENT_STATUSES


def build_cwns_presence_mask(series):
    """Return boolean mask for CWNS presence values (any detectable status, including FUTURE/PAST)."""
    return series.map(parse_status).isin({'PRESENT', 'PRESENT_AND_FUTURE', 'FUTURE', 'PAST'})


def extract_leaves(processes_dict, group_id=None, ignore_disposal=True):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if not isinstance(details, dict):
            continue
        if ignore_disposal and name == 'Disposal':
            continue
        if 'alt_names' in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name, ignore_disposal=ignore_disposal))
    return leaves


def get_leaf_names(cat_name, cat_val):
    """Return leaf process names for a category from the keywords hierarchy."""
    if isinstance(cat_val, dict) and 'alt_names' in cat_val:
        return [cat_name]
    return [name for name, _, _ in extract_leaves(cat_val)]


def get_cwns_unit_process_names(process_name, process_details):
    """Get CWNS unit process names for a given process from keywords"""
    if isinstance(process_details, dict) and 'cwns_processes' in process_details:
        names = process_details['cwns_processes']
        return names if isinstance(names, list) else [names]
    return []


def find_process_details(process_name, unitprocess_keywords):
    """Find process details in keywords hierarchy, searching both top-level and nested"""
    if process_name in unitprocess_keywords:
        return unitprocess_keywords[process_name]
    for category, cat_keywords in unitprocess_keywords.items():
        if not isinstance(cat_keywords, dict):
            continue
        if process_name in cat_keywords:
            return cat_keywords[process_name]
        for parent_name, parent_details in cat_keywords.items():
            if isinstance(parent_details, dict) and process_name in parent_details:
                return parent_details[process_name]
    return None


def extract_cwns_processes(row, proc_cols):
    """Extract set of processes marked as present in a CWNS row."""
    return {p for p in proc_cols
            if str(row.get(p, '')).strip().lower() not in {'', '0', '0.0', 'nan'}}


def extract_npdes_processes(row, proc_cols):
    """Extract set of processes marked as present in an NPDES row."""
    return {p for p in proc_cols
            if str(row.get(p, '')).strip().lower().startswith('PRESENT')}


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes. Not directly used in this codebase"""
    el_abbadi_dir = os.path.join(os.path.dirname(__file__), 'data', 'el_abbadi', 'input')
    werf_codes_df = pd.read_csv(os.path.join(el_abbadi_dir, 'UNIT_PROCESS_EI_CODES_WERF_modified.csv'))
    matching = werf_codes_df[werf_codes_df['FINAL_UNIT_PROCESS_NAME'] == cwns_process_name]
    return matching['WERF_CODE'].unique().tolist() if not matching.empty else []


def load_ciwqs_to_cwns_table(mapping_csv_path: str) -> pd.DataFrame:
    """Load ``ciwqs_to_cwns.csv`` with string cells stripped of surrounding whitespace."""
    df = pd.read_csv(mapping_csv_path, dtype=str, keep_default_na=False).fillna('')
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def _mapping_cw_join_key(cw_cell: str) -> str | None:
    s = str(cw_cell).strip()
    if not s or s.upper() == 'NA':
        return None
    return s


def cwns_process_column_names(cwns_df: pd.DataFrame) -> list[str]:
    """Unit-process columns on the CWNS CA export (everything except facility meta)."""
    meta = {'CWNS_ID', 'FACILITY_NAME', 'PERMIT_NUMBER', 'STATE_CODE', 'NPDES_PERMIT'}
    return [c for c in cwns_df.columns if c not in meta]


def merge_mapping_with_cwns_processes(
    mapping_df: pd.DataFrame,
    cwns_ca_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join each mapping row to CA CWNS process data on (CWNS_ID, CWNS_Facility_Name)."""
    m = mapping_df.copy()
    c = cwns_ca_df.copy()

    proc_cols = cwns_process_column_names(c)  # compute before adding temp columns

    m['_cw_id']   = m['CWNS_ID'].apply(lambda x: _mapping_cw_join_key(str(x)))
    m['_cw_name'] = m['CWNS_Facility_Name'].astype(str).str.strip().str.upper()
    c['_cw_id']   = c['CWNS_ID'].astype(str).str.strip()
    c['_cw_name'] = c['FACILITY_NAME'].astype(str).str.strip().str.upper()
    right_cols = ['_cw_id', '_cw_name'] + proc_cols + (['FACILITY_NAME'] if 'FACILITY_NAME' in c.columns else [])
    right = c[right_cols].drop_duplicates(subset=['_cw_id', '_cw_name'], keep='first').rename(
        columns={'_cw_id': '_r_id', '_cw_name': '_r_name'}
    )

    out = m.merge(
        right,
        left_on=['_cw_id', '_cw_name'],
        right_on=['_r_id', '_r_name'],
        how='left',
        indicator='_cwns_merge',
    )
    return out.drop(columns=['_cw_id', '_cw_name', '_r_id', '_r_name'])


def mapping_facility_cwns_sets(mapping_df: pd.DataFrame) -> tuple[set, set]:
    """Returns (with_cwns, no_cwns) — sets of (WDID, Facility_Name) tuples.

    with_cwns: rows declaring a non-empty, non-NA CWNS_ID.
    no_cwns:   rows explicitly marked CWNS_ID == 'NA'.
    """
    with_cwns, no_cwns = set(), set()
    for _, row in mapping_df.iterrows():
        fac = (str(row.get('WDID', '')).strip(), str(row.get('Facility_Name', '')).strip())
        cid = str(row.get('CWNS_ID', '')).strip().upper()
        if _mapping_cw_join_key(cid) is not None:
            with_cwns.add(fac)
        elif cid == 'NA':
            no_cwns.add(fac)
    return with_cwns, no_cwns


def union_cwns_processes(merged: pd.DataFrame, proc_cols: list[str]) -> pd.DataFrame:
    """One row per (WDID, Facility_Name), unioning process statuses across mapping edges.

    Carries the first non-blank NPDES_No per group for downstream permit-based bridges.
    """
    def merge_column_statuses(column: pd.Series) -> str:
        tokens = {parse_status(v) for v in column}
        if 'PRESENT_AND_FUTURE' in tokens or ('PRESENT' in tokens and 'FUTURE' in tokens):
            return 'PRESENT_AND_FUTURE'
        for token in ('PRESENT', 'FUTURE', 'PAST', 'OFFSITE'):
            if token in tokens:
                return token
        return ''

    present_proc = [c for c in proc_cols if c in merged.columns]
    if not present_proc:
        return pd.DataFrame(columns=['WDID', 'Facility_Name', 'NPDES_No'])

    chunks = []
    for (wdid, fname), grp in merged.groupby(['WDID', 'Facility_Name'], dropna=False, sort=False):
        if not str(wdid).strip() and not str(fname).strip():
            continue
        npdes_vals = [v for v in grp.get('NPDES_No', []) if str(v).strip()]
        d = {'WDID': wdid, 'Facility_Name': fname, 'NPDES_No': npdes_vals[0] if npdes_vals else ''}
        for pc in present_proc:
            d[pc] = merge_column_statuses(grp[pc])
        chunks.append(d)
    return pd.DataFrame(chunks) if chunks else pd.DataFrame(columns=['WDID', 'Facility_Name', 'NPDES_No'] + present_proc)
