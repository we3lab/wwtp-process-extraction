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


def prepare_cwns_ca(cwns_proc_df, manual_csv_path, facility_name_matches_path):
    """Consolidate CWNS CA process data with facility names and clean NPDES permits.

    Input df must include FACILITY_NAME and NPDES_PERMIT columns (provided by step0 output).

    Matching tiers (applied in order, later tiers override earlier):
    1. NPDES_PERMIT from step0 output (FACILITY_PERMIT.csv, NPDES source only)
    2. Manual overrides from cwns_permits_match_manual.csv (CWNS_ID-keyed)
    3. Name-based matches from cwns_facility_name_match_manual.csv
    """
    ca = cwns_proc_df[cwns_proc_df['STATE_CODE'] == 'CA'].copy()
    src_cols = ['CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE', 'FACILITY_NAME', 'NPDES_PERMIT']
    proc_cols = [c for c in ca.columns if c not in src_cols]

    # Consolidate by CWNS_ID
    consolidated = ca.groupby('CWNS_ID').agg(
        raw_permit_list=('PERMIT_NUMBER', lambda x: list(x.dropna().unique())),
        FACILITY_NAME=('FACILITY_NAME', 'first'),
        NPDES_PERMIT=('NPDES_PERMIT', 'first'),
        **{col: (col, 'first') for col in proc_cols}
    ).reset_index()
    consolidated['CWNS_ID'] = consolidated['CWNS_ID'].astype(str)

    # Apply manual CSV overrides (CWNS_ID-keyed: CWNS_ID,NPDES_PERMIT,FACILITY_NAME)
    manual = pd.read_csv(manual_csv_path, dtype=str).fillna('')
    cwns_id_map = (manual[manual['NPDES_PERMIT'].str.strip() != '']
                   .drop_duplicates('CWNS_ID').set_index('CWNS_ID')['NPDES_PERMIT'])
    mask = consolidated['CWNS_ID'].isin(cwns_id_map.index)
    consolidated.loc[mask, 'NPDES_PERMIT'] = consolidated.loc[mask, 'CWNS_ID'].map(cwns_id_map)

    # Apply name-based matches
    facility_name_manual = pd.read_csv(facility_name_matches_path, dtype=str).fillna('')
    if len(facility_name_manual) > 0:
        facility_name_manual_map = facility_name_manual.drop_duplicates('CWNS_ID').set_index('CWNS_ID')['NPDES_PERMIT']
        missing = consolidated['NPDES_PERMIT'].isna() | (consolidated['NPDES_PERMIT'].str.strip() == '')
        fmask = consolidated['CWNS_ID'].isin(facility_name_manual_map.index) & missing
        consolidated.loc[fmask, 'NPDES_PERMIT'] = consolidated.loc[fmask, 'CWNS_ID'].map(facility_name_manual_map)

    return consolidated


_SUFFIX_RE = re.compile(
    r'\b(WWTF|WWTP|WRP|WPCF|WWRF|WQCP|WPCP|WRF|WWRP|STP|SD|CSD|'
    r'CITY\s+OF|TOWN\s+OF|COUNTY\s+OF|DISTRICT|SANITARY|SANITATION|'
    r'WATER\s+RECLAMATION|WATER\s+POLLUTION\s+CONTROL|'
    r'TREATMENT\s+PLANT|TREATMENT\s+FACILITY|RECLAMATION\s+FACILITY|'
    r'RECLAMATION\s+PLANT)\b',
    re.IGNORECASE,
)


def normalize_facility_name(name):
    """Normalize a facility name: uppercase, strip type suffixes and punctuation."""
    if not name or (isinstance(name, float) and name != name):  # NaN check
        return ''
    s = str(name).upper().strip()
    s = _SUFFIX_RE.sub('', s)
    s = re.sub(r'[^\w\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def match_cwns_to_npdes(consolidated_cwns, npdes_permits_set,
                         npdes_name_to_permit=None, npdes_permit_to_name=None):
    """Match consolidated CWNS facilities to NPDES permits.

    Tiers applied in order (first match wins for unmatched rows):
    1. NPDES_PERMIT column (from FACILITY_PERMIT.csv or manual CSV)
    2. raw_permit_list scan (any CWNS permit in npdes_permits_set)
    3. Exact normalized facility name match (if npdes_name_to_permit provided)
    
    4. Duplicate resolution
        when >1 CWNS row links to the same permit, use
       word-overlap on normalized facility names to keep the best match and
       unlink the others (requires npdes_permit_to_name: permit → NPDES name).

    Adds 'matched' and 'linking_permit' columns.
    """
    df = consolidated_cwns.copy()

    # Tier 1: match on clean NPDES permit
    df['linking_permit'] = df['NPDES_PERMIT'].where(
        df['NPDES_PERMIT'].fillna('').str.strip().isin(npdes_permits_set)
    )

    # Tier 2: check raw permit list for unmatched rows
    unmatched = df['linking_permit'].isna()
    if unmatched.any():
        df.loc[unmatched, 'linking_permit'] = df.loc[unmatched, 'raw_permit_list'].apply(
            lambda permits: next((p for p in permits if str(p).strip() in npdes_permits_set), None)
        )

    # Tier 3: exact normalized facility name match
    if npdes_name_to_permit:
        unmatched = df['linking_permit'].isna()
        if unmatched.any():
            norm_map = {normalize_facility_name(k): v for k, v in npdes_name_to_permit.items()
                        if normalize_facility_name(k)}
            df.loc[unmatched, 'linking_permit'] = df.loc[unmatched, 'FACILITY_NAME'].apply(
                lambda n: norm_map.get(normalize_facility_name(n))
            )

    # Tier 4: resolve permit collisions via facility-name word overlap
    if npdes_permit_to_name:
        linked = df[df['linking_permit'].notna()]
        dup_permits = linked['linking_permit'].value_counts()
        dup_permits = set(dup_permits[dup_permits > 1].index)
        resolved = 0
        for permit in dup_permits:
            mask = df['linking_permit'] == permit
            candidates = df[mask]
            npdes_words = set(normalize_facility_name(
                npdes_permit_to_name.get(permit, '')).split())
            if not npdes_words:
                continue
            scores = candidates['FACILITY_NAME'].apply(
                lambda n: len(set(normalize_facility_name(n).split()) & npdes_words)
            )
            best_score = scores.max()
            if best_score == 0:
                continue
            best_idx = scores.idxmax()
            losers = candidates.index[candidates.index != best_idx]
            df.loc[losers, 'linking_permit'] = None
            resolved += len(losers)
        if resolved:
            print(f"  Tier 4: unlinked {resolved} lower-scoring CWNS duplicates across "
                  f"{len(dup_permits)} colliding permits")

    df['matched'] = df['linking_permit'].notna()
    return df

