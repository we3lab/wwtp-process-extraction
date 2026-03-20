import re
import pandas as pd
import os


def extract_leaves(processes_dict, group_id=None):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if not isinstance(details, dict):
            continue
        if 'alt_names' in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name))
    return leaves


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
            if str(row.get(p, '')).strip().lower().startswith('present')}


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes. Not directly used in this codebase"""
    el_abbadi_dir = os.path.join(os.path.dirname(__file__), 'data', 'el_abbadi', 'input')
    werf_codes_df = pd.read_csv(os.path.join(el_abbadi_dir, 'UNIT_PROCESS_EI_CODES_WERF_modified.csv'))
    matching = werf_codes_df[werf_codes_df['FINAL_UNIT_PROCESS_NAME'] == cwns_process_name]
    return matching['WERF_CODE'].unique().tolist() if not matching.empty else []


def prepare_cwns_ca(cwns_proc_df, facilities_path, permit_path, manual_csv_path,
                    facilities_2012_path=None, facility_name_matches_path=None):
    """Consolidate CWNS CA process data with facility names and clean NPDES permits.

    Matching tiers (applied in order, later tiers override earlier):
    1. NPDES_PERMIT from FACILITY_PERMIT.csv (primary, from official CWNS data)
    2. Manual overrides from cwns_permits_match_manual.csv (CWNS_ID-keyed)
    3. Name-based matches from cwns_facility_name_match_manual.csv (only rows with VALIDATED=Y)
    """
    ca = cwns_proc_df[cwns_proc_df['STATE_CODE'] == 'CA'].copy()
    meta_cols = ['CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE']
    proc_cols = [c for c in ca.columns if c not in meta_cols]

    # Consolidate by CWNS_ID
    consolidated = ca.groupby('CWNS_ID').agg(
        raw_permit_list=('PERMIT_NUMBER', lambda x: list(x.dropna().unique())),
        **{col: (col, 'first') for col in proc_cols}
    ).reset_index()
    consolidated['CWNS_ID'] = consolidated['CWNS_ID'].astype(str)

    # Add facility names from 2022 CWNS data
    facilities = pd.read_csv(facilities_path, dtype=str)
    consolidated = consolidated.merge(facilities[['CWNS_ID', 'FACILITY_NAME']], on='CWNS_ID', how='left')

    # Fill missing FACILITY_NAME from 2012 data (older facilities not in 2022 survey)
    if facilities_2012_path:
        fac12 = pd.read_csv(facilities_2012_path, dtype=str)
        ca_fac12 = fac12[fac12['State'] == 'CA'][['Facility/Project Name', 'CWNS Number']].copy()
        ca_fac12.columns = ['FACILITY_NAME_2012', 'CWNS_ID12']
        # Ensure 11-digit format (leading zero for single-digit state codes)
        ca_fac12['CWNS_ID12'] = ca_fac12['CWNS_ID12'].apply(
            lambda x: '0' + str(x) if len(str(x)) < 11 else str(x))
        fac12_map = ca_fac12.drop_duplicates('CWNS_ID12').set_index('CWNS_ID12')['FACILITY_NAME_2012']
        null_name = consolidated['FACILITY_NAME'].isna()
        consolidated.loc[null_name, 'FACILITY_NAME'] = consolidated.loc[null_name, 'CWNS_ID'].map(fac12_map)

    # Add clean NPDES permits from FACILITY_PERMIT
    permits = pd.read_csv(permit_path, dtype=str)
    npdes_permits = (permits[permits['PERMIT_SOURCE'] == 'NPDES'][['CWNS_ID', 'PERMIT_NUMBER']]
                     .drop_duplicates(subset='CWNS_ID', keep='first')
                     .rename(columns={'PERMIT_NUMBER': 'NPDES_PERMIT'}))
    consolidated = consolidated.merge(npdes_permits, on='CWNS_ID', how='left')

    # Apply manual CSV overrides (CWNS_ID-keyed: CWNS_ID,NPDES_PERMIT,FACILITY_NAME)
    manual = pd.read_csv(manual_csv_path, dtype=str).fillna('')
    cwns_id_map = (manual[manual['NPDES_PERMIT'].str.strip() != '']
                   .drop_duplicates('CWNS_ID').set_index('CWNS_ID')['NPDES_PERMIT'])
    mask = consolidated['CWNS_ID'].isin(cwns_id_map.index)
    consolidated.loc[mask, 'NPDES_PERMIT'] = consolidated.loc[mask, 'CWNS_ID'].map(cwns_id_map)

    # Apply name-based matches
    if facility_name_matches_path and os.path.exists(facility_name_matches_path):
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


def match_cwns_to_npdes(consolidated_cwns, npdes_permits_set, npdes_name_to_permit=None):
    """Match consolidated CWNS facilities to NPDES permits.

    Three tiers (first match wins):
    1. NPDES_PERMIT column (from FACILITY_PERMIT.csv or manual CSV)
    2. raw_permit_list scan (any CWNS permit in npdes_permits_set)
    3. Exact normalized facility name match (if npdes_name_to_permit dict provided)

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

    df['matched'] = df['linking_permit'].notna()
    return df
