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


def prepare_cwns_ca(cwns_proc_df, facilities_path, permit_path, manual_csv_path, facilities_2012_path=None):
    """Consolidate CWNS CA process data with facility names and clean NPDES permits.

    1. Filter to CA, consolidate by CWNS_ID (merge duplicate rows)
    2. Add FACILITY_NAME from 2022_FACILITIES.csv; fall back to 2012 data if provided
    3. Add NPDES_PERMIT from FACILITY_PERMIT.csv (NPDES-sourced permits)
    4. Override with manual CSV corrections (format: CWNS_ID,NPDES_PERMIT,FACILITY_NAME)
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
        ca_fac12['CWNS_ID12'] = ca_fac12['CWNS_ID12'].str.lstrip('0')
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

    return consolidated


def match_cwns_to_npdes(consolidated_cwns, npdes_permits_set):
    """Match consolidated CWNS facilities to NPDES permits.

    Matches on NPDES_PERMIT first, then raw_permit_list as fallback.
    Adds 'matched' and 'linking_permit' columns.
    """
    df = consolidated_cwns.copy()

    # Primary: match on clean NPDES permit
    df['linking_permit'] = df['NPDES_PERMIT'].where(
        df['NPDES_PERMIT'].fillna('').str.strip().isin(npdes_permits_set)
    )

    # Fallback: check raw permits for unmatched rows
    unmatched = df['linking_permit'].isna()
    if unmatched.any():
        df.loc[unmatched, 'linking_permit'] = df.loc[unmatched, 'raw_permit_list'].apply(
            lambda permits: next((p for p in permits if str(p).strip() in npdes_permits_set), None)
        )

    df['matched'] = df['linking_permit'].notna()
    return df
