#!/usr/bin/env python3
"""
Create a DataFrame that, for each CWNS->NPDES matched row, marks all
potential site processes (columns taken from `site_data.csv`) as:
 - 'present' if any CWNS unit process mapped to that defined process has Present=='Y'
 - 'planned' if any CWNS unit process mapped to that defined process has Projected=='Y' and no present
 - 0 otherwise

Saves output to `output/2025-10-31/cwns_process_presence.csv` (path derived from site_data file location).
"""
import os
import json
import pandas as pd
from utils import get_all_keys, find_process_details, get_cwns_unit_process_names

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = ROOT

def build_cwns_process_presence(
    site_data_path: str,
    cwns_permit_path: str,
    cwns_process_path: str,
    keywords_json_path: str,
) -> pd.DataFrame:
    print('Loading data...')
    site_df = pd.read_csv(site_data_path, dtype=str).fillna('')
    cwns_process_df = pd.read_csv(cwns_process_path, dtype=str).fillna('')
    cwns_permit = pd.read_csv(cwns_permit_path, dtype=str).fillna('')

    with open(keywords_json_path, 'r') as f:
        unitprocess_keywords = json.load(f)

    # Extract all leaf process names from JSON
    process_cols = get_all_keys(unitprocess_keywords)
    print(f'Extracted {len(process_cols)} processes from keywords JSON')

    # Build CWNS->process mapping {cwns_name_lower: [defined_process1, ...]}
    print('Building CWNS to process mapping...')
    cwns_to_process = {}
    for process_name in process_cols:
        details = find_process_details(process_name, unitprocess_keywords)
        if details:
            cwns_names = get_cwns_unit_process_names(process_name, details)
            for cwns_name in cwns_names:
                key = cwns_name.lower().strip()
                cwns_to_process.setdefault(key, []).append(process_name)
    print(f'Mapped {len(cwns_to_process)} CWNS processes to defined taxonomy')

    # Match CWNS facilities to NPDES permits
    print('Matching CWNS facilities to NPDES permits...')
    site_df['NPDES_No'] = site_df['NPDES_No'].astype(str).str.strip()
    cwns_permit['PERMIT_NUMBER'] = cwns_permit['PERMIT_NUMBER'].astype(str).str.strip()

    cwns_to_npdes = site_df.merge(
        cwns_permit,
        left_on='NPDES_No',
        right_on='PERMIT_NUMBER',
        how='left'
    ).rename(columns={
        'CWNS_ID': 'CWNS Number'
    })
    print(f'Matched {cwns_to_npdes["CWNS Number"].notna().sum()} facilities')

    # Generate process presence dataframe
    print('Generating CWNS process presence...')
    rows = []

    for _, facility in cwns_to_npdes.iterrows():
        cwns_id = str(facility.get('CWNS Number', '')).strip()

        row = {
            'CWNS Number': cwns_id,
            'NPDES_No': facility.get('NPDES_No', ''),
            'Agency': facility.get('Agency', ''),
            'Facility_Name': facility.get('Facility_Name', ''),
            'PDF_File': facility.get('PDF_File', '')
        }

        status = {p: 0 for p in process_cols}

        if cwns_id:
            unit_processes = cwns_process_df[
                cwns_process_df['CWNS Number'].astype(str).str.strip() == cwns_id
            ]

            for _, up in unit_processes.iterrows():
                unit_proc_name = str(up.get('Unit Process', '')).strip()
                present = str(up.get('Present', '')).strip().upper() == 'Y'
                planned = str(up.get('Projected', '')).strip().upper() == 'Y'

                key = unit_proc_name.lower()
                mapped_processes = cwns_to_process.get(key, [])

                for proc in mapped_processes:
                    if proc in status:
                        if present:
                            status[proc] = 2
                        elif planned and status[proc] < 1:
                            status[proc] = 1

        for p in process_cols:
            row[p] = 'present' if status[p] == 2 else ('planned' if status[p] == 1 else 0)

        rows.append(row)

    return pd.DataFrame(rows)


# Default paths
SITE_DATA = os.path.join(BASE, 'npdes_permits', 'output', '2025-10-31', 'site_data.csv')
CWNS_PERMIT = os.path.join(BASE, 'npdes_permits', 'data', 'cwns', '2022', 'FACILITY_PERMIT.csv')
CWNS_PROCESS = os.path.join(BASE, 'npdes_permits', 'data', 'cwns', '2012', 'Unit_Process_Details.csv')
KEYWORDS_JSON = os.path.join(BASE, 'npdes_permits', 'data', 'unitprocess_keywords.json')
OUT_PATH = os.path.join(BASE, 'npdes_permits', 'output', '2025-10-31', 'cwns_process_presence.csv')


if __name__ == '__main__':
    out_df = build_cwns_process_presence(
        site_data_path=SITE_DATA,
        cwns_permit_path=CWNS_PERMIT,
        cwns_process_path=CWNS_PROCESS,
        keywords_json_path=KEYWORDS_JSON,
    )
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f'Saved: {OUT_PATH}')
    print(f'Total facilities: {len(out_df)}')
