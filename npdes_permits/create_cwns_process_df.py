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
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE = ROOT  # repo root at npdes_permits

# Paths (adjust if your working layout differs)
SITE_DATA = os.path.join(BASE,'npdes_permits', 'output', '2025-10-31', 'site_data.csv')
CWNS_TO_NPDES = os.path.join(BASE, 'npdes_permits', 'output', '2025-10-31', 'matched_cwns_npdes_ca.csv')
CWNS_PROCESS = os.path.join(BASE, 'npdes_permits', 'data', 'cwns', '2012', 'Unit_Process_Details.csv')
PROCESS_MAP = os.path.join(BASE, 'npdes_permits', 'data', 'process_cwns.csv')
PROCESS_LIST = ["Screening/Microstrainer","Grit Removal","Flow Equalization","Primary Clarifier","Flotation","SBR","Oxidation Ditch","Pure Oxygen Activated Sludge","Extended Aeration","Stepfeed","Unspecified CAS","Biofilter Trickling Filter","Rotating Biological Contactor","Anaerobic Filter","Unspecified Trickling Filter","Facultative Lagoon","Aerated Lagoon","Anaerobic Lagoon","Polishing Lagoon","Unspecified Lagoon","Membrane Bioreactor","Membrane Aerated Biofilm Reactor","AO","A2O","MLE","Bardenpho","Unspecified BNR","Chemical N Removal","Chemical P Removal","Tertiary Filtration","Constructed Wetland","Surface Wetland","Unspecified Nature-Based Solution","UV-AOP","Coagulation","Flocculation","Ion Exchange","Activated Carbon","Sedimentation","Media Filtration","Nanofiltration","Electrodialysis Reversal","Electrodialysis","Reverse Osmosis","Ultrafiltration","Microfiltration","Chlorination","Dechlorination","UV Disinfection","Ozonation","Thermal Disinfection","Unspecified Disinfection","Alum Addition","Ferric Chloride Addition","Polymer Addition","Other Chemical Addition","Anaerobic Digestion","Aerobic Digestion","Unspecified Digestion","Mechanical Dewatering","Centrifuge","Drying","Thickening","Lime Treatment","Land Treatment","Stabilization Pond","Biosolids Lagoon","MHI","FBI","Unspecified Incineration","Biogas Production","Cogeneration","Unspecified Energy Recovery"]
OUT_PATH = os.path.join(BASE, 'npdes_permits', 'output', '2025-10-31', 'cwns_process_presence.csv')
print('Reading files...')
site_df = pd.read_csv(SITE_DATA, dtype=str).fillna('')
cwns_to_npdes = pd.read_csv(CWNS_TO_NPDES, dtype=str).fillna('')
cwns_process = pd.read_csv(CWNS_PROCESS, dtype=str).fillna('')
process_map = pd.read_csv(PROCESS_MAP, dtype=str).fillna('')

try:
    # try to import a python module that defines PROCESS_LIST
    from process_list import PROCESS_LIST  # type: ignore
    print('Loaded PROCESS_LIST from process_list.py')
except Exception:
    try:
        pl_path = os.path.join(BASE, 'data', 'process_list.json')
        if os.path.exists(pl_path):
            PROCESS_LIST = pd.read_json(pl_path, typ='series').tolist() if pl_path.endswith('.json') else None
            print(f'Loaded PROCESS_LIST from {pl_path}')
    except Exception:
        PROCESS_LIST = None

if PROCESS_LIST is None:
    # fallback: infer process columns from site_data (assume first 3 are meta)
    meta_cols = list(site_df.columns[:3])
    PROCESS_LIST = [c for c in site_df.columns if c not in meta_cols]
    print('PROCESS_LIST not found externally; inferred from site_data columns')

process_cols = PROCESS_LIST
meta_cols = list(site_df.columns[:3])
print(f'Using {len(process_cols)} process columns')

# build mapping: normalized CWNS_Process -> list of Defined_Process
map_dict = {}
for _, r in process_map.iterrows():
    c_proc = str(r.get('CWNS_Process','')).strip()
    d_proc = str(r.get('Defined_Process','')).strip()
    if not c_proc or not d_proc:
        continue
    key = c_proc.lower()
    map_dict.setdefault(key, set()).add(d_proc)

# For robustness also add trimmed duplicates

# Prepare result rows
rows = []

# For each matched CWNS->NPDES row
for idx, r in cwns_to_npdes.iterrows():
    cw_id = str(r.get('CWNS Number','')).strip()
    # initialize row with meta columns if present in matched file
    row = {}
    # copy some provenance metadata if present
    for k in ['CWNS Number','Permit Number_x','Agency','Facility_Name','PDF_File']:
        if k in r.index:
            row[k] = r.get(k,'')
    # initialize all process_cols to 0
    for p in process_cols:
        row[p] = 0

    # select unit processes for this CWNS number
    ups = cwns_process[cwns_process['CWNS Number'].astype(str).str.strip() == cw_id]
    if ups.empty:
        rows.append(row)
        continue

    # keep track of status per defined process: 2=present,1=planned,0=none
    status = {p:0 for p in process_cols}

    for _, up in ups.iterrows():
        unit_proc = str(up.get('Unit Process','')).strip()
        present_flag = str(up.get('Present','')).strip().upper() == 'Y'
        proj_flag = str(up.get('Projected','')).strip().upper() == 'Y'
        key = unit_proc.lower()
        # find mapped defined processes
        mapped = map_dict.get(key, set())
        if not mapped:
            # try a looser match: exact-insensitive substring match against keys
            for k_map in map_dict.keys():
                if k_map in key or key in k_map:
                    mapped = mapped.union(map_dict[k_map])
        # apply statuses
        for d in mapped:
            if d not in status:
                # if the defined process isn't one of the site process columns, skip
                continue
            if present_flag:
                status[d] = 2
            elif proj_flag and status[d] < 1:
                status[d] = 1

    # map numeric status to strings requested
    for p in process_cols:
        if status[p] == 2:
            row[p] = 'present'
        elif status[p] == 1:
            row[p] = 'planned'
        else:
            row[p] = 0

    rows.append(row)

# Build DataFrame and save
out_df = pd.DataFrame(rows)
# Ensure columns order: meta then process_cols
out_cols = [c for c in ['CWNS Number','Permit Number_x','Agency','Facility_Name','PDF_File'] if c in out_df.columns]
out_cols.extend(process_cols)
out_df = out_df[out_cols]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)
print('Saved', OUT_PATH)
print('Rows:', len(out_df))
print(out_df.head())
