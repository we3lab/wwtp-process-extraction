#!/usr/bin/env python3
"""
Compare processes between CWNS unit_processes_by_facility.csv (mapped to taxonomy)
and NPDES unit_processes.csv.

Uses unit_processes_by_facility.csv directly (filtered to CA, matched by permit number),
then compares the set of processes marked as present vs the unit processes file.

Outputs:
 - output/2026-2-18/compare_cwns_unitprocesses_detailed.csv : per-match detailed rows
 - output/2026-2-18/compare_cwns_unitprocesses_summary.json : aggregated metrics
"""
import os, json
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(__file__)
OUTDIR = os.path.join(BASE, 'output', '2026-2-18')
UNIT_PATH = os.path.join(OUTDIR, 'unit_processes.csv')
CWNS_PATH = os.path.join(BASE, 'output', 'unit_processes_by_facility.csv')
SITE_DATA = os.path.join(OUTDIR, 'site_data.csv')

os.makedirs(OUTDIR, exist_ok=True)

# Load NPDES text extraction results
unit = pd.read_csv(UNIT_PATH, dtype=str).fillna('')

# Load CWNS data (already mapped to taxonomy by el_abbadi_tt_assignment.py)
cwns_all = pd.read_csv(CWNS_PATH, dtype=str).fillna('')

# Filter CWNS to California only
cwns_ca = cwns_all[cwns_all['STATE_CODE'] == 'CA'].copy()
print(f'CWNS CA facilities: {len(cwns_ca)}')

# Load site_data to get Agency/Facility_Name for CWNS matching
site_df = pd.read_csv(SITE_DATA, dtype=str).fillna('')
site_df['NPDES_No'] = site_df['NPDES_No'].astype(str).str.strip()

# Match CWNS to site_data by permit number
cwns_ca['PERMIT_NUMBER'] = cwns_ca['PERMIT_NUMBER'].astype(str).str.strip()
cwns = cwns_ca.merge(
    site_df[['NPDES_No', 'Agency', 'Facility_Name']],
    left_on='PERMIT_NUMBER',
    right_on='NPDES_No',
    how='inner'
)
print(f'CWNS matched to NPDES: {len(cwns)}')

# normalize keys
def norm(s):
    return str(s).strip().lower()

cwns['_key'] = cwns['Agency'].astype(str).str.strip().str.lower() + '||' + cwns['Facility_Name'].astype(str).str.strip().str.lower()
unit['_key'] = unit['AGENCY_NAME'].astype(str).str.strip().str.lower() + '||' + unit['FACILITY_NAME'].astype(str).str.strip().str.lower()

# Build lookups
cwns_map = {k:df for k, df in cwns.groupby('_key')}
unit_map = {k:df for k, df in unit.groupby('_key')}

# Determine process columns for each file (exclude meta)
meta_cwns = {'CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE', 'NPDES_No', 'Agency', 'Facility_Name', '_key'}
proc_cols_cwns = [c for c in cwns.columns if c not in meta_cwns]

meta_unit = {'AGENCY_NAME','FACILITY_NAME','PERMIT_NUMBER', '_key'}
proc_cols_unit = [c for c in unit.columns if c not in meta_unit]

# We'll iterate over keys present in either set
all_keys = sorted(set(cwns_map.keys()) | set(unit_map.keys()))

rows = []
miss_count = 0
hall_count = 0
both_count = 0
no_gt = 0
for k in all_keys:
    crow = cwns_map.get(k, pd.DataFrame()).iloc[0] if k in cwns_map else None
    urow = unit_map.get(k, pd.DataFrame()).iloc[0] if k in unit_map else None
    agency = crow['Agency'] if crow is not None else (urow['AGENCY_NAME'] if urow is not None else '')
    fac = crow['Facility_Name'] if crow is not None else (urow['FACILITY_NAME'] if urow is not None else '')

    # extract sets
    gt_set = set()
    if crow is not None:
        for p in proc_cols_cwns:
            v = str(crow.get(p, '')).strip().lower()
            # CWNS data uses numeric 0/1; treat any non-zero as present
            if v not in {'', '0', '0.0', 'nan'}:
                gt_set.add(p)
    else:
        no_gt += 1

    pred_set = set()
    if urow is not None:
        for p in proc_cols_unit:
            v = str(urow.get(p, '')).strip().lower()
            # unit_processes may use 'present','future','present_and_future' etc.; treat 'present' as predicted
            if v in {'present','present_and_future','present_and_planned'} or v.startswith('present'):
                pred_set.add(p)

    missed = sorted(gt_set - pred_set)
    hall = sorted(pred_set - gt_set)
    inter = sorted(gt_set & pred_set)

    if missed or hall:
        miss_count += len(missed)
        hall_count += len(hall)
    if gt_set and pred_set:
        both_count += 1

    rows.append({
        'key': k,
        'Agency': agency,
        'Facility_Name': fac,
        'ground_truth_count': len(gt_set),
        'predicted_count': len(pred_set),
        'intersection_count': len(inter),
        'missed': '|'.join(missed),
        'hallucinated': '|'.join(hall),
    })

# Save detailed CSV
out_df = pd.DataFrame(rows)
out_csv = os.path.join(OUTDIR, 'compare_cwns_unitprocesses_detailed.csv')
out_df.to_csv(out_csv, index=False)

# Summary metrics
summary = {
    'rows_total': len(rows),
    'rows_with_gt': len([r for r in rows if r['ground_truth_count']>0]),
    'rows_with_pred_and_gt': both_count,
    'total_missed_items': miss_count,
    'total_hallucinated_items': hall_count,
}
with open(os.path.join(OUTDIR,'compare_cwns_unitprocesses_summary.json'),'w') as f:
    json.dump(summary, f, indent=2)

print('Saved detailed:', out_csv)
print('Saved summary:', os.path.join(OUTDIR,'compare_cwns_unitprocesses_summary.json'))
print('Summary:', summary)
