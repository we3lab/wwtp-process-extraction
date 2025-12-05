#!/usr/bin/env python3
"""
Compare processes between `output/2025-10-31/cwns_process_presence.csv` and
`output/2025-10-31/unit_processes_v2.csv` (or `unit_processes.csv` if v2 missing).

Matches rows by `Agency` and `Facility_Name` (normalized), then compares the set
of processes marked as present/planned in the CWNS-derived file vs the unit
processes file.

Outputs:
 - output/2025-10-31/compare_cwns_unitprocesses_detailed.csv : per-match detailed rows
 - output/2025-10-31/compare_cwns_unitprocesses_summary.json : aggregated metrics
"""
import os, json
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(__file__)
OUTDIR = os.path.join(BASE, 'output', '2025-10-31')
CWNS_PATH = os.path.join(OUTDIR, 'cwns_process_presence.csv')
UNIT_V2 = os.path.join(OUTDIR, 'unit_processes_v2.csv')
UNIT = os.path.join(OUTDIR, 'unit_processes.csv')

if os.path.exists(UNIT_V2):
    UNIT_PATH = UNIT_V2
elif os.path.exists(UNIT):
    UNIT_PATH = UNIT
else:
    raise SystemExit('No unit_processes_v2.csv or unit_processes.csv found in output folder')

os.makedirs(OUTDIR, exist_ok=True)

cwns = pd.read_csv(CWNS_PATH, dtype=str).fillna('')
unit = pd.read_csv(UNIT_PATH, dtype=str).fillna('')

# normalize keys
def norm(s):
    return str(s).strip().lower()

cwns['_key'] = cwns['Agency'].astype(str).str.strip().str.lower() + '||' + cwns['Facility_Name'].astype(str).str.strip().str.lower()
unit['_key'] = unit['AGENCY_NAME'].astype(str).str.strip().str.lower() + '||' + unit['FACILITY_NAME'].astype(str).str.strip().str.lower()

# Build lookups
cwns_map = {k:df for k, df in cwns.groupby('_key')}
unit_map = {k:df for k, df in unit.groupby('_key')}

# Determine process columns for each file (exclude meta)
meta_cwns = {'CWNS Number','Permit Number_x','Agency','Facility_Name','PDF_File','_key'}
proc_cols_cwns = [c for c in cwns.columns if c not in meta_cwns]

meta_unit = {'AGENCY_NAME','FACILITY_NAME','PERMIT_NUMBER'}
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
            if v in {'present','planned','present_and_future','present_and_planned'} or v.startswith('present') or v=='planned':
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
