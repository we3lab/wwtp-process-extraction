#!/usr/bin/env python3
"""
Compare processes between CWNS and NPDES at the facility level.

Uses consolidated CWNS data (by CWNS_ID, enriched with facility names and clean
NPDES permits) matched to NPDES unit_processes.csv by linking permit.

Outputs per-facility detailed CSV and aggregated summary JSON.
"""
import os, json
import pandas as pd
from helpers.utils import (prepare_cwns_ca, match_cwns_to_npdes,
                           extract_cwns_processes, extract_npdes_processes)

BASE = os.path.dirname(__file__)
DATE_FOLDER = '2026-2-18'
OUTDIR = os.path.join(BASE, 'output', DATE_FOLDER)
os.makedirs(OUTDIR, exist_ok=True)

# Load NPDES results
unit = pd.read_csv(os.path.join(OUTDIR, 'unit_processes.csv'), dtype=str).fillna('')

# Load, consolidate, and match CWNS
cwns_all = pd.read_csv(os.path.join(BASE, 'output', 'unit_processes_by_facility.csv'), dtype=str).fillna('')
cwns = prepare_cwns_ca(
    cwns_all,
    os.path.join(BASE, 'data', 'cwns', '2022', '2022_FACILITIES.csv'),
    os.path.join(BASE, 'data', 'cwns', '2022', 'FACILITY_PERMIT.csv'),
    os.path.join(BASE, 'data', 'cwns', 'cwns_facilities_match_manual.csv'),
)
cwns = match_cwns_to_npdes(cwns, set(unit['PERMIT_NUMBER'].dropna().unique()))
cwns_matched = cwns[cwns['matched']].copy()
print(f'CWNS CA: {len(cwns)}, matched: {len(cwns_matched)}')

# Determine process columns
meta_cwns = {'CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE', 'FACILITY_NAME',
             'NPDES_PERMIT', 'raw_permit_list', 'linking_permit', 'matched'}
proc_cols_cwns = [c for c in cwns_matched.columns if c not in meta_cwns]
meta_unit = {'AGENCY_NAME', 'FACILITY_NAME', 'PERMIT_NUMBER', 'FACILITY_KEY',
             'PDF_File', 'Shared_PDF'}
proc_cols_unit = [c for c in unit.columns if c not in meta_unit]

# Merge CWNS and NPDES on permit, then compare per facility
merged = cwns_matched.merge(
    unit, left_on='linking_permit', right_on='PERMIT_NUMBER',
    how='outer', suffixes=('_cwns', '_npdes'), indicator=True
)

rows = []
for _, row in merged.iterrows():
    has_cwns = row['_merge'] in ('both', 'left_only')
    has_npdes = row['_merge'] in ('both', 'right_only')
    fac = row.get('FACILITY_NAME_cwns') or row.get('FACILITY_NAME_npdes', '')

    gt_set = extract_cwns_processes(row, proc_cols_cwns) if has_cwns else set()
    pred_set = extract_npdes_processes(row, proc_cols_unit) if has_npdes else set()

    rows.append({
        'CWNS_ID': row.get('CWNS_ID', ''),
        'PERMIT_NUMBER': row.get('linking_permit') or row.get('PERMIT_NUMBER', ''),
        'Facility_Name': fac,
        'ground_truth_count': len(gt_set),
        'predicted_count': len(pred_set),
        'intersection_count': len(gt_set & pred_set),
        'missed': '|'.join(sorted(gt_set - pred_set)),
        'hallucinated': '|'.join(sorted(pred_set - gt_set)),
    })

out_df = pd.DataFrame(rows)
out_csv = os.path.join(OUTDIR, 'compare_cwns_unitprocesses_detailed.csv')
out_df.to_csv(out_csv, index=False)

both = out_df[(out_df['ground_truth_count'] > 0) & (out_df['predicted_count'] > 0)]
summary = {
    'rows_total': len(out_df),
    'rows_with_gt': (out_df['ground_truth_count'] > 0).sum(),
    'rows_with_pred_and_gt': len(both),
    'total_missed_items': out_df['missed'].str.count(r'\|').sum() + (out_df['missed'] != '').sum(),
    'total_hallucinated_items': out_df['hallucinated'].str.count(r'\|').sum() + (out_df['hallucinated'] != '').sum(),
}
with open(os.path.join(OUTDIR, 'compare_cwns_unitprocesses_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Saved: {out_csv}')
print(f'Summary: {summary}')
