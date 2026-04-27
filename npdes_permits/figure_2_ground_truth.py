import json
import os
import pandas as pd
from collections import defaultdict

from helpers.utils import (
    extract_leaves,
    build_cwns_presence_mask,
    is_present,
    get_leaf_names,
    load_ciwqs_to_cwns_table,
    mapping_facility_cwns_sets,
    merge_mapping_with_cwns_processes,
    union_cwns_processes,
    cwns_process_column_names,
)
from helpers.plotting import create_ground_truth_plot

DATE_FOLDER = '2026-4-26'
figures_dir = f'npdes_permits/output/{DATE_FOLDER}/final'
os.makedirs(figures_dir, exist_ok=True)

def build_category_facility_sets(process_cols, gt_common, text_common, cwns_common,
                                  leaf_to_category):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set): gt_fac, npdes_fac, cwns_fac.
    Each set contains (WDID, Facility_Name) tuples.
    """
    gt_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        if col in gt_common.columns:
            for _, row in gt_common.iterrows():
                if is_present(row.get(col, '')):
                    gt_fac[category].add(row['_fac'])
        if col in text_common.columns:
            for _, row in text_common.iterrows():
                if is_present(row.get(col, '')):
                    npdes_fac[category].add(row['_fac'])
        if col in cwns_common.columns:
            mask = build_cwns_presence_mask(cwns_common[col])
            for _, r in cwns_common.loc[mask, ['WDID', 'Facility_Name']].iterrows():
                cwns_fac[category].add((r['WDID'], r['Facility_Name']))

    return gt_fac, npdes_fac, cwns_fac


def build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_facilities):
    """Build summary rows for create_ground_truth_plot from per-category facility sets."""
    all_cats = sorted(set(list(gt_fac.keys()) + list(npdes_fac.keys()) + list(cwns_fac.keys())))
    rows = []
    for cat in all_cats:
        ground_truth = len(gt_fac[cat])
        npdes = len(npdes_fac[cat])
        cwns = len(cwns_fac[cat])
        if ground_truth == 0 and npdes == 0 and cwns == 0:
            continue
        if ground_truth > 0:
            npdes_str = f"{(npdes - ground_truth) / ground_truth * 100:+.0f}%"
            cwns_str = f"{(cwns - ground_truth) / ground_truth * 100:+.0f}%"
        else:
            npdes_str = f"+{npdes}" if npdes > 0 else "0"
            cwns_str = f"+{cwns}" if cwns > 0 else "0"
        gt_p    = gt_fac[cat]    & common_facilities
        npdes_p = npdes_fac[cat] & common_facilities
        cwns_p  = cwns_fac[cat]  & common_facilities
        rows.append({
            'Process_Category': cat,
            'GroundTruth': ground_truth,
            'NPDES_Manual': npdes,
            'NPDES_vs_GT': npdes_str,
            'NPDES_FP': len(npdes_p - gt_p),
            'NPDES_FN': len(gt_p - npdes_p),
            'CWNS': cwns,
            'CWNS_vs_GT': cwns_str,
            'CWNS_FP': len(cwns_p - gt_p),
            'CWNS_FN': len(gt_p - cwns_p),
        })
    return rows


with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

MAPPING_CSV = 'npdes_permits/data/ciwqs_to_cwns.csv'
CWNS_CA_CSV = 'npdes_permits/output/cwns_processes_by_facility.csv'

mapping_tbl = load_ciwqs_to_cwns_table(MAPPING_CSV)
facilities_with_cwns, _ = mapping_facility_cwns_sets(mapping_tbl)

# bridge: NPDES_No → (WDID, Facility_Name)
permit_to_fac = {
    str(r['NPDES_No']).strip(): (str(r['WDID']).strip(), str(r['Facility_Name']).strip())
    for _, r in mapping_tbl.iterrows() if str(r.get('NPDES_No', '')).strip()
}

def main():

    ca_cwns_data = pd.read_csv(CWNS_CA_CSV, dtype=str, low_memory=False)
    ca_cwns_data['CWNS_ID'] = ca_cwns_data['CWNS_ID'].astype(str).str.strip()
    merged_mapping_cwns = merge_mapping_with_cwns_processes(mapping_tbl, ca_cwns_data)
    n_with_cwns = int((merged_mapping_cwns['_cwns_merge'] == 'both').sum())

    print(f"CIWQS mapping rows with CA CWNS survey attach: {n_with_cwns} / {len(merged_mapping_cwns)}")

    # Build leaf → top-level category mapping
    leaf_to_category = {
        leaf: cat_name
        for cat_name, cat_value in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat_name, cat_value)
    }

    # Load Google Sheets
    ground_truth_df  = pd.read_csv('npdes_permits/data/train_set_ground_truth.csv', dtype=str)
    npdes_text_df       = pd.read_csv('npdes_permits/data/train_set_npdes_manual.csv', dtype=str)

    print(f"GroundTruth sheet: {len(ground_truth_df)} facilities")
    print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")

    meta_cols = ['Agency', 'Facility_Name', 'NPDES_No', 'PDF_File', "Ground Truth Sources"]

    ground_truth_process_cols    = [c for c in ground_truth_df.columns    if c not in meta_cols]
    npdes_text_process_cols      = [c for c in npdes_text_df.columns      if c not in meta_cols]

    disposal_leaves = {name for name, _, _ in extract_leaves(
        unitprocess_keywords['Solids Processing']['Disposal'], ignore_disposal=False)}
    all_sheet_process_cols = [c for c in dict.fromkeys(ground_truth_process_cols + npdes_text_process_cols)
                            if c not in disposal_leaves]

    # TRAIN ground truth comparison — identify common facilities via (WDID, Facility_Name) tuples

    ground_truth_permits = {str(x).strip() for x in ground_truth_df['NPDES_No'].dropna().unique()}
    text_permits = {str(x).strip() for x in npdes_text_df['NPDES_No'].dropna().unique()}

    gt_facilities   = {permit_to_fac[p] for p in ground_truth_permits if p in permit_to_fac}
    text_facilities = {permit_to_fac[p] for p in text_permits         if p in permit_to_fac}
    common_facilities = gt_facilities & text_facilities & facilities_with_cwns
    print(f"Facilities in all 3 sources (Train): {len(common_facilities)}")

    # Add facility tuple column to source DataFrames for accumulation
    ground_truth_df['_fac'] = ground_truth_df['NPDES_No'].str.strip().map(permit_to_fac)
    npdes_text_df['_fac']   = npdes_text_df['NPDES_No'].str.strip().map(permit_to_fac)

    ground_truth_common = ground_truth_df[ground_truth_df['_fac'].isin(common_facilities)].copy()
    text_common = npdes_text_df[npdes_text_df['_fac'].isin(common_facilities)].copy()

    slice_attached = merged_mapping_cwns.loc[
        merged_mapping_cwns['CWNS_ID'].apply(lambda x: bool(str(x).strip() and str(x).strip().upper() != 'NA'))
        & merged_mapping_cwns.apply(lambda r: (r['WDID'], r['Facility_Name']), axis=1).isin(common_facilities)
    ]
    _proc_cols = [c for c in cwns_process_column_names(ca_cwns_data) if c in slice_attached.columns]
    cwns_common = union_cwns_processes(slice_attached, _proc_cols)

    gt_fac, npdes_fac, cwns_fac = build_category_facility_sets(
        all_sheet_process_cols, ground_truth_common, text_common, cwns_common,
        leaf_to_category,
    )

    gt_simple_rows = build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_facilities)

    gt_simple_df = pd.DataFrame(gt_simple_rows).sort_values('GroundTruth', ascending=False).reset_index(drop=True)
    print(gt_simple_df.to_string(index=False))

    create_ground_truth_plot(
        gt_simple_rows,
        n_facilities=len(common_facilities),
        save_path=f'{figures_dir}/figure_2_ground_truth_ground_truth_vs_npdes_text_vs_cwns.png',
    )

    print("FACILITY-LEVEL COMPARISON TO GROUND TRUTH (GroundTruth)")

    # Reverse bridge: (WDID, Facility_Name) → NPDES_No
    fac_to_permit = {v: k for k, v in permit_to_fac.items()}

    facility_rows = []
    for fac in sorted(common_facilities):
        npdes = fac_to_permit.get(fac, '')
        gt_rows_match   = ground_truth_common[ground_truth_common['_fac'] == fac]
        text_rows_match = text_common[text_common['_fac'] == fac]
        cwns_rows_match = cwns_common[(cwns_common['WDID'] == fac[0]) & (cwns_common['Facility_Name'] == fac[1])]
        if gt_rows_match.empty or text_rows_match.empty or cwns_rows_match.empty:
            continue

        ground_truth_row = gt_rows_match.iloc[0]
        text_row         = text_rows_match.iloc[0]
        cwns_row         = cwns_rows_match.iloc[0]
        facility_name    = ground_truth_row.get('Facility_Name', npdes)

        gt_set = {col for col in all_sheet_process_cols
                if col in ground_truth_common.columns and is_present(ground_truth_row.get(col, ''))}
        npdes_set = {col for col in all_sheet_process_cols
                    if col in text_common.columns and is_present(text_row.get(col, ''))}

        cwns_set = {col for col in all_sheet_process_cols
                    if col in cwns_common.columns
                    and build_cwns_presence_mask(pd.Series([cwns_row.get(col, '')])).iloc[0]}

        npdes_tp = len(gt_set & npdes_set)
        npdes_fp = len(npdes_set - gt_set)
        npdes_fn = len(gt_set - npdes_set)
        npdes_precision = npdes_tp / (npdes_tp + npdes_fp) if (npdes_tp + npdes_fp) else 0
        npdes_recall    = npdes_tp / (npdes_tp + npdes_fn) if (npdes_tp + npdes_fn) else 0
        npdes_f1 = (2 * npdes_precision * npdes_recall / (npdes_precision + npdes_recall)
                    if (npdes_precision + npdes_recall) else 0)

        cwns_tp = len(gt_set & cwns_set)
        cwns_fp = len(cwns_set - gt_set)
        cwns_fn = len(gt_set - cwns_set)
        cwns_precision = cwns_tp / (cwns_tp + cwns_fp) if (cwns_tp + cwns_fp) else 0
        cwns_recall    = cwns_tp / (cwns_tp + cwns_fn) if (cwns_tp + cwns_fn) else 0
        cwns_f1 = (2 * cwns_precision * cwns_recall / (cwns_precision + cwns_recall)
                if (cwns_precision + cwns_recall) else 0)

        facility_rows.append({
            'NPDES_No': npdes,
            'Facility_Name': facility_name,
            'GT_Count': len(gt_set),
            'NPDES_TP': npdes_tp, 'NPDES_FP': npdes_fp, 'NPDES_FN': npdes_fn,
            'NPDES_Precision': npdes_precision, 'NPDES_Recall': npdes_recall, 'NPDES_F1': npdes_f1,
            'NPDES_Missed': '|'.join(sorted(gt_set - npdes_set)),
            'NPDES_Extra': '|'.join(sorted(npdes_set - gt_set)),
            'CWNS_TP': cwns_tp, 'CWNS_FP': cwns_fp, 'CWNS_FN': cwns_fn,
            'CWNS_Precision': cwns_precision, 'CWNS_Recall': cwns_recall, 'CWNS_F1': cwns_f1,
            'CWNS_Missed': '|'.join(sorted(gt_set - cwns_set)),
            'CWNS_Extra': '|'.join(sorted(cwns_set - gt_set)),
        })

    gt_comparison_df = pd.DataFrame(facility_rows)
    gt_comparison_csv = f'npdes_permits/output/{DATE_FOLDER}/ground_truth_comparison_by_facility.csv'
    gt_comparison_df.to_csv(gt_comparison_csv, index=False)
    print(f"Saved facility-level comparison: {os.path.basename(gt_comparison_csv)}")


    # Overall error vs ground truth — only for categories where GT > 0
    gt_rows_with_gt = [r for r in gt_simple_rows if r['GroundTruth'] > 0]
    if gt_rows_with_gt:
        cwns_overall_error = sum(r['CWNS_FP'] + r['CWNS_FN'] for r in gt_rows_with_gt) / sum(r['GroundTruth'] for r in gt_rows_with_gt)
        npdes_overall_error = sum(r['NPDES_FP'] + r['NPDES_FN'] for r in gt_rows_with_gt) / sum(r['GroundTruth'] for r in gt_rows_with_gt)
        gt_zero_cwns_fp = sum(r['CWNS_FP'] for r in gt_simple_rows if r['GroundTruth'] == 0)
        print(f"\nOverall error vs Ground Truth (categories with GT>0): CWNS = {cwns_overall_error:.1%}, NPDES Text = {npdes_overall_error:.1%}")
        if gt_zero_cwns_fp:
            print(f"  (CWNS also has {gt_zero_cwns_fp} FP detections in categories with no ground truth annotations)")

if __name__ == '__main__':
    main()