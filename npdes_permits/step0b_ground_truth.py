import json
import os
import pandas as pd
from collections import defaultdict

from helpers.utils import (extract_leaves, prepare_cwns_ca, match_cwns_to_npdes,
                           build_cwns_presence_mask, is_yes, count_yes)
from helpers.plotting import create_ground_truth_plot
from helpers.load_google_sheet import load_google_sheet_csv

DATE_FOLDER = '2026-2-18'
GOOGLE_SHEET_ID = '18U4IlfAiNH1UNdUYH5fF35fX99ll9SciKYRUuHUdT8w'
BACWA_PLOT_CATEGORIES = ['Activated Sludge', 'Lagoon', 'Nutrient Removal', 'Filtration']


def build_category_facility_sets(process_cols, gt_common, text_common, cwns_common,
                                  leaf_to_category):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set): gt_fac, npdes_fac, cwns_fac.
    """
    gt_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        if col in gt_common.columns:
            for _, row in gt_common.iterrows():
                if is_yes(row.get(col, '')):
                    gt_fac[category].add(row['NPDES_No'])
        if col in text_common.columns:
            for _, row in text_common.iterrows():
                if is_yes(row.get(col, '')):
                    npdes_fac[category].add(row['NPDES_No'])
        if col in cwns_common.columns:
            mask = build_cwns_presence_mask(cwns_common[col])
            for permit in cwns_common.loc[mask, 'linking_permit']:
                cwns_fac[category].add(permit)

    return gt_fac, npdes_fac, cwns_fac


def build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_permits):
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
        gt_p    = gt_fac[cat]    & common_permits
        npdes_p = npdes_fac[cat] & common_permits
        cwns_p  = cwns_fac[cat]  & common_permits
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

# Re-load and match CWNS data (mirrors step5 prep)
cwns_data = pd.read_csv('npdes_permits/output/unit_processes_by_facility.csv',
                         low_memory=False, dtype={'CWNS_ID': str})
ca_cwns_data = prepare_cwns_ca(
    cwns_data,
    'npdes_permits/data/cwns/cwns_permits_match_manual.csv',
    'npdes_permits/data/cwns/cwns_facility_name_match_manual.csv',
)

all_ca_npdes = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/all_ca_npdes.csv', dtype=str)
npdes_name_to_permit = (
    all_ca_npdes[all_ca_npdes['NPDES No.'].notna()]
    [['NPDES No.', 'Facility Name']]
    .rename(columns={'NPDES No.': 'PERMIT_NUMBER', 'Facility Name': 'FACILITY_NAME'})
    .drop_duplicates(subset='FACILITY_NAME', keep='first')
    .set_index('FACILITY_NAME')['PERMIT_NUMBER']
    .to_dict()
)
npdes_permit_numbers = set(all_ca_npdes['NPDES No.'].dropna().unique())
ca_cwns_data = match_cwns_to_npdes(ca_cwns_data, npdes_permit_numbers,
                                    npdes_name_to_permit=npdes_name_to_permit)

figures_dir = f'npdes_permits/output/{DATE_FOLDER}/figures'
os.makedirs(figures_dir, exist_ok=True)

# Build leaf → top-level category mapping
leaf_to_category = {}
for cat_name, cat_value in unitprocess_keywords.items():
    if isinstance(cat_value, dict) and 'alt_names' in cat_value:
        leaf_to_category[cat_name] = cat_name
    else:
        for leaf_name, _, _ in extract_leaves(cat_value):
            leaf_to_category[leaf_name] = cat_name

# Load Google Sheets

ground_truth_df     = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - Ground Truth')
npdes_text_df       = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - From NPDES Text')
bacwa_ground_truth_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'BACWA - Ground Truth')
bacwa_npdes_text_df   = load_google_sheet_csv(GOOGLE_SHEET_ID, 'BACWA - From NPDES Text')

print(f"GroundTruth sheet: {len(ground_truth_df)} facilities")
print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")
print(f"BACWA Ground Truth sheet: {len(bacwa_ground_truth_df)} facilities")
print(f"BACWA NPDES Text sheet: {len(bacwa_npdes_text_df)} facilities")

meta_cols = ['Agency', 'Facility_Name', 'NPDES_No', 'PDF_File', "Ground Truth Sources"]

ground_truth_process_cols    = [c for c in ground_truth_df.columns    if c not in meta_cols]
npdes_text_process_cols      = [c for c in npdes_text_df.columns      if c not in meta_cols]
bacwa_ground_truth_process_cols = [c for c in bacwa_ground_truth_df.columns if c not in meta_cols]
bacwa_npdes_text_process_cols   = [c for c in bacwa_npdes_text_df.columns   if c not in meta_cols]

all_sheet_process_cols = list(dict.fromkeys(ground_truth_process_cols + npdes_text_process_cols))
all_bacwa_process_cols = list(dict.fromkeys(bacwa_ground_truth_process_cols + bacwa_npdes_text_process_cols))

# TRAIN ground truth comparison

cwns_permits_gt = set(ca_cwns_data['linking_permit'].dropna().str.strip())
ground_truth_permits = set(ground_truth_df['NPDES_No'].dropna().str.strip())
text_permits         = set(npdes_text_df['NPDES_No'].dropna().str.strip())
common_permits       = ground_truth_permits & text_permits & cwns_permits_gt
print(f"Facilities in all 3 sources (Train): {len(common_permits)}")

ground_truth_common = ground_truth_df[ground_truth_df['NPDES_No'].str.strip().isin(common_permits)].copy()
text_common         = npdes_text_df[npdes_text_df['NPDES_No'].str.strip().isin(common_permits)].copy()
cwns_common         = ca_cwns_data[ca_cwns_data['linking_permit'].str.strip().isin(common_permits)].copy()

gt_fac, npdes_fac, cwns_fac = build_category_facility_sets(
    all_sheet_process_cols, ground_truth_common, text_common, cwns_common,
    leaf_to_category,
)

gt_simple_rows = build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_permits)

gt_simple_df = pd.DataFrame(gt_simple_rows).sort_values('GroundTruth', ascending=False).reset_index(drop=True)
print(gt_simple_df.to_string(index=False))

create_ground_truth_plot(
    gt_simple_rows,
    n_facilities=len(common_permits),
    save_path=f'{figures_dir}/ground_truth_ground_truth_vs_npdes_text_vs_cwns.png',
)

print("FACILITY-LEVEL COMPARISON TO GROUND TRUTH (GroundTruth)")

facility_rows = []
for permit in sorted(common_permits):
    ground_truth_row = ground_truth_common[ground_truth_common['NPDES_No'].str.strip() == permit].iloc[0]
    text_row         = text_common[text_common['NPDES_No'].str.strip() == permit].iloc[0]
    cwns_row         = cwns_common[cwns_common['linking_permit'].str.strip() == permit].iloc[0]

    facility_name = ground_truth_row.get('Facility_Name', permit)

    gt_set = {col for col in all_sheet_process_cols
              if col in ground_truth_common.columns and is_yes(ground_truth_row.get(col, ''))}
    npdes_set = {col for col in all_sheet_process_cols
                 if col in text_common.columns and is_yes(text_row.get(col, ''))}

    cwns_set = set()
    for col in all_sheet_process_cols:
        if col in cwns_common.columns:
            val = cwns_row.get(col, '')
            s = str(val).strip().lower()
            try:
                if float(val) > 0:
                    cwns_set.add(col)
                    continue
            except (ValueError, TypeError):
                pass
            if s in ('present', 'planned', 'present_and_future', 'present_and_planned') or s.startswith('present'):
                cwns_set.add(col)

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
        'NPDES_No': permit,
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


# BACWA ground truth comparison (4-category bar plot)

bacwa_gt_permits   = set(bacwa_ground_truth_df['NPDES_No'].dropna().str.strip())
bacwa_text_permits = set(bacwa_npdes_text_df['NPDES_No'].dropna().str.strip())
bacwa_cwns_permits = set(ca_cwns_data['linking_permit'].dropna().str.strip())
bacwa_common       = bacwa_gt_permits & bacwa_text_permits & bacwa_cwns_permits
print(f"\nFacilities in all 3 sources (BACWA): {len(bacwa_common)}")

bacwa_gt_common   = bacwa_ground_truth_df[bacwa_ground_truth_df['NPDES_No'].str.strip().isin(bacwa_common)].copy()
bacwa_text_common = bacwa_npdes_text_df[bacwa_npdes_text_df['NPDES_No'].str.strip().isin(bacwa_common)].copy()
bacwa_cwns_common = ca_cwns_data[ca_cwns_data['linking_permit'].str.strip().isin(bacwa_common)].copy()

bacwa_gt_fac, bacwa_npdes_fac, bacwa_cwns_fac = build_category_facility_sets(
    all_bacwa_process_cols, bacwa_gt_common, bacwa_text_common, bacwa_cwns_common,
    leaf_to_category,
)

bacwa_gt_rows = build_gt_rows(bacwa_gt_fac, bacwa_npdes_fac, bacwa_cwns_fac, bacwa_common)
bacwa_filtered_rows = [r for r in bacwa_gt_rows if r['Process_Category'] in BACWA_PLOT_CATEGORIES]

create_ground_truth_plot(
    bacwa_filtered_rows,
    n_facilities=len(bacwa_common),
    save_path=f'{figures_dir}/bacwa_ground_truth_vs_npdes_text_vs_cwns.png',
)

# pring overall % error vs ground truth for both data sources (summary)
cwns_overall_error = sum(r['CWNS_FP'] + r['CWNS_FN'] for r in gt_simple_rows) / sum(r['GroundTruth'] for r in gt_simple_rows)
npdes_overall_error = sum(r['NPDES_FP'] + r['NPDES_FN'] for r in gt_simple_rows) / sum(r['GroundTruth'] for r in gt_simple_rows)
print(f"\nOverall error vs Ground Truth: CWNS = {cwns_overall_error:.1%}, NPDES Text = {npdes_overall_error:.1%}")  