import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os
import seaborn as sns

from helpers.utils import *
from helpers.metrics import METRIC_SCORE_COLUMNS, compute_metrics
from helpers.utils import parse_status, is_present, PRESENT_STATUSES
from helpers.plotting import COLORS, HATCH_PATTERNS, plot_stacked_counts
from helpers.utils import get_leaf_names

DATE_FOLDER = '2026-2-18'
GOOGLE_SHEET_ID = '18U4IlfAiNH1UNdUYH5fF35fX99ll9SciKYRUuHUdT8w'

MANUAL_SOURCE_ORDER = ['Manual', 'LLM', 'Keyword']
MANUAL_SOURCE_COLORS = {
    'Manual':  '#8c8c8c',
    'LLM':     COLORS['npdes_total'],
    'Keyword': COLORS['npdes'],
}
MANUAL_STATUS_ORDER = ['PRESENT', 'FUTURE', 'PAST', 'off_site']

# Subset of METRIC_SCORE_COLUMNS for the facility violin (CSV / tables still use full set).
VIOLIN_METRIC_COLUMNS = tuple(
    c for c in METRIC_SCORE_COLUMNS if c not in {'Precision', 'Recall'}
)


def build_status_mask(df, process_name, unitprocess_keywords, status_filter):
    statuses = df[process_name].map(parse_status)
    if status_filter == 'any':
        return statuses.isin(PRESENT_STATUSES)
    return statuses == str(status_filter).upper()


def build_binary_mask(df, process_name, unitprocess_keywords):
    return build_status_mask(df, process_name, unitprocess_keywords, 'any')


def get_status_counts(process_name, unit_process_results):
    """Extract status breakdown for a process (PRESENT_AND_FUTURE folded into PRESENT)."""
    s = unit_process_results[process_name].map(parse_status)
    return {
        'PRESENT': int(s.isin(PRESENT_STATUSES).sum()),
        'FUTURE':  int((s == 'FUTURE').sum()),
    }


def create_method_deviation_plot(process_names, manual_df, llm_df, keyword_df,
                                 category_name, figsize=(12, 5), fontsize=14, save_path=None):
    """Plot LLM and keyword deviations from manual readings (above y=0: extra; below: missed)."""
    manual_permits = set(manual_df['NPDES_No'].dropna())
    llm_common = manual_permits & set(llm_df['PERMIT_NUMBER'].dropna()) if llm_df is not None else set()
    kw_common  = manual_permits & set(keyword_df['PERMIT_NUMBER'].dropna())

    rows = []
    for process in process_names:
        manual_proc = set(manual_df.loc[
            manual_df[process].map(is_present), 'NPDES_No'])
        manual_count = len(manual_proc)

        llm_fp = llm_fn = 0
        if llm_df is not None:
            sub  = llm_df[llm_df['PERMIT_NUMBER'].isin(llm_common)]
            mask = build_binary_mask(sub, process, None)
            llm_proc = set(sub.loc[mask, 'PERMIT_NUMBER'])
            m = manual_proc & llm_common
            llm_fp, llm_fn = len(llm_proc - m), len(m - llm_proc)

        kw_fp = kw_fn = 0
        sub  = keyword_df[keyword_df['PERMIT_NUMBER'].isin(kw_common)]
        mask = build_binary_mask(sub, process, None)
        kw_proc = set(sub.loc[mask, 'PERMIT_NUMBER'])
        m = manual_proc & kw_common
        kw_fp, kw_fn = len(kw_proc - m), len(m - kw_proc)

        if not any([llm_fp, llm_fn, kw_fp, kw_fn, manual_count]):
            continue
        rows.append({'Process': process, 'Manual_Count': manual_count,
                     'LLM_FP': llm_fp, 'LLM_FN': llm_fn, 'KW_FP': kw_fp, 'KW_FN': kw_fn})

    if not rows:
        print(f"No deviation data for '{category_name}'")
        return

    df = pd.DataFrame(rows).sort_values('Manual_Count', ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    w = 0.18

    for idx, row in df.iterrows():
        for x_off, fp, fn, color in [(-w, row['LLM_FP'], row['LLM_FN'], COLORS['npdes_total']),
                                      (+w, row['KW_FP'],  row['KW_FN'],  COLORS['npdes'])]:
            if fp: ax.bar(idx + x_off, fp,  w * 2, color=color, edgecolor='black', linewidth=0.5)
            if fn: ax.bar(idx + x_off, -fn, w * 2, color=color, hatch='///', edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Process'], rotation=45, ha='right', fontsize=fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('WWTP Count vs Manual Reading', fontsize=16)
    ax.set_title(f'{category_name.replace("_", " ").title()} – Method vs Manual Reading', fontsize=18)

    legend_handles = [
        Patch(color='none', label='Method'),
        Patch(facecolor=COLORS['npdes_total'], edgecolor='black', linewidth=0.5, label='  NPDES - LLM Extraction'),
        Patch(facecolor=COLORS['npdes'],       edgecolor='black', linewidth=0.5, label='  NPDES Keyword'),
        Patch(color='none', label='vs Manual Reading'),
        Patch(facecolor='gray', edgecolor='black', linewidth=0.5, label='  Extra (above)'),
        Patch(facecolor='gray', hatch='///', edgecolor='black', linewidth=0.5, label='  Missed (below)'),
    ]
    leg = ax.legend(handles=legend_handles, loc='upper left',
                    bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=11)
    for i, (h, t) in enumerate(zip(leg.legend_handles, leg.get_texts())):
        if i in {0, 3}:
            h.set_visible(False)
            t.set_fontweight('bold')

    plt.subplots_adjust(bottom=0.25)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_manual_status_counts(process_name, manual_df):
    """Status counts from manual readings for a single process."""
    counts = {s: 0 for s in MANUAL_STATUS_ORDER}
    for val in manual_df[process_name]:
        cat = parse_status(val)
        if not cat:
            continue
        key = 'off_site' if cat == 'OFFSITE' else cat
        if key in counts:
            counts[key] += 1
    return counts


def build_method_metric_inputs(process_names, manual_df, pred_df, pred_permit_col):
    """Build key-aligned manual/pred dataframes for helpers.metrics.compute_metrics."""
    common_keys = sorted(set(manual_df['NPDES_No'].dropna()) & set(pred_df[pred_permit_col].dropna()))

    manual_sub = (manual_df[manual_df['NPDES_No'].isin(common_keys)]
                  .drop_duplicates(subset='NPDES_No')
                  .set_index('NPDES_No'))
    pred_sub = (pred_df[pred_df[pred_permit_col].isin(common_keys)]
                .drop_duplicates(subset=pred_permit_col)
                .set_index(pred_permit_col))

    manual_metric_df = pd.DataFrame({'key': common_keys})
    pred_metric_df = pd.DataFrame({'key': common_keys})

    for process in process_names:
        manual_metric_df[process] = manual_sub.reindex(common_keys)[process].map(parse_status).values
        pred_metric_df[process] = pred_sub.reindex(common_keys)[process].map(parse_status).values

    return manual_metric_df, pred_metric_df, common_keys


def compute_facility_metric_rows(manual_metric_df, pred_metric_df, process_names, source_name):
    """Compute per-facility metrics used for split violin density."""
    manual_idx = manual_metric_df.set_index('key')
    pred_idx = pred_metric_df.set_index('key')
    rows = []
    for key in manual_idx.index:
        tp = fp = fn = tn = 0
        state_correct = state_total = 0
        for process in process_names:
            manual_state = manual_idx.at[key, process]
            pred_state = pred_idx.at[key, process]
            manual_pos = is_present(manual_state)
            pred_pos = is_present(pred_state)

            tp += int(manual_pos and pred_pos)
            fp += int((not manual_pos) and pred_pos)
            fn += int(manual_pos and (not pred_pos))
            tn += int((not manual_pos) and (not pred_pos))

            if manual_pos:
                state_total += 1
                state_correct += int(manual_state == pred_state)

        precision = tp / (tp + fp) if (tp + fp) else float('nan')
        recall    = tp / (tp + fn) if (tp + fn) else float('nan')
        f1        = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float('nan')
        missed_rate = fn / (tp + fn) if (tp + fn) else float('nan')
        hallucinated_rate = fp / (tp + fp) if (tp + fp) else float('nan')
        state_accuracy = state_correct / state_total if state_total else float('nan')
        scores = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
            'Missed_Rate': missed_rate,
            'Hallucinated_Rate': hallucinated_rate,
            'State_Accuracy': state_accuracy,
        }
        row = {'Source': source_name, 'key': key}
        for col in METRIC_SCORE_COLUMNS:
            row[col] = scores[col]
        rows.append(row)
    return rows


def create_split_violin_plot(facility_metrics_df, save_path):
    score_cols = [c for c in VIOLIN_METRIC_COLUMNS if c in facility_metrics_df.columns]
    plot_df = facility_metrics_df[facility_metrics_df['Source'].isin(['LLM', 'Keyword'])].melt(
        id_vars=['Source', 'key'],
        value_vars=score_cols,
        var_name='Metric',
        value_name='Value',
    ).dropna(subset=['Value'])

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(
        data=plot_df,
        x='Metric',
        y='Value',
        hue='Source',
        order=score_cols,
        hue_order=['LLM', 'Keyword'],
        split=True,
        palette={'LLM': COLORS['npdes_total'], 'Keyword': COLORS['npdes']},
        inner=None,
        cut=0,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel('Facility-level score', fontsize=12)
    ax.set_title('Keyword vs LLM Facility-level Metric Distributions', fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Source', loc='upper right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {os.path.basename(save_path)}")


# Load all required data
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load keyword-based NPDES results
unit_process_results = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv')

nan_permit_mask = unit_process_results['PERMIT_NUMBER'].isna()
if nan_permit_mask.any():
    extracted = unit_process_results.loc[nan_permit_mask, 'PDF_File'].apply(
        lambda f: m.group(0).upper() if (m := re.search(r'CA\d{7}', str(f), re.IGNORECASE)) else None
    )
    filled = extracted.notna().sum()
    unit_process_results.loc[extracted.index, 'PERMIT_NUMBER'] = extracted
    print(f"Resolved {filled} of {nan_permit_mask.sum()} NaN PERMIT_NUMBERs from PDF filenames")

print(f"NPDES data: Loaded {len(unit_process_results)} rows")

unit_full = unit_process_results.copy()

# Load LLM results
llm_results_path = 'npdes_permits/output/llm_unit_processes_by_facility.csv'
llm_results = pd.read_csv(llm_results_path) if os.path.exists(llm_results_path) else None
if llm_results is not None:
    print(f"LLM results: {len(llm_results)} facilities")
else:
    print(f"LLM results not found at {llm_results_path}; breakdown plots will show keyword only")

# Filter to facilities processed by BOTH methods
if llm_results is not None:
    llm_permit_numbers = set(llm_results['PERMIT_NUMBER'].dropna())
    kw_permit_numbers  = set(unit_full['PERMIT_NUMBER'].dropna())
    both_permit_numbers = llm_permit_numbers & kw_permit_numbers
    llm_results_both = llm_results[llm_results['PERMIT_NUMBER'].isin(both_permit_numbers)].copy()
    unit_full_both   = unit_full[unit_full['PERMIT_NUMBER'].isin(both_permit_numbers)].copy()
    print(f"Facilities processed by both LLM and keyword: {len(both_permit_numbers)} "
          f"(LLM only: {len(llm_permit_numbers - kw_permit_numbers)}, "
          f"keyword only: {len(kw_permit_numbers - llm_permit_numbers)})")
else:
    llm_results_both = None
    unit_full_both   = unit_full

# Load manual readings (train + test) as the deviation baseline
train_manual = pd.read_csv('npdes_permits/data/train_set_npdes_manual.csv', dtype=str)
test_manual  = pd.read_csv('npdes_permits/data/test_set_npdes_manual.csv', dtype=str)
manual_combined = (pd.concat([train_manual, test_manual])
                   .drop_duplicates(subset='NPDES_No').reset_index(drop=True))
manual_permits = set(manual_combined['NPDES_No'].dropna())
llm_results_manual = (llm_results_both[llm_results_both['PERMIT_NUMBER'].isin(manual_permits)].copy()
                      if llm_results_both is not None else None)
unit_full_manual = unit_full_both[unit_full_both['PERMIT_NUMBER'].isin(manual_permits)].copy()
print(f"Manual baseline: {len(manual_combined)} facilities "
      f"({len(manual_permits & set(unit_full_both['PERMIT_NUMBER']))} matched to keyword, "
      f"{len(manual_permits & set(llm_results_both['PERMIT_NUMBER'])) if llm_results_both is not None else 0} matched to LLM)")

figures_dir = f'npdes_permits/output/{DATE_FOLDER}/figures'
os.makedirs(figures_dir, exist_ok=True)

for category in categories_to_plot:
    print(f"\nProcessing category: {category}")
    safe_category = category.replace("/", "_").replace(os.sep, "_")

    process_names = get_leaf_names(category, unitprocess_keywords[category])

    # Method vs manual reading: deviation bars (FP above zero, FN below)
    create_method_deviation_plot(
        process_names,
        manual_combined,
        llm_results_manual,
        unit_full_manual,
        category,
        save_path=f'{figures_dir}/{safe_category}_npdes_method_comparison_deviation.png'
    )
    print(f"  Saved {safe_category}_npdes_method_comparison_deviation.png")

    # Method vs manual reading: stacked absolute counts per status
    counts_by_source = {
        'Manual':  {p: get_manual_status_counts(p, manual_combined) for p in process_names},
        'Keyword': {p: get_status_counts(p, unit_full_manual)       for p in process_names},
    }
    if llm_results_manual is not None:
        counts_by_source['LLM'] = {p: get_status_counts(p, llm_results_manual) for p in process_names}
    src_order = [s for s in MANUAL_SOURCE_ORDER if s in counts_by_source]
    plot_stacked_counts(
        counts_by_source, process_names,
        f'{figures_dir}/{safe_category}_npdes_method_comparison_counts.png',
        f'{category.replace("_", " ").title()} – Manual vs LLM vs Keyword',
        MANUAL_SOURCE_COLORS, src_order, MANUAL_STATUS_ORDER,
    )
    print(f"  Saved {safe_category}_npdes_method_comparison_counts.png")


# ── Method comparison metrics ─────────────────────────────────────────────────
all_process_list = [p for cat in categories_to_plot
                    for p in get_leaf_names(cat, unitprocess_keywords[cat])]
metrics_frames = []
facility_metric_rows = []

manual_metric_kw, pred_metric_kw, _ = build_method_metric_inputs(
    all_process_list, manual_combined, unit_full_both, 'PERMIT_NUMBER'
)
kw_metrics = compute_metrics(manual_metric_kw, pred_metric_kw, all_process_list, 'Keyword')
metrics_frames.append(kw_metrics.rename(columns={'Label': 'Process'}))
facility_metric_rows.extend(
    compute_facility_metric_rows(manual_metric_kw, pred_metric_kw, all_process_list, 'Keyword')
)

if llm_results_both is not None:
    manual_metric_llm, pred_metric_llm, _ = build_method_metric_inputs(
        all_process_list, manual_combined, llm_results_both, 'PERMIT_NUMBER'
    )
    llm_metrics = compute_metrics(manual_metric_llm, pred_metric_llm, all_process_list, 'LLM')
    metrics_frames.append(llm_metrics.rename(columns={'Label': 'Process'}))
    facility_metric_rows.extend(
        compute_facility_metric_rows(manual_metric_llm, pred_metric_llm, all_process_list, 'LLM')
    )

metrics_df = pd.concat(metrics_frames, ignore_index=True)
metrics_path = f'npdes_permits/output/{DATE_FOLDER}/npdes_method_comparison_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"\nSaved npdes_method_comparison_metrics.csv")
summary = metrics_df.groupby('Source')[[
    'Precision', 'Recall', 'F1', 'Accuracy', 'Missed_Rate', 'Hallucinated_Rate', 'State_Accuracy'
]].mean()
print(summary.to_string(float_format=lambda x: f'{x:.3f}'))

facility_metrics_df = pd.DataFrame(facility_metric_rows)
final_dir = f'npdes_permits/output/{DATE_FOLDER}/final'
os.makedirs(final_dir, exist_ok=True)
create_split_violin_plot(
    facility_metrics_df,
    f'{final_dir}/figure_3_npdes_method_comparison_metrics_violin.png',
)

# Overall status summary
total_present = total_present_and_future = total_future = 0
for category in categories_to_plot:
    for process_name in get_leaf_names(category, unitprocess_keywords[category]):
        s = unit_full[process_name].astype(str).str.strip().str.upper()
        total_present            += int((s == 'PRESENT').sum())
        total_present_and_future += int((s == 'PRESENT_AND_FUTURE').sum())
        total_future             += int((s == 'FUTURE').sum())

print(f"Total process instances marked as 'PRESENT': {total_present}")
print(f"Total process instances marked as 'PRESENT_AND_FUTURE': {total_present_and_future}")
print(f"Total process instances marked as 'FUTURE': {total_future}")
print(f"Grand total: {total_present + total_present_and_future + total_future}")
