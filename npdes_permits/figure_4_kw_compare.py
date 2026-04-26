"""
Grouped stacked bar-chart comparison of unit process detection across two data sources:
  - CWNS (California facilities from output/cwns_processes_by_facility.csv)
  - Keyword Search (output/{DATE_FOLDER}/unit_processes.csv)

Same structure as figure_4_source_comparison.py; shared helpers imported from there.
Only PRESENT status exists in keyword search results.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figure_4_source_comparison import (
    # constants / data loaded at module level
    keywords, merged_map, mapping_tbl, ca_cwns,
    DATE_FOLDER, OUTPUT_DIR, MIN_COUNT, PLOT_GROUPS,
    # pure helper functions
    get_facility_counts, total_count,
    filter_zero_leaf_items, compact_positions_by_category,
    build_plot_items, set_axes, sort_leaf_items_by_count,
)
from helpers.utils import (
    rows_mapping_declares_cwns, mapping_npdes_with_declared_cw,
    union_cwns_processes_by_npdes_no, cwns_process_column_names,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, draw_stacked_bar


# ── Load and normalise keyword search data ────────────────────────────────────

print("Loading keyword search data …")
kw_path = f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv'
kw_df = pd.read_csv(kw_path, dtype=str)
_meta = {'AGENCY_NAME', 'FACILITY_NAME', 'PERMIT_NUMBER', 'PDF_File', 'Shared_PDF'}
proc_cols_kw = [c for c in kw_df.columns if c not in _meta]
for col in proc_cols_kw:
    kw_df[col] = kw_df[col].str.strip().str.upper().replace({'0': '', 'NAN': ''})
kw_df['PERMIT_NUMBER'] = kw_df['PERMIT_NUMBER'].astype(str).str.strip()
print(f"  Keyword rows (pre-dedup): {len(kw_df)}")


def deduplicate_kw_facilities(df, proc_cols):
    """Collapse multiple rows per permit: PRESENT if any row has it."""
    rows = []
    for permit, group in df.groupby('PERMIT_NUMBER', dropna=False, sort=False):
        out = {
            'PERMIT_NUMBER': permit,
            'FACILITY_NAME': next((v for v in group.get('FACILITY_NAME', []) if pd.notna(v) and v), ''),
            'AGENCY_NAME':   next((v for v in group.get('AGENCY_NAME',   []) if pd.notna(v) and v), ''),
        }
        for col in proc_cols:
            out[col] = 'PRESENT' if (group[col] == 'PRESENT').any() else ''
        rows.append(out)
    return pd.DataFrame(rows)


before = len(kw_df)
kw_df = deduplicate_kw_facilities(kw_df, proc_cols_kw)
print(f"  Keyword facilities deduplicated: {before} rows → {len(kw_df)} facilities")


# ── Rebuild cwns_df scoped to keyword permits ─────────────────────────────────

kw_permits = {str(x).strip() for x in kw_df['PERMIT_NUMBER'].dropna().unique()}
matched_permits = mapping_npdes_with_declared_cw(mapping_tbl)

declared_cw = rows_mapping_declares_cwns(merged_map)
slice_kw = merged_map.loc[
    declared_cw & merged_map['NPDES_No'].astype(str).str.strip().isin(kw_permits)
]
_proc_cols = [c for c in cwns_process_column_names(ca_cwns) if c in slice_kw.columns]
cwns_df_kw = union_cwns_processes_by_npdes_no(slice_kw, _proc_cols)
cwns_df_kw['PERMIT_NUMBER'] = cwns_df_kw['NPDES_No'].astype(str).str.strip()
print(f"  CWNS rows for plot (matched to keyword permits): {len(cwns_df_kw)}")

kw_df = kw_df[kw_df['PERMIT_NUMBER'].isin(matched_permits)].copy()
print(f"  Keyword facilities after CWNS-match filter: {len(kw_df)}")

N_CWNS_KW = len(cwns_df_kw)
N_KW      = len(kw_df)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _stack_spec():
    return [
        ('PRESENT',  HATCH_PATTERNS['PRESENT'],  1.00),
        ('PAST',     HATCH_PATTERNS['PAST'],     1.00),
        ('FUTURE',   HATCH_PATTERNS['FUTURE'],   1.00),
        ('OFFSITE', HATCH_PATTERNS['OFFSITE'],  0.85),
    ]


def draw_group_kw(ax, x, bar_w, cwns_counts, kw_counts, alpha_scale=1.0):
    spec = _stack_spec()
    draw_stacked_bar(ax, x - bar_w / 2, bar_w, cwns_counts, COLORS['cwns'],  spec, alpha_scale)
    draw_stacked_bar(ax, x + bar_w / 2, bar_w, kw_counts,   COLORS['npdes'], spec, alpha_scale)


def make_legend_kw(ax, fontsize=12):
    source_handles = [
        mpatches.Patch(facecolor=COLORS['cwns'],  edgecolor='black', lw=0.5, label='CWNS'),
        mpatches.Patch(facecolor=COLORS['npdes'], edgecolor='black', lw=0.5, label='NPDES - Keyword Search'),
    ]
    status_handles = [
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['PRESENT'],  edgecolor='black', lw=0.5, label='Present'),
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['PAST'],     edgecolor='black', lw=0.5, label='Past'),
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['FUTURE'],   edgecolor='black', lw=0.5, label='Future'),
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['OFFSITE'], edgecolor='black', lw=0.5, alpha=0.85, label='Offsite'),
    ]
    header_source = mpatches.Patch(color='none', label='Data Source')
    header_status = mpatches.Patch(color='none', label='Status')
    handles = [header_source] + list(source_handles) + [header_status] + list(status_handles)
    header_indices = {0, 1 + len(source_handles)}
    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0, fontsize=fontsize, framealpha=0.85)
    for i, (h, t) in enumerate(zip(leg.legend_handles, leg.get_texts())):
        if i in header_indices:
            h.set_visible(False)
            t.set_fontweight('bold')
        if i == 0:
            t.set_horizontalalignment('left')
    return leg


# ── 1. Per-treatment-stage plots ──────────────────────────────────────────────

print("\nGenerating per-treatment-stage plots …")
bar_w = 0.35

for group_title, json_cats in PLOT_GROUPS.items():
    items, positions, labels = build_plot_items(json_cats, keywords)
    if not items:
        continue

    cwns_counts_list = [get_facility_counts(cwns_df_kw, item['cols']) for item in items]
    kw_counts_list   = [get_facility_counts(kw_df,      item['cols']) for item in items]

    items, positions, labels, cwns_counts_list, kw_counts_list = filter_zero_leaf_items(
        items, positions, labels, cwns_counts_list, kw_counts_list
    )
    items, labels, cwns_counts_list, kw_counts_list = sort_leaf_items_by_count(
        items, labels, cwns_counts_list, kw_counts_list
    )
    positions = compact_positions_by_category(items, gap=0.25)

    if not items or max(total_count(c) for c in cwns_counts_list + kw_counts_list) == 0:
        print(f"  {group_title}: all zeros, skipping")
        continue

    x_span = positions[-1] - positions[0] + 1 if positions else 1
    fig, ax = plt.subplots(figsize=(max(7, x_span * 0.85), 5))

    for item, pos, cwns_c, kw_c in zip(items, positions, cwns_counts_list, kw_counts_list):
        alpha = 1.0 if item['is_total'] else 0.60
        draw_group_kw(ax, pos, bar_w, cwns_c, kw_c, alpha_scale=alpha)

    set_axes(ax, labels, positions)

    cat_spans = {}
    for item, pos in zip(items, positions):
        cat = item['cat']
        if cat not in cat_spans:
            cat_spans[cat] = [pos, pos]
        else:
            cat_spans[cat][1] = pos

    ylim = ax.get_ylim()
    first = True
    for cat, (s, e) in cat_spans.items():
        if not first:
            ax.axvline(s - 0.5, color='#999999', lw=0.8, linestyle='--', zorder=1)
        leaves_in_cat = [it for it in items if it['cat'] == cat and not it['is_total']]
        if len(leaves_in_cat) > 1:
            ax.text((s + e) / 2, ylim[1] * 0.97, cat,
                    ha='center', va='top', fontsize=11, color='#444444', style='italic')
        first = False
    ax.set_ylim(ylim)
    make_legend_kw(ax)
    plt.tight_layout()

    safe = group_title.replace('/', '_').replace(' ', '_')
    path = f'{OUTPUT_DIR}/{safe}_kw_compare.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ── 2. Major-categories plot ──────────────────────────────────────────────────

print("\nGenerating major-categories plot …")
cat_labels = list(keywords.keys())

all_cwns_kw = []
all_kw      = []
from helpers.utils import get_leaf_names
for cat_name, cat_val in keywords.items():
    leaves = get_leaf_names(cat_name, cat_val)
    all_cwns_kw.append(get_facility_counts(cwns_df_kw, leaves))
    all_kw.append(get_facility_counts(kw_df, leaves))

cat_filtered = [
    (lbl, cwns_c, kw_c)
    for lbl, cwns_c, kw_c in zip(cat_labels, all_cwns_kw, all_kw)
    if total_count(cwns_c) >= MIN_COUNT or total_count(kw_c) >= MIN_COUNT
]
if cat_filtered:
    cat_labels, all_cwns_kw, all_kw = map(list, zip(*cat_filtered))

order = sorted(range(len(cat_labels)),
               key=lambda i: -(total_count(all_cwns_kw[i]) + total_count(all_kw[i])))
cat_labels  = [cat_labels[i]  for i in order]
all_cwns_kw = [all_cwns_kw[i] for i in order]
all_kw      = [all_kw[i]      for i in order]

n = len(cat_labels)
fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))

for i, (cwns_c, kw_c) in enumerate(zip(all_cwns_kw, all_kw)):
    draw_group_kw(ax, i, bar_w, cwns_c, kw_c)

set_axes(ax, cat_labels, list(range(n)), rotation=45)
make_legend_kw(ax)

final_dir = f'npdes_permits/output/{DATE_FOLDER}/final'
os.makedirs(final_dir, exist_ok=True)
path = f'{final_dir}/figure_4_major_categories_kw_compare.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {os.path.basename(path)}")

print("\nDone.")
