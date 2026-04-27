"""
Grouped stacked bar-chart comparison of unit process detection across two data sources:
  - CWNS (California facilities from output/cwns_processes_by_facility.csv)
  - LLM Search (output/llm_unit_processes_by_facility.csv)

CWNS rows join ``ciwqs_to_cwns.csv`` to the CA step0 export (exact ``CWNS_ID``).
The export includes placeholder rows for CA ``CWNS_ID`` values missing from CWNS
inventory aggregation (blank ``0`` processes). LLM rows are kept when
``PERMIT_NUMBER`` matches mapping ``NPDES_No`` on a row that declares a
``CWNS_ID``. No fuzzy matching.

Each bar is stacked by status (process columns must already be normalized: stripped, uppercase).
PRESENT_AND_FUTURE is counted as PRESENT only (not split into Future).

Plot stacks (both sources): Present | Past | Future | Offsite
  CWNS : PRESENT | PRESENT_AND_FUTURE | FUTURE | PAST | OFFSITE | 0
  LLM  : PRESENT | PRESENT_AND_FUTURE | FUTURE | PAST | OFFSITE

Produces:
  1. One graph per treatment-stage group (combined leaves)
  2. One graph of major categories (top-level JSON keys)
  Y-axis: summed status counts over the leaves in each bar group.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.utils import get_leaf_names, PRESENT_STATUSES
from helpers.utils import (
    load_ciwqs_to_cwns_table,
    mapping_facility_cwns_sets,
    merge_mapping_with_cwns_processes,
    union_cwns_processes,
    cwns_process_column_names,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, draw_stacked_bar
from figure_2_ground_truth import permit_to_fac

DATE_FOLDER = '2026-4-26'
OUTPUT_DIR  = f'npdes_permits/output/{DATE_FOLDER}/figures'
MIN_COUNT   = 15  # drop bar groups where both sources are below this threshold

# ── Data helpers ──────────────────────────────────────────────────────────────


def _any_flag(df, cols, statuses):
    """Boolean Series: True if any col in cols has any value in statuses."""
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            mask |= df[col].isin(statuses)
    return mask


def get_facility_counts(df, leaf_cols):
    """Unique-facility counts: each facility counted once at highest-priority status."""
    has_present = _any_flag(df, leaf_cols, PRESENT_STATUSES)
    has_past    = _any_flag(df, leaf_cols, {'PAST'})
    has_future  = _any_flag(df, leaf_cols, {'FUTURE'})
    has_offsite = _any_flag(df, leaf_cols, {'OFFSITE'})
    return {
        'PRESENT':  int(has_present.sum()),
        'PAST':     int((has_past    & ~has_present).sum()),
        'FUTURE':   int((has_future  & ~has_present).sum()),
        'OFFSITE': int((has_offsite & ~has_present & ~has_future).sum()),
    }


def total_count(counts):
    return sum(counts.values())


def filter_zero_leaf_items(items, positions, labels, cwns_counts_list, llm_counts_list):
    """Drop items where both sources are below MIN_COUNT."""
    kept = []
    for item, pos, label, cwns_c, llm_c in zip(items, positions, labels, cwns_counts_list, llm_counts_list):
        if total_count(cwns_c) < MIN_COUNT and total_count(llm_c) < MIN_COUNT:
            continue
        kept.append((item, pos, label, cwns_c, llm_c))

    if not kept:
        return [], [], [], [], []

    new_items, new_positions, new_labels, new_cwns, new_llm = zip(*kept)
    return list(new_items), list(new_positions), list(new_labels), list(new_cwns), list(new_llm)


def compact_positions_by_category(items, gap=0.25):
    """Assign compact x-positions with a small gap only between categories."""
    if not items:
        return []

    positions = []
    x = 0.0
    prev_cat = None
    for item in items:
        cat = item.get('cat')
        if prev_cat is not None and cat != prev_cat:
            x += gap
        positions.append(x)
        x += 1.0
        prev_cat = cat
    return positions


def _first_non_empty(series):
    for val in series:
        if pd.notna(val) and val != '':
            return val
    return ''


_STATUS_PRIORITY = ['PRESENT', 'FUTURE', 'OFFSITE', 'PAST']


def deduplicate_llm_facilities(llm_df):
    """Collapse multiple LLM rows for the same facility/permit into one row."""
    df = llm_df.copy()
    for col in ['PERMIT_NUMBER', 'Facility_Name', 'Agency']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    key_cols = [c for c in ['PERMIT_NUMBER', 'Facility_Name'] if c in df.columns]
    meta_cols = [c for c in ['PERMIT_NUMBER', 'Agency', 'Facility_Name'] if c in df.columns]
    proc_cols = [c for c in df.columns if c not in meta_cols]

    if not key_cols:
        return df

    rows = []
    for _, group in df.groupby(key_cols, dropna=False, sort=False):
        out = {}
        for col in meta_cols:
            out[col] = _first_non_empty(group[col])
        for col in proc_cols:
            vals = {v for v in group[col] if pd.notna(v) and v != ''}
            out[col] = next((s for s in _STATUS_PRIORITY if s in vals), '')
        rows.append(out)

    ordered_cols = [c for c in ['PERMIT_NUMBER', 'Agency', 'Facility_Name'] if c in df.columns]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]
    return pd.DataFrame(rows).reindex(columns=ordered_cols)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _plot_stack_spec():
    """Bottom-to-top: Present, Past, Future, Offsite."""
    off_h = HATCH_PATTERNS['OFFSITE']
    return [
        ('PRESENT',  HATCH_PATTERNS['PRESENT'],  1.00),
        ('PAST',     HATCH_PATTERNS['PAST'],     1.00),
        ('FUTURE',   HATCH_PATTERNS['FUTURE'],   1.00),
        ('OFFSITE', off_h,                      0.85),
    ]


def draw_group(ax, x, bar_w, cwns_counts, llm_counts, alpha_scale=1.0):
    """Draw CWNS | LLM bars centred at position x."""
    spec = _plot_stack_spec()
    draw_stacked_bar(ax, x - bar_w / 2, bar_w, cwns_counts, COLORS['cwns'], spec, alpha_scale)
    draw_stacked_bar(ax, x + bar_w / 2, bar_w, llm_counts, COLORS['npdes_total'], spec, alpha_scale)


def set_axes(ax, labels, positions, tick_fontsize=12, ylabel_fontsize=14, rotation=45,
             ylabel='WWTP Count'):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha='right', fontsize=tick_fontsize)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis='y', which='major', labelsize=tick_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)


# ── Item-list builder for per-category plots ──────────────────────────────────
#
# Each "item" is one bar group on the x-axis. Items come in two kinds:
#   is_total=True  → "Category (Total)" – facilities with ANY leaf in that category
#   is_total=False → individual leaf process
#
# A half-position gap is inserted between JSON categories inside a combined plot.

def build_plot_items(json_cats, keywords):
    """Return (items, positions, labels) for a combined treatment-stage plot.

    items   – list of dicts with keys: label, is_total, cols, cat
    positions – x-coordinate for each item (gaps inserted between categories)
    labels  – x-tick label for each item (bold suffix for totals)
    """
    items = []
    x = 0.0
    positions = []
    labels = []

    for ci, cat_name in enumerate(json_cats):
        if cat_name not in keywords:
            continue
        leaves = get_leaf_names(cat_name, keywords[cat_name])

        # Total bar only when the category contributes >1 leaf
        if len(leaves) > 1:
            items.append({'label': f'{cat_name}\n(Total)', 'is_total': True,
                          'cols': leaves, 'cat': cat_name})
            positions.append(x);  labels.append(f'{cat_name}\n(Total)');  x += 1

        for leaf in leaves:
            items.append({'label': leaf, 'is_total': False,
                          'cols': [leaf], 'cat': cat_name})
            positions.append(x);  labels.append(leaf);  x += 1

        # Quarter-position gap between categories (except after the last one)
        if ci < len(json_cats) - 1:
            x += 0.25

    return items, positions, labels


def make_legend(ax, fontsize=12):
    """Build legend with data sources and status encoding."""
    source_handles = [
        mpatches.Patch(facecolor=COLORS['cwns'],        edgecolor='black', lw=0.5, label='CWNS'),
        mpatches.Patch(facecolor=COLORS['npdes_total'], edgecolor='black', lw=0.5, label='NPDES - LLM extraction'),
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


def draw_group_kw(ax, x, bar_w, cwns_counts, kw_counts, alpha_scale=1.0):
    """Draw CWNS | KW bars centred at position x."""
    spec = _plot_stack_spec()
    draw_stacked_bar(ax, x - bar_w / 2, bar_w, cwns_counts, COLORS['cwns'],  spec, alpha_scale)
    draw_stacked_bar(ax, x + bar_w / 2, bar_w, kw_counts,   COLORS['npdes'], spec, alpha_scale)

def make_legend_kw(ax, fontsize=12):
    """Build legend for keyword comparison."""
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


def sort_leaf_items_by_count(items, labels, cwns_counts_list, llm_counts_list):
    """Sort leaf items by total count descending within each category group; totals stay first."""
    tuples = list(zip(items, labels, cwns_counts_list, llm_counts_list))
    result = []
    i = 0
    while i < len(tuples):
        cat = tuples[i][0]['cat']
        group = []
        while i < len(tuples) and tuples[i][0]['cat'] == cat:
            group.append(tuples[i])
            i += 1
        totals = [t for t in group if t[0]['is_total']]
        leaves = [t for t in group if not t[0]['is_total']]
        leaves.sort(key=lambda t: -(total_count(t[2]) + total_count(t[3])))
        result.extend(totals + leaves)
    if not result:
        return [], [], [], []
    items_out, labels_out, cwns_out, llm_out = zip(*result)
    return list(items_out), list(labels_out), list(cwns_out), list(llm_out)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading unitprocess_keywords.json …")
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    keywords = json.load(f)

print("Loading LLM search data …")
_llm_path = f'npdes_permits/output/{DATE_FOLDER}/llm_unit_processes_by_facility.csv'
if not os.path.exists(_llm_path):
    print(f"  LLM data not found for {DATE_FOLDER} — skipping LLM")
    llm_df = pd.DataFrame()
else:
    llm_df = pd.read_csv(_llm_path)
    before_dedup = len(llm_df)
    llm_df = deduplicate_llm_facilities(llm_df)
    print(f"  LLM rows deduplicated by facility/permit: {before_dedup} -> {len(llm_df)}")
    proc_cols_llm = [c for c in llm_df.columns if c not in {'PERMIT_NUMBER', 'Agency', 'Facility_Name'}]
    past_count   = sum((llm_df[c] == 'PAST').sum() for c in proc_cols_llm)
    future_count = sum((llm_df[c] == 'FUTURE').sum() for c in proc_cols_llm)
    print(f"  LLM facilities: {len(llm_df)}  |  'PAST' (in plot stacks): {past_count}"
          f"  |  'FUTURE': {future_count}")


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

before_kw = len(kw_df)
kw_df = deduplicate_kw_facilities(kw_df, proc_cols_kw)
print(f"  Keyword facilities deduplicated: {before_kw} rows → {len(kw_df)} facilities")

print("Loading and matching CWNS data …")
mapping_tbl = load_ciwqs_to_cwns_table('npdes_permits/data/ciwqs_to_cwns.csv')
facilities_with_cwns, facilities_without_cwns = mapping_facility_cwns_sets(mapping_tbl)

ca_cwns = pd.read_csv(
    'npdes_permits/output/cwns_processes_by_facility.csv',
    low_memory=False,
    dtype={'CWNS_ID': str},
)
ca_cwns['CWNS_ID'] = ca_cwns['CWNS_ID'].astype(str).str.strip()

if not llm_df.empty:
    llm_permits = {str(x).strip() for x in llm_df['PERMIT_NUMBER'].dropna().unique()}
    llm_facilities = {permit_to_fac[p] for p in llm_permits if p in permit_to_fac}
else:
    llm_permits = set()
    llm_facilities = set()

kw_permits_raw = set(kw_df['PERMIT_NUMBER'].astype(str).str.strip())
kw_facilities  = {permit_to_fac[p] for p in kw_permits_raw if p in permit_to_fac}

# Coverage summary — before filtering to overlap only
print(f"\nFacility coverage (unique WDID + Facility_Name, before overlap filter):")
print(f"  CWNS in mapping:            {len(facilities_with_cwns)}")
print(f"  Keyword:                    {len(kw_facilities)}")
print(f"  Both CWNS + Keyword:        {len(kw_facilities & facilities_with_cwns)}")
print(f"  Keyword only (no CWNS):     {len(kw_facilities - facilities_with_cwns)}")
print(f"  CWNS only (not in KW):      {len(facilities_with_cwns - kw_facilities)}")
if not llm_df.empty:
    print(f"  LLM:                        {len(llm_facilities)}")
    print(f"  Both CWNS + LLM:            {len(llm_facilities & facilities_with_cwns)}")
    print(f"  LLM only (no CWNS):         {len(llm_facilities - facilities_with_cwns)}")
    print(f"  CWNS only (not in LLM):     {len(facilities_with_cwns - llm_facilities)}")

merged_map = merge_mapping_with_cwns_processes(mapping_tbl, ca_cwns)
n_attach = int((merged_map['_cwns_merge'] == 'both').sum())
print(f"\n  CIWQS mapping rows with CWNS survey attach: {n_attach} / {len(merged_map)}")

fac_col = merged_map.apply(lambda r: (r['WDID'], r['Facility_Name']), axis=1)
declared_cw = merged_map['CWNS_ID'].apply(lambda x: bool(str(x).strip() and str(x).strip().upper() != 'NA'))
target_facs = (llm_facilities | kw_facilities) & facilities_with_cwns
slice_map = merged_map.loc[declared_cw & fac_col.isin(target_facs)]
_proc_cols = [c for c in cwns_process_column_names(ca_cwns) if c in slice_map.columns]
cwns_df = union_cwns_processes(slice_map, _proc_cols)
cwns_df['PERMIT_NUMBER'] = cwns_df['NPDES_No'].astype(str).str.strip()

# Save unmatched LLM permits (no CWNS counterpart declared) for manual review
if not llm_df.empty:
    unmatched_facs = llm_facilities - facilities_with_cwns - facilities_without_cwns
    unmatched_npdes = sorted(p for p, f in permit_to_fac.items() if f in unmatched_facs)
    if unmatched_npdes:
        unmatched_path = f'npdes_permits/output/{DATE_FOLDER}/unmatched_npdes_no_cwns.csv'
        (llm_df[llm_df['PERMIT_NUMBER'].astype(str).isin(unmatched_npdes)]
         [['PERMIT_NUMBER', 'Facility_Name']].drop_duplicates()
         .to_csv(unmatched_path, index=False))
        print(f"  Unmatched LLM (no CWNS): {len(unmatched_npdes)} → {os.path.basename(unmatched_path)}")

# Filter LLM/KW to only facilities with a declared CWNS
matched_permits = {p for p, f in permit_to_fac.items() if f in facilities_with_cwns}
if not llm_df.empty:
    llm_df = llm_df[llm_df['PERMIT_NUMBER'].astype(str).isin(matched_permits)].copy()
kw_df = kw_df[kw_df['PERMIT_NUMBER'].isin(matched_permits)].copy()

N_CWNS = len(cwns_df)
N_LLM  = len(llm_df)
N_KW   = len(kw_df)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Treatment-stage groupings for per-category plots ─────────────────────────
# Each entry: plot_title → [list of top-level JSON category names to combine]
# Leaf processes from all listed categories are shown together on one plot,
# with shaded bands separating categories that have multiple leaves.
PLOT_GROUPS = {
    'Primary Treatment': [
        'Screening/Microstrainer', 'Comminution', 'Grit Removal',
        'Equalization', 'Flotation',
    ],
    'Clarification': ['Clarification'],
    'Secondary Treatment': ['Activated Sludge', 'Lagoon'],
    'Nutrient Removal': ['Nutrient Removal'],
    'Filtration': ['Filtration'],
    'Disinfection': ['Disinfection'],
    'Chemical Treatment': ['Coagulation', 'Flocculation', 'Chemical Addition'],
    'Advanced Treatment': ['Ion Exchange', 'Activated Carbon', 'UV-AOP', 'Wetland'],
    'Solids Processing': ['Anaerobic Digestion', 'Aerobic Digestion'],
}


# ── 1. Per-treatment-stage plots ──────────────────────────────────────────────

print("\nGenerating per-treatment-stage plots …")
bar_w = 0.35

for group_title, json_cats in PLOT_GROUPS.items():
    items, positions, labels = build_plot_items(json_cats, keywords)

    if not items:
        continue

    # Compute counts for each item
    cwns_counts_list, llm_counts_list = [], []
    for item in items:
        cwns_counts_list.append(get_facility_counts(cwns_df, item['cols']))
        llm_counts_list.append(get_facility_counts(llm_df, item['cols']))

    items, positions, labels, cwns_counts_list, llm_counts_list = filter_zero_leaf_items(
        items, positions, labels, cwns_counts_list, llm_counts_list
    )
    items, labels, cwns_counts_list, llm_counts_list = sort_leaf_items_by_count(
        items, labels, cwns_counts_list, llm_counts_list
    )
    positions = compact_positions_by_category(items, gap=0.25)

    if not items or max(total_count(c) for c in cwns_counts_list + llm_counts_list) == 0:
        print(f"  {group_title}: all zeros, skipping")
        continue

    # Figure width based on number of bar groups (positions span)
    x_span = positions[-1] - positions[0] + 1 if positions else 1
    fig_w = max(7, x_span * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    for item, pos, cwns_c, llm_c in zip(items, positions, cwns_counts_list, llm_counts_list):
        alpha = 1.0 if item['is_total'] else 0.60
        draw_group(ax, pos, bar_w, cwns_c, llm_c, alpha_scale=alpha)

    set_axes(ax, labels, positions)

    # Shade category bands using item position info
    # Build band spans: [first_pos, last_pos] per cat
    cat_spans = {}   # cat_name → [min_pos, max_pos]
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
        # Only label the band if there are multiple leaves (otherwise x-tick already shows it)
        leaves_in_cat = [it for it in items if it['cat'] == cat and not it['is_total']]
        if len(leaves_in_cat) > 1:
            ax.text((s + e) / 2, ylim[1] * 0.97, cat,
                    ha='center', va='top', fontsize=11, color='#444444', style='italic')
        first = False
    ax.set_ylim(ylim)
    make_legend(ax)
    plt.tight_layout()

    safe = group_title.replace('/', '_').replace(' ', '_')
    path = f'{OUTPUT_DIR}/{safe}_source_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ── 2. Major-categories plot ─────────────────────────────────────────────────
# One bar group per top-level JSON category. Stacks sum status counts over all
# leaves in that category (same rule as combined treatment-stage bars).

print("\nGenerating major-categories plots …")

# REPLACE WITH:
for comparison_type in ['llm', 'kw']:
    if comparison_type == 'llm' and llm_df.empty:
        continue
    print(f"\n  Generating {comparison_type.upper()} major-categories plot…")
    comparison_df = llm_df if comparison_type == 'llm' else kw_df
    draw_fn = draw_group if comparison_type == 'llm' else draw_group_kw
    legend_fn = make_legend if comparison_type == 'llm' else make_legend_kw
    suffix = 'source_comparison' if comparison_type == 'llm' else 'kw_compare'
    
    cat_labels = list(keywords.keys())

    all_cwns = []
    all_comp  = []
    for cat_name, cat_val in keywords.items():
        leaves = get_leaf_names(cat_name, cat_val)
        all_cwns.append(get_facility_counts(cwns_df, leaves))
        all_comp.append(get_facility_counts(comparison_df, leaves))

    cat_filtered = [
        (lbl, cwns_c, comp_c)
        for lbl, cwns_c, comp_c in zip(cat_labels, all_cwns, all_comp)
        if total_count(cwns_c) >= MIN_COUNT or total_count(comp_c) >= MIN_COUNT
    ]
    if cat_filtered:
        cat_labels, all_cwns, all_comp = map(list, zip(*cat_filtered))

    order = sorted(range(len(cat_labels)),
                   key=lambda i: -(total_count(all_cwns[i]) + total_count(all_comp[i])))
    cat_labels = [cat_labels[i] for i in order]
    all_cwns   = [all_cwns[i]   for i in order]
    all_comp   = [all_comp[i]   for i in order]

    n = len(cat_labels)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))

    for i, (cwns_c, comp_c) in enumerate(zip(all_cwns, all_comp)):
        draw_fn(ax, i, bar_w, cwns_c, comp_c)

    set_axes(ax, cat_labels, list(range(n)), rotation=45)
    legend_fn(ax)

    final_dir = f'npdes_permits/output/{DATE_FOLDER}/final'
    os.makedirs(final_dir, exist_ok=True)
    path = f'{final_dir}/figure_4_major_categories_{suffix}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {os.path.basename(path)}")
