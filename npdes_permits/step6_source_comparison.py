"""
step6_source_comparison_comparison.py

Grouped stacked bar-chart comparison of unit process detection across two data sources:
  - CWNS (California facilities from output/unit_processes_by_facility.csv)
  - LLM Search (output/<DATE>/llm_unit_processes_by_facility.csv)

Each bar is stacked by status:
  CWNS : present (solid) | present_and_future (xxx) | future (//)
  LLM  : present (solid) | future (//)              | off_site (transparent)
  'past' (LLM) : excluded from display.

Produces:
  1. One graph per major category (all leaf processes in that category)
  2. One graph of all processes (all leaves from unitprocess_keywords.json)
  3. One graph of major categories only (each top-level key, facility counted once)
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
from helpers.utils import extract_leaves, prepare_cwns_ca, match_cwns_to_npdes, get_leaf_names
from helpers.plotting import COLORS, HATCH_PATTERNS, draw_stacked_bar, make_source_status_legend

DATE_FOLDER = '2026-2-18'
OUTPUT_DIR  = f'npdes_permits/output/{DATE_FOLDER}/figures'
MIN_COUNT   = 15  # drop bar groups where both sources are below this threshold


# ── Category background palette (all-processes overview) ──────────────────────
CATEGORY_BG_COLORS = [
    '#f0f4ff', '#fff8f0', '#f0fff4', '#fff0f8',
    '#f8f0ff', '#fffff0', '#f0ffff', '#fff4f0',
    '#f4fff0', '#f0f8ff', '#fff0f0', '#f0fff8',
    '#fff0ff', '#fffff8', '#f8fff0', '#f0f0ff',
    '#fff8ff', '#f8f8f0', '#f0f8f8',
]

# ── Status visual encoding ────────────────────────────────────────────────────
#  (status_key, hatch, alpha)  – stacked bottom-to-top
CWNS_STACK = [
    ('present',            HATCH_PATTERNS['present'],            1.00),
    ('present_and_future', HATCH_PATTERNS['present_and_future'], 1.00),
    ('future',             HATCH_PATTERNS['future'],             1.00),
]
LLM_STACK = [
    ('present',  HATCH_PATTERNS['present'],  1.00),
    ('future',   HATCH_PATTERNS['future'],   1.00),
    ('off_site', '',                          0.30),  # transparent – off-site treatment
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def _best_cwns_status(series):
    """Aggregate multiple CWNS rows for one CWNS_ID: keep the highest-priority status."""
    vals = set(series.astype(str).str.lower())
    for s in ('present', 'present_and_future', 'future'):
        if s in vals:
            return s
    return '0'


def get_cwns_kw_counts(df, col):
    """Per-process status counts for CWNS or keyword data."""
    empty = {'present': 0, 'present_and_future': 0, 'future': 0}
    if col not in df.columns:
        return empty
    s = df[col].astype(str).str.lower()
    return {
        'present':            int((s == 'present').sum()),
        'present_and_future': int((s == 'present_and_future').sum()),
        'future':             int((s == 'future').sum()),
    }


def get_llm_counts(df, col):
    """Per-process status counts for LLM data."""
    empty = {'present': 0, 'future': 0, 'off_site': 0}
    if col not in df.columns:
        return empty
    s = df[col].astype(str).str.lower()
    return {
        'present':  int((s == 'present').sum()),
        'future':   int((s == 'future').sum()),
        'off_site': int((s == 'off_site').sum()),
    }


def _any_flag(df, cols, statuses):
    """Boolean Series: True if any col in cols has any value in statuses."""
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().isin(statuses)
    return mask


def get_cwns_kw_category_counts(df, leaf_cols):
    """Category-level status counts for CWNS/keyword. Each facility counted once at highest priority."""
    has_present = _any_flag(df, leaf_cols, {'present'})
    has_paf     = _any_flag(df, leaf_cols, {'present_and_future'})
    has_future  = _any_flag(df, leaf_cols, {'future'})
    return {
        'present':            int(has_present.sum()),
        'present_and_future': int((has_paf & ~has_present).sum()),
        'future':             int((has_future & ~has_present & ~has_paf).sum()),
    }


def get_llm_category_counts(df, leaf_cols):
    """Category-level status counts for LLM. 'future' shown separately as dashed."""
    has_present = _any_flag(df, leaf_cols, {'present'})
    has_future  = _any_flag(df, leaf_cols, {'future'})
    has_offsite = _any_flag(df, leaf_cols, {'off_site'})
    return {
        'present':  int(has_present.sum()),
        'future':   int((has_future & ~has_present).sum()),
        'off_site': int((has_offsite & ~has_present & ~has_future).sum()),
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
    for val in series.astype(str):
        text = val.strip()
        if text and text.lower() != 'nan':
            return text
    return ''


def _merge_llm_status(series):
    vals = set(series.astype(str).str.strip().str.lower())
    if 'present' in vals:
        return 'present'
    if 'future' in vals:
        return 'future'
    if 'off_site' in vals:
        return 'off_site'
    if 'past' in vals:
        return 'past'
    return ''


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
            out[col] = _merge_llm_status(group[col])
        rows.append(out)

    ordered_cols = [c for c in ['PERMIT_NUMBER', 'Agency', 'Facility_Name'] if c in df.columns]
    ordered_cols += [c for c in df.columns if c not in ordered_cols]
    return pd.DataFrame(rows).reindex(columns=ordered_cols)


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

def get_all_leaves_ordered(keywords):
    result = []
    for cat_name, cat_val in keywords.items():
        for leaf in get_leaf_names(cat_name, cat_val):
            result.append((leaf, cat_name))
    return result


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_group(ax, x, bar_w, cwns_counts, llm_counts, alpha_scale=1.0):
    """Draw CWNS | LLM bars centred at position x."""
    draw_stacked_bar(ax, x - bar_w / 2, bar_w, cwns_counts, COLORS['cwns'],        CWNS_STACK, alpha_scale)
    draw_stacked_bar(ax, x + bar_w / 2, bar_w, llm_counts,  COLORS['npdes_total'], LLM_STACK,  alpha_scale)


def set_axes(ax, labels, positions, fontsize=9, rotation=45):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha='right', fontsize=fontsize)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylabel('WWTP Count', fontsize=11)


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


def make_legend(ax, fontsize=10):
    """Build legend with data sources and status encoding."""
    source_handles = [
        mpatches.Patch(facecolor=COLORS['cwns'],        edgecolor='black', lw=0.5, label='CWNS'),
        mpatches.Patch(facecolor=COLORS['npdes_total'], edgecolor='black', lw=0.5, label='NPDES - LLM extraction'),
    ]
    status_handles = [
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['present'],            edgecolor='black', lw=0.5, label='Present'),
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['present_and_future'], edgecolor='black', lw=0.5, label='Present & Future'),
        mpatches.Patch(facecolor='grey', hatch=HATCH_PATTERNS['future'],             edgecolor='black', lw=0.5, label='Future'),
        mpatches.Patch(facecolor='grey', hatch='',                                   edgecolor='black', lw=0.5, alpha=0.30, label='Off-site (LLM)'),
    ]
    return make_source_status_legend(ax, source_handles, status_handles, fontsize=fontsize)


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
llm_df = pd.read_csv(f'npdes_permits/output/llm_unit_processes_by_facility.csv')
before_dedup = len(llm_df)
llm_df = deduplicate_llm_facilities(llm_df)
after_dedup = len(llm_df)
print(f"  LLM rows deduplicated by facility/permit: {before_dedup} -> {after_dedup}")
proc_cols_llm = [c for c in llm_df.columns if c not in {'PERMIT_NUMBER', 'Agency', 'Facility_Name'}]
past_count   = sum((llm_df[c].astype(str).str.lower() == 'past').sum()   for c in proc_cols_llm)
future_count = sum((llm_df[c].astype(str).str.lower() == 'future').sum() for c in proc_cols_llm)
print(f"  LLM facilities: {len(llm_df)}  |  'past' (excluded): {past_count}"
      f"  |  'future': {future_count}")

print("Loading and matching CWNS data (CA only) …")
cwns_raw = pd.read_csv('npdes_permits/output/unit_processes_by_facility.csv',
                        low_memory=False, dtype={'CWNS_ID': str})

# Build CWNS CA dataset with properly resolved NPDES permit numbers
ca_cwns = prepare_cwns_ca(
    cwns_raw,
    'npdes_permits/data/cwns/cwns_permits_match_manual.csv',
    'npdes_permits/data/cwns/cwns_facility_name_match_manual.csv',
)
print(f"  CA CWNS facilities (consolidated): {len(ca_cwns)}")

# Match CWNS facilities to NPDES permits (3-tier: official permit → raw list → name)
llm_permits = set(llm_df['PERMIT_NUMBER'].dropna().astype(str).unique())
print(f"  LLM permits: {len(llm_permits)}")

ca_cwns = match_cwns_to_npdes(ca_cwns, llm_permits)
cwns_df = ca_cwns[ca_cwns['matched']].copy()
matched_permits = set(cwns_df['linking_permit'].dropna().astype(str))
print(f"  CWNS facilities matched: {len(cwns_df)} / {len(ca_cwns)}")

# Filter LLM to only matched permit numbers
llm_df = llm_df[llm_df['PERMIT_NUMBER'].astype(str).isin(matched_permits)].copy()
print(f"  LLM rows after filter: {len(llm_df)}")

N_CWNS = len(cwns_df)
N_LLM  = len(llm_df)

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
    'Solids Processing': ['Solids Processing'],
}


def shade_cat_bands(ax, leaf_cats, ylim, fontsize=8):
    """Shade alternating bands per source-category within a combined plot."""
    unique = list(dict.fromkeys(leaf_cats))
    palette = CATEGORY_BG_COLORS
    bg = {c: palette[i % len(palette)] for i, c in enumerate(unique)}

    prev, start = None, 0
    blocks = []
    for i, c in enumerate(leaf_cats):
        if c != prev:
            if prev is not None:
                blocks.append((start, i - 1, prev))
            start, prev = i, c
    blocks.append((start, len(leaf_cats) - 1, prev))

    for s, e, cat in blocks:
        ax.axvspan(s - 0.5, e + 0.5, facecolor=bg[cat], alpha=0.30, zorder=0)
        if e > s:          # only label multi-leaf bands
            ax.text((s + e) / 2, ylim[1] * 0.97, cat,
                    ha='center', va='top', fontsize=fontsize,
                    color='#444444', style='italic')
        if s > 0:
            ax.axvline(s - 0.5, color='#999999', lw=0.7, linestyle='--', zorder=1)


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
        if item['is_total']:
            cwns_counts_list.append(get_cwns_kw_category_counts(cwns_df, item['cols']))
            llm_counts_list.append( get_llm_category_counts(llm_df,      item['cols']))
        else:
            col = item['cols'][0]
            cwns_counts_list.append(get_cwns_kw_counts(cwns_df, col))
            llm_counts_list.append( get_llm_counts(llm_df,      col))

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

    set_axes(ax, labels, positions, fontsize=9)

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
    unique_cats_in_plot = list(dict.fromkeys(item['cat'] for item in items))
    cat_bg = {cat: CATEGORY_BG_COLORS[i % len(CATEGORY_BG_COLORS)]
              for i, cat in enumerate(unique_cats_in_plot)}
    first = True
    for cat, (s, e) in cat_spans.items():
        ax.axvspan(s - 0.5, e + 0.5, facecolor=cat_bg[cat], alpha=0.25, zorder=0)
        if not first:
            ax.axvline(s - 0.5, color='#999999', lw=0.8, linestyle='--', zorder=1)
        # Only label the band if there are multiple leaves (otherwise x-tick already shows it)
        leaves_in_cat = [it for it in items if it['cat'] == cat and not it['is_total']]
        if len(leaves_in_cat) > 1:
            ax.text((s + e) / 2, ylim[1] * 0.97, cat,
                    ha='center', va='top', fontsize=8, color='#444444', style='italic')
        first = False
    ax.set_ylim(ylim)

    ax.set_title(
        f'{group_title}  –  CWNS vs LLM',
        fontsize=12)
    make_legend(ax, fontsize=9)
    plt.tight_layout()

    safe = group_title.replace('/', '_').replace(' ', '_')
    path = f'{OUTPUT_DIR}/{safe}_source_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ── 2. All-processes plot ─────────────────────────────────────────────────────

print("\nGenerating all-processes plot …")
all_leaves  = get_all_leaves_ordered(keywords)
leaf_labels = [name for name, _ in all_leaves]
leaf_cats   = [cat  for _, cat  in all_leaves]

all_cwns = [get_cwns_kw_counts(cwns_df, p) for p in leaf_labels]
all_llm  = [get_llm_counts(llm_df,       p) for p in leaf_labels]

filtered = [
    (label, cat, cwns_c, llm_c)
    for label, cat, cwns_c, llm_c in zip(leaf_labels, leaf_cats, all_cwns, all_llm)
    if total_count(cwns_c) >= MIN_COUNT or total_count(llm_c) >= MIN_COUNT
]

if not filtered:
    print("  All Processes: all below threshold, skipping")
else:
    leaf_labels, leaf_cats, all_cwns, all_llm = map(list, zip(*filtered))
    n = len(leaf_labels)

    fig, ax = plt.subplots(figsize=(max(14, n * 0.38), 7))

    for i, (cwns_c, llm_c) in enumerate(zip(all_cwns, all_llm)):
        draw_group(ax, i, bar_w, cwns_c, llm_c)

    set_axes(ax, leaf_labels, list(range(n)), fontsize=7, rotation=55)
    ax.set_title(
        f'All Processes  –  CWNS vs LLM',
        fontsize=14)
    make_legend(ax, fontsize=9)

    # Shaded category bands
    unique_cats = list(dict.fromkeys(leaf_cats))
    cat_bg = {cat: CATEGORY_BG_COLORS[i % len(CATEGORY_BG_COLORS)]
              for i, cat in enumerate(unique_cats)}

    prev_cat, block_start = None, 0
    blocks = []
    for i, cat in enumerate(leaf_cats):
        if cat != prev_cat:
            if prev_cat is not None:
                blocks.append((block_start, i - 1, prev_cat))
            block_start, prev_cat = i, cat
    blocks.append((block_start, n - 1, prev_cat))

    ylim = ax.get_ylim()
    for start, end, cat in blocks:
        ax.axvspan(start - 0.5, end + 0.5, facecolor=cat_bg[cat], alpha=0.35, zorder=0)
        ax.text((start + end) / 2, ylim[1] * 0.98, cat,
                ha='center', va='top', fontsize=6.5, color='#333333', style='italic')
        if start > 0:
            ax.axvline(start - 0.5, color='#888888', linewidth=0.7, linestyle='--', zorder=1)
    ax.set_ylim(ylim)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/all_processes_source_comparison.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ── 3. Major-categories plot ──────────────────────────────────────────────────

print("\nGenerating major-categories plot …")
cat_labels = list(keywords.keys())

all_cwns = []
all_llm  = []
for cat_name, cat_val in keywords.items():
    leaves = get_leaf_names(cat_name, cat_val)
    all_cwns.append(get_cwns_kw_category_counts(cwns_df, leaves))
    all_llm.append( get_llm_category_counts(llm_df,      leaves))

cat_filtered = [
    (lbl, cwns_c, llm_c)
    for lbl, cwns_c, llm_c in zip(cat_labels, all_cwns, all_llm)
    if total_count(cwns_c) >= MIN_COUNT or total_count(llm_c) >= MIN_COUNT
]
if cat_filtered:
    cat_labels, all_cwns, all_llm = map(list, zip(*cat_filtered))

n = len(cat_labels)
fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 5))

for i, (cwns_c, llm_c) in enumerate(zip(all_cwns, all_llm)):
    draw_group(ax, i, bar_w, cwns_c, llm_c)

set_axes(ax, cat_labels, list(range(n)), fontsize=9)
ax.set_title(
    f'Major Categories  –  CWNS vs LLM',
    fontsize=12)
make_legend(ax, fontsize=9)
plt.tight_layout()

path = f'{OUTPUT_DIR}/major_categories_source_comparison.png'
plt.savefig(path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {os.path.basename(path)}")

print("\nDone.")
