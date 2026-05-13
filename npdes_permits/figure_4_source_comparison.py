"""
Grouped stacked bar-chart comparison of unit process detection across two data sources:
  - CWNS (California facilities from output/cwns_processes_by_facility.csv)
  - LLM Search (output/date/llm_unit_processes_by_facility.csv)

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
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from helpers.utils import get_leaf_names, PRESENT_STATUSES, get_unspecified_leaf_names
from helpers.utils import (
    cwns_mapping,
    no_cwns_pids,
    build_cwns_facility_processes,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, make_grouped_legend, save_and_close, set_thick_spines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATE_FOLDER = "2026-4-26"
DATA_DIR = f"npdes_permits/output/{DATE_FOLDER}"
OUTPUT_DIR = f"npdes_permits/output/{DATE_FOLDER}/figures"
MIN_COUNT = 20  # drop bar groups where both sources are below this threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_GROUPS = {
    "Primary Treatment": [
        "Screening/Microstrainer",
        "Comminution",
        "Grit Removal",
        "Equalization",
        "Flotation",
    ],
    "Clarification": ["Clarification"],
    "Secondary Treatment": ["Activated Sludge", "Lagoon"],
    "Nutrient Removal": ["Nutrient Removal"],
    "Filtration": ["Filtration"],
    "Disinfection": ["Disinfection"],
    "Chemical Treatment": ["Coagulation", "Flocculation", "Chemical Addition"],
    "Advanced Treatment": ["Ion Exchange", "Activated Carbon", "UV-AOP", "Wetland"],
    "Solids Processing": ["Anaerobic Digestion", "Aerobic Digestion"],
}

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
    has_past = _any_flag(df, leaf_cols, {"PAST"})
    has_future = _any_flag(df, leaf_cols, {"FUTURE"})
    has_offsite = _any_flag(df, leaf_cols, {"OFFSITE"})
    present_count = int(has_present.sum())
    past_count = int((has_past & ~has_present).sum())
    future_count = int((has_future & ~has_present).sum())
    offsite_count = int((has_offsite & ~has_present & ~has_future).sum())
    not_present_count = len(df) - (present_count + past_count + future_count + offsite_count)
    return {
        "PRESENT": present_count,
        "PAST": past_count,
        "FUTURE": future_count,
        "OFFSITE": offsite_count,
        "NOT_PRESENT": not_present_count,
    }


def draw_stacked_bar(ax, xpos, width, counts, color, stack_order):
    bottom = 0
    for key, hatch in stack_order:
        val = counts.get(key, 0)
        if val > 0:
            facecolor = "white" if key == "NOT_PRESENT" else color
            base = ax.bar(
                xpos,
                val,
                width,
                bottom=bottom,
                color=facecolor,
                edgecolor="black",
                linewidth=1.2,
            )
            if hatch:
                hatch_bar = ax.bar(
                    xpos,
                    val,
                    width,
                    bottom=bottom,
                    color="none",
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.0,
                )
                for patch in hatch_bar:
                    patch.set_hatch_linewidth(1.0)
            bottom += val


def set_standard_axes(
    ax,
    labels,
    positions,
    tick_fontsize=12,
    ylabel_fontsize=14,
    rotation=45,
    ylabel="WWTP Count",
):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=tick_fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", which="major", labelsize=tick_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    set_thick_spines(ax, linewidth=1.6)


def render_source_plot(
    ax,
    labels,
    positions,
    source_counts,
    source_items,
    stack_order,
    status_legend_items,
    bar_width,
):
    n_sources = len(source_counts)
    offsets = [(idx - (n_sources - 1) / 2) * bar_width for idx in range(n_sources)]

    for pos, *counts in zip(positions, *source_counts):
        for count, (_, color_key), offset in zip(counts, source_items, offsets):
            draw_stacked_bar(ax, pos + offset, bar_width, count, COLORS[color_key], stack_order)
    set_standard_axes(ax, labels, positions, rotation=45)
    make_grouped_legend(
        ax,
        groups=[
            {
                "header": "Data Source",
                "items": [
                    (label, {"facecolor": COLORS[color_key]}) for label, color_key in source_items
                ],
            },
            {"header": "Status", "items": status_legend_items},
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=12,
    )


def build_major_category_sources(common_facilities, include_kw=False):
    cwns_compare_df = cwns_df[cwns_df["Place ID"].isin(common_facilities)].copy()
    llm_compare_df = llm_df[llm_df["Place ID"].isin(common_facilities)].copy()
    kw_compare_df = kw_df[kw_df["Place ID"].isin(common_facilities)].copy() if include_kw else None

    cat_labels = []
    cwns_counts = []
    llm_counts = []
    kw_counts = []

    for cat_name, cat_val in keywords.items():
        leaves = get_leaf_names(cat_name, cat_val)
        cat_labels.append(cat_name)
        cwns_counts.append(get_facility_counts(cwns_compare_df, leaves))
        llm_counts.append(get_facility_counts(llm_compare_df, leaves))
        if include_kw:
            kw_counts.append(get_facility_counts(kw_compare_df, leaves))

    if include_kw:
        kept = [
            (label, cwns_c, llm_c, kw_c)
            for label, cwns_c, llm_c, kw_c in zip(cat_labels, cwns_counts, llm_counts, kw_counts)
            if sum(cwns_c.values()) >= MIN_COUNT
            or sum(llm_c.values()) >= MIN_COUNT
            or sum(kw_c.values()) >= MIN_COUNT
        ]
        if kept:
            cat_labels, cwns_counts, llm_counts, kw_counts = map(list, zip(*kept))
        order = sorted(range(len(cat_labels)), key=lambda i: (llm_counts[i]["NOT_PRESENT"], cat_labels[i]))
        return [cat_labels[i] for i in order], [cwns_counts[i] for i in order], [llm_counts[i] for i in order], [kw_counts[i] for i in order]

    kept = [
        (label, cwns_c, llm_c)
        for label, cwns_c, llm_c in zip(cat_labels, cwns_counts, llm_counts)
        if sum(cwns_c.values()) >= MIN_COUNT or sum(llm_c.values()) >= MIN_COUNT
    ]
    if kept:
        cat_labels, cwns_counts, llm_counts = map(list, zip(*kept))
    order = sorted(range(len(cat_labels)), key=lambda i: (llm_counts[i]["NOT_PRESENT"], cat_labels[i]))
    return [cat_labels[i] for i in order], [cwns_counts[i] for i in order], [llm_counts[i] for i in order], []


# ── Item-list builder for per-category plots ──────────────────────────────────
def build_sorted_plot_items(json_cats, keywords, cwns_df, llm_df):
    """Build leaf items, counts, filter by threshold; sort leaves by NPDES NOT_PRESENT (asc)."""
    items = []
    labels = []
    cwns_counts_list, llm_counts_list = [], []
    for cat_name in json_cats:
        if cat_name not in keywords:
            continue
        leaves = [l for l in get_leaf_names(cat_name, keywords[cat_name]) if l not in excluded_unspecified]
        for leaf in leaves:
            items.append({"label": leaf, "cols": [leaf], "cat": cat_name})
            labels.append(leaf)
            cwns_counts_list.append(get_facility_counts(cwns_df, [leaf]))
            llm_counts_list.append(get_facility_counts(llm_df, [leaf]))
    kept = [
        (item, label, cwns_c, llm_c)
        for item, label, cwns_c, llm_c in zip(items, labels, cwns_counts_list, llm_counts_list)
        if sum(cwns_c.values()) >= MIN_COUNT or sum(llm_c.values()) >= MIN_COUNT
    ]
    if not kept:
        return [], [], [], []
    tuples = list(kept)
    result = []
    i = 0
    while i < len(tuples):
        cat = tuples[i][0]["cat"]
        group = []
        while i < len(tuples) and tuples[i][0]["cat"] == cat:
            group.append(tuples[i])
            i += 1
        # X order: lowest → highest NPDES (right bar) "Not Present"
        group.sort(key=lambda t: (t[3]["NOT_PRESENT"], t[1]))
        result.extend(group)
    items_out, labels_out, cwns_out, llm_out = zip(*result)
    return list(items_out), list(labels_out), list(cwns_out), list(llm_out)


# ── Load data ─────────────────────────────────────────────────────────────────

with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
    keywords = json.load(f)

excluded_unspecified = get_unspecified_leaf_names(keywords)

all_leaf_processes = {
    leaf
    for cat_name, cat_val in keywords.items()
    for leaf in get_leaf_names(cat_name, cat_val)
}
proc_cols = sorted(all_leaf_processes)

llm_df = pd.read_csv(
    f"{DATA_DIR}/llm_unit_processes_by_facility.csv", dtype=str
    )

kw_df = pd.read_csv(
    f"{DATA_DIR}/kw_unit_processes_by_facility.csv", dtype=str
    )

ca_cwns = pd.read_csv(
    "npdes_permits/output/cwns_processes_by_facility.csv",
    dtype=str,
)
ca_cwns["CWNS_ID"] = ca_cwns["CWNS_ID"].str.strip()

llm_facilities = set(llm_df["Place ID"])
kw_facilities = set(kw_df["Place ID"])
cwns_pids = set(cwns_mapping["Place ID"])

# Coverage summary — before filtering to overlap only
print(f"\nFacility coverage (unique WDID + Facility Name, before overlap filter):")
print(f"  CWNS in mapping:            {len(cwns_pids)}")
print(f"  Keyword:                    {len(kw_facilities)}")
print(f"  Both CWNS + Keyword:        {len(kw_facilities & cwns_pids)}")
print(f"  Keyword only (no CWNS):     {len(kw_facilities - cwns_pids)}")
print(f"  CWNS only (not in KW):      {len(cwns_pids - kw_facilities)}")
print(f"  LLM:                        {len(llm_facilities)}")
print(f"  Both CWNS + LLM:            {len(llm_facilities & cwns_pids)}")
print(f"  All 3 sources:              {len(llm_facilities & kw_facilities & cwns_pids)}")

cwns_df, merged_map = build_cwns_facility_processes(ca_cwns, target_facilities=llm_facilities | kw_facilities)

n_attach = int((merged_map["_cwns_merge"] == "both").sum())
print(f"\n  CIWQS mapping rows with CWNS survey attach: {n_attach} / {len(merged_map)}")

# Save facilities with no CWNS match
site_data_path = f"npdes_permits/output/{DATE_FOLDER}/site_data.csv"
pid_to_name = {}
_site = pd.read_csv(site_data_path, dtype=str).fillna("")
site_facs = set(_site["Place ID"]) - {""}
for _, r in _site.iterrows():
    pid = r["Place ID"]
    if pid:
        pid_to_name.setdefault(pid, r["Facility Name"])
for _, r in kw_df.iterrows():
    pid = r["Place ID"]
    if pid:
        pid_to_name.setdefault(pid, r["Facility Name"])
candidate_facs = (kw_facilities | site_facs) - {""}

unmatched_pids = [
    pid for pid in sorted(candidate_facs)
    if pid not in cwns_pids and pid not in no_cwns_pids
]

kw_has_data = set(
    kw_df.loc[(kw_df[proc_cols].ne("")).any(axis=1), "Place ID"]
)
unmatched_df = pd.DataFrame({
    "Place ID": unmatched_pids,
    "FACILITY_NAME": [pid_to_name.get(pid, "") for pid in unmatched_pids],
    "has_kw_unit_process_data": ["yes" if pid in kw_has_data else "no" for pid in unmatched_pids],
}).sort_values("has_kw_unit_process_data", ascending=True, key=lambda s: s.map({"yes": 0, "no": 1}))
unmatched_df.to_csv(f"{DATA_DIR}/unmatched_kw_no_cwns.csv", index=False)
print(f"  Unmatched KW/site_data (no CWNS): {len(unmatched_pids)} → unmatched_kw_no_cwns.csv")


# CWNS rows with no declared match in ciwqs_to_cwns (by CWNS_ID)
cwns_csv = Path("npdes_permits/output/cwns_processes_by_facility.csv")
mapping_csv = Path("npdes_permits/data/ciwqs_to_cwns.csv")
cwns_unmatched_df = pd.read_csv(cwns_csv, dtype=str).fillna("")
mapping_df = pd.read_csv(mapping_csv, dtype=str, keep_default_na=False).fillna("")

mapped_cwns_ids = {
    cw.strip()
    for cw in mapping_df["CWNS_ID"]
    if cw.strip() and cw.strip().upper() != "NA"
}
cwns_ids = cwns_unmatched_df["CWNS_ID"].str.strip()
_unmatched_cols = [c for c in ["CWNS_ID", "FACILITY_ID", "FACILITY_NAME"] if c in cwns_unmatched_df.columns]
unmatched_cwns_no_kw = cwns_unmatched_df[
    cwns_ids.ne("") & cwns_ids.str.upper().ne("NA") & ~cwns_ids.isin(mapped_cwns_ids)
][_unmatched_cols].drop_duplicates()

unmatched_cwns_no_kw.to_csv(f"{DATA_DIR}/unmatched_cwns_no_kw.csv", index=False)


cwns_facilities_all = set(cwns_df["Place ID"])
kw_df, llm_df = [df[df["Place ID"].isin(cwns_facilities_all)].copy() for df in [kw_df, llm_df]]
llm_common_facilities, kw_common_facilities = [set(df["Place ID"]) for df in [llm_df, kw_df]]

# ── Treatment-stage groupings for per-category plots ─────────────────────────
# Each entry: plot_title → [list of top-level JSON category names to combine]
# Leaf processes from all listed categories are shown together on one plot,
# with shaded bands separating categories that have multiple leaves.


# ── 1. Per-treatment-stage plots ──────────────────────────────────────────────

bar_w = 0.35
status_order = ["PRESENT", "PAST", "FUTURE", "OFFSITE"]
stack_order = [(status, HATCH_PATTERNS.get(status, "")) for status in status_order]
status_order_with_not_present = ["PRESENT", "PAST", "FUTURE", "OFFSITE", "NOT_PRESENT"]
stack_order_with_not_present = [
    (status, HATCH_PATTERNS.get(status, "")) for status in status_order_with_not_present
]
status_legend_items = [
    (status.title(), {
        "facecolor": "grey", 
        "hatch": HATCH_PATTERNS.get(status, ""),
        "edgecolor": "white" if HATCH_PATTERNS.get(status, "") else "black"
    })
    for status in status_order
]
status_legend_items_with_not_present = status_legend_items + [
    ("Not Present", {"facecolor": "white", "edgecolor": "black"}),
]

for group_title, json_cats in PLOT_GROUPS.items():
    cwns_plot_df = cwns_df[cwns_df["Place ID"].isin(llm_common_facilities)].copy()
    llm_plot_df = llm_df[llm_df["Place ID"].isin(llm_common_facilities)].copy()
    items, labels, cwns_counts_list, llm_counts_list = build_sorted_plot_items(
        json_cats, keywords, cwns_plot_df, llm_plot_df
    )
    if not items:
        print(f"  {group_title}: all below threshold, skipping")
        continue
    positions = []
    x = 0.0
    prev_cat = None
    for item in items:
        cat = item["cat"]
        if prev_cat is not None and cat != prev_cat:
            x += 0.25
        positions.append(x)
        x += 1.0
        prev_cat = cat

    if not items or max(sum(c.values()) for c in cwns_counts_list + llm_counts_list) == 0:
        print(f"  {group_title}: all zeros, skipping")
        continue

    # Figure width based on number of bar groups (positions span)
    x_span = positions[-1] - positions[0] + 1 if positions else 1
    fig_w = max(7, x_span * 0.85)
    safe = group_title.replace("/", "_").replace(" ", "_")
    path = f"{OUTPUT_DIR}/{safe}_source_comparison.png"
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    render_source_plot(
        ax=ax,
        labels=labels,
        positions=positions,
        source_counts=[cwns_counts_list, llm_counts_list],
        source_items=[("CWNS", "cwns"), ("NPDES - LLM extraction", "npdes_llm")],
        stack_order=stack_order_with_not_present,
        status_legend_items=status_legend_items_with_not_present,
        bar_width=bar_w,
    )
    cat_spans = {}
    for item, pos in zip(items, positions):
        cat = item["cat"]
        if cat not in cat_spans:
            cat_spans[cat] = [pos, pos]
        else:
            cat_spans[cat][1] = pos
    ylim = ax.get_ylim()
    first = True
    for cat, (s, e) in cat_spans.items():
        if not first:
            ax.axvline(s - 0.5, color="#999999", lw=0.8, linestyle="--", zorder=1)
        if len([it for it in items if it["cat"] == cat]) > 1:
            ax.text(
                (s + e) / 2,
                ylim[1] * 0.97,
                cat,
                ha="center",
                va="top",
                fontsize=11,
                color="#444444",
                style="italic",
            )
        first = False
    ax.set_ylim(ylim)
    make_grouped_legend(
        ax,
        groups=[
            {
                "header": "Data Source",
                "items": [
                    ("CWNS", {"facecolor": COLORS["cwns"]}),
                    ("NPDES - LLM extraction", {"facecolor": COLORS["npdes_llm"]}),
                ],
            },
            {"header": "Status", "items": status_legend_items},
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=12,
    )
    plt.tight_layout()
    save_and_close(fig, path, dpi=200)


# ── 2. Major-categories plot ─────────────────────────────────────────────────
# One bar group per top-level JSON category. Stacks sum status counts over all
# leaves in that category (same rule as combined treatment-stage bars).

for comparison_type in ["llm", "kw"]:
    include_kw = comparison_type == "kw"
    common_facilities = llm_common_facilities if not include_kw else kw_common_facilities & llm_common_facilities
    cat_labels, all_cwns, all_llm, all_kw = build_major_category_sources(common_facilities, include_kw=include_kw)
    suffix = "source_comparison" if comparison_type == "llm" else "kw_compare"
    source_items = [
        ("CWNS", "cwns"),
        ("NPDES - LLM extraction", "npdes_llm"),
    ] if not include_kw else [
        ("CWNS", "cwns"),
        ("NPDES - Keyword Search", "npdes_kw"),
        ("NPDES - LLM extraction", "npdes_llm"),
    ]

    n = len(cat_labels)
    final_dir = f"npdes_permits/output/{DATE_FOLDER}/final"
    os.makedirs(final_dir, exist_ok=True)
    path = f"{final_dir}/figure_4_major_categories_{suffix}.png"
    fig, ax = plt.subplots(figsize=(max(14, n * (0.55 if not include_kw else 0.7)), 6))
    render_source_plot(
        ax=ax,
        labels=cat_labels,
        positions=list(range(n)),
        source_items=source_items,
        stack_order=stack_order_with_not_present,
        status_legend_items=status_legend_items_with_not_present,
        source_counts=[all_cwns, all_llm] if not include_kw else [all_cwns, all_kw, all_llm],
        bar_width=bar_w if not include_kw else 0.24,
    )
    plt.tight_layout()
    save_and_close(fig, path, dpi=200)
    print(f"    Saved {os.path.basename(path)}")
