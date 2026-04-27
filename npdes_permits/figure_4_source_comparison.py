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
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.utils import get_leaf_names, PRESENT_STATUSES
from helpers.utils import (
    name_to_fac,
    facilities_with_cwns,
    facilities_without_cwns,
    map_facility_names_to_tuples,
    build_cwns_facility_processes,
    collapse_facility_processes,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, make_grouped_legend, save_and_close, set_thick_spines

DATE_FOLDER = "2026-4-26"
OUTPUT_DIR = f"npdes_permits/output/{DATE_FOLDER}/figures"
MIN_COUNT = 15  # drop bar groups where both sources are below this threshold

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
            ax.bar(
                xpos,
                val,
                width,
                bottom=bottom,
                color=facecolor,
                hatch=hatch,
                edgecolor="black",
                linewidth=1.2,
            )
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


def render_source_pair_plot(
    ax,
    labels,
    positions,
    left_counts,
    right_counts,
    right_color_key,
    source_items,
    stack_order,
    status_legend_items,
):
    for pos, left_c, right_c in zip(positions, left_counts, right_counts):
        draw_stacked_bar(ax, pos - bar_w / 2, bar_w, left_c, COLORS["cwns"], stack_order)
        draw_stacked_bar(ax, pos + bar_w / 2, bar_w, right_c, COLORS[right_color_key], stack_order)
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


# ── Item-list builder for per-category plots ──────────────────────────────────
def build_sorted_plot_items(json_cats, keywords, cwns_df, llm_df):
    """Build leaf items, counts, filter by threshold; sort leaves by NPDES NOT_PRESENT (asc)."""
    items = []
    labels = []
    cwns_counts_list, llm_counts_list = [], []
    for cat_name in json_cats:
        if cat_name not in keywords:
            continue
        leaves = get_leaf_names(cat_name, keywords[cat_name])
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

print("Loading unitprocess_keywords.json …")
with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
    keywords = json.load(f)

print("Loading LLM search data …")
_llm_path = f"npdes_permits/output/{DATE_FOLDER}/llm_unit_processes_by_facility.csv"
if not os.path.exists(_llm_path):
    print(f"  LLM data not found for {DATE_FOLDER} — skipping LLM")
    llm_df = pd.DataFrame()
else:
    llm_df = pd.read_csv(_llm_path, dtype=str).fillna("")
    before_collapse = len(llm_df)
    llm_df = collapse_facility_processes(
        llm_df,
        key_cols=["PERMIT_NUMBER", "Facility_Name"],
        meta_cols=["Agency"],
    )
    print(f"  LLM rows collapsed: {before_collapse} → {len(llm_df)} facilities")
    proc_cols_llm = [
        c for c in llm_df.columns if c not in {"PERMIT_NUMBER", "Agency", "Facility_Name"}
    ]
    past_count = sum((llm_df[c] == "PAST").sum() for c in proc_cols_llm)
    future_count = sum((llm_df[c] == "FUTURE").sum() for c in proc_cols_llm)
    print(f"  LLM facilities: {len(llm_df)}  |  'PAST': {past_count}  |  'FUTURE': {future_count}")


print("Loading keyword search data …")
kw_path = f"npdes_permits/output/{DATE_FOLDER}/kw_unit_processes_by_facility.csv"
kw_df = pd.read_csv(kw_path, dtype=str).fillna("")
all_leaf_processes = {
    leaf
    for cat_name, cat_val in keywords.items()
    for leaf in get_leaf_names(cat_name, cat_val)
}
proc_cols_kw = sorted(all_leaf_processes)
for col in proc_cols_kw:
    kw_df[col] = kw_df[col].str.strip().str.upper().replace({"0": "", "NAN": ""})
kw_df["PERMIT_NUMBER"] = kw_df["PERMIT_NUMBER"].astype(str).str.strip()
print(f"  Keyword facilities (pre-collapsed by step2): {len(kw_df)}")

print("Loading and matching CWNS data …")

ca_cwns = pd.read_csv(
    "npdes_permits/output/cwns_processes_by_facility.csv",
    low_memory=False,
    dtype={"CWNS_ID": str},
)
ca_cwns["CWNS_ID"] = ca_cwns["CWNS_ID"].astype(str).str.strip()

if not llm_df.empty:
    llm_facilities = map_facility_names_to_tuples(llm_df, "Facility_Name", name_to_fac)
else:
    llm_facilities = set()

kw_facilities = map_facility_names_to_tuples(kw_df, "FACILITY_NAME", name_to_fac)

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

target_facs = (llm_facilities | kw_facilities) & facilities_with_cwns
cwns_df, merged_map = build_cwns_facility_processes(ca_cwns, target_facilities=target_facs)
n_attach = int((merged_map["_cwns_merge"] == "both").sum())
print(f"\n  CIWQS mapping rows with CWNS survey attach: {n_attach} / {len(merged_map)}")
cwns_df["PERMIT_NUMBER"] = cwns_df["NPDES_No"].astype(str).str.strip()
cwns_df["_fac"] = list(zip(cwns_df["WDID"], cwns_df["Facility_Name"]))

# Save facilities with no declared CWNS counterpart, for manual mapping review
kw_unmatched = kw_facilities - facilities_with_cwns - facilities_without_cwns
if kw_unmatched:
    kw_unmatched_path = f"npdes_permits/output/{DATE_FOLDER}/unmatched_kw_no_cwns.csv"
    kw_df[
        kw_df["FACILITY_NAME"]
        .apply(lambda n: name_to_fac.get(str(n).strip(), ("", str(n).strip())))
        .isin(kw_unmatched)
    ][["FACILITY_NAME"]].drop_duplicates().to_csv(kw_unmatched_path, index=False)
    print(f"  Unmatched KW (no CWNS): {len(kw_unmatched)} → unmatched_kw_no_cwns.csv")
else:
    print("  Unmatched KW (no CWNS): 0")

# Filter LLM/KW to only facilities with a declared CWNS
kw_df = kw_df[
    kw_df["FACILITY_NAME"].apply(lambda n: name_to_fac.get(str(n).strip()) in facilities_with_cwns)
].copy()
kw_df["_fac"] = kw_df["FACILITY_NAME"].apply(lambda n: name_to_fac.get(str(n).strip()))
kw_df = kw_df[kw_df["_fac"].notna()].copy()
kw_df["WDID"] = kw_df["_fac"].map(lambda t: t[0])
kw_df["FACILITY_NAME"] = kw_df["_fac"].map(lambda t: t[1])
kw_df = collapse_facility_processes(
    kw_df,
    key_cols=["WDID", "FACILITY_NAME", "_fac"],
    meta_cols=["AGENCY_NAME", "PERMIT_NUMBER", "PDF_File", "Shared_PDF"],
)
if not llm_df.empty:
    llm_df = llm_df[
        llm_df["Facility_Name"].apply(
            lambda n: name_to_fac.get(str(n).strip()) in facilities_with_cwns
        )
    ].copy()
    llm_df["_fac"] = llm_df["Facility_Name"].apply(lambda n: name_to_fac.get(str(n).strip()))
    llm_df = llm_df[llm_df["_fac"].notna()].copy()
    llm_df["WDID"] = llm_df["_fac"].map(lambda t: t[0])
    llm_df["Facility_Name"] = llm_df["_fac"].map(lambda t: t[1])
    llm_df = collapse_facility_processes(
        llm_df,
        key_cols=["WDID", "Facility_Name", "_fac"],
        meta_cols=["Agency", "PERMIT_NUMBER"],
    )

cwns_facilities_all = set(cwns_df["_fac"])
llm_common_facilities = cwns_facilities_all & set(llm_df["_fac"]) if not llm_df.empty else set()
kw_common_facilities = cwns_facilities_all & set(kw_df["_fac"])

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Treatment-stage groupings for per-category plots ─────────────────────────
# Each entry: plot_title → [list of top-level JSON category names to combine]
# Leaf processes from all listed categories are shown together on one plot,
# with shaded bands separating categories that have multiple leaves.
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


# ── 1. Per-treatment-stage plots ──────────────────────────────────────────────

print("\nGenerating per-treatment-stage plots …")
bar_w = 0.35
status_order = ["PRESENT", "PAST", "FUTURE", "OFFSITE"]
stack_order = [(status, HATCH_PATTERNS.get(status, "")) for status in status_order]
status_order_with_not_present = ["PRESENT", "PAST", "FUTURE", "OFFSITE", "NOT_PRESENT"]
stack_order_with_not_present = [
    (status, HATCH_PATTERNS.get(status, "")) for status in status_order_with_not_present
]
status_legend_items = [
    (status.title(), {"facecolor": "grey", "hatch": HATCH_PATTERNS.get(status, "")})
    for status in status_order
]
status_legend_items_with_not_present = status_legend_items + [
    ("Not Present", {"facecolor": "white"}),
]

for group_title, json_cats in PLOT_GROUPS.items():
    cwns_plot_df = cwns_df[cwns_df["_fac"].isin(llm_common_facilities)].copy()
    llm_plot_df = llm_df[llm_df["_fac"].isin(llm_common_facilities)].copy()
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
    render_source_pair_plot(
        ax=ax,
        labels=labels,
        positions=positions,
        left_counts=cwns_counts_list,
        right_counts=llm_counts_list,
        right_color_key="npdes_total",
        source_items=[("CWNS", "cwns"), ("NPDES - LLM extraction", "npdes_total")],
        stack_order=stack_order_with_not_present,
        status_legend_items=status_legend_items_with_not_present,
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
                    ("NPDES - LLM extraction", {"facecolor": COLORS["npdes_total"]}),
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
    print(f"  Saved {os.path.basename(path)}")


# ── 2. Major-categories plot ─────────────────────────────────────────────────
# One bar group per top-level JSON category. Stacks sum status counts over all
# leaves in that category (same rule as combined treatment-stage bars).

print("\nGenerating major-categories plots …")

# REPLACE WITH:
for comparison_type in ["llm", "kw"]:
    if comparison_type == "llm" and llm_df.empty:
        continue
    print(f"\n  Generating {comparison_type.upper()} major-categories plot…")
    comparison_df = llm_df if comparison_type == "llm" else kw_df
    common_facilities = (
        llm_common_facilities if comparison_type == "llm" else kw_common_facilities
    )
    cwns_compare_df = cwns_df[cwns_df["_fac"].isin(common_facilities)].copy()
    comparison_df = comparison_df[comparison_df["_fac"].isin(common_facilities)].copy()
    right_color_key = "npdes_total" if comparison_type == "llm" else "npdes"
    source_items = (
        [("CWNS", "cwns"), ("NPDES - LLM extraction", "npdes_total")]
        if comparison_type == "llm"
        else [("CWNS", "cwns"), ("NPDES - Keyword Search", "npdes")]
    )
    suffix = "source_comparison" if comparison_type == "llm" else "kw_compare"

    cat_labels = list(keywords.keys())

    all_cwns = []
    all_comp = []
    for cat_name, cat_val in keywords.items():
        leaves = get_leaf_names(cat_name, cat_val)
        all_cwns.append(get_facility_counts(cwns_compare_df, leaves))
        all_comp.append(get_facility_counts(comparison_df, leaves))

    cat_filtered = [
        (lbl, cwns_c, comp_c)
        for lbl, cwns_c, comp_c in zip(cat_labels, all_cwns, all_comp)
        if sum(cwns_c.values()) >= MIN_COUNT or sum(comp_c.values()) >= MIN_COUNT
    ]
    if cat_filtered:
        cat_labels, all_cwns, all_comp = map(list, zip(*cat_filtered))

    order = sorted(
        range(len(cat_labels)),
        key=lambda i: (all_comp[i]["NOT_PRESENT"], cat_labels[i]),
    )
    cat_labels = [cat_labels[i] for i in order]
    all_cwns = [all_cwns[i] for i in order]
    all_comp = [all_comp[i] for i in order]

    n = len(cat_labels)
    final_dir = f"npdes_permits/output/{DATE_FOLDER}/final"
    os.makedirs(final_dir, exist_ok=True)
    path = f"{final_dir}/figure_4_major_categories_{suffix}.png"
    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))
    render_source_pair_plot(
        ax=ax,
        labels=cat_labels,
        positions=list(range(n)),
        left_counts=all_cwns,
        right_counts=all_comp,
        right_color_key=right_color_key,
        source_items=source_items,
        stack_order=stack_order_with_not_present,
        status_legend_items=status_legend_items_with_not_present,
    )
    plt.tight_layout()
    save_and_close(fig, path, dpi=200)
    print(f"    Saved {os.path.basename(path)}")
