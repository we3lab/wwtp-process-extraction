"""
Grouped stacked bar-chart comparison of unit process detection across two data sources:
  - CWNS (California facilities from output/cwns_unit_processes_by_facility.csv)
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
from geopy.distance import geodesic
from pathlib import Path
from helpers.utils import get_leaf_names, PRESENT_STATUSES, get_unspecified_leaf_names
from helpers.utils import (
    cwns_mapping,
    no_cwns_pids,
    build_cwns_facility_processes,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, make_grouped_legend, save_and_close, set_thick_spines

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATE_FOLDER = "2026-5-25"
DATA_DIR = f"wwtp_process_extraction/output/{DATE_FOLDER}"
OUTPUT_DIR = f"wwtp_process_extraction/output/{DATE_FOLDER}/figures"
MIN_COUNT = 20  # drop bar groups where both sources are below this threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_GROUPS = {
    "Primary Treatment": ["Headworks", "Comminution", "Equalization", "Flotation"],
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

with open("wwtp_process_extraction/data/unitprocess_keywords.json", "r") as f:
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
    "wwtp_process_extraction/output/cwns_unit_processes_by_facility.csv",
    dtype=str,
)
ca_cwns["CWNS_ID"] = ca_cwns["CWNS_ID"].str.strip()

llm_facilities = set(llm_df["Place ID"])
kw_facilities = set(kw_df["Place ID"])
cwns_pids = set(cwns_mapping["Place ID"])

# Coverage summary — before filtering to overlap only
print(f"\nFacility coverage (unique WDID + Facility Name, before overlap filter):")
print(f"  Both CWNS + site_data extracted (KW):           {len(kw_facilities & cwns_pids)}")
print(f"  Both CWNS + site_data extracted (LLM):          {len(llm_facilities & cwns_pids)}")
print(f"  site_data extracted only (no CWNS):             {len(kw_facilities - cwns_pids)}")
print(f"  Mapped to CWNS but dropped from site_data:      {len(cwns_pids - kw_facilities)}")
print(f"  All 3 sources:                                  {len(llm_facilities & kw_facilities & cwns_pids)}")

# print the CWNS only as plain text, comma-separated around strings, without ""
cwns_only_facs = cwns_pids - kw_facilities
print("CWNS only facilities:")
print(", ".join(cwns_only_facs))

cwns_df, merged_map = build_cwns_facility_processes(ca_cwns, target_facilities=llm_facilities | kw_facilities)

n_attach = int((merged_map["_cwns_merge"] == "both").sum())
print(f"\n  CIWQS mapping rows with CWNS survey attach: {n_attach} / {len(merged_map)}")

# Save facilities with no CWNS match
site_data_path = f"wwtp_process_extraction/output/{DATE_FOLDER}/site_data.csv"
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
facility_names = unmatched_df['FACILITY_NAME'].tolist()
# print plain text, comma-separated around strings
print(', '.join(facility_names))

# CWNS rows with no declared match in ciwqs_to_cwns (by CWNS_ID)
cwns_csv = Path("wwtp_process_extraction/output/cwns_unit_processes_by_facility.csv")
mapping_csv = Path("wwtp_process_extraction/data/ciwqs_to_cwns.csv")
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

# ── Unit process counts for all processes ─────────────────────────────────────
_cwns_common = cwns_df[cwns_df["Place ID"].isin(llm_common_facilities)].copy()
_llm_common = llm_df[llm_df["Place ID"].isin(llm_common_facilities)].copy()
_count_rows = []
for proc in proc_cols:
    if proc not in _cwns_common.columns and proc not in _llm_common.columns:
        continue
    cwns_c = get_facility_counts(_cwns_common, [proc]) if proc in _cwns_common.columns else {"PRESENT": 0, "PAST": 0, "FUTURE": 0, "OFFSITE": 0, "NOT_PRESENT": len(_cwns_common)}
    llm_c = get_facility_counts(_llm_common, [proc]) if proc in _llm_common.columns else {"PRESENT": 0, "PAST": 0, "FUTURE": 0, "OFFSITE": 0, "NOT_PRESENT": len(_llm_common)}
    _count_rows.append({"process": proc, **{f"cwns_{k.lower()}": v for k, v in cwns_c.items()}, **{f"llm_{k.lower()}": v for k, v in llm_c.items()}})
process_counts_df = pd.DataFrame(_count_rows)
process_counts_df.to_csv(f"{DATA_DIR}/process_counts_cwns_vs_llm.csv", index=False)
print(f"\nSaved process counts: {f'{DATA_DIR}/process_counts_cwns_vs_llm.csv'} ({len(process_counts_df)} processes)")

# TODO: denitrification filter — no dedicated leaf yet in unitprocess_keywords.json.
# "Media Filtration" is too broad (catches sand/cloth/disc filters).
# Proxy: facilities with BOTH Media Filtration PRESENT and Denitrification PRESENT.
# Better fix: load llm_postprocess_ontology JSONs and find items where trigger_process
# contains BOTH "Media Filtration" and "Denitrification" on the same extracted item
# (same equipment row) — then add a "Denitrification Filter" leaf to keywords JSON.
_mf_present = set(_llm_common.loc[_llm_common["Media Filtration"].isin(PRESENT_STATUSES), "Place ID"])
_dn_present = set(_llm_common.loc[_llm_common["Denitrification"].isin(PRESENT_STATUSES), "Place ID"])
_denif_filter_proxy = len(_mf_present & _dn_present)
_cwns_dn_c = get_facility_counts(_cwns_common, ["Denitrification"])
print(f"  {'Denitrif. Filter (proxy: MF+DN both present)':<40} {'n/a':>14} {_denif_filter_proxy:>12}")

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
        stack_order=stack_order,
        status_legend_items=status_legend_items,
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
    final_dir = f"wwtp_process_extraction/output/{DATE_FOLDER}/final"
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


SITE_DATA = f"wwtp_process_extraction/output/{DATE_FOLDER}/site_data.csv"
FACILITIES_JSON = f"wwtp_process_extraction/output/{DATE_FOLDER}/facilities.json"
ALL_NPDES = f"wwtp_process_extraction/output/{DATE_FOLDER}/all_ca_npdes.csv"
CWNS_TABLE = "wwtp_process_extraction/output/cwns_unit_processes_by_facility.csv"
CIWQS_MAP = "wwtp_process_extraction/data/ciwqs_to_cwns.csv"

ciwqs_cols = [
    "WDID", "Place ID", "Facility Name", "NPDES No.", "Region",
    "Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS",
    "CWNS_ID", "FACILITY_ID", "CWNS Facility Name",
]

def normalize(s):
    return str(s).strip().upper() if pd.notna(s) else ""


def coalesce_blank(left, right):
    return left.replace("", pd.NA).fillna(right).fillna("")


site = pd.read_csv(SITE_DATA, dtype=str).fillna("")
cwns = pd.read_csv(CWNS_TABLE, dtype=str).fillna("")
ciwqs = pd.read_csv(CIWQS_MAP, dtype=str, keep_default_na=False).fillna("").rename(
    columns={"Latitude": "Latitude_CIWQS", "Longitude": "Longitude_CIWQS"}
)
all_npdes = pd.read_csv(ALL_NPDES, dtype=str).fillna("").rename(
    columns={"Latitude": "Latitude_CIWQS_from_npdes", "Longitude": "Longitude_CIWQS_from_npdes"}
)
cwns_fac_tp = ca_cwns[["CWNS_ID", "FACILITY_ID", "FACILITY_NAME", "STATE_CODE", "LATITUDE", "LONGITUDE"]].rename(columns={"FACILITY_NAME": "CWNS Facility Name"}).copy()
cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]] = cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]].apply(lambda c: c.str.strip())

cwns_loc_map = cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]].drop_duplicates().merge(
    cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "LATITUDE", "LONGITUDE"]].rename(
        columns={"LATITUDE": "Latitude_CWNS_from_cwns", "LONGITUDE": "Longitude_CWNS_from_cwns"}
    ).drop_duplicates(),
    on=["CWNS_ID", "FACILITY_ID"], how="left"
).drop_duplicates()

# CWNS_ID → FACILITY_ID lookup for populating existing mapping rows
cwns_id_to_fac_id = cwns_fac_tp.drop_duplicates("CWNS_ID").set_index("CWNS_ID")["FACILITY_ID"].to_dict()

for col in ["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]:
    site[col] = site[col].str.strip()
site_lookup_cols = ["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]
site_lookup = site[site_lookup_cols].drop_duplicates()

for col in ["WDID", "Facility Name"]:
    all_npdes[col] = all_npdes[col].str.strip()
ciwqs_lookup = all_npdes[["WDID", "Facility Name", "Latitude_CIWQS_from_npdes", "Longitude_CIWQS_from_npdes"]].drop_duplicates()

site = site.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

ciwqs[["WDID", "Facility Name"]] = ciwqs[["WDID", "Facility Name"]].apply(lambda c: c.str.strip())
# ensure CWNS keys are normalized for merges
for col in ["CWNS_ID", "CWNS Facility Name"]:
    if col in ciwqs.columns:
        ciwqs[col] = ciwqs[col].astype(str).str.strip()

# Populate FACILITY_ID for existing rows that predate this column
if "FACILITY_ID" not in ciwqs.columns:
    ciwqs["FACILITY_ID"] = ""
ciwqs["FACILITY_ID"] = ciwqs["FACILITY_ID"].astype(str).str.strip()
needs_fac_id = ciwqs["FACILITY_ID"].eq("") & ciwqs["CWNS_ID"].ne("") & ciwqs["CWNS_ID"].str.upper().ne("NA")
ciwqs.loc[needs_fac_id, "FACILITY_ID"] = ciwqs.loc[needs_fac_id, "CWNS_ID"].map(cwns_id_to_fac_id).fillna("")

ciwqs = ciwqs.merge(site_lookup, on=["WDID", "Facility Name"], how="left", suffixes=("", "_site"))
ciwqs = ciwqs.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

for dest, src in [("NPDES No.", "NPDES_No_site"), ("Region", "Region_site")]:
    if src in ciwqs.columns:
        ciwqs[dest] = ciwqs[dest] if dest in ciwqs.columns else ciwqs[src]
        ciwqs[dest] = coalesce_blank(ciwqs[dest], ciwqs[src])
        ciwqs = ciwqs.drop(columns=[src])
for dest, src in [("Latitude_CIWQS", "Latitude_CIWQS_from_npdes"), ("Longitude_CIWQS", "Longitude_CIWQS_from_npdes")]:
    if src in ciwqs.columns:
        ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest] if dest in ciwqs.columns else "")
        ciwqs = ciwqs.drop(columns=[src])

ciwqs["Latitude_CWNS"] = ciwqs.get("Latitude_CWNS", "")
ciwqs["Longitude_CWNS"] = ciwqs.get("Longitude_CWNS", "")

already_mapped = set(ciwqs["Facility Name"].map(normalize))
unmapped = (
    site[~site["Facility Name"].map(normalize).isin(already_mapped)]
    .drop_duplicates("Facility Name")
    .copy()
)
print(f"Unmapped facilities: {len(unmapped)}")

# Match unmapped facilities by name against CA Treatment Plant facilities
cwns_fac_ca = cwns_fac_tp[cwns_fac_tp["STATE_CODE"] == "CA"].copy()
cwns_fac_ca["_name"] = cwns_fac_ca["CWNS Facility Name"].map(normalize)
name_idx = cwns_fac_ca.groupby("_name").apply(lambda g: g.to_dict("records")).to_dict()

new_rows = []

for _, row in unmapped.iterrows():
    fac_name = row["Facility Name"].strip()
    permit = str(row.get("NPDES No.", "")).strip().upper()
    base_entry = {
        "WDID": str(row.get("WDID", "")).strip(),
        "Place ID": str(row.get("Place ID", "")).strip(),
        "Facility Name": fac_name,
        "NPDES No.": permit,
        "Region": str(row.get("Region", "")).strip(),
        "Latitude_CIWQS": str(row.get("Latitude_CIWQS_from_npdes", "")).strip(),
        "Longitude_CIWQS": str(row.get("Longitude_CIWQS_from_npdes", "")).strip(),
        "Latitude_CWNS": "",
        "Longitude_CWNS": "",
    }
    cwns_hits = name_idx.get(normalize(fac_name), [])
    if cwns_hits:
        print(f"  [name] {permit} — {fac_name} → {len(cwns_hits)} CWNS row(s)")
        for hit in cwns_hits:
            new_rows.append({**base_entry, "CWNS_ID": hit["CWNS_ID"], "FACILITY_ID": hit["FACILITY_ID"], "CWNS Facility Name": hit["CWNS Facility Name"]})
    else:
        print(f"  [no match] {permit} — {fac_name}")
        new_rows.append({**base_entry, "CWNS_ID": "", "FACILITY_ID": "", "CWNS Facility Name": ""})

ciwqs = ciwqs.merge(
    cwns_loc_map[["CWNS_ID", "FACILITY_ID", "Latitude_CWNS_from_cwns", "Longitude_CWNS_from_cwns"]],
    on=["CWNS_ID", "FACILITY_ID"], how="left"
)

for dest, src in [("Latitude_CWNS", "Latitude_CWNS_from_cwns"), ("Longitude_CWNS", "Longitude_CWNS_from_cwns")]:
    if src in ciwqs.columns:
        ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest])
        ciwqs = ciwqs.drop(columns=[src])

for col in ciwqs_cols:
    if col not in ciwqs.columns:
        ciwqs[col] = ""
ciwqs_out = ciwqs[ciwqs_cols]

if new_rows:
    new_df = pd.DataFrame(new_rows, columns=ciwqs_cols)
    combined = pd.concat([ciwqs_out, new_df], ignore_index=True)
    print(f"\nAdded {len(new_rows)} rows. ciwqs_to_cwns.csv now has {len(combined)} rows.")
else:
    combined = ciwqs_out
    print("\nNo new rows to add.")

# Dedupe on Place ID + FACILITY_ID, preferring rows with NPDES filled, preserving original row order
_save = combined.reset_index(drop=True).rename_axis("_orig_order").reset_index()
_save["_npdes_empty"] = _save["NPDES No."].eq("")
(
    _save.sort_values(["_npdes_empty", "_orig_order"])
    .drop_duplicates(subset=["Place ID", "FACILITY_ID"], keep="first")
    .sort_values("_orig_order")
    [ciwqs_cols]
    .to_csv(CIWQS_MAP, index=False)
)

coord_cols = ["Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS"]
geo = combined[combined[coord_cols].replace("", pd.NA).notna().all(axis=1)].copy()
for col in coord_cols:
    geo[col] = pd.to_numeric(geo[col], errors="coerce")
geo = geo.dropna(subset=coord_cols)

geo["_dist_miles"] = geo.apply(
    lambda r: geodesic((r["Latitude_CIWQS"], r["Longitude_CIWQS"]), (r["Latitude_CWNS"], r["Longitude_CWNS"])).miles,
    axis=1,
)
far = geo[geo["_dist_miles"] > 2].sort_values("_dist_miles", ascending=False)
print(f"\nRows where CWNS and CIWQS coords are >2 miles apart: {len(far)}")
print(far[["Facility Name", "NPDES No.", "CWNS_ID", "FACILITY_ID", "_dist_miles"]].to_string(index=False))
