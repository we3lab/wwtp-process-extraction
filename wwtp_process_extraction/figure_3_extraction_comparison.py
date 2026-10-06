"""
Grouped stacked bar-chart comparison of unit process detection across data sources:
  - CWNS (California facilities from output/unit_processes_by_facility_cwns.csv)
  - LLM Search (output/unit_processes_by_facility_llm.csv)
  - Keyword Search (output/unit_processes_by_facility_kw.csv, SI figure only)

CWNS rows join ``ciwqs_to_cwns.csv`` to the CA step0 export (exact ``CWNS_ID``).
The export includes placeholder rows for CA ``CWNS_ID`` values missing from CWNS
inventory aggregation (blank ``0`` processes). LLM and keyword rows are kept when
their ``Place ID`` has a CWNS match in the mapping. No fuzzy matching.

Each bar is stacked by status (process columns must already be normalized: stripped, uppercase).
PRESENT_AND_FUTURE is counted as PRESENT only (not split into Future).

Plot stacks (all sources): Present | Past | Future | Offsite (| Not Present on major categories)
  CWNS : PRESENT | PRESENT_AND_FUTURE | FUTURE | PAST | OFFSITE | 0
  LLM  : PRESENT | PRESENT_AND_FUTURE | FUTURE | PAST | OFFSITE

Produces:
  1. One graph per treatment-stage group (combined leaves)
  2. One graph of major categories (top-level JSON keys), LLM and LLM + keyword versions
  Y-axis: facilities per bar group, each counted once at its highest-priority status.
Also writes the unmatched-facility CSVs, the merged unit_processes_by_facility.csv, per-process
counts, and rewrites ciwqs_to_cwns.csv with name-matched rows and coordinates.
"""

import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from helpers.utils import (
    get_leaf_names,
    PRESENT_STATUSES,
    cwns_mapping,
    no_cwns_pids,
    build_cwns_facility_processes,
    add_county_and_sort,
    unitprocess_keywords as keywords,
    OUTPUT_DIR,
    FIGURES_DIR,
    BY_CATEGORY_DIR,
    SITE_DATA_RELEVANT_CSV,
    SITE_DATA_ALL_CSV,
    CWNS_TABLE_CSV,
    CIWQS_TO_CWNS_CSV,
)
from helpers.plotting import COLORS, HATCH_PATTERNS, make_grouped_legend, save_and_close, set_thick_spines

MIN_COUNT = 20  # drop bar groups where both sources are below this threshold
MAJOR_MIN_COUNT = 5  # threshold for excluding major-category totals (TOTAL plot)
BAR_WIDTH = 0.35

# Stacked statuses, bottom to top; a bar group counts as detected by their sum
STACKED_STATUSES = ("PRESENT", "PAST", "FUTURE", "OFFSITE")

# Treatment-stage groupings for per-category plots
# Each entry: plot_title → [list of top-level JSON category names to combine]
# Leaf processes from all listed categories are shown together on one plot,
# with shaded bands separating categories that have multiple leaves.
PLOT_GROUPS = {
    "Primary Treatment": ["Headworks", "Comminution", "Equalization", "Flotation"],
    "Clarification": ["Clarification"],
    "Secondary Treatment": ["Activated Sludge", "Lagoon"],
    "Nutrient Removal": ["Nutrient Removal"],
    "Filtration": ["Filtration"],
    "Disinfection": ["Disinfection"],
    "Chemical Treatment": ["Coagulation", "Flocculation", "Chemical Addition"],
    "Advanced Treatment": ["Ion Exchange", "Activated Carbon", "UV-AOP", "Wetland"],
    "Solids Processing": ["Solids Processing"],
}

CWNS_SOURCE = ("Clean Watershed Needs Survey", "Clean Watershed Needs Survey")
LLM_SOURCE = ("Facility Permit - LLM extraction", "npdes_llm")
KW_SOURCE = ("NPDES - Keyword Search", "npdes_kw")

STACK_ORDER = [(status, HATCH_PATTERNS[status]) for status in STACKED_STATUSES]
STACK_ORDER_WITH_NOT_PRESENT = STACK_ORDER + [("NOT_PRESENT", "")]
STATUS_LEGEND_ITEMS = [
    (status.title(), {
        "facecolor": "grey",
        "hatch": HATCH_PATTERNS[status],
        "edgecolor": "white" if HATCH_PATTERNS[status] else "black"
    })
    for status in STACKED_STATUSES
]
STATUS_LEGEND_ITEMS_WITH_NOT_PRESENT = STATUS_LEGEND_ITEMS + [
    ("Not Present", {"facecolor": "white", "edgecolor": "black"}),
]


# Data helpers

def normalize(s):
    return s.strip().upper()


def coalesce_blank(left, right):
    return left.replace("", pd.NA).fillna(right).fillna("")


def any_status(df, cols, statuses):
    """Boolean Series: True if any col in cols has any value in statuses."""
    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask |= df[col].isin(statuses)
    return mask


def get_facility_counts(df, leaf_cols):
    """Unique-facility counts: each facility counted once at highest-priority status."""
    has_present = any_status(df, leaf_cols, PRESENT_STATUSES)
    has_past = any_status(df, leaf_cols, {"PAST"})
    has_future = any_status(df, leaf_cols, {"FUTURE"})
    has_offsite = any_status(df, leaf_cols, {"OFFSITE"})
    present_count = int(has_present.sum())
    future_count = int((has_future & ~has_present).sum())
    offsite_count = int((has_offsite & ~has_present & ~has_future).sum())
    past_count = int((has_past & ~has_present & ~has_future & ~has_offsite).sum())
    not_present_count = len(df) - (present_count + past_count + future_count + offsite_count)
    return {
        "PRESENT": present_count,
        "PAST": past_count,
        "FUTURE": future_count,
        "OFFSITE": offsite_count,
        "NOT_PRESENT": not_present_count,
    }


def detected_count(counts):
    return sum(counts[status] for status in STACKED_STATUSES)


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
            # One stacked bar: solid status segments, with a hatch overlay where the status has one
            bottom = 0
            for key, hatch in stack_order:
                val = count[key]
                if val > 0:
                    facecolor = "white" if key == "NOT_PRESENT" else COLORS[color_key]
                    ax.bar(
                        pos + offset,
                        val,
                        bar_width,
                        bottom=bottom,
                        color=facecolor,
                        edgecolor="black",
                        linewidth=1.2,
                    )
                    if hatch:
                        hatch_bar = ax.bar(
                            pos + offset,
                            val,
                            bar_width,
                            bottom=bottom,
                            color="none",
                            hatch=hatch,
                            edgecolor="white",
                            linewidth=0.0,
                        )
                        for patch in hatch_bar:
                            patch.set_hatch_linewidth(1.0)
                    bottom += val
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.set_ylabel("WWTP Count", fontsize=14)
    set_thick_spines(ax, linewidth=1.6)
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


# Load data

all_leaf_processes = {
    leaf
    for cat_name, cat_val in keywords.items()
    for leaf in get_leaf_names(cat_name, cat_val)
}
proc_cols = sorted(all_leaf_processes)

llm_df = pd.read_csv(OUTPUT_DIR / "unit_processes_by_facility_llm.csv", dtype=str)
kw_df = pd.read_csv(OUTPUT_DIR / "unit_processes_by_facility_kw.csv", dtype=str)
ca_cwns = pd.read_csv(CWNS_TABLE_CSV, dtype=str)
ca_cwns["CWNS_ID"] = ca_cwns["CWNS_ID"].str.strip()
site = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")
ciwqs = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str, keep_default_na=False)
ciwqs_columns = list(ciwqs.columns)  # rewritten with the same column order
all_npdes = pd.read_csv(SITE_DATA_ALL_CSV, dtype=str).fillna("").rename(
    columns={"Latitude": "Latitude_CIWQS_from_npdes", "Longitude": "Longitude_CIWQS_from_npdes"}
)

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

print(f"\n  CIWQS mapping rows with CWNS survey attach: {len(merged_map)}")

# Save facilities with no CWNS match
pid_to_name = {}
site_facs = set(site["Place ID"]) - {""}
for df in (site, kw_df):
    for pid, name in zip(df["Place ID"], df["Facility Name"]):
        if pid:
            pid_to_name.setdefault(pid, name)
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
unmatched_df.to_csv(OUTPUT_DIR / "unmatched_kw_no_cwns.csv", index=False)
print(f"  Unmatched KW/site_data (no CWNS): {len(unmatched_pids)} → unmatched_kw_no_cwns.csv")
# print plain text, comma-separated around strings
print(', '.join(unmatched_df['FACILITY_NAME']))

# CWNS rows with no declared match in ciwqs_to_cwns (by CWNS_ID)
cwns_unmatched_df = ca_cwns.fillna("")
mapped_cwns_ids = {
    cw.strip()
    for cw in ciwqs["CWNS_ID"]
    if cw.strip() and cw.strip().upper() != "NA"
}
cwns_ids = cwns_unmatched_df["CWNS_ID"]
unmatched_cwns_no_kw = cwns_unmatched_df[
    cwns_ids.ne("") & cwns_ids.str.upper().ne("NA") & ~cwns_ids.isin(mapped_cwns_ids)
][["CWNS_ID", "FACILITY_ID", "FACILITY_NAME"]].drop_duplicates()

unmatched_cwns_no_kw.to_csv(OUTPUT_DIR / "unmatched_cwns_no_kw.csv", index=False)


# Final merged unit-process table (CWNS + CIWQS/LLM)
# One row per (facility, source). source ∈ {cwns, ciwqs}. Facilities present in
# only one source get a single row. Built from the full llm_df, before the
# overlap filtering below, so ciwqs-only facilities are kept.
mapping_by_pid = cwns_mapping.drop_duplicates("Place ID").set_index("Place ID")
llm_name_by_pid = llm_df.drop_duplicates("Place ID").set_index("Place ID")["Facility Name"].to_dict()

final_rows = []
for source_name, src_df in [("Clean Watershed Needs Survey", cwns_df), ("ciwqs", llm_df)]:
    for _, r in src_df.iterrows():
        pid = r["Place ID"]
        m = mapping_by_pid.loc[pid] if pid in mapping_by_pid.index else {}
        row = {
            "source": source_name,
            "CWNS FACILITY_ID": m.get("FACILITY_ID", ""),
            "CWNS FACILITY_NAME": m.get("CWNS Facility Name", ""),
            "CIWQS PLACE_ID": pid,
            "CIWQS Facility Name": llm_name_by_pid.get(pid) or m.get("Facility Name", ""),
        }
        for c in proc_cols:
            row[c] = r[c]
        final_rows.append(row)

final_cols = ["source", "CWNS FACILITY_ID", "CWNS FACILITY_NAME", "CIWQS PLACE_ID", "CIWQS Facility Name"] + proc_cols
final_df = pd.DataFrame(final_rows, columns=final_cols)
final_df = add_county_and_sort(final_df, "CIWQS Facility Name", place_id_col="CIWQS PLACE_ID")
final_path = OUTPUT_DIR / "unit_processes_by_facility.csv"
final_df.to_csv(final_path, index=False)
print(f"\nSaved merged unit processes ({len(final_df)} rows, "
      f"{(final_df['source'] == 'Clean Watershed Needs Survey').sum()} cwns + {(final_df['source'] == 'ciwqs').sum()} ciwqs): {final_path}")

cwns_facilities_all = set(cwns_df["Place ID"])
kw_df, llm_df = [df[df["Place ID"].isin(cwns_facilities_all)].copy() for df in [kw_df, llm_df]]
llm_common_facilities, kw_common_facilities = [set(df["Place ID"]) for df in [llm_df, kw_df]]

# Unit process counts for all processes
cwns_common_df = cwns_df[cwns_df["Place ID"].isin(llm_common_facilities)].copy()
llm_common_df = llm_df[llm_df["Place ID"].isin(llm_common_facilities)].copy()
count_rows = []
for proc in proc_cols:
    cwns_c = get_facility_counts(cwns_common_df, [proc])
    llm_c = get_facility_counts(llm_common_df, [proc])
    count_rows.append({"process": proc, **{f"cwns_{k.lower()}": v for k, v in cwns_c.items()}, **{f"llm_{k.lower()}": v for k, v in llm_c.items()}})
process_counts_df = pd.DataFrame(count_rows)
process_counts_path = OUTPUT_DIR / "process_counts_cwns_vs_llm.csv"
process_counts_df.to_csv(process_counts_path, index=False)
print(f"\nSaved process counts: {process_counts_path} ({len(process_counts_df)} processes)")


# 1. Per-treatment-stage plots

for group_title, json_cats in PLOT_GROUPS.items():
    items = []
    for cat_name in json_cats:
        cat_items = []
        for leaf in get_leaf_names(cat_name, keywords[cat_name], exclude_unspecified=True):
            cwns_c = get_facility_counts(cwns_common_df, [leaf])
            llm_c = get_facility_counts(llm_common_df, [leaf])
            if detected_count(cwns_c) >= MIN_COUNT or detected_count(llm_c) >= MIN_COUNT:
                cat_items.append({"label": leaf, "cat": cat_name, "cwns": cwns_c, "llm": llm_c})
        # X order: lowest → highest NPDES (right bar) "Not Present"
        cat_items.sort(key=lambda item: (item["llm"]["NOT_PRESENT"], item["label"]))
        items.extend(cat_items)
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

    # Figure width based on number of bar groups (positions span)
    x_span = positions[-1] - positions[0] + 1
    fig_w = max(7, x_span * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    render_source_plot(
        ax=ax,
        labels=[item["label"] for item in items],
        positions=positions,
        source_counts=[[item["cwns"] for item in items], [item["llm"] for item in items]],
        source_items=[CWNS_SOURCE, LLM_SOURCE],
        stack_order=STACK_ORDER,
        status_legend_items=STATUS_LEGEND_ITEMS,
        bar_width=BAR_WIDTH,
    )
    cat_spans = {}
    for item, pos in zip(items, positions):
        cat_spans.setdefault(item["cat"], [pos, pos])[1] = pos
    ylim = ax.get_ylim()
    for idx, (cat, (span_start, span_end)) in enumerate(cat_spans.items()):
        if idx:
            ax.axvline(span_start - 0.5, color="#999999", lw=0.8, linestyle="--", zorder=1)
        if sum(it["cat"] == cat for it in items) > 1:
            ax.text(
                (span_start + span_end) / 2,
                ylim[1] * 0.97,
                cat,
                ha="center",
                va="top",
                fontsize=11,
                color="#444444",
                style="italic",
            )
    ax.set_ylim(ylim)
    plt.tight_layout()
    save_and_close(fig, BY_CATEGORY_DIR / f"{group_title.replace(' ', '_')}_source_comparison.png", dpi=300)


# 2. Major-categories plot
# One bar group per top-level JSON category. Each facility is counted once per category,
# at its highest-priority status across all leaves in that category.

for comparison_type in ["llm", "kw"]:
    include_kw = comparison_type == "kw"
    if include_kw:
        common_facilities = kw_common_facilities & llm_common_facilities
        source_dfs = [cwns_df, kw_df, llm_df]
        source_items = [CWNS_SOURCE, KW_SOURCE, LLM_SOURCE]
    else:
        common_facilities = llm_common_facilities
        source_dfs = [cwns_df, llm_df]
        source_items = [CWNS_SOURCE, LLM_SOURCE]
    compare_dfs = [df[df["Place ID"].isin(common_facilities)].copy() for df in source_dfs]

    # (category, per-source counts), kept when any source detects it at least MAJOR_MIN_COUNT times
    categories = []
    for cat_name, cat_val in keywords.items():
        leaves = get_leaf_names(cat_name, cat_val)
        counts = [get_facility_counts(df, leaves) for df in compare_dfs]
        if any(detected_count(c) >= MAJOR_MIN_COUNT for c in counts):
            categories.append((cat_name, counts))
    # LLM counts are last in every source order
    categories.sort(key=lambda category: (category[1][-1]["NOT_PRESENT"], category[0]))
    cat_labels = [cat_name for cat_name, _ in categories]
    source_counts = [[counts[i] for _, counts in categories] for i in range(len(source_dfs))]
    filename = "figure_3" if comparison_type == "llm" else "figure_s4"

    n = len(cat_labels)
    # Drop the Offsite legend entry unless some bar actually has an offsite band
    has_offsite = any(c["OFFSITE"] > 0 for counts in source_counts for c in counts)
    legend_items = STATUS_LEGEND_ITEMS_WITH_NOT_PRESENT
    if not has_offsite:
        legend_items = [it for it in legend_items if it[0] != "Offsite"]
    fig, ax = plt.subplots(figsize=(max(14, n * (0.55 if not include_kw else 0.7)), 6))
    render_source_plot(
        ax=ax,
        labels=cat_labels,
        positions=list(range(n)),
        source_items=source_items,
        stack_order=STACK_ORDER_WITH_NOT_PRESENT,
        status_legend_items=legend_items,
        source_counts=source_counts,
        bar_width=BAR_WIDTH if not include_kw else 0.24,
    )
    plt.tight_layout()
    save_and_close(fig, FIGURES_DIR / f"{filename}.png", dpi=300)
    print(f"    Saved {filename}.png")


# Rewrite ciwqs_to_cwns.csv: fill FACILITY_ID, NPDES, Region and coordinates, add name matches
cwns_fac_tp = ca_cwns[["CWNS_ID", "FACILITY_ID", "FACILITY_NAME", "STATE_CODE", "LATITUDE", "LONGITUDE"]].rename(columns={"FACILITY_NAME": "CWNS Facility Name"})
cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]] = cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]].apply(lambda c: c.str.strip())

cwns_loc_map = cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]].drop_duplicates().merge(
    cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "LATITUDE", "LONGITUDE"]].rename(
        columns={"LATITUDE": "Latitude_CWNS_from_cwns", "LONGITUDE": "Longitude_CWNS_from_cwns"}
    ).drop_duplicates(),
    on=["CWNS_ID", "FACILITY_ID"], how="left"
).drop_duplicates()

# CWNS_ID → FACILITY_ID lookup for populating existing mapping rows
cwns_id_to_fac_id = cwns_fac_tp.drop_duplicates("CWNS_ID").set_index("CWNS_ID")["FACILITY_ID"].to_dict()

site_lookup_cols = ["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]
site[site_lookup_cols] = site[site_lookup_cols].apply(lambda c: c.str.strip())
site_lookup = site[site_lookup_cols].drop_duplicates()

all_npdes[["WDID", "Facility Name"]] = all_npdes[["WDID", "Facility Name"]].apply(lambda c: c.str.strip())
ciwqs_lookup = all_npdes[["WDID", "Facility Name", "Latitude_CIWQS_from_npdes", "Longitude_CIWQS_from_npdes"]].drop_duplicates()

site = site.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

# ensure merge keys are normalized
ciwqs_key_cols = ["WDID", "Facility Name", "CWNS_ID", "CWNS Facility Name", "FACILITY_ID"]
ciwqs[ciwqs_key_cols] = ciwqs[ciwqs_key_cols].apply(lambda c: c.str.strip())

# Populate FACILITY_ID for rows that have a CWNS_ID but none yet
needs_fac_id = ciwqs["FACILITY_ID"].eq("") & ciwqs["CWNS_ID"].ne("") & ciwqs["CWNS_ID"].str.upper().ne("NA")
ciwqs.loc[needs_fac_id, "FACILITY_ID"] = ciwqs.loc[needs_fac_id, "CWNS_ID"].map(cwns_id_to_fac_id).fillna("")

ciwqs = ciwqs.merge(site_lookup, on=["WDID", "Facility Name"], how="left", suffixes=("", "_site"))
ciwqs = ciwqs.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

for dest, src in [("NPDES No.", "NPDES No._site"), ("Region", "Region_site")]:
    ciwqs[dest] = coalesce_blank(ciwqs[dest], ciwqs[src])
    ciwqs = ciwqs.drop(columns=[src])
for dest, src in [("Latitude_CIWQS", "Latitude_CIWQS_from_npdes"), ("Longitude_CIWQS", "Longitude_CIWQS_from_npdes")]:
    ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest])
    ciwqs = ciwqs.drop(columns=[src])

already_mapped = set(ciwqs["Facility Name"].map(normalize))
unmapped = (
    site[~site["Facility Name"].map(normalize).isin(already_mapped)]
    .drop_duplicates("Facility Name")
    .copy()
)
print(f"Unmapped facilities: {len(unmapped)}")

# Match unmapped facilities by name against CA Treatment Plant facilities
cwns_fac_ca = cwns_fac_tp[cwns_fac_tp["STATE_CODE"] == "CA"].copy()
cwns_fac_ca["normalized_name"] = cwns_fac_ca["CWNS Facility Name"].map(normalize)
name_idx = cwns_fac_ca.groupby("normalized_name").apply(lambda g: g.to_dict("records")).to_dict()

new_rows = []

for _, row in unmapped.iterrows():
    fac_name = row["Facility Name"].strip()
    permit = row["NPDES No."].strip().upper()
    base_entry = {
        "WDID": row["WDID"].strip(),
        "Place ID": row["Place ID"].strip(),
        "Facility Name": fac_name,
        "NPDES No.": permit,
        "Region": row["Region"].strip(),
        # left-merged coordinates are NaN for facilities missing from site_data_all
        "Latitude_CIWQS": str(row["Latitude_CIWQS_from_npdes"]).strip(),
        "Longitude_CIWQS": str(row["Longitude_CIWQS_from_npdes"]).strip(),
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
    ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest])
    ciwqs = ciwqs.drop(columns=[src])

ciwqs_out = ciwqs[ciwqs_columns]

if new_rows:
    new_df = pd.DataFrame(new_rows, columns=ciwqs_columns)
    combined = pd.concat([ciwqs_out, new_df], ignore_index=True)
    print(f"\nAdded {len(new_rows)} rows. ciwqs_to_cwns.csv now has {len(combined)} rows.")
else:
    combined = ciwqs_out
    print("\nNo new rows to add.")

# Dedupe on Place ID + FACILITY_ID, preferring rows with NPDES filled, preserving original row order
to_save = combined.reset_index(drop=True).rename_axis("orig_order").reset_index()
to_save["npdes_empty"] = to_save["NPDES No."].eq("")
(
    to_save.sort_values(["npdes_empty", "orig_order"])
    .drop_duplicates(subset=["Place ID", "FACILITY_ID"], keep="first")
    .sort_values("orig_order")
    [ciwqs_columns]
    .to_csv(CIWQS_TO_CWNS_CSV, index=False)
)

coord_cols = ["Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS"]
geo = combined[combined[coord_cols].replace("", pd.NA).notna().all(axis=1)].copy()
for col in coord_cols:
    geo[col] = pd.to_numeric(geo[col], errors="coerce")
geo = geo.dropna(subset=coord_cols)

geo["dist_miles"] = geo.apply(
    lambda r: geodesic((r["Latitude_CIWQS"], r["Longitude_CIWQS"]), (r["Latitude_CWNS"], r["Longitude_CWNS"])).miles,
    axis=1,
)
far = geo[geo["dist_miles"] > 2].sort_values("dist_miles", ascending=False)
print(f"\nRows where CWNS and CIWQS coords are >2 miles apart: {len(far)}")
print(far[["Facility Name", "NPDES No.", "CWNS_ID", "FACILITY_ID", "dist_miles"]].to_string(index=False))
