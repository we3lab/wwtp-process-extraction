"""
Grouped stacked bars of unit-process detection by source: CWNS vs permit LLM extraction (+ keyword
search in figure_s4), over facilities with an exact CWNS_ID match in ciwqs_to_cwns.csv (no fuzzy matching).
Each facility counts once at its highest-priority status; PRESENT_AND_FUTURE counts as Present.
Also writes the unmatched-facility CSVs, the merged unit_processes_by_facility.csv, per-process
counts, and rewrites ciwqs_to_cwns.csv with name-matched rows and coordinates.
"""

import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from helpers.paths import *
from helpers.utils import (
    get_leaf_names,
    category_leaf_names,
    PRESENT_STATUSES,
    cwns_mapping,
    no_cwns_pids,
    build_cwns_facility_processes,
    add_county_and_sort,
    unitprocess_keywords as keywords,
)
from helpers.plotting import COLORS, make_grouped_legend, save_and_close, set_thick_spines

# Hatch patterns for status values
HATCH_PATTERNS = {
    "PRESENT": "",  # Solid fill
    "FUTURE": "/////",  # Forward slashes
    "PAST": "\\\\\\\\\\",  # Backslashes
    "OFFSITE": "....",  # Dots
}
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


def get_facility_counts(df, leaf_cols):
    """Unique-facility counts: each facility counted once at highest-priority status."""
    statuses = df[leaf_cols]
    has_present = statuses.isin(PRESENT_STATUSES).any(axis=1)
    has_past = statuses.eq("PAST").any(axis=1)
    has_future = statuses.eq("FUTURE").any(axis=1)
    has_offsite = statuses.eq("OFFSITE").any(axis=1)
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
                if val <= 0:
                    continue
                facecolor = "white" if key == "NOT_PRESENT" else COLORS[color_key]
                ax.bar(pos + offset, val, bar_width, bottom=bottom, color=facecolor,
                       edgecolor="black", linewidth=1.2)
                if hatch:
                    hatch_bar = ax.bar(pos + offset, val, bar_width, bottom=bottom, color="none",
                                       hatch=hatch, edgecolor="white", linewidth=0.0)
                    for patch in hatch_bar:
                        patch.set_hatch_linewidth(1.0)
                bottom += val
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("WWTP Count", fontsize=14)
    set_thick_spines(ax)
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
        fontsize=12,
    )


# Load data

proc_cols = sorted({leaf for leaves in category_leaf_names.values() for leaf in leaves})

llm_df = pd.read_csv(FACILITY_LLM_CSV, dtype=str)
kw_df = pd.read_csv(FACILITY_KW_CSV, dtype=str)
ca_cwns = pd.read_csv(CWNS_TABLE_CSV, dtype=str)
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
print("\nFacility coverage (unique WDID + Facility Name, before overlap filter):")
print(f"  Both CWNS + site_data extracted (KW):           {len(kw_facilities & cwns_pids)}")
print(f"  Both CWNS + site_data extracted (LLM):          {len(llm_facilities & cwns_pids)}")
print(f"  site_data extracted only (no CWNS):             {len(kw_facilities - cwns_pids)}")
print(f"  Mapped to CWNS but dropped from site_data:      {len(cwns_pids - kw_facilities)}")
print(f"  All 3 sources:                                  {len(llm_facilities & kw_facilities & cwns_pids)}")

# CWNS-only facilities as plain comma-separated text
print("CWNS only facilities:")
print(", ".join(cwns_pids - kw_facilities))

cwns_df, merged_map = build_cwns_facility_processes(
    ca_cwns, target_facilities=llm_facilities | kw_facilities
)

print(f"\n  CIWQS mapping rows with CWNS survey attach: {len(merged_map)}")

# Save facilities with no CWNS match
pid_to_name = {}
for df in (site, kw_df):
    for pid, name in zip(df["Place ID"], df["Facility Name"]):
        pid_to_name.setdefault(pid, name)

unmatched_pids = [
    pid for pid in sorted(kw_facilities | set(site["Place ID"]))
    if pid not in cwns_pids and pid not in no_cwns_pids
]

kw_has_data = set(
    kw_df.loc[(kw_df[proc_cols].ne("")).any(axis=1), "Place ID"]
)
unmatched_df = pd.DataFrame({
    "Place ID": unmatched_pids,
    "FACILITY_NAME": [pid_to_name[pid] for pid in unmatched_pids],
    "has_kw_unit_process_data": ["yes" if pid in kw_has_data else "no" for pid in unmatched_pids],
}).sort_values("has_kw_unit_process_data", key=lambda s: s.map({"yes": 0, "no": 1}))
unmatched_df.to_csv(OUTPUT_DIR / "unmatched_kw_no_cwns.csv", index=False)
print(f"  Unmatched KW/site_data (no CWNS): {len(unmatched_pids)} → unmatched_kw_no_cwns.csv")
# print plain text, comma-separated around strings
print(', '.join(unmatched_df['FACILITY_NAME']))

# CWNS rows with no declared match in ciwqs_to_cwns (by CWNS_ID)
unmatched_cwns_no_kw = ca_cwns.loc[
    ~ca_cwns["CWNS_ID"].isin(cwns_mapping["CWNS_ID"]), ["CWNS_ID", "FACILITY_ID", "FACILITY_NAME"]
].drop_duplicates()
unmatched_cwns_no_kw.to_csv(OUTPUT_DIR / "unmatched_cwns_no_kw.csv", index=False)


# Final merged unit-process table (CWNS + CIWQS/LLM)
# One row per (facility, source). source ∈ {cwns, ciwqs}. Facilities present in
# only one source get a single row. Built from the full llm_df, before the
# overlap filtering below, so ciwqs-only facilities are kept.
mapping_by_pid = cwns_mapping.drop_duplicates("Place ID").set_index("Place ID").to_dict("index")
llm_name_by_pid = dict(zip(llm_df["Place ID"], llm_df["Facility Name"]))

final_rows = []
for source_name, src_df in [("Clean Watershed Needs Survey", cwns_df), ("ciwqs", llm_df)]:
    for _, r in src_df.iterrows():
        pid = r["Place ID"]
        # ciwqs-only facilities have no CWNS mapping row
        m = mapping_by_pid.get(pid, {})
        final_rows.append({
            "source": source_name,
            "CWNS FACILITY_ID": m.get("FACILITY_ID", ""),
            "CWNS FACILITY_NAME": m.get("CWNS Facility Name", ""),
            "CIWQS PLACE_ID": pid,
            "CIWQS Facility Name": llm_name_by_pid[pid],
            **r[proc_cols].to_dict(),
        })

final_cols = ["source", "CWNS FACILITY_ID", "CWNS FACILITY_NAME", "CIWQS PLACE_ID", "CIWQS Facility Name"]
final_df = pd.DataFrame(final_rows, columns=final_cols + proc_cols)
final_df = add_county_and_sort(final_df, "CIWQS Facility Name", place_id_col="CIWQS PLACE_ID")
final_df.to_csv(FACILITY_CSV, index=False)
n_cwns_rows = (final_df["source"] == "Clean Watershed Needs Survey").sum()
print(f"\nSaved merged unit processes ({len(final_df)} rows, "
      f"{n_cwns_rows} cwns + {(final_df['source'] == 'ciwqs').sum()} ciwqs): {FACILITY_CSV}")

# Overlap only from here on: permit facilities with a CWNS match, and their CWNS rows
cwns_common_df = cwns_df[cwns_df["Place ID"].isin(llm_facilities)]
llm_common_df = llm_df[llm_df["Place ID"].isin(cwns_df["Place ID"])]

# Unit process counts for all processes, reused by the per-stage plots
counts_by_proc = {}
count_rows = []
for proc in proc_cols:
    cwns_c = get_facility_counts(cwns_common_df, [proc])
    llm_c = get_facility_counts(llm_common_df, [proc])
    counts_by_proc[proc] = {"cwns": cwns_c, "llm": llm_c}
    count_rows.append({
        "process": proc,
        **{f"cwns_{k.lower()}": v for k, v in cwns_c.items()},
        **{f"llm_{k.lower()}": v for k, v in llm_c.items()},
    })
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
            counts = counts_by_proc[leaf]
            if detected_count(counts["cwns"]) >= MIN_COUNT or detected_count(counts["llm"]) >= MIN_COUNT:
                cat_items.append({"label": leaf, "cat": cat_name, **counts})
        # X order: lowest → highest NPDES (right bar) "Not Present"
        cat_items.sort(key=lambda item: (item["llm"]["NOT_PRESENT"], item["label"]))
        items.extend(cat_items)
    if not items:
        print(f"  {group_title}: all below threshold, skipping")
        continue
    # Bar groups 1 apart, plus a 0.25 gap where the category changes
    positions = []
    x = 0.0
    for i, item in enumerate(items):
        if i and item["cat"] != items[i - 1]["cat"]:
            x += 0.25
        positions.append(x)
        x += 1.0

    # Figure width based on number of bar groups (positions span)
    fig, ax = plt.subplots(figsize=(max(7, (positions[-1] + 1) * 0.85), 5))
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
        # name the category only when it has more than one bar group
        if span_end > span_start:
            ax.text((span_start + span_end) / 2, ylim[1] * 0.97, cat, ha="center", va="top",
                    fontsize=11, color="#444444", style="italic")
    ax.set_ylim(ylim)
    plt.tight_layout()
    save_and_close(fig, BY_CATEGORY_DIR / f"{group_title.replace(' ', '_')}_source_comparison.png")


# 2. Major-categories plot
# One bar group per top-level JSON category. Each facility is counted once per category,
# at its highest-priority status across all leaves in that category.

all_three = llm_facilities & kw_facilities & set(cwns_df["Place ID"])
major_plots = [
    # (filename, compare_dfs, source_items, bar_width, figure width per category)
    ("figure_3", [cwns_common_df, llm_common_df], [CWNS_SOURCE, LLM_SOURCE], BAR_WIDTH, 0.55),
    ("figure_s4", [df[df["Place ID"].isin(all_three)] for df in (cwns_df, kw_df, llm_df)],
     [CWNS_SOURCE, KW_SOURCE, LLM_SOURCE], 0.24, 0.7),
]
for filename, compare_dfs, source_items, bar_width, width_per_cat in major_plots:
    # (category, per-source counts), kept when any source detects it at least MAJOR_MIN_COUNT times
    categories = []
    for cat_name, leaves in category_leaf_names.items():
        counts = [get_facility_counts(df, leaves) for df in compare_dfs]
        if any(detected_count(c) >= MAJOR_MIN_COUNT for c in counts):
            categories.append((cat_name, counts))
    # LLM counts are last in every source order
    categories.sort(key=lambda category: (category[1][-1]["NOT_PRESENT"], category[0]))
    cat_labels = [cat_name for cat_name, _ in categories]
    source_counts = [[counts[i] for _, counts in categories] for i in range(len(compare_dfs))]

    n = len(cat_labels)
    # Drop the Offsite legend entry unless some bar actually has an offsite band
    has_offsite = any(c["OFFSITE"] > 0 for counts in source_counts for c in counts)
    legend_items = STATUS_LEGEND_ITEMS_WITH_NOT_PRESENT
    if not has_offsite:
        legend_items = [it for it in legend_items if it[0] != "Offsite"]
    fig, ax = plt.subplots(figsize=(max(14, n * width_per_cat), 6))
    render_source_plot(
        ax=ax,
        labels=cat_labels,
        positions=list(range(n)),
        source_items=source_items,
        stack_order=STACK_ORDER_WITH_NOT_PRESENT,
        status_legend_items=legend_items,
        source_counts=source_counts,
        bar_width=bar_width,
    )
    plt.tight_layout()
    save_and_close(fig, FIGURES_DIR / f"{filename}.png")
    print(f"    Saved {filename}.png")


# Rewrite ciwqs_to_cwns.csv: fill FACILITY_ID, NPDES, Region and coordinates, add name matches
site_lookup = site[["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]].drop_duplicates()
ciwqs_lookup = all_npdes[
    ["WDID", "Facility Name", "Latitude_CIWQS_from_npdes", "Longitude_CIWQS_from_npdes"]
].drop_duplicates()
site = site.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")
cwns_coords = ca_cwns[["CWNS_ID", "FACILITY_ID", "LATITUDE", "LONGITUDE"]].rename(
    columns={"LATITUDE": "Latitude_CWNS_from_cwns", "LONGITUDE": "Longitude_CWNS_from_cwns"}
)

# Populate FACILITY_ID for rows that have a CWNS_ID but none yet (blank/"NA" CWNS_IDs stay blank)
no_fac_id = ciwqs["FACILITY_ID"].eq("")
cwns_fac_id = ca_cwns.set_index("CWNS_ID")["FACILITY_ID"]
ciwqs.loc[no_fac_id, "FACILITY_ID"] = ciwqs.loc[no_fac_id, "CWNS_ID"].map(cwns_fac_id).fillna("")

ciwqs = ciwqs.merge(site_lookup, on=["WDID", "Facility Name"], how="left", suffixes=("", "_site"))
ciwqs = ciwqs.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")
ciwqs = ciwqs.merge(cwns_coords, on=["CWNS_ID", "FACILITY_ID"], how="left")

# Stored NPDES/Region win; freshly merged coordinates win over the stored ones
for dest, src in [("NPDES No.", "NPDES No._site"), ("Region", "Region_site")]:
    ciwqs[dest] = coalesce_blank(ciwqs[dest], ciwqs[src])
for dest, src in [
    ("Latitude_CIWQS", "Latitude_CIWQS_from_npdes"),
    ("Longitude_CIWQS", "Longitude_CIWQS_from_npdes"),
    ("Latitude_CWNS", "Latitude_CWNS_from_cwns"),
    ("Longitude_CWNS", "Longitude_CWNS_from_cwns"),
]:
    ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest])

already_mapped = set(ciwqs["Facility Name"].map(normalize))
unmapped = site[~site["Facility Name"].map(normalize).isin(already_mapped)].drop_duplicates("Facility Name")
print(f"Unmapped facilities: {len(unmapped)}")

# Match unmapped facilities by name against CWNS facilities
cwns_by_name = {}
cwns_names = ca_cwns[["CWNS_ID", "FACILITY_ID", "FACILITY_NAME"]]
for cwns_id, fac_id, cwns_name in cwns_names.itertuples(index=False):
    cwns_name = cwns_name.strip()
    cwns_by_name.setdefault(normalize(cwns_name), []).append(
        {"CWNS_ID": cwns_id, "FACILITY_ID": fac_id, "CWNS Facility Name": cwns_name}
    )

new_rows = []
for _, row in unmapped.iterrows():
    fac_name, permit = row["Facility Name"], row["NPDES No."]
    base_entry = {
        "WDID": row["WDID"],
        "Place ID": row["Place ID"],
        "Facility Name": fac_name,
        "NPDES No.": permit,
        "Region": row["Region"],
        # left-merged coordinates are NaN for facilities missing from site_data_all
        "Latitude_CIWQS": str(row["Latitude_CIWQS_from_npdes"]),
        "Longitude_CIWQS": str(row["Longitude_CIWQS_from_npdes"]),
        "Latitude_CWNS": "",
        "Longitude_CWNS": "",
    }
    cwns_hits = cwns_by_name.get(normalize(fac_name), [])
    if cwns_hits:
        print(f"  [name] {permit} — {fac_name} → {len(cwns_hits)} CWNS row(s)")
        new_rows.extend({**base_entry, **hit} for hit in cwns_hits)
    else:
        print(f"  [no match] {permit} — {fac_name}")
        new_rows.append({**base_entry, "CWNS_ID": "", "FACILITY_ID": "", "CWNS Facility Name": ""})

ciwqs_out = ciwqs[ciwqs_columns]
if new_rows:
    combined = pd.concat([ciwqs_out, pd.DataFrame(new_rows, columns=ciwqs_columns)], ignore_index=True)
    print(f"\nAdded {len(new_rows)} rows. ciwqs_to_cwns.csv now has {len(combined)} rows.")
else:
    combined = ciwqs_out
    print("\nNo new rows to add.")

# Dedupe on Place ID + FACILITY_ID, preferring rows with NPDES filled, preserving original row order
(
    combined.sort_values("NPDES No.", key=lambda s: s.eq(""), kind="stable")
    .drop_duplicates(subset=["Place ID", "FACILITY_ID"])
    .sort_index()
    .to_csv(CIWQS_TO_CWNS_CSV, index=False)
)

coord_cols = ["Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS"]
geo = combined.copy()
for col in coord_cols:
    geo[col] = pd.to_numeric(geo[col], errors="coerce")
geo = geo.dropna(subset=coord_cols)

geo["dist_miles"] = geo.apply(
    lambda r: geodesic(
        (r["Latitude_CIWQS"], r["Longitude_CIWQS"]), (r["Latitude_CWNS"], r["Longitude_CWNS"])
    ).miles,
    axis=1,
)
far = geo[geo["dist_miles"] > 2].sort_values("dist_miles", ascending=False)
print(f"\nRows where CWNS and CIWQS coords are >2 miles apart: {len(far)}")
print(far[["Facility Name", "NPDES No.", "CWNS_ID", "FACILITY_ID", "dist_miles"]].to_string(index=False))
