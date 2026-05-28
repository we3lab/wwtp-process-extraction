import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from helpers.metrics import (
    METRIC_SCORE_COLUMNS,
    compute_metrics,
    compute_facility_metric_rows,
)
from helpers.utils import (
    parse_status,
    is_present,
    PRESENT_STATUSES,
    get_leaf_names,
    merge_column_statuses,
    get_unspecified_leaf_names,
    unitprocess_keywords
)
from helpers.plotting import (
    COLORS,
    HATCH_PATTERNS,
    make_grouped_legend,
    save_and_close,
    set_thick_spines,
)

DATE_FOLDER = "2026-5-25"

MANUAL_STATUS_ORDER = ["PRESENT", "FUTURE", "PAST", "off_site"]

# Subset of METRIC_SCORE_COLUMNS for the facility violin (CSV / tables still use full set).
VIOLIN_METRIC_COLUMNS = tuple(c for c in METRIC_SCORE_COLUMNS if c not in {"Missed_Rate", "Hallucinated_Rate"})



def get_status_counts(process_name, unit_process_results):
    """Extract status breakdown for a process (PRESENT_AND_FUTURE folded into PRESENT)."""
    s = unit_process_results[process_name].map(parse_status)
    return {
        "PRESENT": int(s.isin(PRESENT_STATUSES).sum()),
        "FUTURE": int((s == "FUTURE").sum()),
    }


def create_method_deviation_plot(
    process_names,
    manual_df,
    llm_df,
    keyword_df,
    category_name,
    figsize=(12, 5),
    fontsize=12,
    save_path=None,
):
    """Plot LLM and keyword deviations from manual readings (above y=0: extra; below: missed)."""
    manual_facilities = set(manual_df["Place ID"].dropna())
    llm_common = manual_facilities & set(llm_df["Place ID"].dropna()) if llm_df is not None else set()
    kw_common = manual_facilities & set(keyword_df["Place ID"].dropna())

    rows = []
    for process in process_names:
        manual_proc = set(manual_df.loc[manual_df[process].map(is_present), "Place ID"])
        manual_count = len(manual_proc)

        llm_fp = llm_fn = 0
        if llm_df is not None:
            sub = llm_df[llm_df["Place ID"].isin(llm_common)]
            mask = sub[process].map(parse_status).isin(PRESENT_STATUSES)
            llm_proc = set(sub.loc[mask, "Place ID"])
            m = manual_proc & llm_common
            llm_fp, llm_fn = len(llm_proc - m), len(m - llm_proc)

        kw_fp = kw_fn = 0
        sub = keyword_df[keyword_df["Place ID"].isin(kw_common)]
        mask = sub[process].map(parse_status).isin(PRESENT_STATUSES)
        kw_proc = set(sub.loc[mask, "Place ID"])
        m = manual_proc & kw_common
        kw_fp, kw_fn = len(kw_proc - m), len(m - kw_proc)

        if not any([llm_fp, llm_fn, kw_fp, kw_fn, manual_count]):
            continue
        rows.append(
            {
                "Process": process,
                "Manual_Count": manual_count,
                "LLM_FP": llm_fp,
                "LLM_FN": llm_fn,
                "KW_FP": kw_fp,
                "KW_FN": kw_fn,
            }
        )

    if not rows:
        print(f"No deviation data for '{category_name}'")
        return

    df = pd.DataFrame(rows).sort_values("Manual_Count", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    w = 0.18

    for idx, row in df.iterrows():
        for x_off, fp, fn, color in [
            (-w, row["LLM_FP"], row["LLM_FN"], COLORS["npdes_llm"]),
            (+w, row["KW_FP"], row["KW_FN"], COLORS["npdes_kw"]),
        ]:
            x = idx + x_off
            width = w * 2
            if fp:
                ax.bar(
                    x,
                    fp,
                    width,
                    bottom=0,
                    color=color,
                    edgecolor="black",
                    linewidth=1.2,
                )
            if fn:
                y = -fn
                ax.bar(
                    x,
                    y,
                    width,
                    bottom=0,
                    color=color,
                    hatch=HATCH_PATTERNS["FUTURE"],
                    edgecolor="black",
                    linewidth=1.2,
                )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Process"], rotation=45, ha="right", fontsize=fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel("WWTP Count vs Manual Reading", fontsize=14)
    set_thick_spines(ax, linewidth=1.6)
    make_grouped_legend(
        ax,
        groups=[
            {
                "header": "Method",
                "items": [
                    ("  NPDES - LLM Extraction", {"facecolor": COLORS["npdes_llm"]}),
                    ("  NPDES Keyword", {"facecolor": COLORS["npdes_kw"]}),
                ],
            },
            {
                "header": "vs Manual Reading",
                "items": [
                    ("  Extra (above)", {"facecolor": "gray"}),
                    (
                        "  Missed (below)",
                        {"facecolor": "gray", "hatch": HATCH_PATTERNS["FUTURE"]},
                    ),
                ],
            },
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=11,
    )

    plt.subplots_adjust(bottom=0.25)
    if save_path:
        save_and_close(fig, save_path, dpi=300)
    else:
        plt.close(fig)


def get_manual_status_counts(process_name, manual_df):
    """Status counts from manual readings for a single process."""
    counts = {s: 0 for s in MANUAL_STATUS_ORDER}
    for val in manual_df[process_name]:
        cat = parse_status(val)
        if not cat:
            continue
        key = "off_site" if cat == "OFFSITE" else cat
        if key in counts:
            counts[key] += 1
    return counts


def build_method_metric_inputs(process_names, manual_df, pred_df, pred_key_col="Place ID"):
    """Build key-aligned manual/pred dataframes for helpers.metrics.compute_metrics."""
    common_keys = sorted(set(manual_df["Place ID"].dropna()) & set(pred_df[pred_key_col].dropna()))

    manual_sub = (
        manual_df[manual_df["Place ID"].isin(common_keys)]
        .drop_duplicates(subset="Place ID")
        .set_index("Place ID")
    )
    pred_sub = (
        pred_df[pred_df[pred_key_col].isin(common_keys)].drop_duplicates(subset=pred_key_col).set_index(pred_key_col)
    )

    manual_metric_df = pd.DataFrame({"key": common_keys})
    pred_metric_df = pd.DataFrame({"key": common_keys})

    for process in process_names:
        manual_metric_df[process] = (
            manual_sub.reindex(common_keys)[process].map(parse_status).values
        )
        pred_metric_df[process] = pred_sub.reindex(common_keys)[process].map(parse_status).values

    return manual_metric_df, pred_metric_df, common_keys


def _add_split_violin_gap(ax, gap=0.04):
    """Shift split violin halves away from center to create visible horizontal separation."""
    for poly in ax.collections:
        if not hasattr(poly, "get_paths"):
            continue
        paths = poly.get_paths()
        if not paths:
            continue
        verts = paths[0].vertices
        x_mean = float(np.mean(verts[:, 0]))
        nearest_tick = round(x_mean)
        side = np.sign(x_mean - nearest_tick)
        if side != 0:
            verts[:, 0] += side * gap

_PALETTE_METRIC = {"LLM": COLORS["npdes_llm"], "Keyword": COLORS["npdes_kw"]}

def draw_split_violin(ax, facility_metrics_df, panel_label):
    score_cols = [c for c in VIOLIN_METRIC_COLUMNS if c in facility_metrics_df.columns]
    plot_df = (
        facility_metrics_df[facility_metrics_df["Source"].isin(["LLM", "Keyword"])]
        .melt(
            id_vars=["Source", "key"],
            value_vars=score_cols,
            var_name="Metric",
            value_name="Value",
        )
        .dropna(subset=["Value"])
    )
    n_fac = int(plot_df["key"].nunique())

    sns.violinplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Source",
        order=score_cols,
        hue_order=["LLM", "Keyword"],
        split=True,
        palette=_PALETTE_METRIC,
        inner=None,
        cut=0,
        linewidth=1.2,
        ax=ax,
    )
    for poly in ax.collections:
        if hasattr(poly, "set_edgecolor"):
            poly.set_edgecolor("black")
        if hasattr(poly, "set_linewidth"):
            poly.set_linewidth(1.2)
    _add_split_violin_gap(ax, gap=0.04)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Facility-level score\n(N={n_fac})", fontsize=14)
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels([label.replace("_", " ") for label in score_cols], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("")
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.15), ncol=2, fontsize=11, frameon=False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=16)
    set_thick_spines(ax, linewidth=1.6)

def aggregate_to_category_states(metric_df, category_to_leaves):
    """Collapse leaf-status columns into category-status columns per facility key."""
    out = pd.DataFrame({"key": metric_df["key"]})
    for category, leaves in category_to_leaves.items():
        present_leaves = [c for c in leaves if c in metric_df.columns]
        if not present_leaves:
            out[category] = ""
            continue
        out[category] = metric_df[present_leaves].apply(merge_column_statuses, axis=1)
    return out


categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load keyword-based NPDES results
unit_process_results = pd.read_csv(
    f"wwtp_process_extraction/output/{DATE_FOLDER}/kw_unit_processes_by_facility.csv", dtype=str
).fillna("")
print(f"NPDES keyword data: Loaded {len(unit_process_results)} unique facilities")

unit_full = unit_process_results.copy()

# Load LLM results
llm_results_path = f"wwtp_process_extraction/output/{DATE_FOLDER}/llm_unit_processes_by_facility.csv"
llm_results = pd.read_csv(llm_results_path, dtype=str).fillna("") if os.path.exists(llm_results_path) else None
if llm_results is not None:
    llm_results = llm_results[llm_results["Place ID"].ne("")].copy()
    print(f"LLM results: {len(llm_results)} unique facilities")
else:
    print(f"LLM results not found at {llm_results_path}; breakdown plots will show keyword only")

# Filter to facilities processed by BOTH methods
if llm_results is not None:
    llm_facilities = set(llm_results["Place ID"])
    kw_facilities = set(unit_full["Place ID"])
    both_facilities = llm_facilities & kw_facilities
    llm_results_both = llm_results[llm_results["Place ID"].isin(both_facilities)].copy()
    unit_full_both = unit_full[unit_full["Place ID"].isin(both_facilities)].copy()
    print(
        f"Facilities processed by both LLM and keyword: {len(both_facilities)} "
        f"(LLM only: {len(llm_facilities - kw_facilities)}, "
        f"keyword only: {len(kw_facilities - llm_facilities)})"
    )
else:
    llm_results_both = None
    unit_full_both = unit_full

# Load manual readings (train + test) as the deviation baseline
train_manual = pd.read_csv("wwtp_process_extraction/data/train_set_npdes_manual.csv", dtype=str).fillna("")
train_manual["Place ID"] = train_manual["Place ID"].str.strip()
test_manual = pd.read_csv("wwtp_process_extraction/data/test_set_npdes_manual.csv", dtype=str).fillna("")
test_manual["Place ID"] = test_manual["Place ID"].str.strip()
manual_combined = (
    pd.concat([train_manual, test_manual]).drop_duplicates(subset="Place ID").reset_index(drop=True)
)
manual_combined = manual_combined[manual_combined["Place ID"].ne("")].copy()
manual_facilities = set(manual_combined["Place ID"])
llm_results_manual = (
    llm_results_both[llm_results_both["Place ID"].isin(manual_facilities)].copy()
    if llm_results_both is not None
    else None
)
unit_full_manual = unit_full_both[unit_full_both["Place ID"].isin(manual_facilities)].copy()
print(
    f"Manual baseline: {len(manual_combined)} facilities "
    f"({len(manual_facilities & set(unit_full_both['Place ID']))} matched to keyword, "
    f"{len(manual_facilities & set(llm_results_both['Place ID'])) if llm_results_both is not None else 0} matched to LLM)"
)

figures_dir = f"wwtp_process_extraction/output/{DATE_FOLDER}/figures"
os.makedirs(figures_dir, exist_ok=True)

excluded_unspecified = get_unspecified_leaf_names(unitprocess_keywords)

for category in categories_to_plot:
    print(f"\nProcessing category: {category}")
    safe_category = category.replace("/", "_").replace(os.sep, "_")

    process_names = [
        p for p in get_leaf_names(category, unitprocess_keywords[category])
        if p not in excluded_unspecified
    ]

    # Method vs manual reading: deviation bars (FP above zero, FN below)
    create_method_deviation_plot(
        process_names,
        manual_combined,
        llm_results_manual,
        unit_full_manual,
        category,
        save_path=f"{figures_dir}/{safe_category}_npdes_method_comparison_deviation.png",
    )
    print(f"  Saved {safe_category}_npdes_method_comparison_deviation.png")


# ── Method comparison metrics ─────────────────────────────────────────────────
all_process_list = [
    p for cat in categories_to_plot for p in get_leaf_names(cat, unitprocess_keywords[cat])
]
unit_process_list = [p for p in all_process_list if p not in excluded_unspecified]
metrics_frames = []
facility_metric_rows = []

manual_metric_kw, pred_metric_kw, _ = build_method_metric_inputs(
    all_process_list, manual_combined, unit_full_both, "Place ID"
)
kw_metrics = compute_metrics(manual_metric_kw, pred_metric_kw, unit_process_list, "Keyword")
metrics_frames.append(
    kw_metrics.rename(columns={"Label": "Process"}).assign(Level="Unit_Process")
)
facility_metric_rows.extend(
    compute_facility_metric_rows(manual_metric_kw, pred_metric_kw, unit_process_list, "Keyword")
)

if llm_results_both is not None:
    manual_metric_llm, pred_metric_llm, _ = build_method_metric_inputs(
        all_process_list, manual_combined, llm_results_both, "Place ID"
    )
    llm_metrics = compute_metrics(manual_metric_llm, pred_metric_llm, unit_process_list, "LLM")
    metrics_frames.append(
        llm_metrics.rename(columns={"Label": "Process"}).assign(Level="Unit_Process")
    )
    facility_metric_rows.extend(
        compute_facility_metric_rows(manual_metric_llm, pred_metric_llm, unit_process_list, "LLM")
    )

unit_process_metrics_df = pd.DataFrame(facility_metric_rows)
final_dir = f"wwtp_process_extraction/output/{DATE_FOLDER}/final"
os.makedirs(final_dir, exist_ok=True)

# Build category-level facility metrics by collapsing leaf states to category states.
category_to_leaves = {
    cat: get_leaf_names(cat, unitprocess_keywords[cat]) for cat in categories_to_plot
}
category_metric_rows = []
manual_cat_kw = aggregate_to_category_states(manual_metric_kw, category_to_leaves)
pred_cat_kw = aggregate_to_category_states(pred_metric_kw, category_to_leaves)
cat_metrics_kw = compute_metrics(manual_cat_kw, pred_cat_kw, categories_to_plot, "Keyword")
metrics_frames.append(cat_metrics_kw.rename(columns={"Label": "Process"}).assign(Level="Category"))
category_metric_rows.extend(
    compute_facility_metric_rows(manual_cat_kw, pred_cat_kw, categories_to_plot, "Keyword")
)
if llm_results_both is not None:
    manual_cat_llm = aggregate_to_category_states(manual_metric_llm, category_to_leaves)
    pred_cat_llm = aggregate_to_category_states(pred_metric_llm, category_to_leaves)
    cat_metrics_llm = compute_metrics(manual_cat_llm, pred_cat_llm, categories_to_plot, "LLM")
    metrics_frames.append(
        cat_metrics_llm.rename(columns={"Label": "Process"}).assign(Level="Category")
    )
    category_metric_rows.extend(
        compute_facility_metric_rows(manual_cat_llm, pred_cat_llm, categories_to_plot, "LLM")
    )
category_metrics_df = pd.DataFrame(category_metric_rows)

metrics_df = pd.concat(metrics_frames, ignore_index=True)
metrics_path = f"wwtp_process_extraction/output/{DATE_FOLDER}/npdes_method_comparison_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"\nSaved npdes_method_comparison_metrics.csv")
summary = metrics_df.groupby(["Level", "Source"])[
    [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "Missed_Rate",
        "Hallucinated_Rate",
        "State_Accuracy",
    ]
].mean()
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
kw_hallucinated = (
    metrics_df[(metrics_df["Source"] == "Keyword") & (metrics_df["Level"] == "Unit_Process")][
        ["Process", "Hallucinated_Rate", "FP", "Support_Pred"]
    ]
    .dropna(subset=["Hallucinated_Rate"])
    .sort_values(["Hallucinated_Rate", "FP", "Support_Pred"], ascending=[False, False, False])
    .head(12)
)
if not kw_hallucinated.empty:
    print("\nTop hallucinated unit processes (Keyword):")
    print(kw_hallucinated.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

violin_path = f"{final_dir}/figure_3_npdes_method_comparison_metrics_violin.png"
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6))
draw_split_violin(ax_top, unit_process_metrics_df, "A.")
draw_split_violin(ax_bottom, category_metrics_df, "B.")
# subplot titles
ax_top.set_title("Unit Process-Level Metrics")
ax_bottom.set_title("Category-Level Metrics")
fig.tight_layout(h_pad=2.0)
save_and_close(fig, violin_path, dpi=300)
print(f"Saved {os.path.basename(violin_path)}")

# Overall status summary
total_present = total_present_and_future = total_future = 0
for category in categories_to_plot:
    for process_name in get_leaf_names(category, unitprocess_keywords[category]):
        s = unit_full[process_name].str.upper()
        total_present += int((s == "PRESENT").sum())
        total_present_and_future += int((s == "PRESENT_AND_FUTURE").sum())
        total_future += int((s == "FUTURE").sum())

print(f"Total process instances marked as 'PRESENT': {total_present}")
print(f"Total process instances marked as 'PRESENT_AND_FUTURE': {total_present_and_future}")
print(f"Total process instances marked as 'FUTURE': {total_future}")
print(f"Grand total: {total_present + total_present_and_future + total_future}")
