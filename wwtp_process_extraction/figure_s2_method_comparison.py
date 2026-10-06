import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from step6_postprocess_llm_output import build_model_comparison
from helpers.metrics import (
    METRIC_SCORE_COLUMNS,
    build_metric_inputs,
    aggregate_to_category_states,
    compute_metrics,
    compute_facility_metric_rows,
)
from helpers.utils import (
    get_leaf_names,
    unitprocess_keywords,
    OUTPUT_DIR,
    FIGURES_DIR,
    MANUAL_CSV,
)
from helpers.plotting import (
    COLORS,
    save_and_close,
    set_thick_spines,
)

# Subset of METRIC_SCORE_COLUMNS for the facility violin (CSV / tables still use full set).
VIOLIN_METRIC_COLUMNS = tuple(c for c in METRIC_SCORE_COLUMNS if c not in {"Missed_Rate", "Hallucinated_Rate"})

VIOLIN_SOURCES = ["Keyword", "GPT-5", "GPT-5 mini"]
VIOLIN_PALETTE = {
    "Keyword": COLORS["npdes_kw"],
    "GPT-5": COLORS["gpt-5"],
    "GPT-5 mini": COLORS["gpt-5-mini"],
}


def draw_violin(ax, facility_metrics_df, panel_label):
    score_cols = list(VIOLIN_METRIC_COLUMNS)
    plot_df = (
        facility_metrics_df[facility_metrics_df["Source"].isin(VIOLIN_SOURCES)]
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
        hue_order=VIOLIN_SOURCES,
        palette=VIOLIN_PALETTE,
        inner=None,
        cut=0,
        density_norm="width",
        common_norm=False,
        width=0.8,
        linewidth=1.2,
        ax=ax,
    )
    for poly in ax.collections:
        poly.set_edgecolor("black")
        poly.set_linewidth(1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Facility-level score\n(N={n_fac})", fontsize=14)
    ax.tick_params(axis="x", rotation=0, labelsize=12)
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels([label.replace("_", " ") for label in score_cols], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("")
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.18), ncol=3, fontsize=11, frameon=False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=16)
    set_thick_spines(ax, linewidth=1.6)


categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load keyword-based NPDES results
keyword_results = pd.read_csv(OUTPUT_DIR / "unit_processes_by_facility_kw.csv", dtype=str).fillna("")
print(f"NPDES keyword data: Loaded {len(keyword_results)} unique facilities")

# Load LLM results
llm_results = pd.read_csv(OUTPUT_DIR / "unit_processes_by_facility_llm.csv", dtype=str).fillna("")
print(f"LLM results: {len(llm_results)} unique facilities")

# Filter to facilities processed by BOTH methods
llm_facilities = set(llm_results["Place ID"])
kw_facilities = set(keyword_results["Place ID"])
both_facilities = llm_facilities & kw_facilities
llm_results_both = llm_results[llm_results["Place ID"].isin(both_facilities)].copy()
keyword_results_both = keyword_results[keyword_results["Place ID"].isin(both_facilities)].copy()
print(
    f"Facilities processed by both LLM and keyword: {len(both_facilities)} "
    f"(LLM only: {len(llm_facilities - kw_facilities)}, "
    f"keyword only: {len(kw_facilities - llm_facilities)})"
)

# Load manual readings (train + test) as the baseline
manual = pd.read_csv(MANUAL_CSV, dtype=str).fillna("")

# The manual CSV is exactly the benchmark (manually-read) facility set.
manual_facilities = set(manual["Place ID"])

print(
    f"Manual baseline: {len(manual)} facilities "
    f"({len(manual_facilities & set(keyword_results_both['Place ID']))} matched to keyword, "
    f"{len(manual_facilities & set(llm_results_both['Place ID']))} matched to LLM)"
)


# Method comparison metrics
all_process_list = [
    p for cat in categories_to_plot for p in get_leaf_names(cat, unitprocess_keywords[cat])
]
unit_process_list = [
    p for cat in categories_to_plot for p in get_leaf_names(cat, unitprocess_keywords[cat], exclude_unspecified=True)
]

# Per-model ontology results (gpt-5, gpt-5-mini) for the violin comparison, sourced from
# the same postprocessed JSONs table_1 uses — restricted to the manual-read benchmark facilities.
model_comparison = build_model_comparison()
prediction_sources = [("Keyword", keyword_results_both), ("LLM", llm_results_both)]
for model_label, source_name in [("gpt-5", "GPT-5"), ("gpt-5-mini", "GPT-5 mini")]:
    model_df = model_comparison[
        (model_comparison["Method"] == "Ontology") & (model_comparison["Model"] == model_label)
    ]
    prediction_sources.append((source_name, model_df))
metric_inputs = {
    source_name: build_metric_inputs(all_process_list, manual, pred_df)
    for source_name, pred_df in prediction_sources
}

# Category-level metrics collapse leaf states to category states
category_to_leaves = {
    cat: get_leaf_names(cat, unitprocess_keywords[cat]) for cat in categories_to_plot
}
metrics_frames = []
facility_metric_rows = []
category_metric_rows = []
for source_name, (manual_metric, pred_metric) in metric_inputs.items():
    source_metrics = compute_metrics(manual_metric, pred_metric, unit_process_list, source_name)
    metrics_frames.append(
        source_metrics.rename(columns={"Label": "Process"}).assign(Level="Unit_Process")
    )
    facility_metric_rows.extend(
        compute_facility_metric_rows(manual_metric, pred_metric, unit_process_list, source_name)
    )
    manual_cat = aggregate_to_category_states(manual_metric, category_to_leaves)
    pred_cat = aggregate_to_category_states(pred_metric, category_to_leaves)
    cat_metrics = compute_metrics(manual_cat, pred_cat, categories_to_plot, source_name)
    metrics_frames.append(
        cat_metrics.rename(columns={"Label": "Process"}).assign(Level="Category")
    )
    category_metric_rows.extend(
        compute_facility_metric_rows(manual_cat, pred_cat, categories_to_plot, source_name)
    )
unit_process_metrics_df = pd.DataFrame(facility_metric_rows)
category_metrics_df = pd.DataFrame(category_metric_rows)

metrics_df = pd.concat(metrics_frames, ignore_index=True)
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
print("\nTop hallucinated unit processes (Keyword):")
print(kw_hallucinated.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

violin_path = FIGURES_DIR / "figure_s2.png"
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6))
draw_violin(ax_top, unit_process_metrics_df, "A.")
draw_violin(ax_bottom, category_metrics_df, "B.")
fig.tight_layout(h_pad=3.5)
save_and_close(fig, violin_path, dpi=300)
print(f"Saved {violin_path.name}")

# Overall status summary
total_present = total_present_and_future = total_future = 0
for process_name in all_process_list:
    s = keyword_results[process_name].str.upper()
    total_present += int((s == "PRESENT").sum())
    total_present_and_future += int((s == "PRESENT_AND_FUTURE").sum())
    total_future += int((s == "FUTURE").sum())

print(f"Total process instances marked as 'PRESENT': {total_present}")
print(f"Total process instances marked as 'PRESENT_AND_FUTURE': {total_present_and_future}")
print(f"Total process instances marked as 'FUTURE': {total_future}")
print(f"Grand total: {total_present + total_present_and_future + total_future}")
