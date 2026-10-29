import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from step6_postprocess_llm_output import build_model_comparison
from helpers.metrics import confusion_scores, build_metric_inputs, aggregate_to_category_states, compute_metrics
from helpers.paths import *
from helpers.utils import (
    get_leaf_names,
    category_leaf_names,
    unitprocess_keywords,
)
from helpers.plotting import COLORS, save_and_close, set_thick_spines

# 0-1 facility scores shown in the violins, in plot order
VIOLIN_METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "State_Accuracy"]
VIOLIN_SOURCES = ["Keyword", "GPT-5", "GPT-5 mini"]
VIOLIN_PALETTE = {
    "Keyword": COLORS["npdes_kw"],
    "GPT-5": COLORS["gpt-5"],
    "GPT-5 mini": COLORS["gpt-5-mini"],
}
SUMMARY_METRICS = ["Precision", "Recall", "F1", "Accuracy", "Missed_Rate", "Hallucinated_Rate", "State_Accuracy"]


def compute_facility_metric_rows(manual_df, pred_df, label_cols, source_name):
    # Per-facility scores over label_cols for the violin plot
    manual_indexed = manual_df.set_index("key")
    pred_indexed = pred_df.set_index("key")
    rows = []
    for key in manual_indexed.index:
        scores = confusion_scores(manual_indexed.loc[key, label_cols], pred_indexed.loc[key, label_cols])[-1]
        rows.append({"Source": source_name, "key": key, **scores})
    return rows


def draw_violin(ax, facility_metrics_df, panel_label):
    plot_df = (
        facility_metrics_df[facility_metrics_df["Source"].isin(VIOLIN_SOURCES)]
        .melt(
            id_vars=["Source", "key"],
            value_vars=VIOLIN_METRIC_COLUMNS,
            var_name="Metric",
            value_name="Value",
        )
        .dropna(subset=["Value"])
    )
    n_fac = plot_df["key"].nunique()

    sns.violinplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Source",
        order=VIOLIN_METRIC_COLUMNS,
        hue_order=VIOLIN_SOURCES,
        palette=VIOLIN_PALETTE,
        inner=None,
        cut=0,
        density_norm="width",
        ax=ax,
    )
    for poly in ax.collections:
        poly.set_edgecolor("black")
        poly.set_linewidth(1.2)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Facility-level score\n(N={n_fac})", fontsize=14)
    ax.set_xticks(range(len(VIOLIN_METRIC_COLUMNS)))
    ax.set_xticklabels([label.replace("_", " ") for label in VIOLIN_METRIC_COLUMNS], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("")
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.18), ncol=3, fontsize=11, frameon=False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, va="top", fontsize=16)
    set_thick_spines(ax)


categories_to_plot = list(category_leaf_names)
print(f"Categories: {categories_to_plot}")

# Load keyword-based NPDES results
keyword_results = pd.read_csv(FACILITY_KW_CSV, dtype=str).fillna("")
print(f"NPDES keyword data: Loaded {len(keyword_results)} unique facilities")

# Load LLM results
llm_results = pd.read_csv(FACILITY_LLM_CSV, dtype=str).fillna("")
print(f"LLM results: {len(llm_results)} unique facilities")

# Filter to facilities processed by BOTH methods
llm_facilities = set(llm_results["Place ID"])
kw_facilities = set(keyword_results["Place ID"])
both_facilities = llm_facilities & kw_facilities
llm_results_both = llm_results[llm_results["Place ID"].isin(both_facilities)]
keyword_results_both = keyword_results[keyword_results["Place ID"].isin(both_facilities)]
print(
    f"Facilities processed by both LLM and keyword: {len(both_facilities)} "
    f"(LLM only: {len(llm_facilities - kw_facilities)}, "
    f"keyword only: {len(kw_facilities - llm_facilities)})"
)

# Manual readings: exactly the benchmark (manually-read) facility set
manual = pd.read_csv(MANUAL_CSV, dtype=str).fillna("")
manual_facilities = set(manual["Place ID"])
print(
    f"Manual baseline: {len(manual)} facilities "
    f"({len(manual_facilities & set(keyword_results_both['Place ID']))} matched to keyword, "
    f"{len(manual_facilities & set(llm_results_both['Place ID']))} matched to LLM)"
)

# Method comparison metrics
all_process_list = [p for leaves in category_leaf_names.values() for p in leaves]
unit_process_list = [
    p for cat, val in unitprocess_keywords.items() for p in get_leaf_names(cat, val, exclude_unspecified=True)
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
    manual_cat = aggregate_to_category_states(manual_metric, category_leaf_names)
    pred_cat = aggregate_to_category_states(pred_metric, category_leaf_names)
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
summary = metrics_df.groupby(["Level", "Source"])[SUMMARY_METRICS].mean()
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
save_and_close(fig, violin_path)
print(f"Saved {violin_path.name}")

# Overall keyword status counts
keyword_statuses = keyword_results[all_process_list].map(str.upper)
status_counts = {s: int(keyword_statuses.eq(s).sum().sum()) for s in ("PRESENT", "PRESENT_AND_FUTURE", "FUTURE")}
for status, count in status_counts.items():
    print(f"Total process instances marked as '{status}': {count}")
print(f"Grand total: {sum(status_counts.values())}")
