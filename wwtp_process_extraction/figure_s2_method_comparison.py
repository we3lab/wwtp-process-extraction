import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from step6_postprocess_llm_output import build_model_comparison
from helpers.metrics import confusion_scores, build_metric_inputs, aggregate_to_category_states, compute_metrics
from helpers.paths import *
from helpers.utils import category_leaf_names, scored_leaf_names, unit_leaf_names, facility_table
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
    ax.set_xticks(range(len(VIOLIN_METRIC_COLUMNS)), [label.replace("_", " ") for label in VIOLIN_METRIC_COLUMNS],
                  fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("")
    ax.legend(loc="upper right", bbox_to_anchor=(1.01, 1.18), ncol=3, fontsize=11, frameon=False)
    ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes, va="top", fontsize=16)
    set_thick_spines(ax)


categories_to_plot = list(category_leaf_names)
print(f"Categories: {categories_to_plot}")

# Load keyword-based NPDES results
keyword_results = facility_table(PDF_KW_CSV)
print(f"NPDES keyword data: Loaded {len(keyword_results)} unique facilities")

# Load LLM results
llm_results = facility_table(PDF_LLM_CSV)
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
manual = pd.read_csv(MANUAL_CSV, dtype=str)
manual_facilities = set(manual["Place ID"])
print(
    f"Manual baseline: {len(manual)} facilities "
    f"({len(manual_facilities & set(keyword_results_both['Place ID']))} matched to keyword, "
    f"{len(manual_facilities & set(llm_results_both['Place ID']))} matched to LLM)"
)

# Per-model ontology results (gpt-5, gpt-5-mini) for the violin comparison, sourced from
# the same postprocessed JSONs table_1 uses — restricted to the manual-read benchmark facilities.
runs = build_model_comparison().groupby(["Method", "Model"])
prediction_sources = [
    ("Keyword", keyword_results_both),
    ("LLM", llm_results_both),
    ("GPT-5", runs.get_group(("Ontology", "gpt-5"))),
    ("GPT-5 mini", runs.get_group(("Ontology", "gpt-5-mini"))),
]

# Unit-process and category metrics; category states collapse the leaf states
metrics_frames = []
facility_rows = {"Unit_Process": [], "Category": []}
for source_name, pred_df in prediction_sources:
    manual_leaf, pred_leaf = build_metric_inputs(scored_leaf_names, manual, pred_df)
    levels = [
        ("Unit_Process", manual_leaf, pred_leaf, unit_leaf_names),
        ("Category", aggregate_to_category_states(manual_leaf, category_leaf_names),
         aggregate_to_category_states(pred_leaf, category_leaf_names), categories_to_plot),
    ]
    for level, manual_states, pred_states, labels in levels:
        metrics_frames.append(compute_metrics(manual_states, pred_states, labels, source_name).assign(Level=level))
        # Per-facility scores for the violins
        manual_by_key = manual_states.set_index("key")
        pred_by_key = pred_states.set_index("key")
        for key in manual_by_key.index:
            scores = confusion_scores(manual_by_key.loc[key, labels], pred_by_key.loc[key, labels])[-1]
            facility_rows[level].append({"Source": source_name, "key": key, **scores})

metrics_df = pd.concat(metrics_frames, ignore_index=True).rename(columns={"Label": "Process"})
summary = metrics_df.groupby(["Level", "Source"])[SUMMARY_METRICS].mean()
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
kw_hallucinated = (
    metrics_df[(metrics_df["Source"] == "Keyword") & (metrics_df["Level"] == "Unit_Process")][
        ["Process", "Hallucinated_Rate", "FP", "Support_Pred"]
    ]
    .dropna(subset=["Hallucinated_Rate"])
    .sort_values(["Hallucinated_Rate", "FP", "Support_Pred"], ascending=False)
    .head(12)
)
print("\nTop hallucinated unit processes (Keyword):")
print(kw_hallucinated.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

violin_path = FIGURES_DIR / "figure_s2.png"
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6))
draw_violin(ax_top, pd.DataFrame(facility_rows["Unit_Process"]), "A.")
draw_violin(ax_bottom, pd.DataFrame(facility_rows["Category"]), "B.")
fig.tight_layout(h_pad=3.5)
save_and_close(fig, violin_path)
print(f"Saved {violin_path.name}")

# Overall keyword status counts
status_counts = {s: int(keyword_results[scored_leaf_names].eq(s).sum().sum())
                 for s in ("PRESENT", "PRESENT_AND_FUTURE", "FUTURE")}
for status, count in status_counts.items():
    print(f"Total process instances marked as '{status}': {count}")
print(f"Grand total: {sum(status_counts.values())}")
