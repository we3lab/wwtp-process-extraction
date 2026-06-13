"""SI figure: category-level F1 for gpt-5-mini, ontology vs list method."""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from helpers.metrics import compute_metrics
from helpers.utils import parse_status, get_leaf_names, merge_column_statuses, unitprocess_keywords
from helpers.plotting import COLORS, save_and_close, set_thick_spines

WORKBOOK = "wwtp_process_extraction/output/llm_extraction/model_comparison_all.csv"
MANUAL_PATH = "wwtp_process_extraction/data/unit_processes_by_facility_manual.csv"
FINAL_DIR = "wwtp_process_extraction/output/final"
MODEL = "gpt-5-mini"

# "with ontology" = Ontology method; "without ontology" = List method
METHODS = [("Ontology", "Ontology-based prompt"), ("List", "List-based prompt")]
METHOD_COLORS = {"Ontology": COLORS["npdes_llm"], "List": COLORS["Clean Watershed Needs Survey"]}

categories = [c for c in unitprocess_keywords.keys() if c != "Disposal"]
category_to_leaves = {c: get_leaf_names(c, unitprocess_keywords[c]) for c in categories}
all_leaves = [p for leaves in category_to_leaves.values() for p in leaves]


def to_category_states(metric_df):
    """Collapse leaf states to one present/absent state per category, per facility."""
    out = pd.DataFrame({"key": metric_df["key"]})
    for cat, leaves in category_to_leaves.items():
        cols = [c for c in leaves if c in metric_df.columns]
        out[cat] = metric_df[cols].apply(merge_column_statuses, axis=1) if cols else ""
    return out


def main():
    wb = pd.read_csv(WORKBOOK, dtype=str).fillna("")
    wb["Place ID"] = wb["Place ID"].str.strip()
    manual = pd.read_csv(MANUAL_PATH, dtype=str).fillna("")
    manual["Place ID"] = manual["Place ID"].str.strip()
    # restrict to the benchmark facilities (the workbook's manual-read set), as table_1 does
    bench = set(wb.loc[wb["Method"].eq("Manual Read"), "Place ID"])
    manual = manual[manual["Place ID"].isin(bench)].copy()

    per_method = {}
    for method, _ in METHODS:
        pred = wb[(wb["Method"] == method) & (wb["Model"] == MODEL)].copy()

        keys = sorted(set(manual["Place ID"].dropna()) & set(pred["Place ID"].dropna()))
        m = manual.drop_duplicates("Place ID").set_index("Place ID").reindex(keys)
        p = pred.drop_duplicates("Place ID").set_index("Place ID").reindex(keys)
        manual_df = pd.DataFrame({"key": keys})
        pred_df = pd.DataFrame({"key": keys})
        for leaf in all_leaves:
            manual_df[leaf] = m[leaf].map(parse_status).values if leaf in m.columns else ""
            pred_df[leaf] = p[leaf].map(parse_status).values if leaf in p.columns else ""
        cat_metrics = compute_metrics(
            to_category_states(manual_df), to_category_states(pred_df), categories, method
        )
        per_method[method] = cat_metrics.set_index("Label")

    # keep categories with at least one manual-positive facility (F1 otherwise undefined)
    support = per_method["Ontology"]["Support_Manual"]
    keep = [c for c in categories if support.get(c, 0) > 0]
    keep = sorted(keep, key=lambda c: per_method["Ontology"].loc[c, "F1"])  # ascending → best on top

    # grouped horizontal bars
    y = range(len(keep))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(5, 0.42 * len(keep))))
    for i, (method, label) in enumerate(METHODS):
        offset = (0.5 - i) * h
        vals = [float(per_method[method].loc[c, "F1"]) for c in keep]
        ax.barh(
            [yy + offset for yy in y], vals, height=h, color=METHOD_COLORS[method],
            edgecolor="black", linewidth=1.2, label=label,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c}  (n={int(support[c])})" for c in keep], fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Category-level F1 (gpt-5-mini)", fontsize=13)
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    set_thick_spines(ax, linewidth=1.6)
    fig_path = f"{FINAL_DIR}/figure_s3_category_f1_method.png"
    save_and_close(fig, fig_path, dpi=300)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
