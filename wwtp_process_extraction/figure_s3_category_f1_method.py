"""SI figure: category-level F1 for gpt-5-mini, ontology vs list method."""

import os
import pandas as pd
import matplotlib.pyplot as plt

from step6_postprocess_llm_output import build_model_comparison
from helpers.metrics import build_metric_inputs, aggregate_to_category_states, compute_metrics
from helpers.utils import get_leaf_names, unitprocess_keywords, FINAL_DIR, MANUAL_CSV
from helpers.plotting import COLORS, save_and_close, set_thick_spines

MODEL = "gpt-5-mini"

# "with ontology" = Ontology method; "without ontology" = List method
METHODS = [("Ontology", "Ontology-based prompt"), ("List", "List-based prompt")]
METHOD_COLORS = {"Ontology": COLORS["npdes_llm"], "List": COLORS["Clean Watershed Needs Survey"]}

categories = list(unitprocess_keywords.keys())
category_to_leaves = {c: get_leaf_names(c, unitprocess_keywords[c]) for c in categories}
all_leaves = [p for leaves in category_to_leaves.values() for p in leaves]


def main():
    model_comparison = build_model_comparison()
    # the manual CSV is the benchmark (manually-read) facility set, as table_1 uses
    manual = pd.read_csv(MANUAL_CSV, dtype=str)

    per_method = {}
    for method, _ in METHODS:
        pred = model_comparison[(model_comparison["Method"] == method) & (model_comparison["Model"] == MODEL)]
        manual_df, pred_df = build_metric_inputs(all_leaves, manual, pred)
        cat_metrics = compute_metrics(
            aggregate_to_category_states(manual_df, category_to_leaves),
            aggregate_to_category_states(pred_df, category_to_leaves),
            categories,
            method,
        )
        per_method[method] = cat_metrics.set_index("Label")

    # keep categories with at least one manual-positive facility (F1 otherwise undefined)
    support = per_method["Ontology"]["Support_Manual"]
    keep = sorted((c for c in categories if support.get(c, 0) > 0), key=lambda c: per_method["Ontology"].loc[c, "F1"])  # ascending → best on top

    # Per-category F1 by method; flag categories driving the methods apart (|diff| > 0.05)
    print("\nCategory-level F1 (gpt-5-mini): Ontology vs List")
    print(f"{'Category':25s} {'Ontology':>9s} {'List':>7s} {'Diff':>7s}")
    f1 = {method: per_method[method]["F1"] for method, _ in METHODS}
    for c in sorted(keep, key=lambda c: f1["Ontology"][c] - f1["List"][c]):
        ontology_f1, list_f1 = f1["Ontology"][c], f1["List"][c]
        diff = ontology_f1 - list_f1
        flag = "  <-- driving (ontology higher)" if diff > 0.05 else "  <-- driving (list higher)" if diff < -0.05 else ""
        print(f"{c:25s} {ontology_f1:9.3f} {list_f1:7.3f} {diff:+7.3f}{flag}")

    # grouped horizontal bars
    y = range(len(keep))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, max(5, 0.42 * len(keep))))
    for i, (method, label) in enumerate(METHODS):
        offset = (0.5 - i) * h
        vals = [f1[method][c] for c in keep]
        ax.barh(
            [yy + offset for yy in y], vals, height=h, color=METHOD_COLORS[method],
            edgecolor="black", linewidth=1.2, label=label,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c}  (n={int(support[c])})" for c in keep], fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Category-level F1 (gpt-5-mini)", fontsize=13)
    ax.grid(False)
    ax.legend(loc="lower right", fontsize=11, frameon=False, bbox_to_anchor=(1.02, 1.02))
    set_thick_spines(ax, linewidth=1.6)
    fig_path = FINAL_DIR / "figure_s3.png"
    save_and_close(fig, fig_path, dpi=300)
    print(f"Saved {os.path.relpath(fig_path)}")


if __name__ == "__main__":
    main()
