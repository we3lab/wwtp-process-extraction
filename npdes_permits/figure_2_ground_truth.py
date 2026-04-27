import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from helpers.utils import (
    extract_leaves,
    build_cwns_presence_mask,
    is_present,
    get_leaf_names,
    name_to_fac,
    facilities_with_cwns,
    map_facility_names_to_tuples,
    build_cwns_facility_processes,
)
from helpers.plotting import COLORS
from helpers.plotting import make_grouped_legend, save_and_close, set_thick_spines

DATE_FOLDER = "2026-4-26"
figures_dir = f"npdes_permits/output/{DATE_FOLDER}/final"
os.makedirs(figures_dir, exist_ok=True)


def build_category_facility_sets(
    process_cols, gt_common, text_common, cwns_common, leaf_to_category
):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set): gt_fac, npdes_fac, cwns_fac.
    Each set contains (WDID, Facility_Name) tuples.
    """
    gt_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        if col in gt_common.columns:
            for _, row in gt_common.iterrows():
                if is_present(row.get(col, "")):
                    gt_fac[category].add(row["_fac"])
        if col in text_common.columns:
            for _, row in text_common.iterrows():
                if is_present(row.get(col, "")):
                    npdes_fac[category].add(row["_fac"])
        if col in cwns_common.columns:
            mask = build_cwns_presence_mask(cwns_common[col])
            for _, r in cwns_common.loc[mask, ["WDID", "Facility_Name"]].iterrows():
                cwns_fac[category].add((r["WDID"], r["Facility_Name"]))

    return gt_fac, npdes_fac, cwns_fac


def build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_facilities):
    """Build summary rows for ground-truth comparison plotting from per-category facility sets."""
    all_cats = sorted(set(list(gt_fac.keys()) + list(npdes_fac.keys()) + list(cwns_fac.keys())))
    rows = []
    for cat in all_cats:
        ground_truth = len(gt_fac[cat])
        npdes = len(npdes_fac[cat])
        cwns = len(cwns_fac[cat])
        if ground_truth == 0 and npdes == 0 and cwns == 0:
            continue
        if ground_truth > 0:
            npdes_str = f"{(npdes - ground_truth) / ground_truth * 100:+.0f}%"
            cwns_str = f"{(cwns - ground_truth) / ground_truth * 100:+.0f}%"
        else:
            npdes_str = f"+{npdes}" if npdes > 0 else "0"
            cwns_str = f"+{cwns}" if cwns > 0 else "0"
        gt_p = gt_fac[cat] & common_facilities
        npdes_p = npdes_fac[cat] & common_facilities
        cwns_p = cwns_fac[cat] & common_facilities
        rows.append(
            {
                "Process_Category": cat,
                "GroundTruth": ground_truth,
                "NPDES_Manual": npdes,
                "NPDES_vs_GT": npdes_str,
                "NPDES_FP": len(npdes_p - gt_p),
                "NPDES_FN": len(gt_p - npdes_p),
                "CWNS": cwns,
                "CWNS_vs_GT": cwns_str,
                "CWNS_FP": len(cwns_p - gt_p),
                "CWNS_FN": len(gt_p - cwns_p),
            }
        )
    return rows


with open("npdes_permits/data/unitprocess_keywords.json", "r") as f:
    unitprocess_keywords = json.load(f)

CWNS_CA_CSV = "npdes_permits/output/cwns_processes_by_facility.csv"


def main():

    ca_cwns_data = pd.read_csv(CWNS_CA_CSV, dtype=str, low_memory=False)
    ca_cwns_data["CWNS_ID"] = ca_cwns_data["CWNS_ID"].astype(str).str.strip()
    # Build leaf → top-level category mapping
    leaf_to_category = {
        leaf: cat_name
        for cat_name, cat_value in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat_name, cat_value)
    }

    # Load Google Sheets
    ground_truth_df = pd.read_csv("npdes_permits/data/train_set_ground_truth.csv", dtype=str)
    npdes_text_df = pd.read_csv("npdes_permits/data/train_set_npdes_manual.csv", dtype=str)

    print(f"GroundTruth sheet: {len(ground_truth_df)} facilities")
    print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")

    meta_cols = [
        "Agency",
        "Facility_Name",
        "NPDES_No",
        "PDF_File",
        "Ground Truth Sources",
    ]

    ground_truth_process_cols = [c for c in ground_truth_df.columns if c not in meta_cols]
    npdes_text_process_cols = [c for c in npdes_text_df.columns if c not in meta_cols]

    disposal_leaves = {
        name
        for name, _, _ in extract_leaves(
            unitprocess_keywords["Solids Processing"]["Disposal"], ignore_disposal=False
        )
    }
    all_sheet_process_cols = [
        c
        for c in dict.fromkeys(ground_truth_process_cols + npdes_text_process_cols)
        if c not in disposal_leaves
    ]

    # TRAIN ground truth comparison — identify common facilities via (WDID, Facility_Name) tuples
    gt_facilities = map_facility_names_to_tuples(ground_truth_df, "Facility_Name", name_to_fac)
    text_facilities = map_facility_names_to_tuples(npdes_text_df, "Facility_Name", name_to_fac)
    common_facilities = gt_facilities & text_facilities & facilities_with_cwns
    print(f"Facilities in all 3 sources (Train): {len(common_facilities)}")

    # Add facility tuple column to source DataFrames for accumulation
    ground_truth_df["_fac"] = ground_truth_df["Facility_Name"].str.strip().map(name_to_fac)
    npdes_text_df["_fac"] = npdes_text_df["Facility_Name"].str.strip().map(name_to_fac)

    ground_truth_common = ground_truth_df[ground_truth_df["_fac"].isin(common_facilities)].copy()
    text_common = npdes_text_df[npdes_text_df["_fac"].isin(common_facilities)].copy()

    cwns_common, merged_mapping_cwns = build_cwns_facility_processes(
        ca_cwns_data, target_facilities=common_facilities
    )
    n_with_cwns = int((merged_mapping_cwns["_cwns_merge"] == "both").sum())
    print(
        f"CIWQS mapping rows with CA CWNS survey attach: {n_with_cwns} / {len(merged_mapping_cwns)}"
    )

    gt_fac, npdes_fac, cwns_fac = build_category_facility_sets(
        all_sheet_process_cols,
        ground_truth_common,
        text_common,
        cwns_common,
        leaf_to_category,
    )

    gt_simple_rows = build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_facilities)

    gt_simple_df = (
        pd.DataFrame(gt_simple_rows)
        .sort_values("GroundTruth", ascending=False)
        .reset_index(drop=True)
    )
    print(gt_simple_df.to_string(index=False))

    tick_fontsize = 12
    label_fontsize = 14
    gt_plot_df = pd.DataFrame(gt_simple_rows)
    gt_plot_df = gt_plot_df[gt_plot_df["GroundTruth"] > 0].copy()
    gt_plot_df = gt_plot_df.sort_values("GroundTruth", ascending=False).reset_index(drop=True)

    n_facilities = len(common_facilities)
    for source in ("NPDES", "CWNS"):
        fp_col = f"{source}_FP"
        fn_col = f"{source}_FN"
        total_err_col = f"{source}_Error_Total"
        gt_plot_df[total_err_col] = gt_plot_df[fp_col] + gt_plot_df[fn_col]
        gt_plot_df[f"{source}_Error_Rate"] = gt_plot_df[total_err_col] / n_facilities
        gt_plot_df[f"{source}_Missed_Rate"] = gt_plot_df[fn_col] / n_facilities
        gt_plot_df[f"{source}_Extra_Rate"] = gt_plot_df[fp_col] / n_facilities
        gt_plot_df[f"{source}_Accuracy_Like"] = 1 - gt_plot_df[f"{source}_Error_Rate"]
        denom = gt_plot_df[total_err_col].where(gt_plot_df[total_err_col] > 0, 1)
        gt_plot_df[f"{source}_Missed_Share"] = gt_plot_df[fn_col] / denom
        gt_plot_df[f"{source}_Extra_Share"] = gt_plot_df[fp_col] / denom
    gt_plot_df = gt_plot_df.sort_values("NPDES_Error_Rate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    for i, row in gt_plot_df.iterrows():
        npdes_x = i - w / 2
        cwns_x = i + w / 2
        ax.bar(
            npdes_x,
            row["NPDES_Missed_Rate"] * 100,
            w,
            color=to_rgba(COLORS["npdes"], 0.9),
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        ax.bar(
            npdes_x,
            row["NPDES_Extra_Rate"] * 100,
            w,
            bottom=row["NPDES_Missed_Rate"] * 100,
            color=to_rgba(COLORS["npdes"], 0.45),
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        ax.bar(
            cwns_x,
            row["CWNS_Missed_Rate"] * 100,
            w,
            color=to_rgba(COLORS["cwns"], 0.9),
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        ax.bar(
            cwns_x,
            row["CWNS_Extra_Rate"] * 100,
            w,
            bottom=row["CWNS_Missed_Rate"] * 100,
            color=to_rgba(COLORS["cwns"], 0.45),
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )

    npdes_avg = float(gt_plot_df["NPDES_Error_Rate"].mean() * 100)
    cwns_avg = float(gt_plot_df["CWNS_Error_Rate"].mean() * 100)
    ax.axhline(npdes_avg, color=COLORS["npdes"], linewidth=1.6, linestyle="-", zorder=1)
    ax.axhline(cwns_avg, color=COLORS["cwns"], linewidth=1.6, linestyle="-", zorder=1)

    ax.set_xticks(range(len(gt_plot_df)))
    ax.set_xticklabels(gt_plot_df["Process_Category"], rotation=45, ha="right", fontsize=tick_fontsize)
    ax.set_ylabel(f"Error Rate (%)\n(N={n_facilities})", fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    set_thick_spines(ax, linewidth=1.6)

    ax.set_ylim(0, 100)
    pct_ticks = [i * 20 for i in range(6)]
    ax.set_yticks(pct_ticks)
    ax.set_yticklabels([f"{p}%" for p in pct_ticks], fontsize=tick_fontsize)

    legend_groups = [
        {
            "header": "Data Source",
            "items": [
                ("  NPDES Text", {"facecolor": COLORS["npdes"]}),
                ("  CWNS", {"facecolor": COLORS["cwns"]}),
            ],
        },
        {
            "header": "Error Composition",
            "items": [
                ("  Extra (FP share)", {"facecolor": "gray", "alpha": 0.45}),
                ("  Missed (FN share)", {"facecolor": "gray", "alpha": 0.9}),
            ],
        },
        {
            "header": "Average Error",
            "items": [
                Line2D(
                    [],
                    [],
                    color="black",
                    linewidth=1.6,
                    linestyle="-",
                    label="  Average Aross\n  Categories",
                ),
            ],
        },
    ]
    make_grouped_legend(ax, legend_groups, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=11)

    plt.subplots_adjust(bottom=0.35)
    save_path = f"{figures_dir}/figure_2_ground_truth_ground_truth_vs_npdes_text_vs_cwns.png"
    save_and_close(fig, save_path, dpi=300)
    print(f"  Saved {os.path.basename(save_path)}")

    print("FACILITY-LEVEL COMPARISON TO GROUND TRUTH (GroundTruth)")

    facility_rows = []
    for fac in sorted(common_facilities):
        gt_rows_match = ground_truth_common[ground_truth_common["_fac"] == fac]
        text_rows_match = text_common[text_common["_fac"] == fac]
        cwns_rows_match = cwns_common[
            (cwns_common["WDID"] == fac[0]) & (cwns_common["Facility_Name"] == fac[1])
        ]
        if gt_rows_match.empty or text_rows_match.empty or cwns_rows_match.empty:
            continue

        ground_truth_row = gt_rows_match.iloc[0]
        text_row = text_rows_match.iloc[0]
        cwns_row = cwns_rows_match.iloc[0]
        npdes = str(ground_truth_row.get("NPDES_No", "")).strip()
        facility_name = ground_truth_row.get("Facility_Name", npdes)

        gt_set = {
            col
            for col in all_sheet_process_cols
            if col in ground_truth_common.columns and is_present(ground_truth_row.get(col, ""))
        }
        npdes_set = {
            col
            for col in all_sheet_process_cols
            if col in text_common.columns and is_present(text_row.get(col, ""))
        }

        cwns_set = {
            col
            for col in all_sheet_process_cols
            if col in cwns_common.columns
            and build_cwns_presence_mask(pd.Series([cwns_row.get(col, "")])).iloc[0]
        }

        npdes_tp = len(gt_set & npdes_set)
        npdes_fp = len(npdes_set - gt_set)
        npdes_fn = len(gt_set - npdes_set)
        npdes_precision = npdes_tp / (npdes_tp + npdes_fp) if (npdes_tp + npdes_fp) else 0
        npdes_recall = npdes_tp / (npdes_tp + npdes_fn) if (npdes_tp + npdes_fn) else 0
        npdes_f1 = (
            2 * npdes_precision * npdes_recall / (npdes_precision + npdes_recall)
            if (npdes_precision + npdes_recall)
            else 0
        )

        cwns_tp = len(gt_set & cwns_set)
        cwns_fp = len(cwns_set - gt_set)
        cwns_fn = len(gt_set - cwns_set)
        cwns_precision = cwns_tp / (cwns_tp + cwns_fp) if (cwns_tp + cwns_fp) else 0
        cwns_recall = cwns_tp / (cwns_tp + cwns_fn) if (cwns_tp + cwns_fn) else 0
        cwns_f1 = (
            2 * cwns_precision * cwns_recall / (cwns_precision + cwns_recall)
            if (cwns_precision + cwns_recall)
            else 0
        )

        facility_rows.append(
            {
                "NPDES_No": npdes,
                "Facility_Name": facility_name,
                "GT_Count": len(gt_set),
                "NPDES_TP": npdes_tp,
                "NPDES_FP": npdes_fp,
                "NPDES_FN": npdes_fn,
                "NPDES_Precision": npdes_precision,
                "NPDES_Recall": npdes_recall,
                "NPDES_F1": npdes_f1,
                "NPDES_Missed": "|".join(sorted(gt_set - npdes_set)),
                "NPDES_Extra": "|".join(sorted(npdes_set - gt_set)),
                "CWNS_TP": cwns_tp,
                "CWNS_FP": cwns_fp,
                "CWNS_FN": cwns_fn,
                "CWNS_Precision": cwns_precision,
                "CWNS_Recall": cwns_recall,
                "CWNS_F1": cwns_f1,
                "CWNS_Missed": "|".join(sorted(gt_set - cwns_set)),
                "CWNS_Extra": "|".join(sorted(cwns_set - gt_set)),
            }
        )

    gt_comparison_df = pd.DataFrame(facility_rows)
    gt_comparison_csv = (
        f"npdes_permits/output/{DATE_FOLDER}/ground_truth_comparison_by_facility.csv"
    )
    gt_comparison_df.to_csv(gt_comparison_csv, index=False)
    print(f"Saved facility-level comparison: {os.path.basename(gt_comparison_csv)}")

    # Overall error vs ground truth — only for categories where GT > 0
    gt_rows_with_gt = [r for r in gt_simple_rows if r["GroundTruth"] > 0]
    if gt_rows_with_gt:
        cwns_overall_error = sum(r["CWNS_FP"] + r["CWNS_FN"] for r in gt_rows_with_gt) / sum(
            r["GroundTruth"] for r in gt_rows_with_gt
        )
        npdes_overall_error = sum(r["NPDES_FP"] + r["NPDES_FN"] for r in gt_rows_with_gt) / sum(
            r["GroundTruth"] for r in gt_rows_with_gt
        )
        gt_zero_cwns_fp = sum(r["CWNS_FP"] for r in gt_simple_rows if r["GroundTruth"] == 0)
        print(
            f"\nOverall error vs Ground Truth (categories with GT>0): CWNS = {cwns_overall_error:.1%}, NPDES Text = {npdes_overall_error:.1%}"
        )
        if gt_zero_cwns_fp:
            print(
                f"  (CWNS also has {gt_zero_cwns_fp} FP detections in categories with no ground truth annotations)"
            )


if __name__ == "__main__":
    main()
