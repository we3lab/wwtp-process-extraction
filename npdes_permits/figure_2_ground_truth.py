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
    cwns_mapping,
    build_cwns_facility_processes,
    unitprocess_keywords
)
from helpers.plotting import COLORS
from helpers.plotting import make_grouped_legend, save_and_close, set_thick_spines

DATE_FOLDER = "2026-5-15"
figures_dir = f"npdes_permits/output/{DATE_FOLDER}/final"
os.makedirs(figures_dir, exist_ok=True)


def build_category_facility_sets(
    process_cols, gt_common, text_common, cwns_common, leaf_to_category
):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set): gt_fac, npdes_fac, cwns_fac.
    Each set contains (Place ID, Facility Name) tuples.
    """
    gt_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        if col in gt_common.columns:
            for _, row in gt_common.iterrows():
                if is_present(row.get(col, "")):
                    gt_fac[category].add(row["Place ID"])
        if col in text_common.columns:
            for _, row in text_common.iterrows():
                if is_present(row.get(col, "")):
                    npdes_fac[category].add(row["Place ID"])
        if col in cwns_common.columns:
            mask = build_cwns_presence_mask(cwns_common[col])
            for pid in cwns_common.loc[mask, "Place ID"]:
                cwns_fac[category].add(pid)

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


CWNS_CA_CSV = "npdes_permits/output/cwns_processes_by_facility.csv"


def main():

    ca_cwns_data = pd.read_csv(CWNS_CA_CSV, dtype=str, low_memory=False)
    ca_cwns_data["CWNS_ID"] = ca_cwns_data["CWNS_ID"].str.strip()
    # Build leaf → top-level category mapping
    leaf_to_category = {
        leaf: cat_name
        for cat_name, cat_value in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat_name, cat_value)
    }

    # Load Google Sheets
    ground_truth_df = pd.read_csv("npdes_permits/data/train_set_ground_truth.csv", dtype=str).fillna("")
    ground_truth_df["Place ID"] = ground_truth_df["Place ID"].str.strip()
    npdes_text_df = pd.read_csv("npdes_permits/data/train_set_npdes_manual.csv", dtype=str).fillna("")
    npdes_text_df["Place ID"] = npdes_text_df["Place ID"].str.strip()

    print(f"GroundTruth sheet: {len(ground_truth_df)} facilities")
    print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")

    meta_cols = [
        "Agency",
        "Place ID",
        "WDID",
        "Facility Name",
        "NPDES No.",
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

    common_facilities = set(ground_truth_df["Place ID"]) & set(npdes_text_df["Place ID"]) & set(cwns_mapping["Place ID"])
    ground_truth_common = ground_truth_df[ground_truth_df["Place ID"].isin(common_facilities)].copy()
    text_common = npdes_text_df[npdes_text_df["Place ID"].isin(common_facilities)].copy()
    cwns_common, merged_mapping_cwns = build_cwns_facility_processes(
        ca_cwns_data, target_facilities=common_facilities
    )

    # print facilities from the ground truth dataNOT in common_facilities
    gt_not_common = set(ground_truth_df["Place ID"]) - common_facilities
    if gt_not_common:
        print(f"\nFacilities in Ground Truth but not in common set (N={len(gt_not_common)}):")
        for pid in gt_not_common:
            name = ground_truth_df.loc[ground_truth_df["Place ID"] == pid, "Facility Name"].iloc[0]
            print(f"  {pid}: {name}")
    gt_fac, npdes_fac, cwns_fac = build_category_facility_sets(
        all_sheet_process_cols,
        ground_truth_common,
        text_common,
        cwns_common,
        leaf_to_category,
    )

    gt_simple_rows = build_gt_rows(gt_fac, npdes_fac, cwns_fac, common_facilities)

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
    gt_plot_df = gt_plot_df.sort_values(
        ["CWNS_Error_Rate", "NPDES_Error_Rate"], ascending=[True, True]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    for i, row in gt_plot_df.iterrows():
        npdes_x = i - w / 2
        cwns_x = i + w / 2
        for x, color_key, missed_col, extra_col in [
            (npdes_x, "npdes_kw", "NPDES_Missed_Rate", "NPDES_Extra_Rate"),
            (cwns_x,  "cwns",     "CWNS_Missed_Rate",  "CWNS_Extra_Rate"),
        ]:
            missed = row[missed_col] * 100
            ax.bar(x, missed, w, color=to_rgba(COLORS[color_key], 0.9), edgecolor="black", linewidth=1.2, zorder=2)
            ax.bar(x, row[extra_col] * 100, w, bottom=missed, color=to_rgba(COLORS[color_key], 0.45), edgecolor="black", linewidth=1.2, zorder=2)

    npdes_avg = float(gt_plot_df["NPDES_Error_Rate"].mean() * 100)
    cwns_avg = float(gt_plot_df["CWNS_Error_Rate"].mean() * 100)
    ax.axhline(npdes_avg, color=COLORS["npdes_kw"], linewidth=1.6, linestyle="-", zorder=1)
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
                ("  NPDES Text", {"facecolor": COLORS["npdes_kw"]}),
                ("  CWNS", {"facecolor": COLORS["cwns"]}),
            ],
        },
        {
            "header": "Error Composition",
            "items": [
                ("  False Positive", {"facecolor": "gray", "alpha": 0.45}),
                ("  False Negative", {"facecolor": "gray", "alpha": 0.9}),
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

    facility_rows = []
    for fac in sorted(common_facilities):
        gt_rows_match = ground_truth_common[ground_truth_common["Place ID"] == fac]
        text_rows_match = text_common[text_common["Place ID"] == fac]
        cwns_rows_match = cwns_common[cwns_common["Place ID"] == fac]
        if gt_rows_match.empty or text_rows_match.empty or cwns_rows_match.empty:
            continue

        ground_truth_row = gt_rows_match.iloc[0]
        text_row = text_rows_match.iloc[0]
        cwns_row = cwns_rows_match.iloc[0]
        npdes = ground_truth_row["NPDES No."]
        facility_name = ground_truth_row["Facility Name"]

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

        src_metrics = {}
        for prefix, pred_set in [("NPDES", npdes_set), ("CWNS", cwns_set)]:
            tp = len(gt_set & pred_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            src_metrics[prefix] = dict(
                TP=tp, FP=fp, FN=fn, Precision=p, Recall=r,
                F1=2 * p * r / (p + r) if (p + r) else 0,
                Missed="|".join(sorted(gt_set - pred_set)),
                Extra="|".join(sorted(pred_set - gt_set)),
            )

        nm, cm = src_metrics["NPDES"], src_metrics["CWNS"]
        facility_rows.append(
            {
                "NPDES No.": npdes,
                "Facility Name": facility_name,
                "GT_Count": len(gt_set),
                "NPDES_TP": nm["TP"], "NPDES_FP": nm["FP"], "NPDES_FN": nm["FN"],
                "NPDES_Precision": nm["Precision"], "NPDES_Recall": nm["Recall"], "NPDES_F1": nm["F1"],
                "NPDES_Missed": nm["Missed"], "NPDES_Extra": nm["Extra"],
                "CWNS_TP": cm["TP"], "CWNS_FP": cm["FP"], "CWNS_FN": cm["FN"],
                "CWNS_Precision": cm["Precision"], "CWNS_Recall": cm["Recall"], "CWNS_F1": cm["F1"],
                "CWNS_Missed": cm["Missed"], "CWNS_Extra": cm["Extra"],
            }
        )

    gt_comparison_df = pd.DataFrame(facility_rows)
    gt_comparison_csv = (
        f"npdes_permits/output/{DATE_FOLDER}/ground_truth_comparison_by_facility.csv"
    )
    gt_comparison_df.to_csv(gt_comparison_csv, index=False)

    gt_rows_with_gt = [r for r in gt_simple_rows if r["GroundTruth"] > 0]
    if gt_rows_with_gt:
        gt_zero_cwns_fp = sum(r["CWNS_FP"] for r in gt_simple_rows if r["GroundTruth"] == 0)
        print(
            f"\nAverage error vs Ground Truth (categories with GT>0): CWNS = {cwns_avg:.1f}%, NPDES Text = {npdes_avg:.1f}%"
        )
        for r in gt_simple_rows:
            if r["GroundTruth"] == 0 and r["CWNS_FP"] > 0:
                print(f"  CWNS FP in category '{r['Process_Category']}' with GT=0, FP={r['CWNS_FP']}")


if __name__ == "__main__":
    main()
