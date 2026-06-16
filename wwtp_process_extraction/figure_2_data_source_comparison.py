import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
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
from helpers.plotting import save_and_close, set_thick_spines

figures_dir = f"wwtp_process_extraction/output/final"
os.makedirs(figures_dir, exist_ok=True)



def build_category_facility_sets(
    process_cols, sd_common, text_common, cwns_common, leaf_to_category
):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set): sd_fac, npdes_fac, cwns_fac.
    Each set contains (Place ID, Facility Name) tuples.
    """
    sd_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        print("category:", category)
        print("col:", col)
        if col in sd_common.columns:
            for _, row in sd_common.iterrows():
                if is_present(row.get(col, "")):
                    sd_fac[category].add(row["Place ID"])
        if col in text_common.columns:
            for _, row in text_common.iterrows():
                if is_present(row.get(col, "")):
                    npdes_fac[category].add(row["Place ID"])
        if col in cwns_common.columns:
            mask = build_cwns_presence_mask(cwns_common[col])
            for pid in cwns_common.loc[mask, "Place ID"]:
                cwns_fac[category].add(pid)

    return sd_fac, npdes_fac, cwns_fac


def build_sd_rows(sd_fac, npdes_fac, cwns_fac, common_facilities):
    """Build summary rows for ground-truth comparison plotting from per-category facility sets."""
    all_cats = sorted(set(list(sd_fac.keys()) + list(npdes_fac.keys()) + list(cwns_fac.keys())))
    print(all_cats)
    rows = []
    for cat in all_cats:
        supplemental_data = len(sd_fac[cat])
        npdes = len(npdes_fac[cat])
        cwns = len(cwns_fac[cat])
        if supplemental_data == 0 and npdes == 0 and cwns == 0:
            continue
        if supplemental_data > 0:
            npdes_str = f"{(npdes - supplemental_data) / supplemental_data :+.0f}%"
            cwns_str = f"{(cwns - supplemental_data) / supplemental_data :+.0f}%"
        else:
            npdes_str = f"+{npdes}" if npdes > 0 else "0"
            cwns_str = f"+{cwns}" if cwns > 0 else "0"
        sd_p = sd_fac[cat] & common_facilities
        npdes_p = npdes_fac[cat] & common_facilities
        cwns_p = cwns_fac[cat] & common_facilities
        rows.append(
            {
                "Process_Category": cat,
                "GroundTruth": supplemental_data,
                "NPDES": npdes,
                "NPDES_vs_GT": npdes_str,
                "NPDES_FP": len(npdes_p - sd_p),
                "NPDES_FN": len(sd_p - npdes_p),
                "CWNS": cwns,
                "CWNS_vs_GT": cwns_str,
                "CWNS_FP": len(cwns_p - sd_p),
                "CWNS_FN": len(sd_p - cwns_p),
            }
        )
    return rows


CWNS_CA_CSV = "wwtp_process_extraction/output/unit_processes_by_facility_cwns.csv"


def main():

    ca_cwns_data = pd.read_csv(CWNS_CA_CSV, dtype=str, low_memory=False)
    ca_cwns_data["CWNS_ID"] = ca_cwns_data["CWNS_ID"].str.strip()

    # Build leaf → top-level category mapping
    leaf_to_category = {
        leaf: cat_name
        for cat_name, cat_value in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat_name, cat_value, exclude_categories=[])
    }

    # remove facilities with no data in CWNS
    print(
        "Total CWNS facilities excluded due to lack of data:",
        sum((ca_cwns_data[leaf_to_category.keys()] == '0').all(axis=1))
    )
    ca_cwns_data = ca_cwns_data[~(ca_cwns_data[leaf_to_category.keys()] == '0').all(axis=1)]

    # Load Google Sheets
    supplemental_data_df = pd.read_csv("wwtp_process_extraction/data/supplemental_data_unit_processes_by_facility.csv", dtype=str).fillna("")
    supplemental_data_df["Place ID"] = supplemental_data_df["Place ID"].str.strip()
    npdes_text_df = pd.read_csv("wwtp_process_extraction/data/unit_processes_by_facility_manual.csv", dtype=str).fillna("")
    npdes_text_df["Place ID"] = npdes_text_df["Place ID"].str.strip()

    print(f"Ground Truth sheet: {len(supplemental_data_df)} facilities")
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

    supplemental_data_process_cols = [c for c in supplemental_data_df.columns if c not in meta_cols]
    npdes_text_process_cols = [c for c in npdes_text_df.columns if c not in meta_cols]

    all_sheet_process_cols = [
        c
        for c in dict.fromkeys(supplemental_data_process_cols + npdes_text_process_cols)
    ]

    common_facilities = set(supplemental_data_df["Place ID"]) & set(npdes_text_df["Place ID"]) & set(cwns_mapping["Place ID"])
    supplemental_data_common = supplemental_data_df[supplemental_data_df["Place ID"].isin(common_facilities)].copy()
    text_common = npdes_text_df[npdes_text_df["Place ID"].isin(common_facilities)].copy()
    cwns_common, merged_mapping_cwns = build_cwns_facility_processes(
        ca_cwns_data, target_facilities=common_facilities
    )

    # remove facilities that were inadvertently added back but have no CWNS data
    common_facilities = common_facilities & set(cwns_common["Place ID"])

    # print facilities from the ground truth dataNOT in common_facilities
    sd_not_common = set(supplemental_data_df["Place ID"]) - common_facilities
    if sd_not_common:
        print(f"\nFacilities in Ground Truth but not in common set (N={len(sd_not_common)}):")
        for pid in sd_not_common:
            name = supplemental_data_df.loc[supplemental_data_df["Place ID"] == pid, "Facility Name"].iloc[0]
            print(f"  {pid}: {name}")
    # remove 'Unspecified' columns for unit-process-level analysis
    cols_no_unspec = [col for col in all_sheet_process_cols if "Unspecified" not in col]
    sd_cat, npdes_cat, cwns_cat = build_category_facility_sets(
        all_sheet_process_cols,
        supplemental_data_common,
        text_common,
        cwns_common,
        leaf_to_category,
    )

    sd_simple_rows = build_sd_rows(sd_cat, npdes_cat, cwns_cat, common_facilities)

    tick_fontsize = 12
    label_fontsize = 14
    sd_plot_df = pd.DataFrame(sd_simple_rows)
    sd_plot_df = sd_plot_df[sd_plot_df["GroundTruth"] > 0].copy()
    sd_plot_df = sd_plot_df.sort_values("GroundTruth", ascending=False).reset_index(drop=True)

    n_facilities = len(common_facilities)
    for source in ("NPDES", "CWNS"):
        fp_col = f"{source}_FP"
        fn_col = f"{source}_FN"
        total_err_col = f"{source}_Error_Total"
        #  Error rates normalized by total GT occurrences in the category (per-GT)
        sd_plot_df[total_err_col] = sd_plot_df[fp_col] + sd_plot_df[fn_col]
        sd_plot_df[f"{source}_Error_Rate_perGT"] = sd_plot_df.apply(
            lambda r: (r[fp_col] + r[fn_col]) / r["GroundTruth"] if r["GroundTruth"] else 0,
            axis=1,
        )
        sd_plot_df[f"{source}_Missed_Rate_perGT"] = sd_plot_df.apply(
            lambda r: r[fn_col] / r["GroundTruth"] if r["GroundTruth"] else 0,
            axis=1,
        )
        sd_plot_df[f"{source}_Extra_Rate_perGT"] = sd_plot_df.apply(
            lambda r: r[fp_col] / r["GroundTruth"] if r["GroundTruth"] else 0,
            axis=1,
        )
        denom = sd_plot_df[total_err_col].where(sd_plot_df[total_err_col] > 0, 1)
        sd_plot_df[f"{source}_Missed_Share"] = sd_plot_df[fn_col] / denom
        sd_plot_df[f"{source}_Extra_Share"] = sd_plot_df[fp_col] / denom
    sd_plot_df = sd_plot_df.sort_values(
        ["CWNS_Error_Rate_perGT", "NPDES_Error_Rate_perGT"], ascending=[True, True]
    ).reset_index(drop=True)

    # Average error rates (percent) using per-GT normalization (errors per ground-truth occurrence)
    npdes_avg = float(sd_plot_df["NPDES_Error_Rate_perGT"].mean() * 100) if not sd_plot_df.empty else 0.0
    cwns_avg = float(sd_plot_df["CWNS_Error_Rate_perGT"].mean() * 100) if not sd_plot_df.empty else 0.0

    # Build facility-level comparison rows (used for violin panel)
    facility_rows = []
    for fac in sorted(common_facilities):
        sd_rows_match = supplemental_data_common[supplemental_data_common["Place ID"] == fac]
        text_rows_match = text_common[text_common["Place ID"] == fac]
        cwns_rows_match = cwns_common[cwns_common["Place ID"] == fac]
        if sd_rows_match.empty or text_rows_match.empty or cwns_rows_match.empty:
            continue

        supplemental_data_row = sd_rows_match.iloc[0]
        text_row = text_rows_match.iloc[0]
        cwns_row = cwns_rows_match.iloc[0]
        npdes = supplemental_data_row["NPDES No."]
        facility_name = supplemental_data_row["Facility Name"]

        # TODO: create a categorical plot
        # all_cats = 
        # sd_cat_set = {
            
        # }

        sd_set = {
            col
            for col in cols_no_unspec
            if col in supplemental_data_common.columns and is_present(supplemental_data_row.get(col, ""))
        }

        npdes_set = {
            col
            for col in cols_no_unspec
            if col in text_common.columns and is_present(text_row.get(col, ""))
        }

        cwns_set = {
            col
            for col in cols_no_unspec
            if col in cwns_common.columns
            and build_cwns_presence_mask(pd.Series([cwns_row.get(col, "")])).iloc[0]
        }

        src_metrics = {}
        for prefix, pred_set in [("NPDES", npdes_set), ("CWNS", cwns_set)]:
            tp = len(sd_set & pred_set)
            fp = len(pred_set - sd_set)
            fn = len(sd_set - pred_set)
            p = tp / (tp + fp) if (tp + fp) else 0
            r = tp / (tp + fn) if (tp + fn) else 0
            src_metrics[prefix] = dict(
                TP=tp, FP=fp, FN=fn, Precision=p, Recall=r,
                F1=2 * p * r / (p + r) if (p + r) else 0,
                Missed="|".join(sorted(sd_set - pred_set)),
                Extra="|".join(sorted(pred_set - sd_set)),
            )
        nm, cm = src_metrics["NPDES"], src_metrics["CWNS"]
        facility_rows.append(
            {
                "NPDES No.": npdes,
                "Facility Name": facility_name,
                "Supplemental_Data_Count": len(sd_set),
                "NPDES_TP": nm["TP"], "NPDES_FP": nm["FP"], "NPDES_FN": nm["FN"],
                "NPDES_Precision": nm["Precision"], "NPDES_Recall": nm["Recall"], "NPDES_F1": nm["F1"],
                "NPDES_Missed": nm["Missed"], "NPDES_Extra": nm["Extra"],
                "CWNS_TP": cm["TP"], "CWNS_FP": cm["FP"], "CWNS_FN": cm["FN"],
                "CWNS_Precision": cm["Precision"], "CWNS_Recall": cm["Recall"], "CWNS_F1": cm["F1"],
                "CWNS_Missed": cm["Missed"], "CWNS_Extra": cm["Extra"],
            }
        )

    sd_comparison_df = pd.DataFrame(facility_rows)
    sd_comparison_csv = (
        "wwtp_process_extraction/output/supplemental_data_comparison_by_facility.csv"
    )
    sd_comparison_df.to_csv(sd_comparison_csv, index=False)

    # Build per-facility error-rate metrics for violin (panel A)
    facility_metrics = []
    for _, frow in sd_comparison_df.iterrows():
        if frow["Supplemental_Data_Count"] == 0:
            continue
        key = frow.get("NPDES No.", frow.get("Facility Name", ""))
        for src in ("NPDES", "CWNS"):
            fp = frow.get(f"{src}_FP", 0)
            fn = frow.get(f"{src}_FN", 0)
            Supplemental_Data_Count = frow.get("Supplemental_Data_Count", 0)
            try:
                fp = int(fp)
                fn = int(fn)
                sd_count = int(Supplemental_Data_Count)
                cols_to_ignore = {"ok Agency", "Facility Name", "Place ID", "NPDES No.", "PDF_File"} 
                num_cols = len(set(cols_no_unspec) - cols_to_ignore)
            except Exception:
                continue
            facility_metrics.append(
                {
                    "Source": src,
                    "Key": key,
                    "False_Positive_Rate": fp / (num_cols - sd_count),
                    "False_Negative_Rate": fn / sd_count,
                    "Error_Rate": (fp + fn) / num_cols,
                }
            )

    fac_metrics_df = pd.DataFrame(facility_metrics).dropna()

    # Create two-panel figure: A=split violin per facility, B=category-level stacked FP/FN counts
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1]})

    # Panel A: split violin similar to figure_s2 comparing NPDES vs CWNS
    violin_cols = ["False_Positive_Rate", "False_Negative_Rate", "Error_Rate"]
    plot_df = (
        fac_metrics_df.melt(id_vars=["Source", "Key"], value_vars=violin_cols, var_name="Metric", value_name="Value")
    )
    palette = {"NPDES": COLORS["npdes_kw"], "CWNS": COLORS["Clean Watershed Needs Survey"]}
    sns.boxplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Source",
        palette=palette,
        dodge=True,
        width=0.6,
        linewidth=1.0,
        showfliers=False,
        ax=axA,
    )
    # change x-axis label to ""
    axA.set_xlabel("")
    # Style boxplot elements with black edges and consistent linewidth
    for artist in axA.artists:
        try:
            artist.set_edgecolor("black")
            artist.set_linewidth(1.0)
            artist.set_facecolor(artist.get_facecolor())
        except Exception:
            pass
    for line in axA.lines:
        try:
            line.set_color("black")
            line.set_linewidth(1.0)
        except Exception:
            pass
    axA.set_ylim(0, 1)
    axA.set_ylabel(f"Facility-level rate\n(N={int(fac_metrics_df['Key'].nunique())})", fontsize=label_fontsize)
    axA.set_xticks(range(len(violin_cols)))
    axA.set_xticklabels([c.replace("_", " ") for c in violin_cols], fontsize=12)
    axA.tick_params(axis="y", labelsize=12)
    axA.legend(
        handles=[
            Patch(facecolor=to_rgba(COLORS["npdes_kw"]), edgecolor="black", label="Facility Permit"),
            Patch(facecolor=to_rgba(COLORS["Clean Watershed Needs Survey"]), edgecolor="black", label="CWNS"),
        ],
        loc="upper right", 
        bbox_to_anchor=(1.02, 1.18), 
        ncol=2, 
        frameon=False, 
        fontsize=11
    )
    axA.text(-0.12, 1.05, "A.", transform=axA.transAxes, ha="left", va="top", fontsize=16)
    set_thick_spines(axA, linewidth=1.6)

    # Panel B: category-level stacked FP (extra) and FN (missed) rates for NPDES and CWNS (percent)
    w = 0.35
    x = range(len(sd_plot_df))
    for i, row in sd_plot_df.iterrows():
        npdes_x = i - w / 2
        cwns_x = i + w / 2
        # Use rates normalized by total GT occurrences in the category (per-GT)
        npdes_fn_rate = row.get("NPDES_Missed_Rate_perGT", 0)
        npdes_fp_rate = row.get("NPDES_Extra_Rate_perGT", 0)
        cwns_fn_rate = row.get("CWNS_Missed_Rate_perGT", 0)
        cwns_fp_rate = row.get("CWNS_Extra_Rate_perGT", 0)
        axB.bar(
            npdes_x,
            npdes_fn_rate,
            w,
            color=to_rgba(COLORS["npdes_kw"], 0.9),
            edgecolor="black",
            linewidth=1.2,
            hatch="....",
            zorder=2,
        )
        axB.bar(npdes_x, npdes_fp_rate, w, bottom=npdes_fn_rate, color=to_rgba(COLORS["npdes_kw"], 0.45), edgecolor="black", linewidth=1.2, zorder=2)
        axB.bar(
            cwns_x,
            cwns_fn_rate,
            w,
            color=to_rgba(COLORS["Clean Watershed Needs Survey"], 0.9),
            edgecolor="black",
            linewidth=1.2,
            hatch="....",
            zorder=2,
        )
        axB.bar(cwns_x, cwns_fp_rate, w, bottom=cwns_fn_rate, color=to_rgba(COLORS["Clean Watershed Needs Survey"], 0.45), edgecolor="black", linewidth=1.2, zorder=2)

    axB.set_xticks(range(len(sd_plot_df)))
    axB.set_xticklabels(
        [f"{row['Process_Category']} (N={int(row['GroundTruth'])})" for _, row in sd_plot_df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    axB.set_ylabel(f"Categorical error rate", fontsize=label_fontsize)
    axB.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    set_thick_spines(axB, linewidth=1.6)
    axB.text(-0.12, 1.05, "B.", transform=axB.transAxes, ha="left", va="top", fontsize=16)
    axB.set_ylim(0, 1.0)
    pct_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axB.set_yticks(pct_ticks)
    axB.set_yticklabels([f"{p}" for p in pct_ticks], fontsize=tick_fontsize)

    axB.legend(
        handles=[
            Patch(facecolor=to_rgba(COLORS["npdes_kw"], 0.45), edgecolor="black", label="Facility Permit False Positive"),
            Patch(facecolor=to_rgba(COLORS["npdes_kw"], 0.9), edgecolor="black", hatch="....", label="Facility Permit False Negative"),
            Patch(facecolor=to_rgba(COLORS["Clean Watershed Needs Survey"], 0.45), edgecolor="black", label="CWNS False Positive"),
            Patch(facecolor=to_rgba(COLORS["Clean Watershed Needs Survey"], 0.9), edgecolor="black", hatch="....", label="CWNS False Negative"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.66, 1.18),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.6,
        handlelength=1.8,
    )

    plt.subplots_adjust(hspace=0.35, bottom=0.33, top=0.90)
    save_path = f"{figures_dir}/figure_2_supplemental_data_supplemental_data_vs_npdes_text_vs_cwns.png"
    save_and_close(fig, save_path, dpi=300)

    

    sd_rows_with_gt = [r for r in sd_simple_rows if r["GroundTruth"] > 0]
    if sd_rows_with_gt:
        sd_zero_cwns_fp = sum(r["CWNS_FP"] for r in sd_simple_rows if r["GroundTruth"] == 0)
        print(
            f"\nAverage error vs Ground Truth (categories with GT>0): CWNS = {cwns_avg:.1f}%, NPDES Text = {npdes_avg:.1f}%"
        )
        for r in sd_simple_rows:
            if r["GroundTruth"] == 0 and r["CWNS_FP"] > 0:
                print(f"  CWNS FP in category '{r['Process_Category']}' with GT=0, FP={r['CWNS_FP']}")


if __name__ == "__main__":
    main()
