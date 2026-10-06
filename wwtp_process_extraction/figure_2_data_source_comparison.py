import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

from helpers.utils import (
    is_present,
    get_leaf_names,
    precision_recall_f1,
    cwns_mapping,
    build_cwns_facility_processes,
    unitprocess_keywords,
    CWNS_TABLE_CSV,
    DATA_DIR,
    OUTPUT_DIR,
    FIGURES_DIR,
    MANUAL_CSV,
)
from helpers.plotting import COLORS
from helpers.plotting import save_and_close, set_thick_spines

SOURCE_COLORS = {"NPDES": COLORS["npdes_kw"], "CWNS": COLORS["Clean Watershed Needs Survey"]}
SOURCE_LABELS = {"NPDES": "Facility Permit", "CWNS": "CWNS"}


def f1_error_parts(tp, fp, fn):
    """Split F1 error (1 - F1) into missed (FN) and extra (FP) shares, plus their total.

    All three divide by 2*tp+fp+fn (= |truth| + |prediction|, the F1/Dice denominator), so
    missed + extra equals the total error 1 - F1, bounded in [0, 1] — figure_2's error is
    exactly the complement of table_1's F1. Zeros are returned when the denominator is zero.
    Returns (missed, extra, total).
    """
    denom = 2 * tp + fp + fn
    if not denom:
        return 0, 0, 0
    return fn / denom, fp / denom, (fp + fn) / denom


def build_category_facility_sets(
    process_cols, sd_common, text_common, cwns_common, leaf_to_category
):
    """
    Aggregate facility-level sets per top-level category for GT, NPDES text, and CWNS.
    Returns three defaultdict(set) of facilities where the category is present:
    sd_fac, npdes_fac, cwns_fac.
    """
    sd_fac = defaultdict(set)
    npdes_fac = defaultdict(set)
    cwns_fac = defaultdict(set)

    for col in process_cols:
        category = leaf_to_category.get(col, col)
        for df, fac in ((sd_common, sd_fac), (text_common, npdes_fac), (cwns_common, cwns_fac)):
            if col in df.columns:
                fac[category].update(df.loc[df[col].map(is_present), "Place ID"])

    return sd_fac, npdes_fac, cwns_fac


def build_sd_rows(sd_fac, npdes_fac, cwns_fac, common_facilities):
    """Build summary rows for ground-truth comparison plotting from per-category facility sets."""
    all_cats = sorted(set(sd_fac) | set(npdes_fac) | set(cwns_fac))
    rows = []
    for cat in all_cats:
        sd_p = sd_fac[cat] & common_facilities
        npdes_p = npdes_fac[cat] & common_facilities
        cwns_p = cwns_fac[cat] & common_facilities

        rows.append(
            {
                "Process_Category": cat,
                "GroundTruth": len(sd_p),
                "NPDES_TP": len(npdes_p & sd_p),
                "NPDES_FP": len(npdes_p - sd_p),
                "NPDES_FN": len(sd_p - npdes_p),
                "CWNS_TP": len(cwns_p & sd_p),
                "CWNS_FP": len(cwns_p - sd_p),
                "CWNS_FN": len(sd_p - cwns_p),
            }
        )
    return rows


def main(error_denominator="f1"):
    # "f1" (main figure): error = 1 - F1, split into FP/FN over 2*tp+fp+fn.
    # "columns" (SI figure): old denominators - FP over absent columns, FN over GT count,
    # total over all unit-process columns; panel B normalized per GT occurrence.
    ca_cwns_data = pd.read_csv(CWNS_TABLE_CSV, dtype=str, low_memory=False)
    ca_cwns_data["CWNS_ID"] = ca_cwns_data["CWNS_ID"].str.strip()

    # Build leaf → top-level category mapping
    leaf_to_category = {
        leaf: cat_name
        for cat_name, cat_value in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat_name, cat_value, exclude_categories=[])
    }

    # remove facilities with no data in CWNS
    no_data = (ca_cwns_data[leaf_to_category.keys()] == '0').all(axis=1)
    print("Total CWNS facilities excluded due to lack of data:", sum(no_data))
    ca_cwns_data = ca_cwns_data[~no_data]

    # Load Google Sheets
    supplemental_data_df = pd.read_csv(DATA_DIR / "unit_processes_by_facility_supplemental_data.csv", dtype=str).fillna("")
    supplemental_data_df["Place ID"] = supplemental_data_df["Place ID"].str.strip()
    npdes_text_df = pd.read_csv(MANUAL_CSV, dtype=str).fillna("")
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

    all_sheet_process_cols = list(dict.fromkeys(supplemental_data_process_cols + npdes_text_process_cols))

    common_facilities = set(supplemental_data_df["Place ID"]) & set(npdes_text_df["Place ID"]) & set(cwns_mapping["Place ID"])
    supplemental_data_common = supplemental_data_df[supplemental_data_df["Place ID"].isin(common_facilities)].copy()
    text_common = npdes_text_df[npdes_text_df["Place ID"].isin(common_facilities)].copy()
    cwns_common, _ = build_cwns_facility_processes(
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

    sd_simple_rows = build_sd_rows(
        sd_cat, npdes_cat, cwns_cat, common_facilities
    )

    tick_fontsize = 12
    label_fontsize = 14
    sd_plot_df = pd.DataFrame(sd_simple_rows)
    sd_plot_df = sd_plot_df[sd_plot_df["GroundTruth"] > 0].copy()
    sd_plot_df = sd_plot_df.sort_values("GroundTruth", ascending=False).reset_index(drop=True)

    for source in ("NPDES", "CWNS"):
        tp_col, fp_col, fn_col = f"{source}_TP", f"{source}_FP", f"{source}_FN"
        if error_denominator == "f1":
            # Missed (FN) and extra (FP) shares of 1 - F1; they sum to the total F1 error, so
            # both panels and table_1 share one bounded error definition.
            parts = sd_plot_df.apply(
                lambda r: f1_error_parts(r[tp_col], r[fp_col], r[fn_col]),
                axis=1,
                result_type="expand",
            )
            sd_plot_df[[f"{source}_Missed_Rate", f"{source}_Extra_Rate", f"{source}_Error_Rate"]] = parts
        else:
            # SI: rates normalized by GT occurrences in the category (per-GT)
            gt = sd_plot_df["GroundTruth"]
            sd_plot_df[f"{source}_Missed_Rate"] = sd_plot_df[fn_col] / gt
            sd_plot_df[f"{source}_Extra_Rate"] = sd_plot_df[fp_col] / gt
            sd_plot_df[f"{source}_Error_Rate"] = (sd_plot_df[fp_col] + sd_plot_df[fn_col]) / gt
    sd_plot_df = sd_plot_df.sort_values(
        ["CWNS_Error_Rate", "NPDES_Error_Rate"], ascending=[True, True]
    ).reset_index(drop=True)

    # Per-leaf (unit-process) rows for the printed GT=0 false-positive diagnostic below;
    # identity mapping keeps each leaf separate (panel B above stays category-level by design).
    sd_leaf, npdes_leaf, cwns_leaf = build_category_facility_sets(
        cols_no_unspec, supplemental_data_common, text_common, cwns_common,
        {leaf: leaf for leaf in cols_no_unspec},
    )
    leaf_rows = pd.DataFrame(build_sd_rows(
        sd_leaf, npdes_leaf, cwns_leaf, common_facilities
    ))

    # Build facility-level comparison rows (used for violin panel)
    facility_rows = []
    for fac in sorted(common_facilities):
        supplemental_data_row = supplemental_data_common[supplemental_data_common["Place ID"] == fac].iloc[0]
        text_row = text_common[text_common["Place ID"] == fac].iloc[0]
        cwns_row = cwns_common[cwns_common["Place ID"] == fac].iloc[0]
        truth_pos = {c for c in cols_no_unspec if is_present(supplemental_data_row.get(c, ""))}
        row = {
            "NPDES No.": supplemental_data_row["NPDES No."],
            "Facility Name": supplemental_data_row["Facility Name"],
            "Supplemental_Data_Count": len(truth_pos),
        }
        for prefix, pred_row in [("NPDES", text_row), ("CWNS", cwns_row)]:
            pred_pos = {c for c in cols_no_unspec if is_present(pred_row.get(c, ""))}
            tp, fp, fn = len(truth_pos & pred_pos), len(pred_pos - truth_pos), len(truth_pos - pred_pos)
            p, r, f1, _ = precision_recall_f1(tp, fp, fn, empty=0)
            row.update({
                f"{prefix}_TP": tp, f"{prefix}_FP": fp, f"{prefix}_FN": fn,
                f"{prefix}_Precision": p, f"{prefix}_Recall": r, f"{prefix}_F1": f1,
                f"{prefix}_Missed": "|".join(sorted(truth_pos - pred_pos)),
                f"{prefix}_Extra": "|".join(sorted(pred_pos - truth_pos)),
            })
        facility_rows.append(row)

    sd_comparison_df = pd.DataFrame(facility_rows)
    sd_comparison_df.to_csv(OUTPUT_DIR / "supplemental_data_comparison_by_facility.csv", index=False)

    # Build per-facility error-rate metrics for violin (panel A)
    num_cols = len(cols_no_unspec)
    facility_metrics = []
    for _, frow in sd_comparison_df.iterrows():
        if frow["Supplemental_Data_Count"] == 0:
            continue
        key = frow["NPDES No."]
        for src in ("NPDES", "CWNS"):
            tp = int(frow[f"{src}_TP"])
            fp = int(frow[f"{src}_FP"])
            fn = int(frow[f"{src}_FN"])
            if error_denominator == "f1":
                false_neg_rate, false_pos_rate, total = f1_error_parts(tp, fp, fn)
            else:
                # SI: FP over absent columns, FN over GT count, total over all columns
                sd_count = tp + fn
                false_pos_rate = fp / (num_cols - sd_count)
                false_neg_rate = fn / sd_count
                total = (fp + fn) / num_cols
            facility_metrics.append(
                {
                    "Source": src,
                    "Key": key,
                    "False Negative (FN) Rate": false_neg_rate,
                    "False Positive (FP) Rate": false_pos_rate,
                    "Error Rate": total,
                }
            )

    fac_metrics_df = pd.DataFrame(facility_metrics)

    # Create two-panel figure: A=split violin per facility, B=category-level stacked FP/FN counts
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 1]})

    # Panel A: boxplot comparing NPDES vs CWNS
    panelA_metric = "(1 - Unit Process F1)" if error_denominator == "f1" else "(per unit-process column)"
    boxplot_cols = ["False Positive (FP) Rate", "False Negative (FN) Rate", "Error Rate"]
    plot_df = (
        fac_metrics_df.melt(id_vars=["Source", "Key"], value_vars=boxplot_cols, var_name="Metric", value_name="Value")
    )
    sns.boxplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Source",
        palette=SOURCE_COLORS,
        dodge=True,
        width=0.6,
        linewidth=1.0,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black"},
        ax=axA,
    )
    # change x-axis label to ""
    axA.set_xlabel("")
    # Style boxplot lines with black edges and consistent linewidth
    for line in axA.lines:
        line.set_color("black")
        line.set_linewidth(1.0)
    axA.set_ylim(0, 1)
    axA.set_ylabel(f"Unit Process Error Metric\nPer-facility (N={int(fac_metrics_df['Key'].nunique())})", fontsize=label_fontsize)
    axA.set_xticks(range(len(boxplot_cols)))
    xlabels = ["False Positive (FP) Rate", "False Negative (FN) Rate", f"Overall Error\n{panelA_metric}"]
    axA.set_xticklabels(xlabels, fontsize=12)
    axA.tick_params(axis="y", labelsize=12)
    axA.legend(
        handles=[
            Patch(facecolor=to_rgba(SOURCE_COLORS[src]), edgecolor="black", label=SOURCE_LABELS[src])
            for src in SOURCE_COLORS
        ],
        loc="upper right", 
        bbox_to_anchor=(1.02, 1.18), 
        ncol=2, 
        frameon=False, 
        fontsize=11
    )
    axA.text(-0.17, 1.05, "A.", transform=axA.transAxes, ha="left", va="top", fontsize=16)
    set_thick_spines(axA, linewidth=1.6)

    # Panel B: category-level stacked FP (extra) and FN (missed) rates for NPDES and CWNS (percent)
    w = 0.35
    for i, row in sd_plot_df.iterrows():
        # Missed (FN) and extra (FP) shares of 1 - F1 (same definition as panel A), NPDES left of CWNS
        for src, x in (("NPDES", i - w / 2), ("CWNS", i + w / 2)):
            fn_rate = row[f"{src}_Missed_Rate"]
            axB.bar(
                x,
                fn_rate,
                w,
                color=to_rgba(SOURCE_COLORS[src], 0.9),
                edgecolor="black",
                linewidth=1.2,
                hatch="....",
                zorder=2,
            )
            axB.bar(x, row[f"{src}_Extra_Rate"], w, bottom=fn_rate, color=to_rgba(SOURCE_COLORS[src], 0.45), edgecolor="black", linewidth=1.2, zorder=2)

    axB.set_xticks(range(len(sd_plot_df)))
    axB.set_xticklabels(
        [f"{row['Process_Category']} (N={int(row['GroundTruth'])})" for _, row in sd_plot_df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    panelB_metric = "(1 - Categorical F1)" if error_denominator == "f1" else "(per GT occurrence)"
    axB.set_ylabel(f"Categorical Error Metric\n{panelB_metric}\nAcross Facilities (N={int(fac_metrics_df['Key'].nunique())})", fontsize=label_fontsize)
    axB.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    set_thick_spines(axB, linewidth=1.6)
    axB.text(-0.17, 1.05, "B.", transform=axB.transAxes, ha="left", va="top", fontsize=16)
    axB.set_ylim(0, 1.0)
    pct_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axB.set_yticks(pct_ticks)
    axB.set_yticklabels([f"{p}" for p in pct_ticks], fontsize=tick_fontsize)

    axB.legend(
        handles=[
            Patch(facecolor=to_rgba(SOURCE_COLORS[src], alpha), edgecolor="black", hatch=hatch, label=f"{SOURCE_LABELS[src]} {kind}")
            for src in SOURCE_COLORS
            for kind, alpha, hatch in (("FP", 0.45, None), ("FN", 0.9, "...."))
        ],
        loc="upper center",
        bbox_to_anchor=(0.59, 1.18),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.6,
        handlelength=1.8,
    )

    plt.subplots_adjust(hspace=0.35, bottom=0.33, top=0.90)
    filename = "figure_2" if error_denominator == "f1" else "figure_s1"
    save_and_close(fig, FIGURES_DIR / f"{filename}.png", dpi=300)

    # Mean facility-level F1 error rate (unit-process granularity) — matches panel A's Error Rate box.
    err_by_src = fac_metrics_df.groupby("Source")["Error Rate"].mean() * 100
    print(
        f"\n[{error_denominator}] Mean facility-level error rate (unit processes): Facility Permit = {err_by_src['NPDES']:.1f}%, CWNS = {err_by_src['CWNS']:.1f}%"
    )
    for _, r in leaf_rows.iterrows():
        if r["GroundTruth"] == 0 and r["CWNS_FP"] > 0:
            print(f"  CWNS FP in unit process '{r['Process_Category']}' with GT=0, FP={int(r['CWNS_FP'])}")


if __name__ == "__main__":
    main()  # main paper figure: error = 1 - F1
    main(error_denominator="columns")  # SI figure: number-of-columns denominator
