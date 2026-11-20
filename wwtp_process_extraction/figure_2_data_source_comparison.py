import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

from helpers.paths import *
from helpers.utils import (
    is_present,
    precision_recall_f1,
    cwns_mapping,
    build_cwns_facility_processes,
    column_top_category,
    leaf_names,
)
from helpers.plotting import COLORS, save_and_close, set_thick_spines

SOURCE_COLORS = {"NPDES": COLORS["npdes_kw"], "CWNS": COLORS["Clean Watershed Needs Survey"]}
SOURCE_LABELS = {"NPDES": "Facility Permit", "CWNS": "CWNS"}
META_COLS = ["Agency", "Place ID", "Facility Name", "NPDES No.", "PDF_File"]
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
# error_denominator -> (figure, panel A metric, panel B metric)
# "f1" (main figure): error = 1 - F1, split into FN/FP over 2*tp+fp+fn.
# "columns" (SI figure): FP over absent columns, FN over GT count, total over all unit-process
# columns; panel B per GT occurrence.
ERROR_VARIANTS = {
    "f1": ("figure_2", "(1 - Unit Process F1)", "(1 - Categorical F1)"),
    "columns": ("figure_s1", "(per unit-process column)", "(per GT occurrence)"),
}


def f1_error_parts(tp, fp, fn):
    # 1 - F1 split into missed (FN) and extra (FP) shares over 2*tp+fp+fn, so missed + extra is
    # exactly the complement of table_1's F1. Returns (missed, extra, total).
    denom = 2 * tp + fp + fn
    return fn / denom, fp / denom, (fp + fn) / denom


def category_confusion(process_cols, leaf_to_category, frames, facilities):
    # Per category (columns not in leaf_to_category stay their own category): the ground-truth
    # facility count and each source's TP/FP/FN against it
    present = {src: defaultdict(set) for src in frames}
    for col in process_cols:
        category = leaf_to_category.get(col, col)
        for src, df in frames.items():
            if col in df.columns:
                present[src][category].update(df.loc[df[col].map(is_present), "Place ID"])
    rows = []
    for cat in sorted(set().union(*present.values())):
        gt = present["GT"][cat] & facilities
        row = {"Process_Category": cat, "GroundTruth": len(gt)}
        for src in ("NPDES", "CWNS"):
            pred = present[src][cat] & facilities
            row.update({f"{src}_TP": len(pred & gt), f"{src}_FP": len(pred - gt), f"{src}_FN": len(gt - pred)})
        rows.append(row)
    return pd.DataFrame(rows)

# Main execution
ca_cwns_data = pd.read_csv(CWNS_TABLE_CSV, dtype=str)

# remove facilities with no data in CWNS
no_data = (ca_cwns_data[leaf_names] == '0').all(axis=1)
print("Total CWNS facilities excluded due to lack of data:", sum(no_data))
ca_cwns_data = ca_cwns_data[~no_data]

# Ground truth (supplemental data) and the manual permit reading
supplemental_data_df = pd.read_csv(SUPPLEMENTAL_CSV, dtype=str)
npdes_text_df = pd.read_csv(MANUAL_CSV, dtype=str)

print(f"Ground Truth sheet: {len(supplemental_data_df)} facilities")
print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")

supplemental_data_process_cols = [c for c in supplemental_data_df.columns if c not in META_COLS]
npdes_text_process_cols = [c for c in npdes_text_df.columns if c not in META_COLS]
all_sheet_process_cols = list(dict.fromkeys(supplemental_data_process_cols + npdes_text_process_cols))
# remove 'Unspecified' columns for unit-process-level analysis
cols_no_unspec = [col for col in all_sheet_process_cols if "Unspecified" not in col]

common_facilities = (
    set(supplemental_data_df["Place ID"]) & set(npdes_text_df["Place ID"]) & set(cwns_mapping["Place ID"])
)
supplemental_data_common = supplemental_data_df[supplemental_data_df["Place ID"].isin(common_facilities)]
text_common = npdes_text_df[npdes_text_df["Place ID"].isin(common_facilities)]
cwns_common, _ = build_cwns_facility_processes(ca_cwns_data, target_facilities=common_facilities)

# keep only facilities with CWNS data
common_facilities = common_facilities & set(cwns_common["Place ID"])

# print facilities from the ground truth data NOT in common_facilities
sd_not_common = set(supplemental_data_df["Place ID"]) - common_facilities
print(f"\nFacilities in Ground Truth but not in common set (N={len(sd_not_common)}):")
sd_names = supplemental_data_df.set_index("Place ID")["Facility Name"]
for pid in sorted(sd_not_common):
    print(f"  {pid}: {sd_names[pid]}")

# Category-level counts for panel B
frames = {"GT": supplemental_data_common, "NPDES": text_common, "CWNS": cwns_common}
category_df = category_confusion(all_sheet_process_cols, column_top_category, frames, common_facilities)
category_df = category_df[category_df["GroundTruth"] > 0]
category_df = category_df.sort_values("GroundTruth", ascending=False).reset_index(drop=True)

# Per-leaf (unit-process) rows for the GT=0 false-positive diagnostic; an empty mapping keeps
# each leaf separate (panel B stays category-level by design)
leaf_rows = category_confusion(cols_no_unspec, {}, frames, common_facilities)
print("\nCWNS false positives in unit processes with no ground truth:")
for _, r in leaf_rows.iterrows():
    if r["GroundTruth"] == 0 and r["CWNS_FP"] > 0:
        print(f"  CWNS FP in unit process '{r['Process_Category']}' with GT=0, FP={int(r['CWNS_FP'])}")

# Facility-level comparison rows (CSV) and panel A error rates for each error variant
num_cols = len(cols_no_unspec)
sd_by_id = supplemental_data_common.set_index("Place ID")
text_by_id = text_common.set_index("Place ID")
cwns_by_id = cwns_common.set_index("Place ID")
facility_rows = []
facility_metrics = {variant: [] for variant in ERROR_VARIANTS}
for fac in sorted(common_facilities):
    supplemental_data_row = sd_by_id.loc[fac]
    truth_pos = {c for c in cols_no_unspec if is_present(supplemental_data_row.get(c, ""))}
    row = {
        "NPDES No.": supplemental_data_row["NPDES No."],
        "Facility Name": supplemental_data_row["Facility Name"],
        "Supplemental_Data_Count": len(truth_pos),
    }
    for prefix, pred_row in [("NPDES", text_by_id.loc[fac]), ("CWNS", cwns_by_id.loc[fac])]:
        pred_pos = {c for c in cols_no_unspec if is_present(pred_row.get(c, ""))}
        tp, fp, fn = len(truth_pos & pred_pos), len(pred_pos - truth_pos), len(truth_pos - pred_pos)
        p, r, f1 = precision_recall_f1(tp, fp, fn, empty=0)
        row.update({
            f"{prefix}_TP": tp, f"{prefix}_FP": fp, f"{prefix}_FN": fn,
            f"{prefix}_Precision": p, f"{prefix}_Recall": r, f"{prefix}_F1": f1,
            f"{prefix}_Missed": "|".join(sorted(truth_pos - pred_pos)),
            f"{prefix}_Extra": "|".join(sorted(pred_pos - truth_pos)),
        })
        rates = {
            "f1": f1_error_parts(tp, fp, fn),
            "columns": (fn / len(truth_pos), fp / (num_cols - len(truth_pos)), (fp + fn) / num_cols),
        }
        for variant, (false_neg_rate, false_pos_rate, total) in rates.items():
            facility_metrics[variant].append({
                "Source": prefix,
                "False Negative (FN) Rate": false_neg_rate,
                "False Positive (FP) Rate": false_pos_rate,
                "Error Rate": total,
            })
    facility_rows.append(row)

pd.DataFrame(facility_rows).to_csv(SUPPLEMENTAL_COMPARISON_CSV, index=False)
n_fac = len(facility_rows)

for error_denominator, (filename, panelA_metric, panelB_metric) in ERROR_VARIANTS.items():
    fac_metrics_df = pd.DataFrame(facility_metrics[error_denominator])
    for source in ("NPDES", "CWNS"):
        tp, fp, fn = (category_df[f"{source}_{kind}"] for kind in ("TP", "FP", "FN"))
        if error_denominator == "f1":
            # missed and extra shares of 1 - F1, the same bounded error as panel A and table_1
            rates = f1_error_parts(tp, fp, fn)
        else:
            # SI: rates normalized by GT occurrences in the category
            gt = category_df["GroundTruth"]
            rates = (fn / gt, fp / gt, (fp + fn) / gt)
        for kind, rate in zip(("Missed", "Extra", "Error"), rates):
            category_df[f"{source}_{kind}_Rate"] = rate
    sd_plot_df = category_df.sort_values(["CWNS_Error_Rate", "NPDES_Error_Rate"]).reset_index(drop=True)

    # Two panels: A = per-facility boxplot, B = category-level stacked FN/FP rates
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 10))

    # Panel A: boxplot comparing NPDES vs CWNS
    boxplot_cols = ["False Positive (FP) Rate", "False Negative (FN) Rate", "Error Rate"]
    plot_df = fac_metrics_df.melt(
        id_vars=["Source"], value_vars=boxplot_cols, var_name="Metric", value_name="Value"
    )
    sns.boxplot(
        data=plot_df,
        x="Metric",
        y="Value",
        hue="Source",
        palette=SOURCE_COLORS,
        width=0.6,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "^", "markerfacecolor": "black", "markeredgecolor": "black"},
        ax=axA,
    )
    axA.set_xlabel("")
    # Style boxplot lines with black edges and consistent linewidth
    for line in axA.lines:
        line.set_color("black")
        line.set_linewidth(1.0)
    axA.set_ylim(0, 1)
    axA.set_ylabel(f"Unit Process Error Metric\nPer-facility (N={n_fac})", fontsize=LABEL_FONTSIZE)
    xlabels = ["False Positive (FP) Rate", "False Negative (FN) Rate", f"Overall Error\n{panelA_metric}"]
    axA.set_xticks(range(len(boxplot_cols)), xlabels, fontsize=TICK_FONTSIZE)
    axA.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axA.legend(
        handles=[
            Patch(facecolor=SOURCE_COLORS[src], edgecolor="black", label=SOURCE_LABELS[src])
            for src in SOURCE_COLORS
        ],
        loc="upper right",
        bbox_to_anchor=(1.02, 1.18),
        ncol=2,
        frameon=False,
        fontsize=11
    )
    axA.text(-0.17, 1.05, "A.", transform=axA.transAxes, va="top", fontsize=16)
    set_thick_spines(axA)

    # Panel B: category-level stacked FN (missed, hatched) and FP (extra) rates, NPDES left of CWNS
    w = 0.35
    for i, row in sd_plot_df.iterrows():
        for src, x in (("NPDES", i - w / 2), ("CWNS", i + w / 2)):
            bottom = 0
            for kind, alpha, hatch in (("Missed", 0.9, "...."), ("Extra", 0.45, None)):
                rate = row[f"{src}_{kind}_Rate"]
                axB.bar(x, rate, w, bottom=bottom, color=to_rgba(SOURCE_COLORS[src], alpha),
                        edgecolor="black", linewidth=1.2, hatch=hatch)
                bottom += rate

    axB.set_xticks(
        range(len(sd_plot_df)),
        [f"{row['Process_Category']} (N={int(row['GroundTruth'])})" for _, row in sd_plot_df.iterrows()],
        rotation=45,
        ha="right",
        fontsize=TICK_FONTSIZE,
    )
    axB.set_ylabel(
        f"Categorical Error Metric\n{panelB_metric}\nAcross Facilities (N={n_fac})", fontsize=LABEL_FONTSIZE
    )
    axB.tick_params(labelsize=TICK_FONTSIZE)
    set_thick_spines(axB)
    axB.text(-0.17, 1.05, "B.", transform=axB.transAxes, va="top", fontsize=16)
    axB.set_ylim(0, 1.0)
    pct_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    axB.set_yticks(pct_ticks, [f"{p}" for p in pct_ticks], fontsize=TICK_FONTSIZE)

    axB.legend(
        handles=[
            Patch(
                facecolor=to_rgba(SOURCE_COLORS[src], alpha), edgecolor="black", hatch=hatch,
                label=f"{SOURCE_LABELS[src]} {kind}",
            )
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
    save_and_close(fig, FIGURES_DIR / f"{filename}.png")

    # Mean facility-level error rate (unit-process granularity), as in panel A's Error Rate box
    err_by_src = fac_metrics_df.groupby("Source")["Error Rate"].mean() * 100
    print(
        f"\n[{error_denominator}] Mean facility-level error rate (unit processes): "
        f"Facility Permit = {err_by_src['NPDES']:.1f}%, CWNS = {err_by_src['CWNS']:.1f}%"
    )
