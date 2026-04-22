from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from helpers.utils import normalize_facility_name


DEFAULT_TRUTH = Path("npdes_permits/data/Train _ Test data - Test (9).csv")
DEFAULT_LLM = Path("npdes_permits/output/llm_unit_processes_by_facility.csv")
DEFAULT_KEYWORD = Path("npdes_permits/output/2026-2-18/unit_processes.csv")
DEFAULT_KEYWORDS = Path("npdes_permits/data/unitprocess_keywords.json")
DEFAULT_OUTPUT_DIR = Path("npdes_permits/output/llm_model_comparison/source_comparison")

SOURCE_ORDER = ["Truth", "LLM", "Keyword"]
STATUS_ORDER = ["present", "planned", "past", "off-site"]
SOURCE_COLORS = {
    "Truth": "#8c8c8c",
    "LLM": "#ff8c00",
    "Keyword": "#ff5fa2",
}
FAMILY_SOURCE_COLORS = {
    "Truth": "#555555",
    "LLM": "#ff6a00",
    "Keyword": "#ff1493",
}
STATUS_HATCHES = {
    "present": "",
    "planned": "//",
    "past": "xx",
    "off-site": "..",
}

META_COLS = {"Agency", "AGENCY_NAME", "Facility_Name", "FACILITY_NAME", "NPDES_No", "PERMIT_NUMBER", "PDF_File", "Shared_PDF"}

FAMILY_PROCESS_MAP = {
    "Primary Treatment": [
        "Comminution",
        "Screening/Microstrainer",
        "Grit Removal",
        "Equalization",
        "Primary Clarification",
    ],
    "AS": [
        "SBR",
        "Oxidation Ditch",
        "Pure Oxygen",
        "Extended Aeration",
        "Stepfeed",
        "Unspecified AS",
    ],
    "Lagoon": [
        "Facultative Lagoon",
        "Aerated Lagoon",
        "Anaerobic Lagoon",
        "Polishing Lagoon",
        "Stabilization Pond",
        "Unspecified Lagoon",
    ],
    "Nutrient Removal": [
        "AO",
        "A2O",
        "MLE",
        "UCT",
        "Four Stage Bardenpho",
        "Five Stage Bardenpho",
        "Unspecified BNR",
        "Chemical N Removal",
        "Chemical P Removal",
        "Unspecified Nutrient Removal",
    ],
    "Filtration": [
        "Secondary Filtration",
        "Media Filtration",
        "Biologically Active Filtration",
        "Trickling Filter",
        "Rotating Biological Contactor",
        "Anaerobic Filter",
        "Membrane Aerated Biofilm Reactor",
        "Unspecified FFR",
        "Nanofiltration",
        "Electrodialysis Reversal",
        "Electrodialysis",
        "Reverse Osmosis",
        "Ultrafiltration",
        "Microfiltration",
        "Membrane Bioreactor",
        "Unspecified Membrane Process",
        "Unspecified Filtration",
    ],
    "Disinfection": [
        "Chlorination",
        "Dechlorination",
        "UV Disinfection",
        "Ozonation",
        "Thermal Disinfection",
        "Unspecified Disinfection",
    ],
    "Solids Processing": [
        "Anaerobic Digestion",
        "Aerobic Digestion",
        "Unspecified Digestion",
        "Mechanical Dewatering",
        "Centrifuge",
        "Drying",
        "Elutriation",
        "Unspecified Dewatering",
        "Thickening",
        "Land Application",
        "Biosolids Lagoon",
        "Biosolids Landfill",
        "Unspecified Disposal",
        "MHI",
        "FBI",
        "Unspecified Incineration",
        "Cogeneration",
        "Heat Recovery",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="Ground truth CSV")
    parser.add_argument("--llm", type=Path, default=DEFAULT_LLM, help="LLM postprocessed CSV")
    parser.add_argument("--keyword", type=Path, default=DEFAULT_KEYWORD, help="Keyword-based CSV")
    parser.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS, help="unitprocess_keywords.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def build_label_to_family_map(keywords: dict[str, Any]) -> dict[str, str]:
    label_to_family: dict[str, str] = {}

    def walk(node: Any, root_family: str) -> None:
        if not isinstance(node, dict):
            return

        skip_keys = {
            "alt_names",
            "alt_names_case_sensitive",
            "cwns_processes",
            "ontology_triggers",
            "ontology_triggers_multi",
            "exclude_if_any",
            "priority",
            "global_priority",
            "secondary_category",
        }
        for key, value in node.items():
            if key in skip_keys:
                continue
            if isinstance(value, dict):
                label_to_family.setdefault(key, root_family)
                walk(value, root_family)
            else:
                label_to_family.setdefault(key, root_family)

    for family, node in keywords.items():
        label_to_family.setdefault(family, family)
        walk(node, family)

    return label_to_family


def extract_leaf_labels_in_json_order(keywords: dict[str, Any]) -> list[str]:
    labels: list[str] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for name, details in node.items():
            if not isinstance(details, dict):
                continue
            if "alt_names" in details:
                labels.append(name)
            else:
                walk(details)

    walk(keywords)
    return labels


def normalize_key(value: Any) -> str:
    text = str(value or "").strip()
    return text


def facility_key(row: pd.Series, permit_col: str, name_col: str) -> str:
    permit = normalize_key(row.get(permit_col, ""))
    if permit:
        return f"permit::{permit}"
    facility_name = normalize_facility_name(row.get(name_col, ""))
    if facility_name:
        return f"name::{facility_name}"
    pdf_file = normalize_key(row.get("PDF_File", ""))
    if pdf_file:
        return f"pdf::{pdf_file}"
    return f"row::{id(row)}"


def normalize_status_set(value: Any, source: str) -> set[str]:
    if pd.isna(value):
        return set()

    text = str(value).strip().upper()
    if not text or text in {"0", "NAN", "NONE"}:
        return set()

    if source == "Truth":
        if "OFF-SITE" in text or text == "THIRD-PARTY":
            return {"off-site"}
        if text.startswith("PAST"):
            return {"past"}
        if text.startswith("PLANNED") or "FUTURE" in text:
            return {"planned"}
        if text.startswith("PRESENT") or text == "YES":
            return {"present"}
        return set()

    if source == "LLM":
        if "OFF-SITE" in text or text == "THIRD-PARTY":
            return {"off-site"}
        if text.startswith("PAST"):
            return {"past"}
        if text.startswith("PLANNED"):
            return {"planned"}
        if text.startswith("PRESENT"):
            return {"present"}
        return set()

    if source == "Keyword":
        if text.startswith("PRESENT_AND_FUTURE"):
            return {"present", "planned"}
        if text.startswith("PRESENT"):
            return {"present"}
        if text.startswith("FUTURE"):
            return {"planned"}
        return set()

    return set()


def infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    permit_candidates = ["NPDES_No", "PERMIT_NUMBER"]
    name_candidates = ["Facility_Name", "FACILITY_NAME"]

    permit_col = next((c for c in permit_candidates if c in df.columns), None)
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if permit_col is None or name_col is None:
        raise ValueError("Could not infer permit and facility name columns from input CSV")
    return permit_col, name_col


def build_label_columns(
    truth_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    keyword_df: pd.DataFrame,
    json_leaf_order: list[str],
) -> list[str]:
    label_cols = []
    seen = set()
    for df in [truth_df, llm_df, keyword_df]:
        for col in df.columns:
            if col in META_COLS or col in seen:
                continue
            seen.add(col)
            label_cols.append(col)

    ordered = [col for col in json_leaf_order if col in seen]
    ordered += [col for col in label_cols if col not in ordered]
    return ordered


def aggregate_source(df: pd.DataFrame, source: str, label_cols: list[str], permit_col: str, name_col: str) -> pd.DataFrame:
    grouped: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        key = facility_key(row, permit_col, name_col)
        record = grouped.setdefault(
            key,
            {
                "key": key,
                "PERMIT_NUMBER": normalize_key(row.get(permit_col, "")),
                "FACILITY_NAME": normalize_key(row.get(name_col, "")),
            },
        )

        if not record["PERMIT_NUMBER"]:
            record["PERMIT_NUMBER"] = normalize_key(row.get(permit_col, ""))
        if not record["FACILITY_NAME"]:
            record["FACILITY_NAME"] = normalize_key(row.get(name_col, ""))

        for col in label_cols:
            status_set = normalize_status_set(row.get(col, ""), source)
            if not status_set:
                continue
            record.setdefault(col, set()).update(status_set)

    for record in grouped.values():
        for col in label_cols:
            record.setdefault(col, set())

    return pd.DataFrame(grouped.values())


def collapse_to_families(
    df: pd.DataFrame,
    label_cols: list[str],
    label_to_family: dict[str, str],
    json_family_order: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    present_families = list(dict.fromkeys(label_to_family.get(col, col) for col in label_cols))
    family_cols = [fam for fam in json_family_order if fam in present_families]
    family_cols += [fam for fam in present_families if fam not in family_cols]
    collapsed_rows = []

    for _, row in df.iterrows():
        collapsed = {"key": row["key"], "PERMIT_NUMBER": row["PERMIT_NUMBER"], "FACILITY_NAME": row["FACILITY_NAME"]}
        family_state_sets = {family: set() for family in family_cols}
        for col in label_cols:
            family = label_to_family.get(col, col)
            family_state_sets[family].update(row[col])
        collapsed.update(family_state_sets)
        collapsed_rows.append(collapsed)

    return pd.DataFrame(collapsed_rows), family_cols


def build_counts(df: pd.DataFrame, label_cols: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        label: {status: 0 for status in STATUS_ORDER}
        for label in label_cols
    }
    for _, row in df.iterrows():
        for label in label_cols:
            for status in row[label]:
                counts[label][status] += 1
    return counts


def compute_metrics(truth_df: pd.DataFrame, pred_df: pd.DataFrame, label_cols: list[str], source_name: str) -> pd.DataFrame:
    rows = []
    truth_indexed = truth_df.set_index("key")
    pred_indexed = pred_df.set_index("key")
    keys = list(truth_indexed.index)

    for label in label_cols:
        tp = fp = fn = tn = 0
        state_correct = 0
        state_total = 0
        for key in keys:
            truth_states = truth_indexed.at[key, label]
            pred_states = pred_indexed.at[key, label]
            truth_pos = bool(truth_states)
            pred_pos = bool(pred_states)

            tp += int(truth_pos and pred_pos)
            fp += int((not truth_pos) and pred_pos)
            fn += int(truth_pos and (not pred_pos))
            tn += int((not truth_pos) and (not pred_pos))

            if truth_pos:
                state_total += 1
                state_correct += int(truth_states == pred_states)

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
        missed_rate = fn / (tp + fn) if (tp + fn) else float("nan")
        hallucinated_rate = fp / (tp + fp) if (tp + fp) else float("nan")
        state_accuracy = state_correct / state_total if state_total else float("nan")

        rows.append(
            {
                "Source": source_name,
                "Label": label,
                "Support_Truth": int(tp + fn),
                "Support_Pred": int(tp + fp),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Accuracy": accuracy,
                "Missed_Rate": missed_rate,
                "Hallucinated_Rate": hallucinated_rate,
                "State_Accuracy": state_accuracy,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metric_df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    rows = []
    for source, subset in metric_df.groupby("Source", sort=False):
        usable = subset[subset[["Support_Truth", "Support_Pred", "TP", "FP", "FN"]].sum(axis=1) > 0]
        rows.append(
            {
                "Level": level_name,
                "Source": source,
                "Macro_F1": usable["F1"].mean(),
                "Macro_Accuracy": usable["Accuracy"].mean(),
                "Macro_Missed_Rate": usable["Missed_Rate"].mean(),
                "Macro_Hallucinated_Rate": usable["Hallucinated_Rate"].mean(),
                "Macro_State_Accuracy": usable["State_Accuracy"].mean(),
                "Label_Count": int(len(usable)),
            }
        )
    return pd.DataFrame(rows)


def plot_stacked_counts(
    counts_by_source: dict[str, dict[str, dict[str, int]]],
    label_cols: list[str],
    output_path: Path,
    title: str,
    source_colors: dict[str, str],
) -> None:
    if not label_cols:
        return

    active_labels = []
    for label in label_cols:
        total = 0
        for source in SOURCE_ORDER:
            for status in STATUS_ORDER:
                total += counts_by_source[source][label][status]
        if total > 0:
            active_labels.append(label)

    if not active_labels:
        return

    n_labels = len(active_labels)
    fig_width = max(14, n_labels * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.24 if len(SOURCE_ORDER) == 3 else 0.3
    offsets = {"Truth": -bar_width, "LLM": 0.0, "Keyword": bar_width}

    for source in SOURCE_ORDER:
        for idx, label in enumerate(active_labels):
            bottom = 0
            for status in STATUS_ORDER:
                value = counts_by_source[source][label][status]
                if value <= 0:
                    continue
                ax.bar(
                    idx + offsets[source],
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=source_colors[source],
                    hatch=STATUS_HATCHES[status],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.95 if source == "Truth" else 0.85,
                )
                bottom += value

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(active_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Facility count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = []
    legend_labels = []
    for source in SOURCE_ORDER:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=source_colors[source], edgecolor="black", linewidth=0.4)
        )
        legend_labels.append(source)
    for status in STATUS_ORDER:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch=STATUS_HATCHES[status], edgecolor="black", linewidth=0.4)
        )
        legend_labels.append(f"{status} band")
    ax.legend(legend_handles, legend_labels, loc="upper right")

    plt.subplots_adjust(bottom=0.25, top=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_detail_counts(
    family_name: str,
    processes: list[str],
    counts_by_source: dict[str, dict[str, dict[str, int]]],
    output_path: Path,
    source_colors: dict[str, str],
) -> None:
    ordered_processes = [p for p in processes if p in counts_by_source["Truth"]]
    if not ordered_processes:
        return

    labels = ["Family Total"] + ordered_processes
    n_labels = len(labels)
    fig_width = max(12, n_labels * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.24
    offsets = {"Truth": -bar_width, "LLM": 0.0, "Keyword": bar_width}

    for source in SOURCE_ORDER:
        for idx, label in enumerate(labels):
            bottom = 0
            for status in STATUS_ORDER:
                if label == "Family Total":
                    value = sum(counts_by_source[source][proc][status] for proc in ordered_processes)
                else:
                    value = counts_by_source[source][label][status]

                if value <= 0:
                    continue

                ax.bar(
                    idx + offsets[source],
                    value,
                    width=bar_width,
                    bottom=bottom,
                    color=source_colors[source],
                    hatch=STATUS_HATCHES[status],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.95 if source == "Truth" else 0.85,
                )
                bottom += value

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Facility count", fontsize=12)
    ax.set_title(f"{family_name}: Family Total + Process Breakdown", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = []
    legend_labels = []
    for source in SOURCE_ORDER:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=source_colors[source], edgecolor="black", linewidth=0.4)
        )
        legend_labels.append(source)
    for status in STATUS_ORDER:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch=STATUS_HATCHES[status], edgecolor="black", linewidth=0.4)
        )
        legend_labels.append(f"{status} band")
    ax.legend(legend_handles, legend_labels, loc="upper right")

    plt.subplots_adjust(bottom=0.3, top=0.9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    truth_df = pd.read_csv(args.truth, dtype=str).fillna("")
    llm_df = pd.read_csv(args.llm, dtype=str).fillna("")
    keyword_df = pd.read_csv(args.keyword, dtype=str).fillna("")
    keywords = json.loads(args.keywords.read_text())
    label_to_family = build_label_to_family_map(keywords)
    json_leaf_order = extract_leaf_labels_in_json_order(keywords)
    json_family_order = list(keywords.keys())

    truth_permit_col, truth_name_col = infer_columns(truth_df)
    llm_permit_col, llm_name_col = infer_columns(llm_df)
    keyword_permit_col, keyword_name_col = infer_columns(keyword_df)

    label_cols = build_label_columns(truth_df, llm_df, keyword_df, json_leaf_order)

    truth_agg = aggregate_source(truth_df, "Truth", label_cols, truth_permit_col, truth_name_col)
    llm_agg = aggregate_source(llm_df, "LLM", label_cols, llm_permit_col, llm_name_col)
    keyword_agg = aggregate_source(keyword_df, "Keyword", label_cols, keyword_permit_col, keyword_name_col)

    common_keys = set(truth_agg["key"]) & set(llm_agg["key"]) & set(keyword_agg["key"])
    truth_agg = truth_agg[truth_agg["key"].isin(common_keys)].copy().reset_index(drop=True)
    llm_agg = llm_agg[llm_agg["key"].isin(common_keys)].copy().reset_index(drop=True)
    keyword_agg = keyword_agg[keyword_agg["key"].isin(common_keys)].copy().reset_index(drop=True)

    print(f"Comparison facilities in common: {len(common_keys)}")

    # Leaf-level comparison.
    leaf_counts = {
        "Truth": build_counts(truth_agg, label_cols),
        "LLM": build_counts(llm_agg, label_cols),
        "Keyword": build_counts(keyword_agg, label_cols),
    }
    leaf_metrics = pd.concat(
        [
            compute_metrics(truth_agg, llm_agg, label_cols, "LLM"),
            compute_metrics(truth_agg, keyword_agg, label_cols, "Keyword"),
        ],
        ignore_index=True,
    )
    leaf_summary = summarize_metrics(leaf_metrics, "Leaf")

    leaf_counts_rows = []
    for source in SOURCE_ORDER:
        for label in label_cols:
            row = {"Level": "Leaf", "Source": source, "Label": label}
            row.update(leaf_counts[source][label])
            leaf_counts_rows.append(row)
    leaf_counts_df = pd.DataFrame(leaf_counts_rows)

    # Family-level comparison.
    truth_family_df, family_cols = collapse_to_families(truth_agg, label_cols, label_to_family, json_family_order)
    llm_family_df, _ = collapse_to_families(llm_agg, label_cols, label_to_family, json_family_order)
    keyword_family_df, _ = collapse_to_families(keyword_agg, label_cols, label_to_family, json_family_order)

    family_counts = {
        "Truth": build_counts(truth_family_df, family_cols),
        "LLM": build_counts(llm_family_df, family_cols),
        "Keyword": build_counts(keyword_family_df, family_cols),
    }
    family_metrics = pd.concat(
        [
            compute_metrics(truth_family_df, llm_family_df, family_cols, "LLM"),
            compute_metrics(truth_family_df, keyword_family_df, family_cols, "Keyword"),
        ],
        ignore_index=True,
    )
    family_summary = summarize_metrics(family_metrics, "Family")

    family_counts_rows = []
    for source in SOURCE_ORDER:
        for label in family_cols:
            row = {"Level": "Family", "Source": source, "Label": label}
            row.update(family_counts[source][label])
            family_counts_rows.append(row)
    family_counts_df = pd.DataFrame(family_counts_rows)

    # Save tables.
    leaf_counts_df.to_csv(args.output_dir / "leaf_status_counts.csv", index=False)
    family_counts_df.to_csv(args.output_dir / "family_status_counts.csv", index=False)
    leaf_metrics.to_csv(args.output_dir / "leaf_metrics_by_label.csv", index=False)
    family_metrics.to_csv(args.output_dir / "family_metrics_by_label.csv", index=False)
    pd.concat([leaf_summary, family_summary], ignore_index=True).to_csv(args.output_dir / "metric_summary.csv", index=False)

    # Save plots.
    plot_stacked_counts(
        leaf_counts,
        label_cols,
        args.output_dir / "leaf_status_comparison.png",
        "Truth vs LLM vs Keyword by Process",
        SOURCE_COLORS,
    )
    plot_stacked_counts(
        family_counts,
        family_cols,
        args.output_dir / "family_status_comparison.png",
        "Truth vs LLM vs Keyword by Process Family",
        FAMILY_SOURCE_COLORS,
    )

    family_plot_dir = args.output_dir / "family_plots"
    for family_name, processes in FAMILY_PROCESS_MAP.items():
        safe_name = family_name.lower().replace(" ", "_")
        plot_family_detail_counts(
            family_name,
            processes,
            leaf_counts,
            family_plot_dir / f"{safe_name}_detail.png",
            FAMILY_SOURCE_COLORS,
        )

    print("Saved comparison tables and plots to:")
    print(args.output_dir)
    print()
    print(pd.concat([leaf_summary, family_summary], ignore_index=True).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()