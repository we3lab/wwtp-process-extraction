"""Evaluate LLM process-detection performance against truth labels.

This script compares each (Method, Model) row in the workbook against the
corresponding Truth row for the same PDF.

It reports three useful families of metrics for this sparse multi-label setup:

- PDF-macro F1: average label-presence F1 computed separately for each PDF,
  then averaged over PDFs so every PDF counts equally.
- Family F1: a relaxed score that collapses detailed process labels to their
  top-level ontology family from unitprocess_keywords.json. This gives partial
  credit when the model predicts a close subtype rather than the exact leaf.
- Exact-state accuracy: among truth-positive cells only, the fraction where the
  model predicts the correct state (PRESENT, FUTURE, PAST, OFFSITE).

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step6_postprocess_llm_output import process_json_to_unit_process_dict, build_model_comparison
from helpers.utils import get_leaf_names, precision_recall_f1, UNSCORED_STATUSES, select_json_per_place_id


DEFAULT_KEYWORDS = Path("wwtp_process_extraction/data/unitprocess_keywords.json")
DEFAULT_OUTPUT = Path("wwtp_process_extraction/output/final/table_1.csv")
MODEL_COMPARISON_DIR = Path("wwtp_process_extraction/output/llm_extraction")
MODEL_COSTS_CSV = Path("wwtp_process_extraction/data/model_costs.csv")
MAIN_DIR = Path("wwtp_process_extraction/output/llm_extraction/ontology-based_gpt-5-mini")
ADDITIONAL_DIR = MAIN_DIR / "additional_runs"
MANUAL_PATH = Path("wwtp_process_extraction/data/unit_processes_by_facility_manual.csv")
KEYWORDS_PATH = Path("wwtp_process_extraction/data/unitprocess_keywords.json")
OUTPUT_CSV = Path("wwtp_process_extraction/output/final/table_s2.csv")
METRIC_COLS = ["Macro Unit Process F1", "Macro Category F1", "State Accuracy"]

# Maps dir-name model labels to rows in model_costs.csv
MODEL_COST_MAP = {
    # mappings from run-dir model labels to the Language Model names used in model_costs.csv
    "gpt-5": "GPT 5",
    "gpt-5-mini": "GPT 5 mini",
    "gpt-pro": "GPT 5",
    "gpt-mini": "GPT 5 mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-pro": "Gemini 2.5 Pro",
    "gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "gemini-flash": "Gemini 2.0 Flash",
    "claude-4-5-sonnet": "Claude 4.5 Sonnet",
    "claude-sonnet-4-6-web": "Claude 4.6 Sonnet",
    "claude-sonnet": "Claude 4.5 Sonnet",
    "claude-3-haiku": "Claude 3 Haiku",
    "claude-haiku": "Claude 3 Haiku",
}

# malformed PDF names
BAD_TO_GOOD_PDF_NAMES = {
    "22_0017_Hea+CB8+C1:C8+C2:C8+C3:C8+C4:C+C3:C8": "22_0017_Healdsburg_WWTF_NPDES.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", type=Path, default=DEFAULT_KEYWORDS, help="Path to unitprocess_keywords.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Optional CSV output path")
    return parser.parse_args()


def normalize_status(value: Any) -> str | None:
    """Map workbook values to canonical states.

    Any non-empty cell counts as a positive prediction/label. The exact string is
    only used for state accuracy on truth-positive cells.
    """

    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.startswith("PRESENT"):
        return "PRESENT"
    if text.startswith("FUTURE"):
        return "FUTURE"
    if "OFFSITE" in text or text == "OFFSITE":
        return "OFFSITE"
    return text


def build_label_to_family_map(keywords: dict[str, Any]) -> dict[str, str]:
    """Map each detailed label to its top-level family.

    Example: Secondary Clarification -> Clarification.
    This is the relaxed metric used to score near-misses.
    """

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


def label_presence_f1(truth_row: pd.Series, pred_row: pd.Series, label_cols: list[str]) -> tuple[float, float, float, float]:
    """Compute label-presence precision/recall/F1/Jaccard for one PDF.
    """

    tp = fp = fn = 0
    for col in label_cols:
        truth_state = normalize_status(truth_row[col])
        if truth_state in UNSCORED_STATUSES:
            continue
        truth_positive = truth_state is not None
        pred_positive = pd.notna(pred_row[col])
        tp += int(truth_positive and pred_positive)
        fp += int((not truth_positive) and pred_positive)
        fn += int(truth_positive and (not pred_positive))

    return precision_recall_f1(tp, fp, fn)


def family_presence_f1(truth_row: pd.Series, pred_row: pd.Series, label_cols: list[str], label_to_family: dict[str, str]) -> tuple[float, float, float, float]:
    """Compute presence metrics after collapsing labels to top-level ontology families.

    Labels whose truth state is PAST/OFFSITE don't contribute to marking their
    family truth-positive (same drop-the-cell rule as label_presence_f1).
    """

    truth_families = {
        label_to_family.get(col, col) for col in label_cols
        if normalize_status(truth_row[col]) not in UNSCORED_STATUSES | {None}
    }
    pred_families = {label_to_family.get(col, col) for col in label_cols if pd.notna(pred_row[col])}

    tp = len(truth_families & pred_families)
    fp = len(pred_families - truth_families)
    fn = len(truth_families - pred_families)

    return precision_recall_f1(tp, fp, fn)


def exact_state_accuracy(truth_row: pd.Series, pred_row: pd.Series, label_cols: list[str]) -> float:
    """Accuracy of the predicted state on truth-positive cells only.

    This ignores all-truly-absent cells, which is important when most labels are
    absent. Otherwise a model can look good simply by predicting nothing.
    """

    correct = 0
    total = 0
    for col in label_cols:
        truth_state = normalize_status(truth_row[col])
        if truth_state is None or truth_state in UNSCORED_STATUSES:
            continue
        total += 1
        pred_state = normalize_status(pred_row[col])
        correct += int(pred_state == truth_state)
    return correct / total if total else float("nan")


def evaluate_workbook(comparison: pd.DataFrame, keywords_path: Path) -> pd.DataFrame:
    df = comparison.copy()
    df["PDF_File"] = df["PDF_File"].replace(BAD_TO_GOOD_PDF_NAMES)

    # Match manual readings to predictions by Place ID (a single PDF can cover
    # multiple facilities, so PDF_File is not a unique key).
    manual_ids = set(df.loc[df["Method"].eq("Manual Read"), "Place ID"].dropna())
    df = df[df["Place ID"].isin(manual_ids)].copy()

    meta_cols = {"Method", "Model", "PDF_File", "Place ID", "Agency", "Facility Name", "NPDES No."}
    label_cols = [col for col in df.columns if col not in meta_cols]
    _kw = json.loads(keywords_path.read_text())
    label_to_family = build_label_to_family_map(_kw)
    _included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val)}
    _unit_included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val, exclude_unspecified=True)}
    label_cols = [c for c in label_cols if c in _included]
    unit_cols = [c for c in label_cols if c in _unit_included]

    manual = df[df["Method"].eq("Manual Read")].drop_duplicates("Place ID").set_index("Place ID")
    results: list[dict[str, Any]] = []

    for (method, model), subset in df[df["Method"].ne("Manual Read")].groupby(["Method", "Model"], sort=True):
        subset = subset.drop_duplicates("Place ID").set_index("Place ID").reindex(manual.index)

        per_pdf_label_f1: list[float] = []
        per_pdf_family_f1: list[float] = []
        per_pdf_state_acc: list[float] = []

        for place_id in manual.index:
            manual_row = manual.loc[place_id, label_cols]
            pred_row = subset.loc[place_id, label_cols]

            _, _, label_f1, _ = label_presence_f1(manual_row, pred_row, unit_cols)
            _, _, family_f1, _ = family_presence_f1(manual_row, pred_row, label_cols, label_to_family)
            state_acc = exact_state_accuracy(manual_row, pred_row, unit_cols)

            per_pdf_label_f1.append(label_f1)
            per_pdf_family_f1.append(family_f1)
            per_pdf_state_acc.append(state_acc)

        results.append(
            {
                "Method": method,
                "Model": model,
                "Macro Unit Process F1": pd.Series(per_pdf_label_f1).mean(),
                "Macro Category F1": pd.Series(per_pdf_family_f1).mean(),
                "State Accuracy": pd.Series(per_pdf_state_acc).mean(),
            }
        )

    return pd.DataFrame(results).sort_values(["Method", "Macro Unit Process F1", "Model"], ascending=[True, False, True])


def load_price_per_pdf(benchmark_ids=None) -> pd.DataFrame:
    """Read token_usage_summary.csv from each model comparison dir.

    For web runs (cost_usd column present): uses reported cost directly.
    For API Playground runs: computes cost from prompt/completion tokens using model_costs.csv.
    Restricted to benchmark_ids (the manual-read facilities) so a model like gpt-5-mini that
    accumulates every CA facility isn't averaged over its full set. Returns Method, Model, Price per PDF.
    """
    costs_df = pd.read_csv(MODEL_COSTS_CSV, skiprows=1)
    costs_df.columns = ["model_name", "input_per_m", "output_per_m"]
    costs_df["model_name"] = costs_df["model_name"].str.strip()

    rows = []
    for dir_path in sorted(MODEL_COMPARISON_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        if dir_name.startswith("ontology-based_"):
            method_label, model_label = "Ontology", dir_name[len("ontology-based_"):]
        elif dir_name.startswith("list-based_"):
            method_label, model_label = "List", dir_name[len("list-based_"):]
        else:
            continue
        usage_path = dir_path / "token_usage_summary.csv"
        if not usage_path.exists():
            continue
        usage_df = pd.read_csv(usage_path)
        if benchmark_ids is not None:
            usage_df = usage_df[usage_df["place_id"].astype(str).str.strip().isin(benchmark_ids)]
        # If a precomputed cost column exists and has values, use it. Otherwise
        # compute cost from token counts using MODEL_COSTS_CSV.
        if "cost_usd" in usage_df.columns and usage_df["cost_usd"].notna().any():
            cost = usage_df["cost_usd"].astype(float).mean()
        else:
            cost_name = MODEL_COST_MAP.get(model_label)
            cost_row = costs_df[costs_df["model_name"] == cost_name]
            input_per_m = cost_row["input_per_m"].iloc[0]
            output_per_m = cost_row["output_per_m"].iloc[0]
            # column is "prompt_toke" (typo in source files)
            prompt_col = "prompt_toke" if "prompt_toke" in usage_df.columns else "prompt_token"
            # Ensure numeric conversion; missing values will become NaN
            prompt_vals = pd.to_numeric(usage_df.get(prompt_col, pd.Series(dtype=float)), errors="coerce")
            comp_vals = pd.to_numeric(usage_df.get("completion_token", pd.Series(dtype=float)), errors="coerce")
            per_row_cost = (prompt_vals / 1_000_000 * input_per_m) + (comp_vals / 1_000_000 * output_per_m)
            cost = per_row_cost.mean()
        rows.append({"Method": method_label, "Model": model_label, "Price per PDF": cost})
    return pd.DataFrame(rows)


def load_structured_output_rates(benchmark_ids=None) -> pd.DataFrame:
    """Fraction of raw model outputs that matched the schema before any coercion.

    The saved JSON is coerced to the {"items": [...]} shape before writing, so it
    can't be re-checked here. step5 records the raw-output conformance per
    extraction in token_usage_summary.csv's "structured_output" column; average that.
    """
    rows = []
    for dir_path in sorted(MODEL_COMPARISON_DIR.iterdir()):
        if not dir_path.is_dir():
            continue
        dir_name = dir_path.name
        if dir_name.startswith("ontology-based_"):
            method_label, model_label = "Ontology", dir_name[len("ontology-based_"):]
        elif dir_name.startswith("list-based_"):
            method_label, model_label = "List", dir_name[len("list-based_"):]
        else:
            continue
        usage_path = dir_path / "token_usage_summary.csv"
        if not usage_path.exists():
            continue
        usage = pd.read_csv(usage_path, dtype=str).fillna("")
        if benchmark_ids is not None:
            usage = usage[usage["place_id"].str.strip().isin(benchmark_ids)]
        if "structured_output" not in usage.columns:
            continue
        flags = usage["structured_output"].str.strip().str.lower()
        flags = flags[flags.isin(["true", "false"])]
        if flags.empty:
            continue
        rows.append({
            "Method": method_label,
            "Model": model_label,
            "Fraction Structured Output": flags.eq("true").mean(),
        })
    return pd.DataFrame(rows)


def predictions_for_run(run_dir, label_cols, pdf_stem_by_place_id=None):
    """Postprocess every JSON in run_dir into a {place_id -> {col: status}} frame over label_cols."""
    rows = {}
    for place_id, jf in select_json_per_place_id(run_dir, pdf_stem_by_place_id=pdf_stem_by_place_id).items():
        with open(jf) as f:
            data = json.load(f)
        result = process_json_to_unit_process_dict(data)
        rows[place_id] = {c: (result.get(c) or np.nan) for c in label_cols}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=label_cols)


def run_metrics(pred_df, manual, label_cols, label_to_family, unit_cols=None):
    """Macro-average the three table_1 metrics over the manual facilities for one run.

    unit_cols (defaults to label_cols) is the leaf-level column set for Unit Process F1 and
    State Accuracy — pass the unspecified-excluded list to keep catch-all leaves out of those
    leaf-level metrics while Category F1 still scores over the full label_cols.
    """
    if unit_cols is None:
        unit_cols = label_cols
    pred = pred_df.reindex(manual.index)  # missing facilities -> all-NaN (counts as no prediction)
    label_f1, family_f1, state_acc = [], [], []
    for place_id in manual.index:
        truth_row = manual.loc[place_id, label_cols]
        pred_row = pred.loc[place_id, label_cols]
        _, _, f1, _ = label_presence_f1(truth_row, pred_row, unit_cols)
        _, _, ff1, _ = family_presence_f1(truth_row, pred_row, label_cols, label_to_family)
        label_f1.append(f1)
        family_f1.append(ff1)
        state_acc.append(exact_state_accuracy(truth_row, pred_row, unit_cols))
    return {
        "Macro Unit Process F1": pd.Series(label_f1).mean(),
        "Macro Category F1": pd.Series(family_f1).mean(),
        "State Accuracy": pd.Series(state_acc).mean(),
    }

def main() -> None:
    args = parse_args()
    comparison = build_model_comparison()
    metrics = evaluate_workbook(comparison, args.keywords)

    # Add keyword-search baseline metrics (NPDES keyword method) computed on the
    # same manual facilities (same n=50 baseline).
    manual_ids = set(comparison.loc[comparison["Method"].eq("Manual Read"), "Place ID"].dropna().astype(str).str.strip())
    # Empty cells must stay NaN, not "". pd.notna("") is True, so fillna("") would
    # treat every blank truth cell as a positive label: false positives become
    # impossible and missed labels are massively overcounted. Match the LLM path.
    manual_full = pd.read_csv(MANUAL_PATH, dtype=str)
    manual_full["Place ID"] = manual_full["Place ID"].str.strip()
    meta = {"Method", "Model", "PDF_File", "Place ID", "Agency", "Facility Name", "NPDES No."}
    label_cols = [c for c in manual_full.columns if c not in meta]
    manual = (
        manual_full[manual_full["Place ID"].isin(manual_ids)]
        .drop_duplicates("Place ID")
        .set_index("Place ID")
    )

    # Load keyword predictions and align
    kw_path = Path("wwtp_process_extraction/output/unit_processes_by_facility_kw.csv")
    kw_df = pd.read_csv(kw_path, dtype=str).fillna("")
    kw_df["Place ID"] = kw_df["Place ID"].str.strip()
    kw_df = kw_df.set_index("Place ID").reindex(manual.index)
    # empty strings -> NaN so absence counts as no prediction
    kw_df = kw_df.replace("", np.nan)

    _kw = json.loads(args.keywords.read_text())
    label_to_family = build_label_to_family_map(_kw)
    _included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val)}
    _unit_included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val, exclude_unspecified=True)}
    label_cols = [c for c in label_cols if c in _included]
    unit_cols = [c for c in label_cols if c in _unit_included]
    kw_metrics = run_metrics(kw_df[label_cols], manual, label_cols, label_to_family, unit_cols)
    kw_row = {
        "Method": "Keyword",
        "Model": "NPDES Keyword",
        **kw_metrics,
    }
    metrics = pd.concat([metrics, pd.DataFrame([kw_row])], ignore_index=True, sort=False)

    # Spot-check the keyword and LLM ontology gpt-5 methods per process
    llm_df = pd.read_csv("wwtp_process_extraction/output/unit_processes_by_facility_llm.csv", dtype=str).fillna("")
    llm_df["Place ID"] = llm_df["Place ID"].str.strip()
    llm_df = llm_df.set_index("Place ID").reindex(manual.index)
    # empty strings -> NaN so absence counts as no prediction
    llm_df = llm_df.replace("", np.nan)
    for method, model, df in [("Keyword", "NPDES Keyword", kw_df), ("Ontology", "gpt-5", llm_df)]:
        audit_rows = []
        for col in unit_cols:  # leaf-level only; unspecified catch-alls aren't scored in Unit Process F1
            truth_pos = manual[col].notna()
            pred_pos = df[col].notna()
            fp = int((pred_pos & ~truth_pos).sum())
            fn = int((truth_pos & ~pred_pos).sum())
            tp = int((truth_pos & pred_pos).sum())
            if fp or fn:
                audit_rows.append({"Process": col, "TP": tp, "FP": fp, "FN": fn, "Errors": fp + fn})
        audit = pd.DataFrame(audit_rows).sort_values(["Errors", "FP", "Process"], ascending=[False, False, True])
        print(f"\n{method} spot-check: per-process disagreement with manual truth (n={len(manual)} facilities)")
        print(audit.to_string(index=False))

    benchmark_ids = {str(x).strip() for x in manual_ids}
    cost_df = load_price_per_pdf(benchmark_ids)
    structured_df = load_structured_output_rates(benchmark_ids)
    metrics = metrics.merge(cost_df, on=["Method", "Model"], how="left")
    metrics = metrics.merge(structured_df, on=["Method", "Model"], how="left")
    metrics["Unit Process F1 / Price per PDF"] = np.where(
        pd.to_numeric(metrics["Price per PDF"], errors="coerce") > 0,
        pd.to_numeric(metrics["Macro Unit Process F1"], errors="coerce") / pd.to_numeric(metrics["Price per PDF"], errors="coerce"),
        np.nan,
    ).round(2)
    metrics = metrics.round(3)

    # Reorder rows to the user's requested display order and add placeholders.
    desired_order = [
        ("Keyword", "NPDES Keyword"),
        ("Ontology", "gpt-5-mini"),
        ("Ontology", "gpt-5"),
        ("Ontology", "gemini-2.5-pro"),
        ("Ontology", "gemini-2.0-flash-001"),
        ("Ontology", "claude-3-haiku"),
        ("Ontology", "claude-4-5-sonnet"),
        ("Ontology", "claude-sonnet-4-6-web"),
        ("List", "gpt-5-mini"),
        ("List", "gpt-5"),
        ("List", "gemini-2.5-pro"),
        ("List", "gemini-2.0-flash-001"),
        ("List", "claude-3-haiku"),
        ("List", "claude-4-5-sonnet"),
        ("List", "claude-sonnet-4-6-web"),
    ]

    ordered_rows = []
    cols = list(metrics.columns)
    seen = set()
    for method, model in desired_order:
        key = (method, model)
        match = metrics[(metrics["Method"] == method) & (metrics["Model"] == model)]
        if key not in seen and not match.empty:
            # use real row only once
            ordered_rows.append(match.iloc[0].to_dict())
            seen.add(key)
        else:
            # create a blank placeholder row for duplicates or missing runs
            row = {c: "" for c in cols}
            row["Method"] = method
            row["Model"] = model
            ordered_rows.append(row)

    metrics = pd.DataFrame(ordered_rows)[cols]

    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)

    # ADDITIONAL RUNS VARIANCE
    manual_full = pd.read_csv(MANUAL_PATH, dtype=str)  # empty cells -> NaN, like table_1
    manual_full["Place ID"] = manual_full["Place ID"].str.strip()
    meta = {"Method", "Model", "PDF_File", "Place ID", "Agency", "Facility Name", "NPDES No."}
    label_cols = [c for c in manual_full.columns if c not in meta]
    manual = manual_full.drop_duplicates("Place ID").set_index("Place ID")
    pdf_stem_by_place_id = (
        manual_full[["Place ID", "PDF_File"]].dropna(subset=["PDF_File"]).drop_duplicates("Place ID")
        .set_index("Place ID")["PDF_File"].to_dict()
    )
    _kw = json.loads(KEYWORDS_PATH.read_text())
    label_to_family = build_label_to_family_map(_kw)
    _included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val)}
    _unit_included = {leaf for cat, val in _kw.items() for leaf in get_leaf_names(cat, val, exclude_unspecified=True)}
    label_cols = [c for c in label_cols if c in _included]
    unit_cols = [c for c in label_cols if c in _unit_included]

    run_dirs = [("main", MAIN_DIR)] + [
        (rd.name, rd) for rd in sorted(ADDITIONAL_DIR.glob("run_*")) if rd.is_dir()
    ]

    rows = []
    for label, run_dir in run_dirs:
        n = len(list(run_dir.glob("*.json")))
        print(f"Scoring {label} ({run_dir}) — {n} json")
        metrics = run_metrics(predictions_for_run(run_dir, label_cols, pdf_stem_by_place_id), manual, label_cols, label_to_family, unit_cols)
        rows.append({"run": label, "n_facilities": n, **metrics})

    df = pd.DataFrame(rows)
    desc = df[METRIC_COLS]
    summary = pd.DataFrame([
        {"run": "mean", **desc.mean().to_dict()},
        {"run": "std", **desc.std(ddof=1).to_dict()},
        {"run": "variance", **desc.var(ddof=1).to_dict()},
    ])
    out = pd.concat([df, summary], ignore_index=True)
    out[METRIC_COLS] = out[METRIC_COLS].round(4)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    # print(out.to_string(index=False))

if __name__ == "__main__":
    main()