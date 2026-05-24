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
  model predicts the correct state (PRESENT, PLANNED, OFFSITE).

The workbook includes one malformed Healdsburg PDF name in one row; that is
normalized here so the full 5-PDF evaluation set is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_WORKBOOK = Path("wwtp_process_extraction/output/llm_model_comparison/model_comparison_all.csv")
DEFAULT_KEYWORDS = Path("wwtp_process_extraction/data/unitprocess_keywords.json")
DEFAULT_OUTPUT = Path("wwtp_process_extraction/output/llm_model_comparison/model_performance_metrics.csv")
MODEL_COMPARISON_DIR = Path("wwtp_process_extraction/output/llm_model_comparison")
MODEL_COSTS_CSV = Path("wwtp_process_extraction/data/model_costs.csv")

# Maps dir-name model labels to rows in model_costs.csv
MODEL_COST_MAP = {
    "gpt-pro": "GPT 5",
    "gpt-mini": "GPT 5 mini",
    "gemini-pro": "Gemini 2.5 Pro",
    "gemini-flash": "Gemini 2.0 Flash",
    "claude-sonnet": "Claude 4.5 Sonnet",
    "claude-haiku": "Claude 3 Haiku",
}

BAD_TO_GOOD_PDF_NAMES = {
    "22_0017_Hea+CB8+C1:C8+C2:C8+C3:C8+C4:C+C3:C8": "22_0017_Healdsburg_WWTF_NPDES.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK, help="Path to the model comparison CSV")
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
    if text.startswith("PLANNED"):
        return "PLANNED"
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
    """Compute label-presence precision/recall/F1/Jaccard for one PDF."""

    tp = fp = fn = 0
    for col in label_cols:
        truth_positive = pd.notna(truth_row[col])
        pred_positive = pd.notna(pred_row[col])
        tp += int(truth_positive and pred_positive)
        fp += int((not truth_positive) and pred_positive)
        fn += int(truth_positive and (not pred_positive))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return precision, recall, f1, jaccard


def family_presence_f1(truth_row: pd.Series, pred_row: pd.Series, label_cols: list[str], label_to_family: dict[str, str]) -> tuple[float, float, float, float]:
    """Compute presence metrics after collapsing labels to top-level ontology families."""

    truth_families = {label_to_family.get(col, col) for col in label_cols if pd.notna(truth_row[col])}
    pred_families = {label_to_family.get(col, col) for col in label_cols if pd.notna(pred_row[col])}

    tp = len(truth_families & pred_families)
    fp = len(pred_families - truth_families)
    fn = len(truth_families - pred_families)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    return precision, recall, f1, jaccard


def exact_state_accuracy(truth_row: pd.Series, pred_row: pd.Series, label_cols: list[str]) -> float:
    """Accuracy of the predicted state on truth-positive cells only.

    This ignores all-truly-absent cells, which is important when most labels are
    absent. Otherwise a model can look good simply by predicting nothing.
    """

    correct = 0
    total = 0
    for col in label_cols:
        truth_state = normalize_status(truth_row[col])
        if truth_state is None:
            continue
        total += 1
        pred_state = normalize_status(pred_row[col])
        correct += int(pred_state == truth_state)
    return correct / total if total else float("nan")


def evaluate_workbook(workbook: Path, keywords_path: Path) -> pd.DataFrame:
    df = pd.read_csv(workbook, dtype=str)
    df["PDF_File"] = df["PDF_File"].replace(BAD_TO_GOOD_PDF_NAMES)

    truth_pdfs = set(df.loc[df["Method"].eq("Truth"), "PDF_File"].dropna())
    df = df[df["PDF_File"].isin(truth_pdfs)].copy()

    label_cols = [col for col in df.columns if col not in {"Method", "Model", "PDF_File"}]
    label_to_family = build_label_to_family_map(json.loads(keywords_path.read_text()))

    truth = df[df["Method"].eq("Truth")].set_index("PDF_File")
    results: list[dict[str, Any]] = []

    for (method, model), subset in df[df["Method"].ne("Truth")].groupby(["Method", "Model"], sort=True):
        subset = subset.set_index("PDF_File").reindex(truth.index)

        per_pdf_label_f1: list[float] = []
        per_pdf_family_f1: list[float] = []
        per_pdf_state_acc: list[float] = []

        for pdf in truth.index:
            truth_row = truth.loc[pdf, label_cols]
            pred_row = subset.loc[pdf, label_cols]

            _, _, label_f1, _ = label_presence_f1(truth_row, pred_row, label_cols)
            _, _, family_f1, _ = family_presence_f1(truth_row, pred_row, label_cols, label_to_family)
            state_acc = exact_state_accuracy(truth_row, pred_row, label_cols)

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
DEFAULT_OUTPUT

def load_price_per_pdf() -> pd.DataFrame:
    """Read token_usage_summary.csv from each model comparison dir.

    For web runs (cost_usd column present): uses reported cost directly.
    For API Playground runs: computes cost from prompt/completion tokens using model_costs.csv.
    Returns a DataFrame with columns Method, Model, Price per PDF.
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
        if "cost_usd" in usage_df.columns:
            cost = usage_df["cost_usd"].mean()
        else:
            cost_name = MODEL_COST_MAP.get(model_label)
            cost_row = costs_df[costs_df["model_name"] == cost_name]
            if cost_row.empty or cost_name is None:
                cost = float("nan")
            else:
                input_per_m = cost_row["input_per_m"].iloc[0]
                output_per_m = cost_row["output_per_m"].iloc[0]
                # column is "prompt_toke" (typo in source files)
                prompt_col = "prompt_toke" if "prompt_toke" in usage_df.columns else "prompt_token"
                per_row_cost = (
                    usage_df[prompt_col] / 1_000_000 * input_per_m
                    + usage_df["completion_token"] / 1_000_000 * output_per_m
                )
                cost = per_row_cost.mean()
        rows.append({"Method": method_label, "Model": model_label, "Price per PDF": cost})
    return pd.DataFrame(rows)


def load_structured_output_rates() -> pd.DataFrame:
    """Check what fraction of JSON outputs match the expected {"items": [...]} schema."""
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
        json_files = list(dir_path.glob("*.json"))
        if not json_files:
            continue
        matched = sum(
            1 for f in json_files
            if isinstance((d := json.loads(f.read_text())).get("items"), list)
        )
        rows.append({
            "Method": method_label,
            "Model": model_label,
            "Fraction Structured Output": matched / len(json_files),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    metrics = evaluate_workbook(args.workbook, args.keywords)

    cost_df = load_price_per_pdf()
    structured_df = load_structured_output_rates()
    metrics = metrics.merge(cost_df, on=["Method", "Model"], how="left")
    metrics = metrics.merge(structured_df, on=["Method", "Model"], how="left")
    metrics = metrics.round(3)

    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Metric guide:")
    print("- Macro Unit Process F1: label-presence F1 averaged over the 5 PDFs so each PDF has equal weight.")
    print("- Macro Category F1: same idea, but with labels collapsed to top-level ontology families for partial credit.")
    print("- State Accuracy: on cells that are true labels, how often the model chose the exact state (PRESENT / PLANNED / OFFSITE).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()