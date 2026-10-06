"""Evaluate LLM process-detection performance against manual labels.

This script scores each (Method, Model) run, rebuilt from the raw extraction JSONs by
step6's build_model_comparison, against the Manual Read labels for the same facility,
plus the NPDES keyword baseline.

It reports four metrics for this sparse multi-label setup:

- Macro Unit Process F1: label-presence F1 computed separately for each facility,
  then averaged over facilities so every facility counts equally.
- Micro Unit Process F1: the same F1 pooled over every facility x label pair.
- Macro Category F1: a relaxed score that collapses detailed process labels to their
  top-level ontology family from unitprocess_keywords.json. This gives partial
  credit when the model predicts a close subtype rather than the exact leaf.
- State Accuracy: among manual-positive cells only, the fraction where the
  model predicts the correct state (PRESENT, FUTURE or OFFSITE).

A cell is positive if its status is PRESENT, PRESENT_AND_FUTURE, FUTURE or OFFSITE; PAST
counts as absent on both sides (helpers.utils.DETECTED_STATUSES).

"""

import json
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd

from step6_postprocess_llm_output import process_json_to_unit_process_dict, build_model_comparison, normalize_pdf_name, model_run_dirs
from helpers.utils import (
    get_leaf_names, precision_recall_f1, select_json_per_place_id, is_present, build_secondary_category_lookup,
    DETECTED_STATUSES, DATA_DIR, OUTPUT_DIR, TXT_DIR, FINAL_DIR, LLM_EXTRACTION_DIR, WATERRAG_RETRIEVAL_DIR,
    MANUAL_CSV, SITE_DATA_RELEVANT_CSV, unitprocess_keywords,
)


TABLE_1_CSV = FINAL_DIR / "table_1.csv"
MODEL_COSTS_CSV = DATA_DIR / "model_costs.csv"
MAIN_DIR = LLM_EXTRACTION_DIR / "ontology-based_gpt-5-mini"
ADDITIONAL_DIR = MAIN_DIR / "additional_runs"
TABLE_S5_CSV = FINAL_DIR / "table_s5.csv"
METRIC_COLS = ["Macro Unit Process F1", "Micro Unit Process F1", "Macro Category F1", "State Accuracy"]
META_COLS = {"Method", "Model", "PDF_File", "Place ID", "Agency", "Facility Name", "NPDES No."}

# table_s3: how the labeled sets compare to the full CA dataset on region, size and permit structure
REPRESENTATIVENESS_CSV = FINAL_DIR / "table_s3.csv"
SUPPLEMENTAL_PATH = DATA_DIR / "unit_processes_by_facility_supplemental_data.csv"
# figure_2 restricts the 17 supplemental facilities to those also present in the NPDES text and
# CWNS mappings; its per-facility output is the definitive list of the 15 that survive.
SUPPLEMENTAL_COMPARISON_PATH = OUTPUT_DIR / "supplemental_data_comparison_by_facility.csv"
REGION_COLS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "Unspecified"]
MULTI_PERMIT_MIN_DOCS = 2

# Maps run-dir model labels to the Language Model names used in model_costs.csv
MODEL_COST_MAP = {
    "gpt-5": "GPT 5",
    "gpt-5-mini": "GPT 5 mini",
    "gpt-5-mini-waterrag": "GPT 5 mini",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-4-5-sonnet": "Claude 4.5 Sonnet",
    "claude-3-haiku": "Claude 3 Haiku",
}

# Display order of table_1 rows; a run with no results gets a blank placeholder row
DESIRED_ORDER = [
    ("Keyword", "NPDES Keyword"),
    ("Ontology", "gpt-5-mini"),
    ("Ontology", "gpt-5-mini-waterrag"),
    ("Ontology", "gpt-5"),
    ("Ontology", "gemini-2.5-pro"),
    ("Ontology", "claude-3-haiku"),
    ("Ontology", "claude-4-5-sonnet"),
    ("Ontology", "claude-sonnet-4-6-web"),
    ("List", "gpt-5-mini"),
    ("List", "gpt-5"),
    ("List", "gemini-2.5-pro"),
    ("List", "claude-3-haiku"),
    ("List", "claude-4-5-sonnet"),
    ("List", "claude-sonnet-4-6-web"),
]


def normalize_status(value):
    """Map status values to canonical states, for state accuracy on manual-positive cells."""

    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.startswith("PRESENT"):
        return "PRESENT"
    if text.startswith("FUTURE"):
        return "FUTURE"
    if "OFFSITE" in text:
        return "OFFSITE"
    return text


def token_cost(usage_df, costs_df, cost_name):
    """Mean per-row cost of usage_df's prompt/completion tokens at cost_name's model_costs.csv rates."""
    cost_row = costs_df[costs_df["model_name"] == cost_name]
    input_per_m = cost_row["input_per_m"].iloc[0]
    output_per_m = cost_row["output_per_m"].iloc[0]
    prompt_vals = pd.to_numeric(usage_df["prompt_token"], errors="coerce")
    comp_vals = pd.to_numeric(usage_df["completion_token"], errors="coerce")
    return ((prompt_vals / 1_000_000 * input_per_m) + (comp_vals / 1_000_000 * output_per_m)).mean()


def load_run_usage(benchmark_ids, pdf_stem_by_place_id) -> pd.DataFrame:
    """Price per PDF and structured-output rate from each run dir's token_usage_summary.csv.

    For web runs (cost_usd column present): uses reported cost directly.
    For API Playground runs: computes cost from prompt/completion tokens using model_costs.csv.
    Restricted to benchmark_ids (the manual-read facilities) so a model like gpt-5-mini that
    accumulates every CA facility isn't averaged over its full set.

    The saved JSON is coerced to the {"items": [...]} shape before writing, so schema
    conformance can't be re-checked here. step5 records the raw-output conformance per
    extraction in the "structured_output" column; average that.
    Returns Method, Model, Price per PDF, Fraction Structured Output.
    """
    costs_df = pd.read_csv(MODEL_COSTS_CSV, skiprows=1)
    costs_df.columns = ["model_name", "input_per_m", "output_per_m"]
    costs_df["model_name"] = costs_df["model_name"].str.strip()

    # step5b's reranker calls are billed separately from the extraction call, so add them
    # in or the -waterrag rows understate their true cost.
    rerank_usage = pd.read_csv(WATERRAG_RETRIEVAL_DIR / "token_usage_summary.csv")
    rerank_usage = rerank_usage[rerank_usage["place_id"].astype(str).str.strip().isin(benchmark_ids)]
    rerank_model = rerank_usage["rerank_model"].dropna().unique()[0]
    rerank_cost = token_cost(rerank_usage, costs_df, MODEL_COST_MAP[rerank_model])

    rows = []
    for dir_path, method_label, model_label in model_run_dirs():
        usage_df = pd.read_csv(dir_path / "token_usage_summary.csv")
        usage_df = usage_df[usage_df["place_id"].astype(str).str.strip().isin(benchmark_ids)]
        if "extraction_file" in usage_df.columns:
            pinned = {f.name for f in select_json_per_place_id(dir_path, benchmark_ids, pdf_stem_by_place_id).values()}
            usage_df = usage_df[usage_df["extraction_file"].isna() | usage_df["extraction_file"].isin(pinned)]
        # If a precomputed cost column exists and has values, use it. Otherwise
        # compute cost from token counts using MODEL_COSTS_CSV.
        if usage_df["cost_usd"].notna().any():
            cost = usage_df["cost_usd"].astype(float).mean()
        else:
            cost = token_cost(usage_df, costs_df, MODEL_COST_MAP[model_label])
        if model_label.endswith("-waterrag"):
            cost += rerank_cost
        flags = usage_df["structured_output"].astype(str).str.strip().str.lower()
        flags = flags[flags.isin(["true", "false"])]
        rows.append({
            "Method": method_label,
            "Model": model_label,
            "Price per PDF": cost,
            "Fraction Structured Output": flags.eq("true").mean(),
        })
    return pd.DataFrame(rows)


def run_metrics(pred_df, manual, label_cols, label_to_family, unit_cols):
    """Macro-average the table_1 metrics over the manual facilities for one run.

    unit_cols is the leaf-level column set for Unit Process F1 and State Accuracy — the
    unspecified-excluded list keeps catch-all leaves out of those leaf-level metrics while
    Category F1 still scores over the full label_cols.
    """
    pred = pred_df.reindex(manual.index)  # missing facilities -> all-NaN (counts as no prediction)
    label_f1, family_f1, state_acc = [], [], []
    # pooled over every facility x label pair, so one-label facilities can't swing the score
    micro_tp = micro_fp = micro_fn = 0
    for place_id in manual.index:
        manual_row = manual.loc[place_id]
        pred_row = pred.loc[place_id]
        manual_pos = {c for c in label_cols if is_present(manual_row[c], DETECTED_STATUSES)}
        pred_pos = {c for c in label_cols if is_present(pred_row[c], DETECTED_STATUSES)}
        manual_units = manual_pos & set(unit_cols)
        pred_units = pred_pos & set(unit_cols)
        tp = len(manual_units & pred_units)
        fp = len(pred_units - manual_units)
        fn = len(manual_units - pred_units)
        micro_tp += tp; micro_fp += fp; micro_fn += fn
        label_f1.append(precision_recall_f1(tp, fp, fn)[2])
        # relaxed: collapse labels to their top-level family before comparing
        manual_families = {label_to_family.get(c, c) for c in manual_pos}
        pred_families = {label_to_family.get(c, c) for c in pred_pos}
        family_f1.append(precision_recall_f1(
            len(manual_families & pred_families),
            len(pred_families - manual_families),
            len(manual_families - pred_families),
        )[2])
        # state accuracy only on manual-positive cells, so predicting nothing can't look good
        correct = sum(normalize_status(pred_row[c]) == normalize_status(manual_row[c]) for c in manual_units)
        state_acc.append(correct / len(manual_units) if manual_units else float("nan"))
    return {
        "Macro Unit Process F1": pd.Series(label_f1).mean(),
        "Micro Unit Process F1": precision_recall_f1(micro_tp, micro_fp, micro_fn)[2],
        "Macro Category F1": pd.Series(family_f1).mean(),
        "State Accuracy": pd.Series(state_acc).mean(),
    }


def region_for_row(row):
    # Region is blank or "SB" for ~20% of rows, so prefer the R<n> prefix on the order number
    match = re.search(r"\bR([1-9])[-_ ]", f"{row['Order_No']} {row['PDF_File']}", re.I)
    if match:
        return f"R{match.group(1)}"
    return f"R{row['Region'][0]}" if row["Region"][:1].isdigit() else "Unspecified"


def build_representativeness_table(evaluation_ids) -> pd.DataFrame:
    """Region, size and permit-structure profile of each labeled set against the full pool.

    Benchmark Set is the 15 facilities figure_2 scores (the supplemental read, minus those with
    no CWNS match), Evaluation Set the manually labeled facilities, and Full Dataset every CA
    facility with an extracted permit text.
    """
    site_df = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")
    site_df["Region"] = site_df.apply(region_for_row, axis=1)
    site_df["Place ID"] = site_df["Place ID"].str.strip()
    site_df = site_df.drop_duplicates("Place ID").set_index("Place ID")

    txt_stems = {p.stem for p in TXT_DIR.glob("*.txt")}
    has_text = site_df["PDF_File"].str.replace(r"\.(pdf|txt)$", "", regex=True, case=False).isin(txt_stems)
    scored_names = set(pd.read_csv(SUPPLEMENTAL_COMPARISON_PATH, dtype=str)["Facility Name"])
    supplemental_df = pd.read_csv(SUPPLEMENTAL_PATH, dtype=str).fillna("")
    benchmark_df = supplemental_df[supplemental_df["Facility Name"].isin(scored_names)]

    populations = [
        ("% of Benchmark Set", set(benchmark_df["Place ID"].str.strip())),
        ("% of Evaluation Set", evaluation_ids),
        ("% of Full Dataset", set(site_df[has_text].index)),
    ]

    rows = []
    for label, place_ids in populations:
        subset = site_df.loc[[pid for pid in place_ids if pid in site_df.index]]
        region_share = subset["Region"].value_counts(normalize=True).mul(100)
        # Major/Minor is blank for most non-POTW rows, so that share covers classified rows only
        is_major = subset["Major/Minor"].replace("", pd.NA).dropna().str.startswith("Major")
        n_docs = pd.to_numeric(subset["Total_PDFs_Available"], errors="coerce").fillna(0)
        row = {"": label, "n": len(subset)}
        for region in REGION_COLS:
            row[region] = round(region_share.get(region, 0.0), 1)
        row["EPA Major"] = round(100 * is_major.mean())
        row["EPA Minor"] = round(100 * (~is_major).mean())
        row["Multiple Permits"] = round(100 * (n_docs >= MULTI_PERMIT_MIN_DOCS).mean(), 1)
        row["Shared Permit"] = round(100 * subset["Shared_PDF"].eq("Yes").mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    comparison = build_model_comparison()

    # Empty cells must stay NaN, not "". pd.notna("") is True, so fillna("") would
    # treat every blank manual cell as a positive label: false positives become
    # impossible and missed labels are massively overcounted. Match the LLM path.
    manual = pd.read_csv(MANUAL_CSV, dtype=str).set_index("Place ID")
    pdf_stem_by_place_id = manual["PDF_File"].to_dict()

    top_category_to_columns = build_secondary_category_lookup(unitprocess_keywords)[0]
    # relaxed Category F1 scores each detailed label as its top-level family
    label_to_family = {leaf: family for family, leaves in top_category_to_columns.items() for leaf in leaves}
    included = {leaf for cat, val in unitprocess_keywords.items() for leaf in get_leaf_names(cat, val)}
    unit_included = {leaf for cat, val in unitprocess_keywords.items() for leaf in get_leaf_names(cat, val, exclude_unspecified=True)}
    label_cols = [c for c in manual.columns if c not in META_COLS and c in included]
    unit_cols = [c for c in label_cols if c in unit_included]

    results = []
    predictions = comparison[comparison["Method"].ne("Manual Read")]
    for (method, model), subset in predictions.groupby(["Method", "Model"], sort=True):
        scores = run_metrics(subset.set_index("Place ID"), manual, label_cols, label_to_family, unit_cols)
        results.append({"Method": method, "Model": model, **scores})
    table = pd.DataFrame(results).sort_values(["Method", "Macro Unit Process F1", "Model"], ascending=[True, False, True])

    # Add keyword-search baseline metrics (NPDES keyword method) on same facilities
    # Load keyword predictions for the manually read document and align
    kw_df = pd.read_csv(OUTPUT_DIR / "unit_processes_by_pdf_kw.csv", dtype=str)
    pinned_stem = kw_df["Place ID"].map({pid: normalize_pdf_name(Path(pdf).stem) for pid, pdf in pdf_stem_by_place_id.items()})
    kw_df = kw_df[kw_df["PDF_File"].map(lambda pdf: normalize_pdf_name(Path(pdf).stem)) == pinned_stem]
    kw_df = kw_df.set_index("Place ID").reindex(manual.index)
    kw_scores = run_metrics(kw_df[label_cols], manual, label_cols, label_to_family, unit_cols)
    kw_row = {"Method": "Keyword", "Model": "NPDES Keyword", **kw_scores}
    table = pd.concat([table, pd.DataFrame([kw_row])], ignore_index=True, sort=False)

    # Spot-check the keyword and LLM ontology gpt-5-mini methods per process
    llm_df = pd.read_csv(OUTPUT_DIR / "unit_processes_by_facility_llm.csv", dtype=str)
    llm_df = llm_df.set_index("Place ID").reindex(manual.index)
    for method, model, df in [("Keyword", "NPDES Keyword", kw_df), ("Ontology", "gpt-5-mini", llm_df)]:
        audit_rows = []
        for col in unit_cols:  # leaf-level only; unspecified catch-alls aren't scored in Unit Process F1
            manual_pos = manual[col].map(lambda v: is_present(v, DETECTED_STATUSES))
            pred_pos = df[col].map(lambda v: is_present(v, DETECTED_STATUSES))
            fp = int((pred_pos & ~manual_pos).sum())
            fn = int((manual_pos & ~pred_pos).sum())
            tp = int((manual_pos & pred_pos).sum())
            if fp or fn:
                audit_rows.append({"Process": col, "TP": tp, "FP": fp, "FN": fn, "Errors": fp + fn})
        audit = pd.DataFrame(audit_rows).sort_values(["Errors", "FP", "Process"], ascending=[False, False, True])
        print(f"\n{method} spot-check: per-process disagreement with manual labels (n={len(manual)} facilities)")
        print(audit.to_string(index=False))

    usage_df = load_run_usage(set(manual.index), pdf_stem_by_place_id)
    table = table.merge(usage_df, on=["Method", "Model"], how="left")
    price = table["Price per PDF"].where(table["Price per PDF"] > 0)
    table["Unit Process F1 / Price per PDF"] = (table["Macro Unit Process F1"] / price).round(2)
    table = table.round(3)

    ordered_rows = []
    cols = list(table.columns)
    for method, model in DESIRED_ORDER:
        match = table[(table["Method"] == method) & (table["Model"] == model)]
        if not match.empty:
            ordered_rows.append(match.iloc[0].to_dict())
        else:
            ordered_rows.append({c: "" for c in cols} | {"Method": method, "Model": model})

    table = pd.DataFrame(ordered_rows)[cols]

    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    table.to_csv(TABLE_1_CSV, index=False)

    # ADDITIONAL RUNS VARIANCE
    run_dirs = [("main", MAIN_DIR)] + [(rd.name, rd) for rd in sorted(ADDITIONAL_DIR.glob("run_*"))]

    rows = []
    for label, run_dir in run_dirs:
        # Only the benchmark places are scored, and they are exactly the ones with a pinned
        # document, so the full CA set stays out of the selection.
        predictions = {}
        for place_id, json_file in select_json_per_place_id(run_dir, set(pdf_stem_by_place_id), pdf_stem_by_place_id).items():
            with open(json_file) as f:
                result = process_json_to_unit_process_dict(json.load(f))
            predictions[place_id] = {c: (result.get(c) or np.nan) for c in label_cols}
        predictions = pd.DataFrame.from_dict(predictions, orient="index").reindex(columns=label_cols)
        print(f"Scoring {label} ({os.path.relpath(run_dir)}) — {len(predictions)} facilities")
        scores = run_metrics(predictions, manual, label_cols, label_to_family, unit_cols)
        rows.append({"run": label, "n_facilities": len(predictions), **scores})

    df = pd.DataFrame(rows)
    desc = df[METRIC_COLS]
    summary = pd.DataFrame([
        {"run": "mean", **desc.mean().to_dict()},
        {"run": "std", **desc.std(ddof=1).to_dict()},
        {"run": "variance", **desc.var(ddof=1).to_dict()},
    ])
    out = pd.concat([df, summary], ignore_index=True)
    not_variance = out["run"] != "variance"
    out.loc[not_variance, METRIC_COLS] = out.loc[not_variance, METRIC_COLS].round(4)
    out.to_csv(TABLE_S5_CSV, index=False)

    # SET REPRESENTATIVENESS
    representativeness = build_representativeness_table(set(manual.index))
    representativeness.to_csv(REPRESENTATIVENESS_CSV, index=False)
    print(representativeness.to_string(index=False))


if __name__ == "__main__":
    main()
