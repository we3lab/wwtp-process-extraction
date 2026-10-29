"""Score each (Method, Model) run and the NPDES keyword baseline against the manual labels (table_1).

Runs are rebuilt from the raw extraction JSONs by step6's build_model_comparison. Macro / Micro Unit
Process F1 average per facility / pool every facility x label pair; Macro Category F1 collapses labels
to their top-level category, so a close subtype gets partial credit; State Accuracy is the share of
manual-positive cells with the right state (PRESENT, FUTURE or OFFSITE). PAST counts as absent
(DETECTED_STATUSES). Also writes the run variance (table_s5) and set representativeness (table_s3).
"""

import json
from pathlib import Path
import os
import re
import pandas as pd

from step6_postprocess_llm_output import (
    process_json_to_unit_process_dict, build_model_comparison, normalize_pdf_name, model_run_dirs,
)
from helpers.paths import *
from helpers.utils import (
    get_leaf_names, parse_status, precision_recall_f1, select_json_per_place_id, is_present,
    column_top_category, category_leaf_names, unitprocess_keywords, DETECTED_STATUSES,
)


TABLE_1_CSV = FIGURES_DIR / "table_1.csv"
TABLE_S5_CSV = FIGURES_DIR / "table_s5.csv"
# table_s3: how the labeled sets compare to the full CA dataset on region, size and permit structure
REPRESENTATIVENESS_CSV = FIGURES_DIR / "table_s3.csv"
MODEL_COSTS_CSV = DATA_DIR / "model_costs.csv"
ADDITIONAL_DIR = MAIN_RUN_DIR / "additional_runs"
METRIC_COLS = ["Macro Unit Process F1", "Micro Unit Process F1", "Macro Category F1", "State Accuracy"]
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

# Display order of table_1 rows; every run must exist
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
    # State for state accuracy: PRESENT_AND_FUTURE counts as PRESENT
    status = parse_status(value)
    return "PRESENT" if status == "PRESENT_AND_FUTURE" else status


def token_cost(usage_df, costs_df, cost_name):
    """Mean per-row cost of usage_df's prompt/completion tokens at cost_name's model_costs.csv rates."""
    cost_row = costs_df[costs_df["model_name"] == cost_name]
    input_per_m = cost_row["input_per_m"].iloc[0]
    output_per_m = cost_row["output_per_m"].iloc[0]
    prompt_cost = usage_df["prompt_token"] / 1_000_000 * input_per_m
    completion_cost = usage_df["completion_token"] / 1_000_000 * output_per_m
    return (prompt_cost + completion_cost).mean()


def load_run_usage(benchmark_ids, pdf_stem_by_place_id) -> pd.DataFrame:
    """Price per PDF and structured-output rate per run, from each run dir's token_usage_summary.csv.

    Web runs report cost_usd; API runs are priced from token counts at model_costs.csv rates. Only the
    benchmark facilities count, so a run covering every CA facility isn't averaged over its full set.
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
            pinned = {f.name for f in select_json_per_place_id(dir_path, pdf_stem_by_place_id).values()}
            usage_df = usage_df[usage_df["extraction_file"].isna() | usage_df["extraction_file"].isin(pinned)]
        if usage_df["cost_usd"].notna().any():
            cost = usage_df["cost_usd"].mean()
        else:
            cost = token_cost(usage_df, costs_df, MODEL_COST_MAP[model_label])
        if model_label.endswith("-waterrag"):
            cost += rerank_cost
        # saved JSONs are already coerced to the schema, so use step5's record of the raw output
        flags = usage_df["structured_output"].astype(str).str.strip().str.lower()
        flags = flags[flags.isin(["true", "false"])]
        rows.append({
            "Method": method_label,
            "Model": model_label,
            "Price per PDF": cost,
            "Fraction Structured Output": flags.eq("true").mean(),
        })
    return pd.DataFrame(rows)


def run_metrics(pred_df, manual, label_cols, unit_cols):
    """Macro-average the table_1 metrics over the manual facilities for one run.

    Unit Process F1 and State Accuracy score unit_cols (no "Unspecified" catch-alls);
    Category F1 uses all label_cols.
    """
    pred = pred_df.reindex(manual.index)  # missing facilities -> all-NaN (counts as no prediction)
    unit_set = set(unit_cols)
    label_f1, family_f1, state_acc = [], [], []
    # pooled over every facility x label pair, so one-label facilities can't swing the score
    micro_tp = micro_fp = micro_fn = 0
    for place_id in manual.index:
        manual_row = manual.loc[place_id]
        pred_row = pred.loc[place_id]
        manual_pos = {c for c in label_cols if is_present(manual_row[c], DETECTED_STATUSES)}
        pred_pos = {c for c in label_cols if is_present(pred_row[c], DETECTED_STATUSES)}
        manual_units = manual_pos & unit_set
        pred_units = pred_pos & unit_set
        tp = len(manual_units & pred_units)
        fp = len(pred_units - manual_units)
        fn = len(manual_units - pred_units)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        label_f1.append(precision_recall_f1(tp, fp, fn)[2])
        # relaxed: collapse labels to their top-level category before comparing
        manual_families = {column_top_category[c] for c in manual_pos}
        pred_families = {column_top_category[c] for c in pred_pos}
        family_f1.append(precision_recall_f1(
            len(manual_families & pred_families),
            len(pred_families - manual_families),
            len(manual_families - pred_families),
        )[2])
        # state accuracy only on manual-positive cells, so predicting nothing can't look good
        correct = sum(normalize_status(pred_row[c]) == normalize_status(manual_row[c]) for c in manual_units)
        state_acc.append(correct / len(manual_units))
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
    pdf_stems = site_df["PDF_File"].map(lambda f: f[:-4] if f.lower().endswith((".pdf", ".txt")) else f)
    has_text = pdf_stems.isin(txt_stems)
    # figure_2's per-facility output is the definitive list of the supplemental facilities it scores
    scored_names = set(pd.read_csv(SUPPLEMENTAL_COMPARISON_CSV, dtype=str)["Facility Name"])
    supplemental_df = pd.read_csv(SUPPLEMENTAL_CSV, dtype=str).fillna("")
    benchmark_df = supplemental_df[supplemental_df["Facility Name"].isin(scored_names)]

    populations = [
        ("% of Benchmark Set", set(benchmark_df["Place ID"].str.strip())),
        ("% of Evaluation Set", evaluation_ids),
        ("% of Full Dataset", set(site_df[has_text].index)),
    ]

    rows = []
    for label, place_ids in populations:
        subset = site_df.loc[list(place_ids)]
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
    manual = pd.read_csv(MANUAL_CSV, dtype=str).set_index("Place ID")
    pdf_stem_by_place_id = manual["PDF_File"].to_dict()

    included = {leaf for leaves in category_leaf_names.values() for leaf in leaves}
    unit_included = {
        leaf for cat, val in unitprocess_keywords.items()
        for leaf in get_leaf_names(cat, val, exclude_unspecified=True)
    }
    label_cols = [c for c in manual.columns if c in included]
    unit_cols = [c for c in label_cols if c in unit_included]

    results = []
    predictions = comparison[comparison["Method"].ne("Manual Read")]
    for (method, model), subset in predictions.groupby(["Method", "Model"]):
        scores = run_metrics(subset.set_index("Place ID"), manual, label_cols, unit_cols)
        results.append({"Method": method, "Model": model, **scores})

    # NPDES keyword baseline, scored on the document each facility's manual labels were read from
    kw_df = pd.read_csv(PDF_KW_CSV, dtype=str)
    pinned_stems = {pid: normalize_pdf_name(Path(pdf).stem) for pid, pdf in pdf_stem_by_place_id.items()}
    kw_stems = kw_df["PDF_File"].map(lambda pdf: normalize_pdf_name(Path(pdf).stem))
    kw_df = kw_df[kw_stems == kw_df["Place ID"].map(pinned_stems)]
    kw_df = kw_df.set_index("Place ID").reindex(manual.index)
    kw_scores = run_metrics(kw_df[label_cols], manual, label_cols, unit_cols)
    results.append({"Method": "Keyword", "Model": "NPDES Keyword", **kw_scores})
    table = pd.DataFrame(results)

    # Spot-check the keyword and LLM ontology gpt-5-mini methods per process
    llm_df = pd.read_csv(FACILITY_LLM_CSV, dtype=str).set_index("Place ID").reindex(manual.index)
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
    table = table.set_index(["Method", "Model"]).loc[DESIRED_ORDER].reset_index()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # fixed 3 decimals
    metric_cols = [c for c in table.columns if c not in ("Method", "Model", "Unit Process F1 / Price per PDF")]
    table[metric_cols] = table[metric_cols].map(lambda x: f"{x:.3f}" if isinstance(x, float) and x == x else x)
    table.to_csv(TABLE_1_CSV, index=False)

    # ADDITIONAL RUNS VARIANCE
    run_dirs = [("main", MAIN_RUN_DIR)] + [(rd.name, rd) for rd in sorted(ADDITIONAL_DIR.glob("run_*"))]

    rows = []
    for label, run_dir in run_dirs:
        # only the benchmark places have a pinned document, so the full CA set stays out
        predictions = {}
        for place_id, json_file in select_json_per_place_id(run_dir, pdf_stem_by_place_id).items():
            with open(json_file) as f:
                predictions[place_id] = process_json_to_unit_process_dict(json.load(f))
        predictions = pd.DataFrame.from_dict(predictions, orient="index")[label_cols]
        print(f"Scoring {label} ({os.path.relpath(run_dir)}) — {len(predictions)} facilities")
        scores = run_metrics(predictions, manual, label_cols, unit_cols)
        rows.append({"run": label, "n_facilities": len(predictions), **scores})

    df = pd.DataFrame(rows)
    desc = df[METRIC_COLS]
    summary = pd.DataFrame([
        {"run": "mean", **desc.mean().to_dict()},
        {"run": "std", **desc.std().to_dict()},
        {"run": "variance", **desc.var().to_dict()},
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
