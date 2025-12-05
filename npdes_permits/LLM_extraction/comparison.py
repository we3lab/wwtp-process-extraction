import os
import json
import pandas as pd
import re
import statistics
from rapidfuzz import fuzz, process

# ============================================================
# 1. Load reference JSON and build lookup tables
# ============================================================

def load_reference(ref_path):
    with open(ref_path, "r") as f:
        ref = json.load(f)

    alt_to_generic = {}
    all_generic = set()

    for p in ref["treatment_processes"]:
        generic = p["generic_name"]
        all_generic.add(generic)

        # add generic name
        alt_to_generic[generic.lower()] = generic
        
        # add alternative names
        for alt in p["alternative_names"]:
            alt_to_generic[alt.lower()] = generic

    return alt_to_generic, sorted(list(all_generic))


# ============================================================
# 2. Parse one JSON result into a list of generic processes
# ============================================================

def extract_score(text):
    """Extract numeric score from substring like 'score: 0.85'."""
    match = re.search(r"score:\s*([0-9]*\.?[0-9]+)", text)
    return float(match.group(1)) if match else None


def extract_best_name(text):
    """Extract the best match name from substring '(best: XXX, score:'."""
    match = re.search(r"best:\s*([^,]+)", text)
    return match.group(1).strip() if match else None


def map_to_generic(name, alt_to_generic):
    """Map alternative or generic name → generic name."""
    if name is None:
        return None

    key = name.lower().strip()
    if key in alt_to_generic:
        return alt_to_generic[key]

    # fuzzy match fallback
    best, score, _ = process.extractOne(
        key,
        alt_to_generic.keys(),
        scorer=fuzz.token_sort_ratio
    )
    if score >= 85:  # fuzzy cutoff
        return alt_to_generic[best]
    return None


def json_to_list(llm_json, reference_json, confidence_threshold=0.7):
    results = []

    # --- Build lookup table from reference_json ---
    alt_to_generic = {}
    for proc in reference_json["treatment_processes"]:
        gen = proc["generic_name"].lower()
        alt_to_generic[gen] = gen
        for alt in proc["alternative_names"]:
            alt_to_generic[alt.lower()] = gen

    # --- helper to add a process safely ---
    def add_process(name):
        name = name.lower().strip()
        if name in alt_to_generic:
            results.append(alt_to_generic[name])

    # --- Handle VARIFIED processes ---
    for p in llm_json.get("processes", []):
        if p.get("confidence", 0) >= confidence_threshold:
            add_process(p["process_name"])

    # --- Handle UNKNOWN processes using best-match + score ---
    for unk in llm_json.get("unknown_processes", []):
        reason = unk.get("reason_not_matched", "")
        
        # Extract the best process and score:
        match = re.search(r"best:\s*(.*?),\s*score:\s*([\d.]+)", reason)
        if not match:
            continue
        
        best_name = match.group(1).lower().strip()
        score = float(match.group(2))

        if score >= confidence_threshold:
            add_process(best_name)

    # remove duplicates
    return sorted(set(results))


# ============================================================
# 3. Apply json_to_list() to all JSON files and build DF
# ============================================================

def generate_predictions_df(results_dir, reference_json, confidence_threshold=0.7):
    rows = []

    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(results_dir, file)

        with open(json_path, "r") as f:
            llm_json = json.load(f)

        base = file.replace(".json", "")  # match to XXX.pdf
        processes = json_to_list(llm_json, reference_json, confidence_threshold)

        row = {"Prediction_File": base}
        for p in processes:
            row[p] = "YES"

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.fillna("")
    return df


# ============================================================
# 4. Compare predictions to ground truth
# ============================================================

def compare_results(ground_truth_csv, predictions_df):
    """
    Compare ground-truth CSV to predictions DataFrame (or CSV path).
    Returns a dict keyed by prediction file base name with sets and scores.
    """
    # load ground truth
    gt = pd.read_csv(ground_truth_csv, sep=",") if isinstance(ground_truth_csv, str) else ground_truth_csv.copy()

    # ensure predictions_df is a DataFrame (allow passing a path)
    preds = pd.read_csv(predictions_df, sep=",") if isinstance(predictions_df, str) else predictions_df.copy()

    # normalize Prediction_File in both frames to the base name (without .pdf) and lowercase
    gt["Prediction_File"] = (
        gt["PDF_File"].astype(str)
        .str.strip()
        .str.replace(r"\.pdf$", "", case=False, regex=True)
        .str.lower()
    )
    preds["Prediction_File"] = (
        preds["Prediction_File"].astype(str).str.strip().str.replace(r"\.pdf$", "", case=False, regex=True).str.lower()
    )

    # identify process columns in ground truth (exclude metadata)
    meta_cols = {"Agency", "Facility_Name", "NPDES_No", "PDF_File", "Prediction_File"}
    processes = [c for c in gt.columns if c not in meta_cols]

    # map canonical process names to their lower form
    proc_map = {p.lower().strip(): p for p in processes}

    # build lookup of prediction columns (lower -> original)
    pred_lookup = {c.lower().strip(): c for c in preds.columns}

    # ensure preds has a column for every canonical process name (copy if case-varied, else create empty)
    for proc_lower, proc_canonical in proc_map.items():
        if proc_lower in pred_lookup:
            src = pred_lookup[proc_lower]
            # copy column under canonical name if needed
            if proc_canonical not in preds.columns:
                preds[proc_canonical] = preds[src]
        else:
            # create empty column if missing
            if proc_canonical not in preds.columns:
                preds[proc_canonical] = ""

    # merge on Prediction_File
    merged = gt.merge(preds, on="Prediction_File", how="left", suffixes=("_gt", "_pred"))

    def fetch_val(row, col_name, source=None):
        """Return the value for col_name from merged row.

        source: None (any), 'gt' or 'pred' to prefer ground-truth or prediction
        columns when resolving pandas merge suffixes.
        """
        if source == "gt":
            keys = [col_name + "_gt", col_name + "_x", col_name, col_name + "_y", col_name + "_pred"]
        elif source == "pred":
            keys = [col_name + "_pred", col_name + "_y", col_name, col_name + "_x", col_name + "_gt"]
        else:
            keys = [col_name, col_name + "_x", col_name + "_y", col_name + "_gt", col_name + "_pred"]

        for key in keys:
            if key in row.index:
                return row[key]
        return None

    def is_yes(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s in {"yes", "y", "1", "true"}  # treat common truthy markers as YES

    comparison = {}
    for _, row in merged.iterrows():
        name = row["Prediction_File"]
        # fetch ground-truth values preferring GT-suffixed columns
        gt_set = {p for p in processes if is_yes(fetch_val(row, p, source="gt"))}
        # fetch prediction values preferring PRED-suffixed columns
        pred_set = {p for p in processes if is_yes(fetch_val(row, p, source="pred"))}
        intersection = gt_set & pred_set
        match_score = round(len(intersection) / max(1, len(gt_set)), 3)
        comparison[name] = {
            "ground_truth": sorted(gt_set),
            "predicted": sorted(pred_set),
            "missed": sorted(gt_set - pred_set),
            "hallucinated": sorted(pred_set - gt_set),
            "match_score": match_score,
        }

    return comparison


# ============================================================
# 5. Evaluate across multiple thresholds
# ============================================================

def evaluate_over_thresholds(results_dir, reference_json, ground_truth_csv, thresholds):
    evaluations = {}

    for th in thresholds:
        preds = generate_predictions_df(results_dir, reference_json, th)
        comp = compare_results(ground_truth_csv, preds)
        evaluations[th] = comp

    return evaluations



REFERNCE_JSON_FILE = "data/treatment_processes.json"
LLM_RESULTS_DIR = "output/llm_results"
GROUND_TRUTH_FILE = "data/test_truth.csv"

with open(REFERNCE_JSON_FILE, "r") as f:
    reference_json = json.load(f)


pred_df = generate_predictions_df(
    results_dir=LLM_RESULTS_DIR,
    reference_json=reference_json,
    confidence_threshold=0.80
)

pred_df.to_csv("output/predictions_080.csv", index=False)


comparisons = compare_results(GROUND_TRUTH_FILE, pred_df)

print(json.dumps(comparisons, indent=2))

# --- compute and print simple aggregate scores so they're visible in stdout
scores = [v.get("match_score", 0.0) for v in comparisons.values()]
scores_nonzero = [v.get("match_score", 0.0) for v in comparisons.values() if len(v.get("ground_truth", [])) > 0]
avg_all = round(statistics.mean(scores), 3) if scores else 0.0
avg_nonzero = round(statistics.mean(scores_nonzero), 3) if scores_nonzero else 0.0
median = round(statistics.median(scores), 3) if scores else 0.0

print(f"\nAggregate summary:")
print(f"  files_total: {len(scores)}")
print(f"  avg_all: {avg_all}")
print(f"  avg_nonzero_gt: {avg_nonzero}")
print(f"  median: {median}\n")

# --- attempt to create small plots and save to output/ if matplotlib is available
try:
    import matplotlib.pyplot as plt
    os.makedirs("output", exist_ok=True)

    # histogram of match scores
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=20, color="#4C72B0", edgecolor="black")
    plt.title("Match score distribution")
    plt.xlabel("match_score")
    plt.ylabel("count")
    plt.axvline(avg_all, color="red", linestyle="--", label=f"avg_all {avg_all:.3f}")
    plt.axvline(avg_nonzero, color="green", linestyle="--", label=f"avg_nonzero {avg_nonzero:.3f}")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join("output", "match_score_hist.png")
    plt.savefig(hist_path)
    plt.close()

    # simple bar for averages
    plt.figure(figsize=(4, 3))
    vals = [avg_all, avg_nonzero]
    labels = ["avg_all", "avg_nonzero"]
    bars = plt.bar(labels, vals, color=["#1f77b4", "#2ca02c"])
    plt.ylim(0, 1)
    plt.title("Average match scores")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    avg_path = os.path.join("output", "match_score_avg.png")
    plt.tight_layout()
    plt.savefig(avg_path)
    plt.close()

    print(f"Saved histogram to {hist_path}")
    print(f"Saved averages bar to {avg_path}")
except Exception as e:
    print("Plotting skipped (matplotlib not available or error). ", str(e))


