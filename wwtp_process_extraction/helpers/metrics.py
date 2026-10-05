import pandas as pd
from helpers.utils import is_present, parse_status, precision_recall_f1, merge_column_statuses

# 0–1 scalar metrics (per label or per facility); violin / summaries use this order.
METRIC_SCORE_COLUMNS = (
    "Hallucinated_Rate",
    "Missed_Rate",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "State_Accuracy",
)


def score_from_counts(
    tp: int, fp: int, fn: int, tn: int, state_correct: int, state_total: int
) -> dict:
    """Compute scalar metrics from confusion counts and state-match counts."""
    precision, recall, f1, _ = precision_recall_f1(tp, fp, fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    missed_rate = fn / (tp + fn) if (tp + fn) else float("nan")
    hallucinated_rate = fp / (tp + fp) if (tp + fp) else float("nan")
    state_accuracy = state_correct / state_total if state_total else float("nan")
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Missed_Rate": missed_rate,
        "Hallucinated_Rate": hallucinated_rate,
        "State_Accuracy": state_accuracy,
    }


def confusion_and_state_counts(manual_states, pred_states) -> tuple:
    """Return (tp, fp, fn, tn, state_correct, state_total) over paired statuses."""
    tp = fp = fn = tn = 0
    state_correct = state_total = 0

    for manual_state, pred_state in zip(manual_states, pred_states):
        manual_pos = is_present(manual_state)
        pred_pos = is_present(pred_state)

        tp += int(manual_pos and pred_pos)
        fp += int((not manual_pos) and pred_pos)
        fn += int(manual_pos and (not pred_pos))
        tn += int((not manual_pos) and (not pred_pos))

        if manual_pos:
            state_total += 1
            state_correct += int(manual_state == pred_state)

    return tp, fp, fn, tn, state_correct, state_total


def build_metric_inputs(process_names, manual_df, pred_df):
    """Build key-aligned manual/pred status dataframes (one row per shared Place ID) for compute_metrics."""
    common_keys = sorted(set(manual_df["Place ID"].dropna()) & set(pred_df["Place ID"].dropna()))

    manual_sub = (
        manual_df[manual_df["Place ID"].isin(common_keys)]
        .drop_duplicates(subset="Place ID")
        .set_index("Place ID")
    )
    pred_sub = (
        pred_df[pred_df["Place ID"].isin(common_keys)].drop_duplicates(subset="Place ID").set_index("Place ID")
    )

    manual_metric_df = pd.DataFrame({"key": common_keys})
    pred_metric_df = pd.DataFrame({"key": common_keys})

    for process in process_names:
        manual_metric_df[process] = (
            manual_sub.reindex(common_keys)[process].map(parse_status).values
        )
        pred_metric_df[process] = pred_sub.reindex(common_keys)[process].map(parse_status).values

    return manual_metric_df, pred_metric_df


def aggregate_to_category_states(metric_df, category_to_leaves):
    """Collapse leaf-status columns into category-status columns per facility key."""
    out = pd.DataFrame({"key": metric_df["key"]})
    for category, leaves in category_to_leaves.items():
        out[category] = metric_df[leaves].apply(merge_column_statuses, axis=1)
    return out


def compute_metrics(
    manual_df: pd.DataFrame, pred_df: pd.DataFrame, label_cols: list, source_name: str
) -> pd.DataFrame:
    rows = []
    manual_indexed = manual_df.set_index("key")
    pred_indexed = pred_df.set_index("key")
    keys = list(manual_indexed.index)

    for label in label_cols:
        manual_states = manual_indexed.loc[keys, label]
        pred_states = pred_indexed.loc[keys, label]
        tp, fp, fn, tn, state_correct, state_total = confusion_and_state_counts(
            manual_states, pred_states
        )
        scores = score_from_counts(tp, fp, fn, tn, state_correct, state_total)

        rows.append(
            {
                "Source": source_name,
                "Label": label,
                "Support_Manual": int(tp + fn),
                "Support_Pred": int(tp + fp),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                **scores,
            }
        )

    return pd.DataFrame(rows)


def compute_facility_metric_rows(
    manual_df: pd.DataFrame, pred_df: pd.DataFrame, label_cols: list, source_name: str
) -> list:
    """Compute per-facility metrics for violin/distribution plots."""
    manual_indexed = manual_df.set_index("key")
    pred_indexed = pred_df.set_index("key")
    rows = []

    for key in manual_indexed.index:
        manual_states = manual_indexed.loc[key, label_cols]
        pred_states = pred_indexed.loc[key, label_cols]
        tp, fp, fn, tn, state_correct, state_total = confusion_and_state_counts(
            manual_states, pred_states
        )
        scores = score_from_counts(tp, fp, fn, tn, state_correct, state_total)

        row = {"Source": source_name, "key": key}
        for col in METRIC_SCORE_COLUMNS:
            row[col] = scores[col]
        rows.append(row)

    return rows
