import pandas as pd
from wwtp_process_extraction.helpers.utils import is_present

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


def _score_from_counts(
    tp: int, fp: int, fn: int, tn: int, state_correct: int, state_total: int
) -> dict:
    """Compute scalar metrics from confusion counts and state-match counts."""
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
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


def _confusion_and_state_counts(manual_states, pred_states) -> tuple:
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
        tp, fp, fn, tn, state_correct, state_total = _confusion_and_state_counts(
            manual_states, pred_states
        )
        scores = _score_from_counts(tp, fp, fn, tn, state_correct, state_total)

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
        tp, fp, fn, tn, state_correct, state_total = _confusion_and_state_counts(
            manual_states, pred_states
        )
        scores = _score_from_counts(tp, fp, fn, tn, state_correct, state_total)

        row = {"Source": source_name, "key": key}
        for col in METRIC_SCORE_COLUMNS:
            row[col] = scores[col]
        rows.append(row)

    return rows


def summarize_metrics(metric_df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    rows = []
    for source, subset in metric_df.groupby("Source", sort=False):
        usable = subset[
            subset[["Support_Manual", "Support_Pred", "TP", "FP", "FN"]].sum(axis=1) > 0
        ]
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
