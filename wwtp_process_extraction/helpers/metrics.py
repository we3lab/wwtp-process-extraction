import pandas as pd
from helpers.utils import is_present, parse_status, precision_recall_f1, merge_column_statuses


def confusion_scores(manual_states, pred_states):
    """Confusion counts (tp, fp, fn, tn) and the scalar metrics over paired statuses."""
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

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    scores = {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan"),
        "Missed_Rate": fn / (tp + fn) if (tp + fn) else float("nan"),
        "Hallucinated_Rate": fp / (tp + fp) if (tp + fp) else float("nan"),
        "State_Accuracy": state_correct / state_total if state_total else float("nan"),
    }
    return tp, fp, fn, tn, scores


def build_metric_inputs(process_names, manual_df, pred_df):
    """Build key-aligned manual/pred status dataframes (one row per shared Place ID) for compute_metrics."""
    common_keys = sorted(set(manual_df["Place ID"].dropna()) & set(pred_df["Place ID"].dropna()))

    metric_dfs = []
    for df in (manual_df, pred_df):
        sub = df.drop_duplicates(subset="Place ID").set_index("Place ID").reindex(common_keys)
        metric_dfs.append(pd.DataFrame({"key": common_keys, **{p: sub[p].map(parse_status).values for p in process_names}}))
    return tuple(metric_dfs)


def aggregate_to_category_states(metric_df, category_to_leaves):
    """Collapse leaf-status columns into category-status columns per facility key."""
    out = pd.DataFrame({"key": metric_df["key"]})
    for category, leaves in category_to_leaves.items():
        out[category] = metric_df[leaves].apply(merge_column_statuses, axis=1)
    return out


def compute_metrics(manual_df, pred_df, label_cols, source_name):
    # manual_df and pred_df come from build_metric_inputs (or their category version), so rows line up
    rows = []
    for label in label_cols:
        tp, fp, fn, tn, scores = confusion_scores(manual_df[label], pred_df[label])
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
