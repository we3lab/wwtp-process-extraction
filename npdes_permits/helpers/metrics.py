import pandas as pd


def compute_metrics(manual_df: pd.DataFrame, pred_df: pd.DataFrame, label_cols: list, source_name: str) -> pd.DataFrame:
    rows = []
    manual_indexed = manual_df.set_index("key")
    pred_indexed = pred_df.set_index("key")
    keys = list(manual_indexed.index)

    for label in label_cols:
        tp = fp = fn = tn = 0
        state_correct = state_total = 0
        for key in keys:
            manual_states = manual_indexed.at[key, label]
            pred_states = pred_indexed.at[key, label]
            manual_pos = bool(manual_states)
            pred_pos = bool(pred_states)

            tp += int(manual_pos and pred_pos)
            fp += int((not manual_pos) and pred_pos)
            fn += int(manual_pos and (not pred_pos))
            tn += int((not manual_pos) and (not pred_pos))

            if manual_pos:
                state_total += 1
                state_correct += int(manual_states == pred_states)

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
        missed_rate = fn / (tp + fn) if (tp + fn) else float("nan")
        hallucinated_rate = fp / (tp + fp) if (tp + fp) else float("nan")
        state_accuracy = state_correct / state_total if state_total else float("nan")

        rows.append({
            "Source": source_name,
            "Label": label,
            "Support_Manual": int(tp + fn),
            "Support_Pred": int(tp + fp),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "Missed_Rate": missed_rate,
            "Hallucinated_Rate": hallucinated_rate,
            "State_Accuracy": state_accuracy,
        })

    return pd.DataFrame(rows)


def summarize_metrics(metric_df: pd.DataFrame, level_name: str) -> pd.DataFrame:
    rows = []
    for source, subset in metric_df.groupby("Source", sort=False):
        usable = subset[subset[["Support_Manual", "Support_Pred", "TP", "FP", "FN"]].sum(axis=1) > 0]
        rows.append({
            "Level": level_name,
            "Source": source,
            "Macro_F1": usable["F1"].mean(),
            "Macro_Accuracy": usable["Accuracy"].mean(),
            "Macro_Missed_Rate": usable["Missed_Rate"].mean(),
            "Macro_Hallucinated_Rate": usable["Hallucinated_Rate"].mean(),
            "Macro_State_Accuracy": usable["State_Accuracy"].mean(),
            "Label_Count": int(len(usable)),
        })
    return pd.DataFrame(rows)
