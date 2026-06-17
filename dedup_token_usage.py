import pandas as pd
from pathlib import Path

base = Path("wwtp_process_extraction/output/llm_extraction")
files = sorted(base.rglob("token_usage_summary.csv"))

for f in files:
    df = pd.read_csv(f)
    n_before = len(df)

    is_failed = df["structured_output"].astype(str).str.upper() == "FAILED"

    # for each facility_name, keep last non-FAILED row; fallback to last row
    def keep_row(group):
        non_failed = group[group["structured_output"].astype(str).str.upper() != "FAILED"]
        if len(non_failed) > 0:
            return non_failed.iloc[[-1]]
        return group.iloc[[-1]]

    deduped = df.groupby("facility_name", sort=False).apply(keep_row).reset_index(drop=True)
    n_after = len(deduped)

    if n_before != n_after:
        deduped.to_csv(f, index=False)
        print(f"  {n_before} -> {n_after} rows: {f}")
    else:
        print(f"  no change ({n_before} rows): {f}")
