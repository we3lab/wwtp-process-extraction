# Union the dated site_data_relevant.csv snapshots into the top-level working file.
#
# Each AS_OF run of step2 overwrites output/site_data_relevant.csv with just that year's
# permits, but step3 and step5 --all_facilities both read that one file. Left alone they would
# only process whichever year ran last. Unioning the snapshots makes every document appear
# exactly once, so text extraction and LLM extraction run once per document rather than once
# per document per year -- a given permit's extraction does not change between snapshots.
#
# Run this AFTER all step2 AS_OF runs finish; step2 rewrites the top-level file each time.

import argparse
import glob
import os

import pandas as pd

from helpers.utils import COLLECTIVE_AGENCY_RE

OUT = "wwtp_process_extraction/output"
SNAPSHOT_GLOB = os.path.join(OUT, "site_data", "*", "site_data_relevant.csv")
UNION_PATH = os.path.join(OUT, "site_data_relevant.csv")

# A facility can hold several orders across snapshots, and one order can carry several PDFs;
# general orders (e.g. 2014-0153-DWQ) are shared by hundreds of facilities. Reg_Measure_ID is
# the stable per-order id, with Order_No as fallback where it is blank.
KEY = ["Place ID", "order_key", "PDF_File"]


def load_snapshots(pattern):
    frames = []
    for path in sorted(glob.glob(pattern)):
        as_of = os.path.basename(os.path.dirname(path))
        df = pd.read_csv(path, dtype=str).fillna("")
        df["as_of"] = as_of
        frames.append(df)
        print(f"  {as_of}: {len(df):5} rows")
    if not frames:
        raise SystemExit(f"no snapshots matched {pattern}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", default="*", help="snapshot folder glob, e.g. '20??-06-01'")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write")
    args = ap.parse_args()

    print("snapshots found:")
    allrows = load_snapshots(os.path.join(OUT, "site_data", args.dates, "site_data_relevant.csv"))
    allrows["order_key"] = allrows["Reg_Measure_ID"].where(
        allrows["Reg_Measure_ID"].str.strip().ne(""), allrows["Order_No"])

    # Older snapshots predate step2's collective-permittee filter, so drop those places here too
    collective = allrows["Agency"].str.contains(COLLECTIVE_AGENCY_RE, na=False)
    if collective.any():
        dropped = allrows.loc[collective, ["Place ID", "Facility Name"]].drop_duplicates()
        print(f"\ncollective-permittee places dropped (not facilities): {len(dropped)}")
        print(dropped.to_string(index=False))
        allrows = allrows[~collective]

    # provenance: which snapshots each document appears in, for the year-over-year join later
    prov = (allrows.groupby(KEY)["as_of"]
            .agg(lambda s: ";".join(sorted(set(s))))
            .reset_index().rename(columns={"as_of": "as_of_dates"}))
    prov["n_snapshots"] = prov["as_of_dates"].str.count(";") + 1

    # keep one row per document, preferring the newest snapshot's metadata
    union = (allrows.sort_values("as_of", ascending=False)
             .drop_duplicates(subset=KEY, keep="first"))

    # provenance rides along as columns rather than a side file: the year-over-year figure
    # needs to know which snapshots a document appeared in, and a second file only drifts
    union = union.merge(prov, on=KEY, how="left")
    cols = ([c for c in allrows.columns if c not in ("as_of", "order_key")]
            + ["as_of_dates", "n_snapshots"])
    union = union[cols]

    current = pd.read_csv(UNION_PATH, dtype=str).fillna("") if os.path.exists(UNION_PATH) else pd.DataFrame()
    print()
    print(f"union            : {len(union):5} rows  ({union['PDF_File'].nunique()} distinct PDFs,"
          f" {union['Place ID'].nunique()} facilities)")
    if len(current):
        print(f"current top-level: {len(current):5} rows  ({current['PDF_File'].nunique()} distinct PDFs)")
        new_pdfs = set(union["PDF_File"]) - set(current["PDF_File"])
        print(f"  documents the union adds: {len(new_pdfs)}")
    print()
    print("documents by snapshot coverage:")
    print(prov["n_snapshots"].value_counts().sort_index().rename("documents").to_string())

    missing = [f for f in union["PDF_File"].unique()
               if f and not os.path.exists(os.path.join(OUT, "permits", f))]
    if missing:
        print(f"\nWARNING: {len(missing)} referenced PDFs are not in output/permits/ "
              f"(step3 will skip them): {missing[:3]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    union.to_csv(UNION_PATH, index=False)
    print(f"\nwrote {UNION_PATH} ({len(union)} rows)")
    print("\nnow run step3 -> step5 --all_facilities -> step6; each skips work already done.")


if __name__ == "__main__":
    main()
