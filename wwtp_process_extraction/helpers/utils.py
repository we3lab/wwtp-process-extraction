import pandas as pd
import re
import json
from pathlib import Path
import unicodedata

from helpers.paths import *

# Canonical status vocabulary; '' (or 0 / NaN) means absent
STATUS_TOKENS = frozenset({"PRESENT", "PRESENT_AND_FUTURE", "FUTURE", "PAST", "OFFSITE"})
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"}) # assumed present
DETECTED_STATUSES = PRESENT_STATUSES | {"FUTURE", "OFFSITE"} # included in table_1 for state accuracy
# Which status wins when one process gets several
STATUS_RANK = {"": 0, "PAST": 1, "OFFSITE": 2, "FUTURE": 3, "PRESENT": 4}

SEP = "\n\n===PLANNED CHANGES===\n\n"
# Soft hyphen and zero-width characters, deleted by normalize_text
ZERO_WIDTH_CHARS = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"))

mapping_df = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str, keep_default_na=False)
mapping_df = mapping_df.apply(lambda col: col.str.strip())
mapping_df = mapping_df.sort_values(by="NPDES No.", key=lambda s: s.eq("")).drop_duplicates(
    subset=["Place ID", "FACILITY_ID"]
)

cwns_mapping = mapping_df[
    mapping_df["CWNS_ID"].ne("") & mapping_df["CWNS_ID"].str.upper().ne("NA")
].copy()

no_cwns_pids: set[str] = set(mapping_df.loc[mapping_df["CWNS_ID"].str.upper().eq("NA"), "Place ID"])

unitprocess_keywords = json.loads(KEYWORDS_JSON.read_text())


def extract_leaves(processes_dict, group_id=None, exclude_keys=()):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if name in exclude_keys:
            continue
        if "alt_names" in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name, exclude_keys=exclude_keys))
    return leaves


# Leaf-process lookups from unitprocess_keywords, shared by steps 1, 4 and 6
leaves = extract_leaves(unitprocess_keywords)
leaf_names = [name for name, _, _ in leaves]
column_priority = {name: details.get("priority", 1) for name, details, _ in leaves}
# leaves sharing a parent group compete on priority
group_to_columns = {}
for name, _, group_id in leaves:
    if group_id:
        group_to_columns.setdefault(group_id, []).append(name)
top_category_to_columns = {}
column_top_category = {}
column_secondary_categories = {}
column_global_priority = {}
for top_cat, cat_val in unitprocess_keywords.items():
    for name, details, _ in extract_leaves({top_cat: cat_val}):
        top_category_to_columns.setdefault(top_cat, []).append(name)
        column_top_category[name] = top_cat
        column_global_priority[name] = details.get("global_priority", 1)
        if details.get("secondary_category"):
            column_secondary_categories[name] = details["secondary_category"]


def normalize_id(value):
    # Place IDs / CWNS_IDs show up as both "219530" and "219530.0" across files
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def select_json_per_place_id(json_dir, pdf_stem_by_place_id):
    # place_id -> the step5 json ({pdf_stem}_{place_id}.json) of the document the manual labels were read from
    selected = {}
    for json_file in Path(json_dir).glob("*.json"):
        stem, _, place_id = json_file.stem.rpartition("_")
        if place_id in pdf_stem_by_place_id and stem == Path(pdf_stem_by_place_id[place_id]).stem:
            selected[place_id] = json_file
    return selected


def is_general_order(text):
    # A statewide general order (e.g. 2014-0153-DWQ), not a facility permit: "general waste discharge
    # requirements" in the title block, issued by the State Board (regional boards title some individual
    # permits the same way), and no Notice of Applicability (an enrollee's NOA cites the general order)
    head = re.sub(r"===PAGE \d+===\n?|\[Page \d+\]\n?", "", text[:1800])
    title = " ".join(head[:300].split()).lower()
    first_page = " ".join(head[:900].split()).lower()
    return (
        "general waste discharge requirements" in title
        and "state water resources control board" in first_page
        and "notice of applicability" not in first_page
    )


def normalize_text(text, lower=True):
    # NFKC, drop zero-width chars, collapse whitespace; lower=False keeps case for acronym keywords
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH_CHARS)
    text = " ".join(text.split())
    return text.lower() if lower else text


def parse_status(val) -> str:
    # Canonical status token for a cell, or '' if blank; fails fast on anything else
    status = str(val).strip().upper()
    if status in ("", "0", "0.0", "NAN", "NONE"):
        return ""
    if status not in STATUS_TOKENS:
        raise ValueError(f"Unrecognized status {val!r}")
    return status


def is_present(val, statuses=PRESENT_STATUSES) -> bool:
    # The single presence rule for all scoring; table_1 passes DETECTED_STATUSES
    return parse_status(val) in statuses


def precision_recall_f1(tp, fp, fn, empty=float("nan")):
    # empty is returned when a denominator is zero: 0 for plots, nan to drop a PDF from a macro-average
    precision = tp / (tp + fp) if (tp + fp) else empty
    recall = tp / (tp + fn) if (tp + fn) else empty
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else empty
    return precision, recall, f1


def get_leaf_names(cat_name, cat_val, exclude_categories=("Disposal",), exclude_unspecified=False):
    # Leaf process names under a category; exclude_unspecified drops the "Unspecified X" catch-alls
    names = [name for name, _, _ in extract_leaves({cat_name: cat_val}, exclude_keys=exclude_categories)]
    if exclude_unspecified:
        names = [name for name in names if not name.lower().startswith("unspecified")]
    return names


# Leaf names per top-level category (without the Disposal group)
category_leaf_names = {cat: get_leaf_names(cat, val) for cat, val in unitprocess_keywords.items()}


def keep_best_priority(status_dict, cols, priority, cleared=""):
    """Among the present cols, clear every one ranked below the best (lowest) priority value."""
    present_cols = [c for c in cols if status_dict.get(c) in PRESENT_STATUSES]
    if len(present_cols) <= 1:
        return
    best_priority = min(priority[c] for c in present_cols)
    for col in present_cols:
        if priority[col] > best_priority:
            status_dict[col] = cleared


def apply_secondary_category_backfill(status_dict, ontology_resolve_fn=None, excluded_cols=()):
    # A PRESENT process whose secondary category has nothing PRESENT backfills one column there: the
    # ontology's matches (step6 hook) if any, else the category's own "Unspecified X", else any catch-all.
    # excluded_cols (cleared by exclude_if_any for this item) are never backfilled.
    present_cols = [c for c, v in status_dict.items() if v in PRESENT_STATUSES]
    for source_col in present_cols:
        for sec_cat in column_secondary_categories.get(source_col, []):
            sec_cols = top_category_to_columns.get(sec_cat, [])
            if any(status_dict.get(c) in PRESENT_STATUSES for c in sec_cols):
                continue
            available = [c for c in sec_cols if c in status_dict and c not in excluded_cols]
            if not available:
                continue
            matched = ontology_resolve_fn(available) if ontology_resolve_fn else []
            target_unspecified = f"Unspecified {sec_cat}".lower()
            exact_unspecified = [c for c in available if c.lower() == target_unspecified]
            unspecified = [c for c in available if "Unspecified" in c]
            pool = matched or exact_unspecified or unspecified or available
            chosen = min(pool, key=lambda c: (column_priority[c], column_global_priority[c], c))
            status_dict[chosen] = "PRESENT"


def get_werf_codes_for_cwns_process(cwns_process_name):
    # for future mapping back to El Abbadi codes
    werf_codes_df = pd.read_csv(DATA_DIR / "el_abbadi" / "UNIT_PROCESS_EI_CODES_WERF_modified.csv", dtype=str)
    matching = werf_codes_df[werf_codes_df["FINAL_UNIT_PROCESS_NAME"] == cwns_process_name]
    return matching["WERF_CODE"].unique().tolist()


def merge_column_statuses(column) -> str:
    """Highest-priority status across all values in column."""
    tokens = {parse_status(v) for v in column}
    if "PRESENT_AND_FUTURE" in tokens or ("PRESENT" in tokens and "FUTURE" in tokens):
        return "PRESENT_AND_FUTURE"
    return max(tokens, key=STATUS_RANK.get, default="")


def collapse_facility_processes(
    df: pd.DataFrame, key_cols: list[str], meta_cols: list[str]
) -> pd.DataFrame:
    # One row per key_cols group: process columns merged by merge_column_statuses, meta columns
    # take the first non-empty value; column order kept
    all_fixed = set(key_cols) | set(meta_cols)
    proc_cols = [c for c in df.columns if c not in all_fixed]
    rows = []
    for _, grp in df.groupby(key_cols, dropna=False, sort=False):
        out = {
            col: next((v for v in grp[col] if pd.notna(v) and str(v).strip()), "")
            for col in (key_cols + meta_cols)
            if col in df.columns
        }
        for col in proc_cols:
            out[col] = merge_column_statuses(grp[col])
        rows.append(out)
    return pd.DataFrame(rows).reindex(columns=list(df.columns))


def build_cwns_facility_processes(ca_cwns_df, target_facilities):
    left = cwns_mapping[["Place ID", "WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]]
    left = left[left["Place ID"].isin(target_facilities)]
    right = collapse_facility_processes(ca_cwns_df[["CWNS_ID"] + leaf_names], ["CWNS_ID"], [])
    merged = left.merge(right, on="CWNS_ID")
    cwns_by_facility = collapse_facility_processes(
        merged, ["Place ID"], ["WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]
    ).drop(columns=["CWNS_ID", "FACILITY_ID"])
    return cwns_by_facility, merged


def current_permit_mask(df, as_of=None):
    # Per facility, keep only the documents of the permits in force on as_of (default: newest snapshot).
    # Each document gets a tier and each facility keeps its best. For docs without any
    # process drop below every tier, an empty current permit won't replace an informative old one.
    # 0 = its order number is one CIWQS listed for the facility that day
    # 1 = no readable order number
    # 2 = superseded. 
    rel = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str, keep_default_na=False)
    target = as_of or max(d for v in rel["as_of_dates"] for d in v.split(";") if d)
    held = {}
    for place_id, order, dates in zip(rel["Place ID"], rel["Order_No"].str.strip(), rel["as_of_dates"]):
        if order and target in dates.split(";"):
            held.setdefault(normalize_id(place_id), set()).add(order)

    tiers = []
    for _, row in df.iterrows():
        doc = str(row["document_order_no"]).strip()
        if not doc:
            tiers.append(1)
            continue
        # the row's own Order_No is only a fallback for facilities missing from the snapshot
        current = held.get(normalize_id(row["Place ID"])) or {str(row["Order_No"]).strip()}
        # "R5-2007-0090", "r5 2007 0090" and "WQ 2007-0090" are one order written three ways:
        # keep letters/digits only and drop a leading WQ
        normalized = []
        for value in [doc] + [o for o in current if o]:
            text = "".join(ch for ch in value if ch.isascii() and ch.isalnum()).upper()
            normalized.append(text.removeprefix("WQ") or text)
        doc_order, *held_orders = normalized
        # containment, not equality: CIWQS adds the region ("R5-2007-0090" vs "2007-0090")
        # or an enrollee suffix ("2014-0153-DWQ-R5348" vs "WQ-2014-0153-DWQ")
        tiers.append(0 if any(doc_order in o or o in doc_order for o in held_orders) else 2)
    content = df[leaf_names].isin(STATUS_TOKENS).any(axis=1)
    tiers = pd.Series(tiers, index=df.index) + (~content).astype(int) * 3
    best = tiers.groupby(df["Place ID"]).transform("min")
    keep = tiers == best
    # A facility whose documents are all superseded would keep its whole history. This is
    # common under general orders (no document prints the general order number), so keep
    # only the newest order's documents.
    fallback = keep & (tiers % 3 == 2)
    if fallback.any():
        # Adoption year, 0 if unreadable. Try a four-digit year first ("R9-2020-0191" holds "92"),
        # then a leading two-digit one ("97-10-DWQ", "05-025", "5-00-080")
        years = []
        for order in df.loc[fallback, "document_order_no"]:
            text = re.sub(r"^R\d{1,2}[A-Z]?-?", "", "".join(str(order).split()).upper())
            four_digit = re.search(r"(?:19|20)\d{2}", text)
            two_digit = re.match(r"(?:\d-)?(\d{2})(?:\D|$)", text)
            if four_digit:
                years.append(int(four_digit.group(0)))
            elif two_digit:
                year = int(two_digit.group(1))
                years.append(1900 + year if year >= 90 else 2000 + year)
            else:
                years.append(0)
        years = pd.Series(years, index=df.index[fallback])
        newest = years.groupby(df.loc[fallback, "Place ID"]).transform("max")
        keep.loc[fallback] = years == newest
    return keep


def add_county_and_sort(df, name_col, place_id_col=None, wdid_col=None, cwns_id_col=None):
    # Insert a 'County' column after name_col and sort by (County, name_col), blanks last. County comes
    # from site_data_all by WDID; mapping_df links WDID to Place ID and CWNS_ID; first non-blank wins.
    site = pd.read_csv(SITE_DATA_ALL_CSV, dtype=str, keep_default_na=False)
    site = pd.DataFrame({"WDID": site["WDID"].str.strip(), "County": site["County"].str.strip()})
    site = site[site["WDID"].ne("") & site["County"].ne("")].drop_duplicates("WDID")
    county_by_wdid = dict(zip(site["WDID"], site["County"]))

    mapped = pd.DataFrame({
        "Place ID": mapping_df["Place ID"].map(normalize_id),
        "CWNS_ID": mapping_df["CWNS_ID"].map(normalize_id),
        "County": mapping_df["WDID"].map(county_by_wdid),
    }).dropna(subset=["County"])
    county_by = {}
    for key in ("Place ID", "CWNS_ID"):
        first = mapped[mapped[key].ne("")].drop_duplicates(key)
        county_by[key] = dict(zip(first[key], first["County"]))

    county = pd.Series(pd.NA, index=df.index, dtype=object)
    if place_id_col:
        county = county.fillna(df[place_id_col].map(normalize_id).map(county_by["Place ID"]))
    if wdid_col:
        county = county.fillna(df[wdid_col].str.strip().map(county_by_wdid))
    if cwns_id_col:
        county = county.fillna(df[cwns_id_col].map(normalize_id).map(county_by["CWNS_ID"]))
    df.insert(df.columns.get_loc(name_col) + 1, "County", county.fillna(""))
    n_missing = df["County"].eq("").sum()
    print(f"  add_county_and_sort: {len(df) - n_missing}/{len(df)} rows got a county ({n_missing} blank)")
    return df.sort_values(
        by=["County", name_col],
        key=lambda col: col.map(lambda v: "\uffff" if not str(v).strip() else str(v).lower()),
    ).reset_index(drop=True)


def build_txt_jobs(facilities_csv):
    facilities_df = pd.read_csv(facilities_csv, dtype=str).fillna("")

    jobs = []
    for row_idx, row in facilities_df.iterrows():
        facility_name = row["Facility Name"].strip()
        pdf_file_value = row["PDF_File"].strip()
        if not facility_name or not pdf_file_value:
            continue

        txt_path = TXT_DIR / Path(pdf_file_value).with_suffix(".txt").name
        if not txt_path.is_file():
            print(f"No txt for '{facility_name}': {txt_path.name}, skipping.")
            continue

        jobs.append((row_idx, txt_path, facility_name, row["Place ID"].strip()))

    return jobs
