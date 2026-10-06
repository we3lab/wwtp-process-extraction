import pandas as pd
import os
import re
import json
from functools import cache
from pathlib import Path
import unicodedata

# Canonical status vocabulary; '' (or 0 / NaN) means absent
STATUS_TOKENS = frozenset({"PRESENT", "PRESENT_AND_FUTURE", "FUTURE", "PAST", "OFFSITE"})
# Figures compare currently installed (PRESENT, PRESENT_AND_FUTURE); everything else is absent
# table_1 F1 also counts planned (FUTURE) and off-site (OFFSITE); only PAST is absent there.
# table_1 state accuracy then checks the exact state on those detected cells.
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"})
DETECTED_STATUSES = PRESENT_STATUSES | {"FUTURE", "OFFSITE"}
# Which status wins when one process gets several
STATUS_RANK = {"": 0, "PAST": 1, "OFFSITE": 2, "FUTURE": 3, "PRESENT": 4}

SEP = "\n\n===PLANNED CHANGES===\n\n"
# Soft hyphen and zero-width characters, deleted by normalize_text
ZERO_WIDTH_CHARS = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"))

# Project paths, resolved from this file so scripts work from any directory
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"
TXT_DIR = OUTPUT_DIR / "permits" / "text"
FIGURES_DIR = OUTPUT_DIR / "figures"
BY_CATEGORY_DIR = FIGURES_DIR / "by_category"
LLM_EXTRACTION_DIR = OUTPUT_DIR / "llm_extraction"
WATERRAG_RETRIEVAL_DIR = OUTPUT_DIR / "waterrag_retrieval"
CIWQS_TO_CWNS_CSV = DATA_DIR / "ciwqs_to_cwns.csv"
KEYWORDS_JSON = DATA_DIR / "unitprocess_keywords.json"
MANUAL_CSV = DATA_DIR / "unit_processes_by_facility_manual.csv"
SITE_DATA_ALL_CSV = OUTPUT_DIR / "site_data_all.csv"
SITE_DATA_RELEVANT_CSV = OUTPUT_DIR / "site_data_relevant.csv"
CWNS_TABLE_CSV = OUTPUT_DIR / "unit_processes_by_facility_cwns.csv"

mapping_df = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str, keep_default_na=False)

for c in mapping_df.columns:
    mapping_df[c] = mapping_df[c].str.strip()

mapping_df = mapping_df.sort_values(
    by="NPDES No.", key=lambda s: s.eq(""), ascending=True
).drop_duplicates(subset=["Place ID", "FACILITY_ID"], keep="first")

cwns_mapping = mapping_df[
    mapping_df["CWNS_ID"].ne("") & mapping_df["CWNS_ID"].str.upper().ne("NA")
].copy()

no_cwns_pids: set[str] = set(mapping_df.loc[mapping_df["CWNS_ID"].str.upper().eq("NA"), "Place ID"])

with open(KEYWORDS_JSON, "r") as f:
    unitprocess_keywords = json.load(f)


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
column_secondary_categories = {}
column_global_priority = {}
for top_cat, cat_val in unitprocess_keywords.items():
    for name, details, _ in extract_leaves({top_cat: cat_val}):
        top_category_to_columns.setdefault(top_cat, []).append(name)
        column_global_priority[name] = details.get("global_priority", 1)
        if details.get("secondary_category"):
            column_secondary_categories[name] = details["secondary_category"]


@cache
def document_recency():
    """(place_id, pdf_stem) -> newest snapshot date (from as_of_dates) the document appears in."""
    recency = {}
    rel = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str, keep_default_na=False)
    for _, row in rel.iterrows():
        pdf = row["PDF_File"].strip()
        if not pdf:
            continue
        key = (row["Place ID"].strip(), Path(pdf).stem)
        recency[key] = max(recency.get(key, ""), max(row["as_of_dates"].split(";")))
    return recency


def select_json_per_place_id(json_dir, place_id_filter=None, pdf_stem_by_place_id=None):
    """Map place_id -> json Path for one model directory, one file per facility.

    step5 names each output {pdf_stem}_{place_id}.json, so a facility with several documents has
    several files. pdf_stem_by_place_id (the manual CSV's PDF_File) pins the document the manual
    labels were read from; otherwise keep the most current one (newest snapshot). A tie raises
    rather than falling back to filename order, which once picked a superseded NOA.
    """
    candidates = {}
    for json_file in Path(json_dir).glob("*.json"):
        parts = json_file.stem.split("_")
        place_id = parts[-1]
        if len(parts) < 2 or not place_id.isdigit():
            continue
        if place_id_filter is not None and place_id not in place_id_filter:
            continue
        if pdf_stem_by_place_id and place_id in pdf_stem_by_place_id:
            stem = Path(pdf_stem_by_place_id[place_id]).stem
            if json_file.name != f"{stem}_{place_id}.json":
                continue
        candidates.setdefault(place_id, []).append(json_file)

    recency = document_recency()
    selected = {}
    for place_id, files in candidates.items():
        file_recency = {f: recency.get((place_id, f.name[: -(len(place_id) + 6)]), "") for f in files}
        ranked = sorted(files, key=file_recency.get, reverse=True)
        best = file_recency[ranked[0]]
        tied = [f for f in ranked if file_recency[f] == best]
        if len(tied) > 1:
            raise ValueError(
                f"Place {place_id} in {Path(json_dir).name}: {len(tied)} documents are equally "
                f"current (snapshot {best or 'none'}), cannot pick one: "
                f"{sorted(f.name for f in tied)}. Pass pdf_stem_by_place_id to disambiguate."
            )
        selected[place_id] = ranked[0]
    return selected


def is_general_order(text):
    """True if text opens as a statewide general order (e.g. 2014-0153-DWQ), not a facility permit.

    Checked on content, since an enrollee's order number is the general order number. All three:
      - "general waste discharge requirements" in the title block
      - issued by the State Water Resources Control Board (regional boards title some
        individual permits and enrollment letters the same way)
      - no "Notice of Applicability" (an enrollee's own NOA cites the general order up top)
    """
    head = re.sub(r"===PAGE \d+===\n?|\[Page \d+\]\n?", "", text[:1800])
    if not re.search(r"general\s+waste\s+discharge\s+requirements", head[:300], re.IGNORECASE):
        return False
    if not re.search(r"state\s+water\s+resources\s+control\s+board", head[:900], re.IGNORECASE):
        return False
    return not re.search(r"notice\s+of\s+applicability", head[:900], re.IGNORECASE)


def normalize_text(text, lower=True):
    """Normalize for matching: NFKC, drop zero-width chars, collapse whitespace, lowercase.

    lower=False keeps original case, for acronym keywords that must match case-sensitively.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH_CHARS)
    text = " ".join(text.split())
    return text.lower() if lower else text


def parse_status(val) -> str:
    """Canonical status token for a cell (see STATUS_TOKENS), or '' if blank.

    Fails fast on any other value, so a typo can't silently count as absent.
    """
    status = str(val).strip().upper()
    if status in ("", "0", "0.0", "NAN", "NONE"):
        return ""
    if status not in STATUS_TOKENS:
        raise ValueError(f"Unrecognized status {val!r}")
    return status


def is_present(val, statuses=PRESENT_STATUSES) -> bool:
    """True if val's status is in statuses (default: currently installed). Everything else is absent.

    The single presence rule for all scoring. table_1 passes DETECTED_STATUSES.
    """
    return parse_status(val) in statuses


def precision_recall_f1(tp, fp, fn, empty=float("nan")):
    """Precision, recall, F1, and Jaccard overlap from TP/FP/FN counts.

    empty is returned for any metric whose denominator is zero: pass 0 for plots
    that should show no error, leave the nan default to drop that PDF from a macro-average.
    """
    precision = tp / (tp + fp) if (tp + fp) else empty
    recall = tp / (tp + fn) if (tp + fn) else empty
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else empty
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else empty
    return precision, recall, f1, jaccard


def get_leaf_names(cat_name, cat_val, exclude_categories=("Disposal",), exclude_unspecified=False):
    """Return leaf process names for a category from the keywords hierarchy.

    exclude_unspecified drops catch-all 'Unspecified X' leaves (priority 1000) — use for
    leaf-level comparisons where matching a catch-all exactly would be unfair.
    """
    if cat_name in exclude_categories:
        return []
    if "alt_names" in cat_val:
        return [cat_name]
    leaves = extract_leaves(cat_val, exclude_keys=exclude_categories)
    if exclude_unspecified:
        # 'Unspecified X' catch-alls are nested leaves, never top-level categories
        leaves = [(n, d, g) for n, d, g in leaves
                  if not (n.lower().startswith("unspecified") and d.get("priority") == 1000)]
    return [name for name, _, _ in leaves]


def keep_best_priority(status_dict, cols, priority, cleared=""):
    """Among the present cols, clear every one ranked below the best (lowest) priority value."""
    present_cols = [c for c in cols if status_dict.get(c) in PRESENT_STATUSES]
    if len(present_cols) <= 1:
        return
    best_priority = min(priority.get(c, 1) for c in present_cols)
    for col in present_cols:
        if priority.get(col, 1) > best_priority:
            status_dict[col] = cleared


def apply_secondary_category_backfill(status_dict, ontology_resolve_fn=None, excluded_cols=()):
    """Backfill secondary categories: if a PRESENT process requests a secondary category
    that has no PRESENT process, mark the best fallback (unspecified-first) as PRESENT.

    ontology_resolve_fn(sec_cols) -> str | None: optional hook for
    ontology-based selection (used by step6). Returns the chosen column name, or None to
    fall back to unspecified-first heuristic.
    excluded_cols: columns cleared by exclude_if_any for this item — never backfilled,
    so the backfill can't resurrect a column an exclusion just removed.
    """
    present_cols = [c for c, v in status_dict.items() if v in PRESENT_STATUSES]
    for source_col in present_cols:
        for sec_cat in column_secondary_categories.get(source_col, []):
            sec_cols = top_category_to_columns.get(sec_cat, [])
            if any(status_dict.get(c) in PRESENT_STATUSES for c in sec_cols):
                continue
            available = [c for c in sec_cols if c in status_dict and c not in excluded_cols]
            if not available:
                continue
            chosen = ontology_resolve_fn(sec_cols) if ontology_resolve_fn else None
            if chosen is None:
                # Prefer the category-level catch-all (e.g., "Unspecified Filtration")
                # before nested catch-alls (e.g., "Unspecified FFR").
                target_unspecified = f"Unspecified {sec_cat}".lower()
                exact_unspecified = [c for c in available if c.lower() == target_unspecified]
                unspecified = [c for c in available if "Unspecified" in c]
                pool = exact_unspecified or unspecified or available
                chosen = min(pool, key=lambda c: (column_priority.get(c, 1), column_global_priority.get(c, 1), c))
            status_dict[chosen] = "PRESENT"


def get_werf_codes_for_cwns_process(cwns_process_name):
    """for future mapping back to El Abbadi codes."""
    el_abbadi_dir = os.path.join(os.path.dirname(__file__), "data", "el_abbadi", "input")
    werf_codes_df = pd.read_csv(
        os.path.join(el_abbadi_dir, "UNIT_PROCESS_EI_CODES_WERF_modified.csv"), dtype=str
    )
    matching = werf_codes_df[werf_codes_df["FINAL_UNIT_PROCESS_NAME"] == cwns_process_name]
    return matching["WERF_CODE"].unique().tolist() if not matching.empty else []


def merge_column_statuses(column) -> str:
    """Highest-priority status across all values in column."""
    tokens = {parse_status(v) for v in column}
    if "PRESENT_AND_FUTURE" in tokens or ("PRESENT" in tokens and "FUTURE" in tokens):
        return "PRESENT_AND_FUTURE"
    return max(tokens, key=STATUS_RANK.get, default="")


def collapse_facility_processes(
    df: pd.DataFrame, key_cols: list[str], meta_cols: list[str]
) -> pd.DataFrame:
    """One row per unique key_cols group; highest-priority status per process column.

    Process columns (everything not in key_cols or meta_cols) are merged via
    merge_column_statuses. Meta columns take the first non-empty value. Column order preserved.
    """
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


def build_cwns_facility_processes(ca_cwns_df, target_facilities=None):
    proc_cols = leaf_names
    left = cwns_mapping[["Place ID", "WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]]
    if target_facilities is not None:
        left = left[left["Place ID"].isin(target_facilities)]
    right = collapse_facility_processes(ca_cwns_df[["CWNS_ID"] + proc_cols], ["CWNS_ID"], [])
    merged = left.merge(right, on="CWNS_ID", how="inner")
    cwns_by_facility = collapse_facility_processes(
        merged, ["Place ID"], ["WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]
    ).drop(columns=["CWNS_ID", "FACILITY_ID"]).fillna("")
    return cwns_by_facility, merged


@cache
def orders_in_force(as_of=None):
    """place_id -> set of Order_No values CIWQS listed for it on a snapshot date (default: newest).

    Read from site_data_relevant's as_of_dates rather than site_data/<date>/, because some dates
    have no snapshot folder. Pass as_of to get the permits as they stood then (e.g. 2022 for CWNS).
    """
    rel = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str, keep_default_na=False)
    dates = {d for v in rel["as_of_dates"] for d in v.split(";") if d}
    target = as_of or max(dates)
    held = {}
    for _, row in rel.iterrows():
        if target not in row["as_of_dates"].split(";"):
            continue
        order = row["Order_No"].strip()
        if order:
            held.setdefault(normalize_id(row["Place ID"]), set()).add(order)
    return held


def current_permit_mask(df, order_col="Order_No", doc_order_col="document_order_no", as_of=None,
                        content=None):
    """Per facility, keep only the documents from the permits in force as of `as_of`.

    Each document gets a tier, and each facility keeps only its best tier, so it is never
    left with nothing:
      0. the document's order number is one the facility held (from orders_in_force)
      1. no order number could be read from the document
      2. the order number is not one it held (superseded)

    Compare to the snapshot's order set, not the row's own `order_col`: site_data_relevant has
    a row per (facility, order), so an old permit would match its own old Order_No. `order_col`
    is only the fallback for facilities missing from the snapshot. A set also keeps concurrent
    permits (e.g. a plant's own order plus a joint-authority order).

    `content` (optional boolean Series) marks rows that yielded processes. Rows without content
    drop below every tier, so an empty current permit can't hide an informative superseded one.
    """
    held = orders_in_force(as_of)
    tiers = []
    for _, row in df.iterrows():
        doc = str(row.get(doc_order_col, "")).strip()
        if not doc:
            tiers.append(1)
            continue
        current = held.get(normalize_id(row["Place ID"])) or {str(row.get(order_col, "")).strip()}
        # "R5-2007-0090", "r5 2007 0090" and "WQ 2007-0090" are one order written three ways:
        # keep letters/digits only and drop a leading region/WQ prefix
        normalized = []
        for value in [doc] + [o for o in current if o]:
            text = "".join(ch for ch in value if ch.isascii() and ch.isalnum()).upper()
            normalized.append(re.sub(r"^(R\d{1,2}[A-Z]?)?WQ", "", text) or text)
        doc_order, *held_orders = normalized
        # containment, not equality: CIWQS adds the region ("R5-2007-0090" vs "2007-0090")
        # or an enrollee suffix ("2014-0153-DWQ-R5348" vs "WQ-2014-0153-DWQ")
        tiers.append(0 if any(doc_order in o or o in doc_order for o in held_orders) else 2)
    tiers = pd.Series(tiers, index=df.index)
    if content is not None:
        tiers = tiers + (~content.reindex(df.index).fillna(False)).astype(int) * 3
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
        for order in df.loc[fallback, doc_order_col]:
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


def normalize_id(value):
    # Place IDs / CWNS_IDs show up as both "219530" and "219530.0" across files
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def add_county_and_sort(df, name_col, place_id_col=None, wdid_col=None, cwns_id_col=None):
    """Insert a 'County' column after name_col and sort by (County, name_col); blanks sort last.

    County comes from site_data_all by WDID; mapping_df links WDID to Place ID and CWNS_ID,
    so any of the three keys can find it. The first non-blank county per key wins.
    """
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
        key=lambda col: col.map(lambda v: "￿" if not str(v).strip() else str(v).lower()),
    ).reset_index(drop=True)


def build_txt_jobs(facilities_information):
    facilities_df = pd.read_csv(facilities_information, dtype=str).fillna("")

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
