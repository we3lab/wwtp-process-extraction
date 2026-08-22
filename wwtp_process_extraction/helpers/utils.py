import pandas as pd
import os
import re
import json
import io
import PyPDF2
from pathlib import Path
import unicodedata

# Canonical status vocabulary: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, '' (absent)
PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE"})

# States excluded entirely from accuracy/F1 scoring (neither a required positive nor a
# false-positive trap if predicted) — a process that's decommissioned (PAST) or whose
# equipment is physically elsewhere (OFFSITE) isn't a clean presence/absence signal.
UNSCORED_STATUSES = frozenset({"PAST", "OFFSITE"})

PLACE_ID_RE = re.compile(r"_(\d+)\.json$")


_document_recency_cache = None


def document_recency():
    """(place_id, pdf_stem) -> newest snapshot date that document appears in, '' if none.

    step2 writes a dated site_data_relevant.csv per AS_OF run and step2c unions them into
    as_of_dates. The document a facility holds in the newest snapshot is its current permit;
    one absent from every snapshot is a leftover from a superseded order.
    """
    global _document_recency_cache
    if _document_recency_cache is None:
        recency = {}
        rel = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str, keep_default_na=False).fillna("")
        dates_col = "as_of_dates" if "as_of_dates" in rel.columns else None
        for _, row in rel.iterrows():
            pdf = row["PDF_File"].strip()
            if not pdf:
                continue
            key = (row["Place ID"].strip(), Path(pdf).stem)
            newest = max(row[dates_col].split(";")) if dates_col and row[dates_col] else ""
            recency[key] = max(recency.get(key, ""), newest)
        _document_recency_cache = recency
    return _document_recency_cache


def select_json_per_place_id(json_dir, place_id_filter=None, pdf_stem_by_place_id=None):
    """Map place_id -> json Path for one model directory, one file per facility.

    step5 encodes each output's Place ID and source PDF as a {pdf_stem}_{id}.json filename. A
    facility with several permit documents (an original plus later modifications) gets one json
    per document, so a single one has to be chosen:

      1. pdf_stem_by_place_id (the manual CSV's PDF_File) pins the exact document ground truth
         was read from, so every caller scores the same source.
      2. otherwise keep the most current document, by newest snapshot it appears in.

    Never resolve ties by filename order -- sorted() picked "12-9-25_IndianSprings..." over the
    current "2026-06-01_WDR_NOA_Revised_IndianSpringsWWTP...", silently scoring a superseded NOA.
    A tie that recency cannot break raises rather than guessing.
    """
    candidates = {}
    for json_file in Path(json_dir).glob("*.json"):
        m = PLACE_ID_RE.search(json_file.name)
        place_id = m.group(1) if m else ""
        if not place_id:
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
        if len(files) == 1:
            selected[place_id] = files[0]
            continue
        ranked = sorted(
            files,
            key=lambda f: recency.get((place_id, f.name[: -(len(place_id) + 6)]), ""),
            reverse=True,
        )
        best = recency.get((place_id, ranked[0].name[: -(len(place_id) + 6)]), "")
        tied = [f for f in ranked if recency.get((place_id, f.name[: -(len(place_id) + 6)]), "") == best]
        if len(tied) > 1:
            raise ValueError(
                f"Place {place_id} in {Path(json_dir).name}: {len(tied)} documents are equally "
                f"current (snapshot {best or 'none'}), cannot pick one: "
                f"{sorted(f.name for f in tied)}. Pass pdf_stem_by_place_id to disambiguate."
            )
        selected[place_id] = ranked[0]
    return selected

SEP = "\n\n===PLANNED CHANGES===\n\n"

# A statewide general order (2014-0153-DWQ, 97-010-DWQ) describes no single plant — its generic
# process list ("septic tank, Imhoff tank, package treatment tank...") would otherwise be
# attributed to every enrolled facility. Detection has to be content-based: the order number
# can't discriminate, because an enrollee's order_no IS the general order number.
_GENERAL_ORDER_RE = re.compile(r"general\s+waste\s+discharge\s+requirements", re.IGNORECASE)
_NOA_HEADER_RE = re.compile(r"notice\s+of\s+applicability", re.IGNORECASE)
_STATE_BOARD_RE = re.compile(r"state\s+water\s+resources\s+control\s+board", re.IGNORECASE)
_ANY_PAGE_MARKER_RE = re.compile(r"===PAGE \d+===\n?|\[Page \d+\]\n?")
GENERAL_ORDER_TITLE_CHARS = 300
NOA_HEADER_CHARS = 900


# Drop watershed permit "agency"
COLLECTIVE_AGENCY_RE = re.compile(r"\borganizations?\s+under\b", re.IGNORECASE)


def is_general_order(text):
    """True if text opens as a statewide general order rather than a facility-specific permit.

    Three conditions, each ruling out a distinct look-alike:
      - title phrase in the opening block, because a general order announces itself there
      - issued by the State Water Resources Control Board, because a REGIONAL board titles
        individual permits the same way ("General WDRs for the Top O'Topanga Community
        Association WWTS at 3360 N Topanga Canyon Blvd"), as do enrollment cover letters
      - no "Notice of Applicability" heading, because an enrollee's own NOA cites the general
        order in its header and would otherwise match
    """
    head = _ANY_PAGE_MARKER_RE.sub("", text[:NOA_HEADER_CHARS * 2])
    if not _GENERAL_ORDER_RE.search(head[:GENERAL_ORDER_TITLE_CHARS]):
        return False
    if not _STATE_BOARD_RE.search(head[:NOA_HEADER_CHARS]):
        return False
    return not _NOA_HEADER_RE.search(head[:NOA_HEADER_CHARS])

# Canonical project paths, resolved from this file so they survive any os.chdir.
PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"
CIWQS_TO_CWNS_CSV = DATA_DIR / "ciwqs_to_cwns.csv"
KEYWORDS_JSON = DATA_DIR / "unitprocess_keywords.json"
SITE_DATA_ALL_CSV = OUTPUT_DIR / "site_data_all.csv"
SITE_DATA_RELEVANT_CSV = OUTPUT_DIR / "site_data_relevant.csv"
FACILITIES_JSON = OUTPUT_DIR / "facilities.json"
CWNS_TABLE_CSV = OUTPUT_DIR / "unit_processes_by_facility_cwns.csv"
# Output column order for rewriting ciwqs_to_cwns.csv (figure_3)
CIWQS_TO_CWNS_COLUMNS = [
    "WDID", "Place ID", "Facility Name", "NPDES No.", "Region",
    "Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS",
    "CWNS_ID", "FACILITY_ID", "CWNS Facility Name",
]

mapping_df = pd.read_csv(
    CIWQS_TO_CWNS_CSV, dtype=str, keep_default_na=False
).fillna("")

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


def package_sub_readers(reader):
    """For a PDF Package/Portfolio, yield a PdfReader for each embedded PDF sub-file."""
    try:
        root = reader.trailer['/Root'].get_object()
        names_obj = root['/Names'].get_object()
        emb_node = names_obj.get('/EmbeddedFiles')
        if not emb_node:
            return
        emb_names = emb_node.get_object()['/Names']
        for i in range(0, len(emb_names), 2):
            try:
                fspec = emb_names[i + 1].get_object()
                ef = fspec.get('/EF', {}).get_object()
                fstream = ef.get('/F') or ef.get('/UF')
                if fstream:
                    yield PyPDF2.PdfReader(io.BytesIO(fstream.get_object().get_data()))
            except Exception:
                continue
    except Exception:
        return
    

def hasprocess_fragments(graph, cls, watr_ns, sh_ns):
    """Process fragments declared by a class's own SHACL hasProcess shape(s).

    The ontology declares "this equipment implies this process" two equivalent ways:
    `sh:hasValue watr:Process-X` or `sh:qualifiedValueShape [ sh:class watr:Process-X ]`.
    """
    fragments = set()
    for prop in graph.objects(cls, sh_ns.property):
        for path in graph.objects(prop, sh_ns.path):
            if path != watr_ns.hasProcess:
                continue
            for val in graph.objects(prop, sh_ns.hasValue):
                if val.fragment:
                    fragments.add(val.fragment)
            for qualified_shape in graph.objects(prop, sh_ns.qualifiedValueShape):
                for val in graph.objects(qualified_shape, sh_ns["class"]):
                    if val.fragment:
                        fragments.add(val.fragment)
    return fragments


def normalize_text(text, lower=True):
    """Normalize for matching: NFKC, drop zero-width chars, collapse whitespace, lowercase.

    lower=False keeps original case, for acronym keywords that must match case-sensitively.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[­​‌‍﻿]", "", text)  # zero-width chars
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lower else text

def parse_status(val) -> str:
    """Normalize any status cell to a canonical token.

    Handles manual sheet values (messy text), LLM output (clean tokens), and CWNS values.
    Returns: PRESENT, PRESENT_AND_FUTURE, FUTURE, PAST, OFFSITE, or ''.
    """
    if val is None or (isinstance(val, float) and val != val):
        return ""
    s = str(val).strip()
    if not s or s in ("0", "0.0"):
        return ""
    t = s.upper().replace("-", "_")
    if t in ("NAN", "NONE"):
        return ""
    if "PRESENT" in t and "FUTURE" in t:
        return "PRESENT_AND_FUTURE"
    for keyword in ["PRESENT", "FUTURE", "PAST", "OFFSITE"]:
        if keyword in t:
            return keyword
    return ""


def is_present(val) -> bool:
    """True if val indicates the process is currently installed (PRESENT or PRESENT_AND_FUTURE).

    FUTURE is excluded — it means planned but not yet in service.
    """
    return parse_status(val) in PRESENT_STATUSES


def is_unscored(val) -> bool:
    """True if val (PAST or OFFSITE) should be dropped entirely from accuracy/F1 scoring."""
    return parse_status(val) in UNSCORED_STATUSES


def presence_diff(truth_row, pred_row, cols, truth_cols=None, pred_cols=None,
                   truth_present_fn=is_present, pred_present_fn=is_present):
    """Per-column TP/FP/FN between a truth row and a prediction row.

    A column is dropped entirely (counted toward neither TP, FP, nor FN) if either
    side's status is PAST/OFFSITE (see UNSCORED_STATUSES) — same rule used for the
    table_1 F1 metrics, centralized here so every comparison applies it consistently.
    truth_present_fn/pred_present_fn let callers swap in a different presence
    definition per side (e.g. CWNS counts FUTURE/PAST as detected; is_present doesn't).
    Returns (tp, fp, fn, missed, extra) where missed/extra are sorted column-name lists.
    """
    truth_cols = truth_row.index if truth_cols is None else truth_cols
    pred_cols = pred_row.index if pred_cols is None else pred_cols
    tp = fp = fn = 0
    missed, extra = [], []
    for col in cols:
        truth_val = truth_row.get(col, "") if col in truth_cols else ""
        pred_val = pred_row.get(col, "") if col in pred_cols else ""
        if is_unscored(truth_val) or is_unscored(pred_val):
            continue
        truth_positive = truth_present_fn(truth_val)
        pred_positive = pred_present_fn(pred_val)
        if truth_positive and pred_positive:
            tp += 1
        elif pred_positive:
            fp += 1
            extra.append(col)
        elif truth_positive:
            fn += 1
            missed.append(col)
    return tp, fp, fn, sorted(missed), sorted(extra)


CWNS_PRESENT_STATUSES = frozenset({"PRESENT", "PRESENT_AND_FUTURE", "FUTURE", "PAST"})


def is_present_cwns(val) -> bool:
    """Scalar counterpart to build_cwns_presence_mask, for use with presence_diff."""
    return parse_status(val) in CWNS_PRESENT_STATUSES


def build_cwns_presence_mask(series):
    """Return boolean mask for CWNS presence values (any detectable status, including FUTURE/PAST)."""
    return series.map(parse_status).isin(CWNS_PRESENT_STATUSES)


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


def f1_error_parts(tp, fp, fn, empty=0):
    """Split F1 error (1 - F1) into missed (FN) and extra (FP) shares, plus their total.

    All three divide by 2*tp+fp+fn (= |truth| + |prediction|, the F1/Dice denominator), so
    missed + extra equals the total error 1 - F1, bounded in [0, 1] — figure_2's error is
    exactly the complement of table_1's F1. empty is returned when the denominator is zero.
    Returns (missed, extra, total).
    """
    denom = 2 * tp + fp + fn
    if not denom:
        return empty, empty, empty
    return fn / denom, fp / denom, (fp + fn) / denom


def extract_leaves(processes_dict, group_id=None, exclude_keys=()):
    """Return list of (name, details_dict, group_id) for all leaf entries."""
    leaves = []
    for name, details in processes_dict.items():
        if name in exclude_keys:
            continue
        if not isinstance(details, dict):
            continue
        if "alt_names" in details:
            leaves.append((name, details, group_id))
        else:
            leaves.extend(extract_leaves(details, group_id=name, exclude_keys=exclude_keys))
    return leaves


def get_leaf_names(cat_name, cat_val, exclude_categories=("Disposal",), exclude_unspecified=False):
    """Return leaf process names for a category from the keywords hierarchy.

    exclude_unspecified drops catch-all 'Unspecified X' leaves (priority 1000) — use for
    leaf-level comparisons where matching a catch-all exactly would be unfair.
    """
    if exclude_categories and cat_name in exclude_categories:
        return []
    if isinstance(cat_val, dict) and "alt_names" in cat_val:
        return [cat_name]
    leaves = extract_leaves(cat_val, exclude_keys=exclude_categories)
    if exclude_unspecified:
        # 'Unspecified X' catch-alls are nested leaves, never top-level categories
        leaves = [(n, d, g) for n, d, g in leaves
                  if not (str(n).lower().startswith("unspecified") and d.get("priority") == 1000)]
    return [name for name, _, _ in leaves]


def build_secondary_category_lookup(keywords_dict):
    """Return (top_category_to_columns, column_secondary_categories, column_global_priority)."""
    top_category_to_columns = {}
    column_secondary_categories = {}
    column_global_priority = {}
    for top_cat, cat_val in keywords_dict.items():
        for name, details, _ in extract_leaves({top_cat: cat_val}):
            top_category_to_columns.setdefault(top_cat, []).append(name)
            if isinstance(details, dict):
                column_global_priority[name] = details.get("global_priority", 1)
                sc = details.get("secondary_category", [])
                if sc and isinstance(sc, list):
                    column_secondary_categories[name] = sc
    return top_category_to_columns, column_secondary_categories, column_global_priority


def apply_secondary_category_backfill(
    status_dict,
    column_secondary_categories,
    top_category_to_columns,
    column_global_priority,
    column_priority,
    ontology_resolve_fn=None,
    excluded_cols=(),
):
    """Backfill secondary categories: if a PRESENT process requests a secondary category
    that has no PRESENT process, mark the best fallback (unspecified-first) as PRESENT.

    ontology_resolve_fn(source_col, sec_cat, sec_cols) -> str | None: optional hook for
    ontology-based selection (used by step4). Returns the chosen column name, or None to
    fall back to unspecified-first heuristic.
    excluded_cols: columns cleared by exclude_if_any for this item — never backfilled,
    so the backfill can't resurrect a column an exclusion just removed.
    """
    present_cols = [c for c, v in status_dict.items() if v in PRESENT_STATUSES]
    for source_col in present_cols:
        for sec_cat in column_secondary_categories.get(source_col, []):
            sec_cols = top_category_to_columns.get(sec_cat, [])
            if not sec_cols or any(status_dict.get(c) in PRESENT_STATUSES for c in sec_cols):
                continue
            available = [c for c in sec_cols if c in status_dict and c not in excluded_cols]
            if not available:
                continue
            chosen = ontology_resolve_fn(source_col, sec_cat, sec_cols) if ontology_resolve_fn else None
            if chosen is None:
                # Prefer the category-level catch-all (e.g., "Unspecified Filtration")
                # before nested catch-alls (e.g., "Unspecified FFR").
                target_unspecified = f"Unspecified {sec_cat}".strip().lower()
                exact_unspecified = [
                    c for c in available
                    if str(c).strip().lower() == target_unspecified
                ]
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
    for token in ("PRESENT", "FUTURE", "PAST", "OFFSITE"):
        if token in tokens:
            return token
    return ""


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
    proc_cols = list({name for name, _, _ in extract_leaves(unitprocess_keywords)})
    left = cwns_mapping[["Place ID", "WDID", "Facility Name", "CWNS_ID", "FACILITY_ID"]]
    if target_facilities is not None:
        left = left[left["Place ID"].isin(target_facilities)]
    right = collapse_facility_processes(ca_cwns_df[["CWNS_ID"] + proc_cols], ["CWNS_ID"], [])
    merged = left.merge(right, on="CWNS_ID", how="inner", indicator="_cwns_merge")
    cwns_by_facility = collapse_facility_processes(
        merged, ["Place ID"], ["WDID", "Facility Name", "CWNS_ID", "FACILITY_ID", "_cwns_merge"]
    ).drop(columns=["CWNS_ID", "FACILITY_ID", "_cwns_merge"], errors="ignore").fillna("")
    return cwns_by_facility, merged


def normalize_order_no(value):
    # "R5-2007-0090", "r5 2007 0090" and "WQ 2007-0090" are one order written three ways
    text = re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()
    return re.sub(r"^(R\d{1,2}[A-Z]?)?WQ", "", text) or text


def same_order_no(a, b):
    """Whether two order numbers name the same order. None if either is missing.

    Containment, not equality, because the two sides are written at different levels of
    detail: a document's title block prints "2007-0090" where CIWQS carries the region
    ("R5-2007-0090"), and a general order enrollee's CIWQS order appends the enrollee number
    ("2014-0153-DWQ-R5348") to the order the document itself prints ("WQ-2014-0153-DWQ").
    Requiring equality called 28 such pairs superseded and left 13 facilities with no
    document at all.
    """
    if not a or not b:
        return None
    a, b = normalize_order_no(a), normalize_order_no(b)
    return a in b or b in a


_orders_in_force_cache = {}


def orders_in_force(as_of=None):
    """place_id -> set of Order_No values CIWQS listed for it as of a snapshot date.

    step2c unions each dated scrape into site_data_relevant.csv's as_of_dates, so the orders
    a facility held on a given date are the Order_No values on rows carrying that date. Read
    from the union rather than output/site_data/<date>/ because as_of_dates carries dates that
    have no snapshot directory.

    as_of=None uses the newest date present. Pass an explicit date to reconstruct the fleet as
    it stood then -- comparisons against CWNS 2022 want the 2022 permits, not today's.
    """
    key = as_of or ""
    if key not in _orders_in_force_cache:
        rel = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str, keep_default_na=False).fillna("")
        dates = set()
        for v in rel["as_of_dates"]:
            dates |= {d for d in str(v).split(";") if d}
        target = as_of or (max(dates) if dates else "")
        held = {}
        for _, row in rel.iterrows():
            if target and target not in str(row["as_of_dates"]).split(";"):
                continue
            order = str(row["Order_No"]).strip()
            if order:
                held.setdefault(normalize_id(row["Place ID"]), set()).add(order)
        _orders_in_force_cache[key] = held
    return _orders_in_force_cache[key]


def order_year(order):
    """Adoption year of an order number, 0 if unreadable.

    Handles "R5-2017-0085", "2014-0153-DWQ", "97-10-DWQ" (two-digit year), "05-025" and the
    old region-prefixed "5-00-080" form. Match the four-digit year first: a naive scan finds
    "92" inside "R9-2020-0191" and reads it as 1992.
    """
    t = re.sub(r"\s+", "", str(order).strip().upper())
    t = re.sub(r"^R\d{1,2}[A-Z]?[-\s]?", "", t)
    m = re.search(r"(?:19|20)\d{2}", t)
    if m:
        return int(m.group(0))
    t = re.sub(r"^\d[-\s]", "", t)          # bare region prefix, e.g. "5-00-080"
    m = re.match(r"(\d{2})\D", t + "-")
    if m:
        y = int(m.group(1))
        return 1900 + y if y >= 90 else 2000 + y
    return 0


def current_permit_mask(df, order_col="Order_No", doc_order_col="document_order_no", as_of=None,
                        content=None):
    """Per facility, keep only the documents belonging to the permits then in force.

    A facility accumulates documents across permit cycles -- CIWQS attaches superseded orders
    to the current order's page, and earlier snapshots contribute their own. Ranked preference
    rather than a hard filter, so a facility is never left with nothing:

      0. the document's own order number is one the facility held as of `as_of`
      1. no order number could be read from the document
      2. the order number is not one it held (superseded)

    Each facility keeps only its best available tier.

    The order set comes from orders_in_force(), NOT from each row's own `order_col`. Comparing
    a document to the Order_No sitting on its own row let 186 of 618 facilities keep documents
    from more than one order cycle (223 superseded documents): site_data_relevant carries a row
    per (facility, order), so a 2018 permit paired with its own 2018 Order_No scored tier 0
    against a facility whose current order is from 2024. Ukiah's 2018 "solar drying bed" was
    reaching a facility whose permit no longer mentions it. `order_col` is now only the
    fallback for facilities absent from the snapshot.

    Comparing to a set also preserves genuinely concurrent permits -- a plant covered by both
    its own order and a joint-authority order keeps both, because the snapshot lists both.

    `content`, when given, is a boolean Series marking rows that actually yielded processes.
    A document that extracted nothing carries no information, so it must not outrank an
    informative superseded one: JWPCP (260 MGD) and EchoWater (115 MGD) were being represented
    by a current permit with zero extracted processes while their superseded documents held
    14-25 each, leaving both facilities empty and pushed onto the regional fallback. Contentless
    rows are demoted below every informative tier and used only if nothing else exists.
    """
    held = orders_in_force(as_of)
    tiers = []
    for _, row in df.iterrows():
        doc = str(row.get(doc_order_col, "")).strip()
        if not doc:
            tiers.append(1)
            continue
        current = held.get(normalize_id(row["Place ID"])) or {str(row.get(order_col, "")).strip()}
        tiers.append(0 if any(same_order_no(doc, o) for o in current if o) else 2)
    tiers = pd.Series(tiers, index=df.index)
    if content is not None:
        tiers = tiers + (~content.reindex(df.index).fillna(False)).astype(int) * 3
    best = tiers.groupby(df["Place ID"]).transform("min")
    keep = tiers == best
    # Where every document is superseded (tier 2), the fallback would otherwise keep the whole
    # history. Facilities enrolled under a general order hit this routinely: CIWQS lists the
    # current general order, which no individual permit document prints. Keep only the newest
    # order's documents -- the best available proxy when nothing matches.
    fallback = keep & (tiers % 3 == 2)
    if fallback.any():
        years = df.loc[fallback, doc_order_col].map(order_year)
        newest = years.groupby(df.loc[fallback, "Place ID"]).transform("max")
        keep.loc[fallback] = years == newest
    return keep


def normalize_id(value):
    # Place IDs / CWNS_IDs show up as both "219530" and "219530.0" across files
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def add_county_and_sort(df, name_col, place_id_col=None, wdid_col=None, cwns_id_col=None):
    """Insert a 'County' column (right after name_col) and sort by (County, name_col).

    County comes from site_data_all (keyed on WDID); mapping_df bridges WDID to
    Place ID and CWNS_ID so any of the three keys can resolve a county. Rows with no
    county sort last.
    """
    site = pd.read_csv(SITE_DATA_ALL_CSV, dtype=str, keep_default_na=False).fillna("")
    county_by_wdid = {}
    for wdid, county in zip(site["WDID"].str.strip(), site["County"].str.strip()):
        if wdid and county and wdid not in county_by_wdid:
            county_by_wdid[wdid] = county

    county_by_place_id, county_by_cwns_id = {}, {}
    for _, row in mapping_df.iterrows():
        county = county_by_wdid.get(row["WDID"], "")
        place_id, cwns_id = normalize_id(row["Place ID"]), normalize_id(row["CWNS_ID"])
        if county and place_id:
            county_by_place_id.setdefault(place_id, county)
        if county and cwns_id:
            county_by_cwns_id.setdefault(cwns_id, county)

    def county_for(row):
        place_id = normalize_id(row[place_id_col]) if place_id_col else ""
        wdid = row[wdid_col].strip() if wdid_col else ""
        cwns_id = normalize_id(row[cwns_id_col]) if cwns_id_col else ""
        return county_by_place_id.get(place_id) or county_by_wdid.get(wdid) or county_by_cwns_id.get(cwns_id) or ""

    df.insert(df.columns.get_loc(name_col) + 1, "County", df.apply(county_for, axis=1))
    n_missing = (df["County"].str.strip() == "").sum()
    print(f"  add_county_and_sort: {len(df) - n_missing}/{len(df)} rows got a county ({n_missing} blank)")
    sort_key = lambda col: col.map(lambda v: "￿" if not str(v).strip() else str(v).lower())
    return df.sort_values(by=["County", name_col], key=sort_key).reset_index(drop=True)


def build_txt_jobs(txt_folder: str, facilities_information: str):
    txt_folder_path = Path(txt_folder)
    facilities_path = Path(facilities_information)
    facilities_df = pd.read_csv(facilities_path, dtype=str).fillna("")
    required_columns = {"Facility Name", "PDF_File"}
    missing_columns = required_columns.difference(set(facilities_df.columns))
    if missing_columns:
        raise ValueError(
            "--facilities_information is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    jobs = []
    for row_idx, row in facilities_df.iterrows():
        facility_name = str(row["Facility Name"]).strip()
        pdf_file_value = str(row["PDF_File"]).strip()

        if not facility_name or facility_name.lower() == "nan":
            continue
        if not pdf_file_value or pdf_file_value.lower() == "nan":
            continue

        txt_path = Path(pdf_file_value)
        if not txt_path.is_absolute():
            path_value = Path(pdf_file_value)
            path_value = txt_folder_path / path_value
            txt_name = path_value.with_suffix(".txt").name
            txt_path = txt_folder_path / txt_name

        if not txt_path.exists() or not txt_path.is_file():
            print(f"No txt for '{facility_name}': {txt_path.name}, skipping.")
            continue

        jobs.append((row_idx, txt_path, txt_path.name, facility_name))

    return jobs
