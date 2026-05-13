import pandas as pd
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter
from rdflib import Graph, Namespace, RDFS

WATR = Namespace("urn:nawi-water-ontology#")
ontology = Graph()

GITHUB_BASE = (
    "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.utils import parse_status, extract_leaves, collapse_facility_processes, build_secondary_category_lookup, apply_secondary_category_backfill

DATE_STR = "2026-4-26"

input_dir = Path(f"npdes_permits/output/{DATE_STR}/llm_extraction")
output_csv = Path(f"npdes_permits/output/{DATE_STR}/llm_unit_processes_by_pdf.csv")
output_fac_csv = Path(f"npdes_permits/output/{DATE_STR}/llm_unit_processes_by_facility.csv")
site_data_csv = Path(f"npdes_permits/output/{DATE_STR}/site_data.csv")
output_json_dir = Path(f"npdes_permits/output/{DATE_STR}/llm_postprocess_ontology")
output_json_dir.mkdir(parents=True, exist_ok=True)

with open("npdes_permits/data/unitprocess_keywords.json") as f:
    keywords = json.load(f)

leaves = []
for top_cat, cat_val in keywords.items():
    for name, details, group_id in extract_leaves({top_cat: cat_val}, ignore_disposal=False):
        leaves.append((name, details, group_id, top_cat))
columns = [name for name, _, _, _ in leaves]
group_to_columns = {}
column_priority = {}
column_exclude_if_any = {}
column_trigger_clauses = {}
top_category_to_columns, column_secondary_categories, column_global_priority = \
    build_secondary_category_lookup(keywords)
for name, _, group_id, _ in leaves:
    if group_id:
        group_to_columns.setdefault(group_id, []).append(name)
for name, details, _, _ in leaves:
    column_priority[name] = details.get("priority", 1)
    exclude_tokens = details.get("exclude_if_any", [])
    if exclude_tokens and isinstance(exclude_tokens, list):
        column_exclude_if_any[name] = exclude_tokens
    trigger = details.get("ontology_triggers")
    if trigger:
        column_trigger_clauses[name] = [trigger] if isinstance(trigger[0], str) else trigger

# ontology_triggers rules sorted by priority
trigger_rules = []
for name, details, group_id, _ in leaves:
    trigger = details.get("ontology_triggers")
    if not trigger:
        continue
    # Normalize: single list → list-of-lists
    clauses = [trigger] if isinstance(trigger[0], str) else trigger
    priority = details.get("priority", 1)
    trigger_rules.append((name, clauses, priority, group_id))
trigger_rules.sort(key=lambda r: r[2])

# ontology_triggers_multi rules: sorted by priority
multi_rules = []
for name, details, group_id, _ in leaves:
    me = details.get("ontology_triggers_multi")
    if me:
        for rule in (me if isinstance(me, list) else [me]):
            multi_rules.append((name, rule, group_id))
multi_rules.sort(key=lambda r: r[1]["priority"])

# Load ontology from GitHub
for filename in [
    "ontology.ttl",
    "equipment.ttl",
    "processtypes.ttl",
    "enumerationkinds.ttl",
    "substances.ttl",
]:
    try:
        ontology.parse(f"{GITHUB_BASE}/{filename}", format="turtle")
    except Exception as e:
        print(f"Could not download {filename} from GitHub")

site_df = pd.read_csv(site_data_csv, dtype=str).fillna("")


def _norm_pdf(s):
    """Normalize a PDF stem for lookup: lowercase, spaces → underscores."""
    return s.lower().replace(" ", "_")


def _norm_fac(s):
    """Normalize a facility name for lookup: lowercase, spaces/slashes/hyphens → underscores."""
    return re.sub(r"[\s/\-]+", "_", s.lower()).strip("_")


def _row_info(row):
    return {
        "Place ID": row["Place ID"],
        "WDID": row["WDID"],
        "Order_No": row["Order_No"],
        "NPDES No.": row["NPDES No."],
        "Agency": row["Agency"],
        "Facility Name": row["Facility Name"],
    }


pdf_map = {
    _norm_pdf(row["PDF_File"].replace(".pdf", "")): _row_info(row)
    for _, row in site_df.iterrows()
}
facility_name_map = {
    _norm_fac(row["Facility Name"]): _row_info(row)
    for _, row in site_df.iterrows()
}


def resolve_identity(filename):
    stem = Path(filename).stem

    # Primary: {pdf_stem}__N__{facility_name} pattern — extract each part explicitly.
    m = re.match(r"^(.+)__\d+__(.+)$", stem)
    if m:
        pdf_stem, fac_part = m.group(1), m.group(2)
        if pdf_stem.lower().endswith(".pdf"):
            pdf_stem = pdf_stem[:-4]
        info = pdf_map.get(_norm_pdf(pdf_stem))
        if info:
            return info
        # PDF stem didn't match — try matching the facility name part instead.
        info = facility_name_map.get(_norm_fac(fac_part))
        if info:
            return info

    # Fallback: files named {pdf_stem}_{facility_name} with no __N__ separator.
    # Try progressively shorter underscore-delimited prefixes (longest first)
    # so the most specific PDF-stem match wins.
    norm_stem = _norm_pdf(stem)
    if norm_stem.endswith(".pdf"):
        norm_stem = norm_stem[:-4]
    parts = norm_stem.split("_")
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in pdf_map:
            return pdf_map[candidate]

    return {}


results = []
unmatched_files = []


def normalize_records(json_data):
    if not isinstance(json_data, dict):
        return []
    items = json_data.get("items")
    if isinstance(items, list):
        return items
    return []


def get_field(item, field_name):
    return item.get(field_name) or item.get(field_name.lower()) or item.get(field_name.upper())


def normalize_values(value):
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    normalized = []
    for entry in values:
        if not entry:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def apply_implementation(existing, impl_value, location=None):
    """Normalize impl+location to canonical status and merge with existing facility-level status."""
    text = str(impl_value or "").strip().lower().replace("-", "_")
    is_offsite = False

    if text in {"off_site", "third_party"}:
        new = "OFFSITE"
        is_offsite = True
    elif text == "present":
        new = "OFFSITE" if is_offsite else "PRESENT"
    elif text == "planned":
        new = "" if is_offsite else "FUTURE"
    elif text == "past":
        new = "" if is_offsite else "PAST"
    else:
        new = ""

    rank = {"": 0, "PAST": 1, "OFFSITE": 2, "FUTURE": 3, "PRESENT": 4}
    return new if rank.get(new, 0) > rank.get(existing, 0) else existing


def normalize_component_name(component_type, name):
    text = str(name or "").strip()
    if not text:
        return ""
    prefix = f"{component_type}-"
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def ontology_labels(component_type, name, include_descendants=False):
    clean_name = normalize_component_name(component_type, name)
    if not clean_name:
        return set()

    uri_candidates = []
    if component_type == "Process":
        uri_candidates.append(WATR[f"Process-{clean_name}"])
    if component_type == "Substance":
        uri_candidates.append(WATR[f"Substance-{clean_name}"])
    uri_candidates.append(WATR[clean_name])

    labels = {clean_name}
    for uri in uri_candidates:
        for ancestor_uri in ontology.transitive_objects(uri, RDFS.subClassOf):
            fragment = ancestor_uri.fragment
            if fragment:
                labels.add(normalize_component_name(component_type, fragment))

        if include_descendants:
            for descendant_uri in ontology.transitive_subjects(RDFS.subClassOf, uri):
                fragment = descendant_uri.fragment
                if fragment:
                    labels.add(normalize_component_name(component_type, fragment))

    return labels


for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue
    with open(input_dir / filename) as f:
        json_data = json.load(f)

    result = {col: "" for col in columns}

    records = normalize_records(json_data)
    output_json_data = {
        "items": [dict(item) if isinstance(item, dict) else item for item in records]
    }

    item_components = []
    for item_idx, item in enumerate(records):
        impl_value = get_field(item, "Implementation")
        impl_location = get_field(item, "Location")
        components = {
            "Process": set(),
            "Role": set(),
            "Equipment": set(),
            "Substance": set(),
        }
        item_role_counts = Counter()
        for comp_type in components:
            for clean in normalize_values(get_field(item, comp_type)):
                components[comp_type].add(clean)
                if comp_type == "Role":
                    item_role_counts[clean] += 1

        for component_type in ["Process", "Equipment", "Substance"]:
            expanded = set()
            for name in components[component_type]:
                expanded.update(
                    ontology_labels(
                        component_type,
                        name,
                        include_descendants=(component_type == "Substance"),
                    )
                )
            components[component_type] = expanded

        item_components.append((item_idx, components, item_role_counts, impl_value, impl_location))

    # ontology_triggers: list = AND, list-of-lists = OR across clauses
    # within sibling group, higher-priority fires first; skip lower-priority siblings
    def matches_item(item, components):
        for comp_type in components:
            prefix = f"{comp_type}-"
            if item.startswith(prefix):
                return normalize_component_name(comp_type, item) in components[comp_type]
        if item in components.get("Substance", set()):
            return True
        return normalize_component_name("Substance", item) in components.get("Substance", set())

    # Pre-group multi rules once; applied per-item below.
    multi_rules_by_group = {}
    for col, rule, group_id in multi_rules:
        if not group_id:
            continue
        multi_rules_by_group.setdefault(group_id, []).append((col, rule))

    # Evaluate each item independently, then merge item-level positives into facility result.
    for item_idx, components, role_counts, impl_value, impl_location in item_components:
        item_result = {col: "" for col in columns}

        for proc in components["Process"]:
            if proc in item_result:
                item_result[proc] = "PRESENT"

        fired_group_best_priority = {}
        for col, clauses, priority, group_id in trigger_rules:
            if col not in item_result:
                continue
            if group_id and priority > fired_group_best_priority.get(group_id, float("inf")):
                continue
            if any(all(matches_item(token, components) for token in clause) for clause in clauses):
                item_result[col] = "PRESENT"
                if group_id:
                    fired_group_best_priority[group_id] = min(
                        fired_group_best_priority.get(group_id, priority),
                        priority,
                    )

        # Optional keyword-level exclusion rules from unitprocess_keywords.json
        # Example: {"exclude_if_any": ["Equipment-GritChamber"]}
        for col, exclusion_tokens in column_exclude_if_any.items():
            if item_result.get(col) != "PRESENT":
                continue
            if any(matches_item(token, components) for token in exclusion_tokens):
                item_result[col] = ""

        for group_id, grouped_rules in multi_rules_by_group.items():
            match_col = None
            for col, rule in grouped_rules:
                if rule.get("Equipment") and not any(
                    eq in components["Equipment"] for eq in rule["Equipment"]
                ):
                    continue
                if rule.get("Role") and not all(r in components["Role"] for r in rule["Role"]):
                    continue
                if not all(
                    role_counts.get(r, 0) >= b.get("min", 0)
                    and role_counts.get(r, 0) <= b.get("max", float("inf"))
                    for r, b in rule.get("role_counts", {}).items()
                ):
                    continue
                match_col = col
                break

            if match_col:
                sibling_cols = group_to_columns.get(group_id, [])
                existing_present = [c for c in sibling_cols if item_result.get(c) == "PRESENT"]
                if existing_present:
                    best_existing_priority = min(
                        column_priority.get(c, 1) for c in existing_present
                    )
                    if (
                        best_existing_priority <= column_priority.get(match_col, 1)
                        and match_col not in existing_present
                    ):
                        continue

                for sibling_col in sibling_cols:
                    if sibling_col in item_result:
                        item_result[sibling_col] = ""
                item_result[match_col] = "PRESENT"

        # Keep highest-priority sibling within this item only.
        for group_id, sibling_cols in group_to_columns.items():
            present_cols = [c for c in sibling_cols if item_result.get(c) == "PRESENT"]
            if len(present_cols) <= 1:
                continue
            best_priority = min(column_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_priority.get(col, 1) > best_priority:
                    item_result[col] = ""

        # Filtration resolution also applies within-item only.
        filtration_cols = top_category_to_columns.get("Filtration", [])
        present_filtration = [c for c in filtration_cols if item_result.get(c) == "PRESENT"]
        if len(present_filtration) > 1:
            best_filtration_priority = min(column_priority.get(c, 1) for c in present_filtration)
            for col in present_filtration:
                if column_priority.get(col, 1) > best_filtration_priority:
                    item_result[col] = ""

        # Global priority resolution (optional per keyword via `global_priority`).
        # Lower value means higher priority. This is used to demote generic
        # processes (e.g., unspecified categories) when a more specific trigger
        # is also present in the same item.
        present_cols = [c for c, value in item_result.items() if value == "PRESENT"]
        if len(present_cols) > 1:
            best_global_priority = min(column_global_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_global_priority.get(col, 1) > best_global_priority:
                    item_result[col] = ""

        # secondary_category backfill (best effort): ontology trigger matching first,
        # then unspecified-first fallback via shared helper.
        def _ontology_resolve(source_col, sec_cat, sec_cols):
            matching = [
                c for c in sec_cols
                if column_trigger_clauses.get(c) and any(
                    all(matches_item(token, components) for token in clause)
                    for clause in column_trigger_clauses[c]
                )
            ]
            if matching:
                return min(matching, key=lambda c: (column_priority.get(c, 1), column_global_priority.get(c, 1), c))
            return None

        apply_secondary_category_backfill(
            item_result, column_secondary_categories, top_category_to_columns,
            column_global_priority, column_priority, ontology_resolve_fn=_ontology_resolve,
        )

        triggered = sorted(col for col, v in item_result.items() if v == "PRESENT")
        if item_idx < len(output_json_data["items"]) and isinstance(
            output_json_data["items"][item_idx], dict
        ):
            output_json_data["items"][item_idx]["trigger_process"] = triggered

        for col, value in item_result.items():
            if value == "PRESENT":
                result[col] = apply_implementation(result.get(col, ""), impl_value, impl_location)

    with open(output_json_dir / filename, "w") as f:
        json.dump(output_json_data, f, indent=2)

    identity = resolve_identity(filename)
    if not identity:
        unmatched_files.append(filename)
    result.update(identity)
    results.append(result)

df = pd.DataFrame(results)
id_cols = ["Place ID", "WDID", "Order_No", "NPDES No.", "Agency", "Facility Name"]
cols = id_cols + [c for c in columns if c in df.columns]
for c in cols:
    if c not in id_cols:
        df[c] = df[c].map(parse_status)
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} rows ({len(results) - len(unmatched_files)} matched, {len(unmatched_files)} unmatched)")

raw_df = pd.read_csv(output_csv, dtype=str).fillna("")
collapsed = collapse_facility_processes(
    raw_df,
    key_cols=["Place ID"],
    meta_cols=["WDID", "Order_No", "NPDES No.", "Agency", "Facility Name"],
)
collapsed.to_csv(output_fac_csv, index=False)
print(f"Collapsed {len(raw_df)} PDF rows → {len(collapsed)} facilities → llm_unit_processes_by_facility.csv")
if unmatched_files:
    print(f"\nNo facility match found in site_data.csv for {len(unmatched_files)} file(s):")
    for f in sorted(unmatched_files):
        print(f"  {f}")
