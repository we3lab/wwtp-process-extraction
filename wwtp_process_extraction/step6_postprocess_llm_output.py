import pandas as pd
import json
import os
from functools import partial
from pathlib import Path
from collections import Counter
from rdflib import RDF, RDFS

from helpers.ontology_to_txt import load_ontology, hasprocess_fragments, WATR
from helpers.utils import (
    parse_status, extract_leaves, collapse_facility_processes, build_secondary_category_lookup,
    apply_secondary_category_backfill, keep_best_priority, add_county_and_sort, select_json_per_place_id,
    current_permit_mask, unitprocess_keywords, OUTPUT_DIR, LLM_EXTRACTION_DIR, MANUAL_CSV,
    SITE_DATA_RELEVANT_CSV, STATUS_TOKENS, STATUS_RANK,
)

ID_COLS = ["Place ID", "WDID", "Order_No", "NPDES No.", "Agency", "Facility Name", "PDF_File",
           "document_order_no"]

INPUT_DIR = LLM_EXTRACTION_DIR / "ontology-based_gpt-5-mini" # Full CA dataset run
PDF_LLM_CSV = OUTPUT_DIR / "unit_processes_by_pdf_llm.csv"
FACILITY_LLM_CSV = OUTPUT_DIR / "unit_processes_by_facility_llm.csv"
POSTPROCESS_DIR_NAME = "ontology_postprocess"  # postprocessed JSONs, in a subfolder of each run dir
RUN_DIR_PREFIXES = {"ontology-based_": "Ontology", "list-based_": "List"}
OFFSITE_WORDS = {"off_site", "third_party", "offsite"}

leaves = extract_leaves(unitprocess_keywords)
columns = [name for name, _, _ in leaves]
group_to_columns = {}
column_priority = {}
column_exclude_if_any = {}
column_trigger_clauses = {}
# ontology_triggers rules sorted by priority
trigger_rules = []
# ontology_triggers_multi: role counts aggregated facility-wide across matching items, so a
# config split across separate basins still fires. List specific reactor classes (not generic
# Reactor/Tank) so an AnaerobicDigester never matches and can't leak its Anaerobic role.
facility_multi_rules = []
top_category_to_columns, column_secondary_categories, column_global_priority = \
    build_secondary_category_lookup(unitprocess_keywords)
for name, details, group_id in leaves:
    if group_id:
        group_to_columns.setdefault(group_id, []).append(name)
    priority = details.get("priority", 1)
    column_priority[name] = priority
    exclude_tokens = details.get("exclude_if_any", [])
    if exclude_tokens:
        column_exclude_if_any[name] = exclude_tokens
    trigger = details.get("ontology_triggers")
    if trigger:
        # Normalize: single list → list-of-lists
        clauses = [trigger] if isinstance(trigger[0], str) else trigger
        column_trigger_clauses[name] = clauses
        trigger_rules.append((name, clauses, priority, group_id))
    multi_rules = details.get("ontology_triggers_multi")
    if multi_rules:
        for rule in (multi_rules if isinstance(multi_rules, list) else [multi_rules]):
            facility_multi_rules.append((name, rule))
trigger_rules.sort(key=lambda r: r[2])
facility_multi_rules.sort(key=lambda r: r[1].get("priority", 1))

# Load ontology from Zenodo
ontology = load_ontology()

# Direct (own-class only) hasProcess fragments, keyed by equipment class fragment.
equipment_own_processes = {
    cls.fragment: fragments
    for cls in ontology.subjects(RDF.type, WATR.Class)
    if (fragments := hasprocess_fragments(ontology, cls))
}


def normalize_pdf_name(s):
    return s.lower().replace(" ", "_")


def model_run_dirs():
    """(dir_path, method_label, model_label) for each ontology-based_/list-based_ run dir."""
    return [
        (dir_path, method_label, dir_path.name[len(prefix):])
        for dir_path in sorted(LLM_EXTRACTION_DIR.iterdir()) if dir_path.is_dir()
        for prefix, method_label in RUN_DIR_PREFIXES.items() if dir_path.name.startswith(prefix)
    ]


def normalize_values(value):
    """A JSON field (string, list or None) as a list of non-blank stripped strings."""
    values = value if isinstance(value, list) else [value]
    return [str(entry).strip() for entry in values if entry and str(entry).strip()]


def apply_implementation(existing, impl_value, location=None):
    text = str(impl_value or "").strip().lower().replace("-", "_")
    is_offsite = str(location or "").strip().lower().replace("-", "_") in OFFSITE_WORDS

    if text in OFFSITE_WORDS:
        new = "OFFSITE"
    elif text == "present":
        new = "OFFSITE" if is_offsite else "PRESENT"
    elif text == "future":
        new = "" if is_offsite else "FUTURE"
    elif text == "past":
        new = "" if is_offsite else "PAST"
    else:
        new = ""

    return new if STATUS_RANK[new] > STATUS_RANK[existing] else existing


def normalize_component_name(component_type, name):
    return str(name or "").strip().removeprefix(f"{component_type}-")


def ontology_labels(component_type, name):
    """Returns own label + rdfs:subClassOf ancestors."""
    clean_name = normalize_component_name(component_type, name)
    uri_candidates = [WATR[clean_name]]
    if component_type in ("Process", "Substance"):
        uri_candidates.insert(0, WATR[f"{component_type}-{clean_name}"])

    labels = {clean_name}
    for uri in uri_candidates:
        for ancestor_uri in ontology.transitive_objects(uri, RDFS.subClassOf):
            labels.add(normalize_component_name(component_type, ancestor_uri.fragment))

    return labels


def matches_item(item, components):
    for comp_type in components:
        prefix = f"{comp_type}-"
        if item.startswith(prefix):
            return normalize_component_name(comp_type, item) in components[comp_type]
    if item in components.get("Substance", set()):
        return True
    return normalize_component_name("Substance", item) in components.get("Substance", set())


def clauses_match(clauses, components):
    return any(all(matches_item(token, components) for token in clause) for clause in clauses)


def resolve_secondary_by_ontology(components, excluded_cols, secondary_cols):
    """ontology_resolve_fn hook for apply_secondary_category_backfill."""
    matching = [
        c for c in secondary_cols
        if c not in excluded_cols and c in column_trigger_clauses
        and clauses_match(column_trigger_clauses[c], components)
    ]
    if matching:
        return min(matching, key=lambda c: (column_priority.get(c, 1), column_global_priority.get(c, 1), c))
    return None


def process_json_to_unit_process_dict(json_data, output_json_path=None):
    """Run ontology mapping on a single JSON extraction result. Returns {col: status} dict."""
    result = {col: "" for col in columns}
    records = json_data["items"]
    if output_json_path is not None:
        output_json_data = {"items": [dict(item) for item in records]}

    item_components = []
    for item_idx, item in enumerate(records):
        impl_value = item.get("Implementation")
        impl_location = item.get("Location")
        components = {
            "Process": set(),
            "Role": set(),
            "Equipment": set(),
            "Substance": set(),
        }
        item_role_counts = Counter()
        for comp_type in components:
            for clean in normalize_values(item.get(comp_type)):
                components[comp_type].add(clean)
                if comp_type == "Role":
                    item_role_counts[clean] += 1

        for component_type in ["Process", "Equipment", "Substance"]:
            expanded = set()
            for name in components[component_type]:
                expanded.update(ontology_labels(component_type, name))
            components[component_type] = expanded

        # Inject each equipment's own + inherited hasProcess (per the ontology's SHACL
        # shapes), then expand those Process fragments' own ancestry too.
        implied_processes = set()
        for equipment_name in components["Equipment"]:
            # transitive_objects includes the class itself
            for cls in ontology.transitive_objects(WATR[equipment_name], RDFS.subClassOf):
                implied_processes |= equipment_own_processes.get(cls.fragment, set())
        for proc_fragment in implied_processes:
            components["Process"].update(ontology_labels("Process", proc_fragment))

        item_components.append((item_idx, components, item_role_counts, impl_value, impl_location))

    for item_idx, components, role_counts, impl_value, impl_location in item_components:
        item_result = {col: "" for col in columns}

        for proc in components["Process"]:
            if proc in item_result:
                item_result[proc] = "PRESENT"

        fired_group_best_priority = {}
        for col, clauses, priority, group_id in trigger_rules:
            if group_id and priority > fired_group_best_priority.get(group_id, float("inf")):
                continue
            if clauses_match(clauses, components):
                item_result[col] = "PRESENT"
                if group_id:
                    # rules are sorted by priority, so the first to fire is the group's best
                    fired_group_best_priority.setdefault(group_id, priority)

        # exclude_if_any: clear a column if the item has any listed token (e.g. Equipment-GritChamber)
        excluded_cols = set()
        for col, exclusion_tokens in column_exclude_if_any.items():
            if item_result.get(col) != "PRESENT":
                continue
            if any(matches_item(token, components) for token in exclusion_tokens):
                item_result[col] = ""
                excluded_cols.add(col)

        for sibling_cols in group_to_columns.values():
            keep_best_priority(item_result, sibling_cols, column_priority)

        # Only the best-priority filtration column survives
        keep_best_priority(item_result, top_category_to_columns.get("Filtration", []), column_priority)

        # global_priority (lower wins): drop generic columns (e.g. Unspecified X) when a specific one fired
        keep_best_priority(item_result, list(item_result), column_global_priority)

        # Fill missing secondary categories: ontology triggers first, else the Unspecified fallback
        apply_secondary_category_backfill(
            item_result, column_secondary_categories, top_category_to_columns,
            column_global_priority, column_priority,
            ontology_resolve_fn=partial(resolve_secondary_by_ontology, components, excluded_cols),
            excluded_cols=excluded_cols,
        )

        if output_json_path is not None:
            output_json_data["items"][item_idx]["trigger_process"] = sorted(
                col for col, v in item_result.items() if v == "PRESENT")

        for col, value in item_result.items():
            if value == "PRESENT":
                result[col] = apply_implementation(result[col], impl_value, impl_location)

    # Full facility-scoped multi-matching rules aggregate the role counts across matching items
    for col, rule in facility_multi_rules:
        eq_set = set(rule.get("Equipment", []))
        matching = [
            (components, role_counts, impl_value, impl_location)
            for _, components, role_counts, impl_value, impl_location in item_components
            if not eq_set or (components["Equipment"] & eq_set)
        ]
        exclusion_tokens = column_exclude_if_any.get(col, [])
        if any(matches_item(token, comps) for comps, _, _, _ in matching for token in exclusion_tokens):
            continue
        role_totals = Counter()
        status = ""
        for comps, role_counts, impl_value, impl_location in matching:
            role_totals.update(role_counts)
            status = apply_implementation(status, impl_value, impl_location)
        counts_ok = all(
            bounds.get("min", 0) <= role_totals[role] <= bounds.get("max", float("inf"))
            for role, bounds in rule.get("role_counts", {}).items()
        )
        if counts_ok and STATUS_RANK[status] > STATUS_RANK[result[col]]:
            result[col] = status

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json_data, f, indent=2)

    return result


def process_list_based_json(json_data):
    """Map list-based LLM output directly to unit process columns.

    List-based items use Process names that are leaf keys from the unit process
    list, so no ontology trigger resolution is needed.
    """
    result = {col: "" for col in columns}
    for item in json_data["items"]:
        impl_value = item.get("Implementation")
        impl_location = item.get("Location")
        for proc in normalize_values(item.get("Process")):
            if proc in result:
                result[proc] = apply_implementation(result[proc], impl_value, impl_location)
    return result


def main():
    site_df = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")
    pdf_map = {}
    for _, row in site_df.iterrows():
        # Use Path.stem so double-extension files (*.pdf.pdf) map to the same stem
        # the LLM JSON generator uses when naming output files.
        key = normalize_pdf_name(Path(row["PDF_File"]).stem)
        pdf_map.setdefault(key, []).append({c: row[c] for c in ID_COLS})

    pdf_stems_by_length = sorted(pdf_map, key=len, reverse=True)

    # ── full dataset ────────────────────────────────────────────────────
    results = []
    unmatched_files = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".json"):
            continue
        with open(INPUT_DIR / filename) as f:
            json_data = json.load(f)

        result = process_json_to_unit_process_dict(json_data, output_json_path=INPUT_DIR / POSTPROCESS_DIR_NAME / filename)

        identity = {}
        norm_stem = normalize_pdf_name(Path(filename).stem)
        for pdf_stem in pdf_stems_by_length:
            if norm_stem == pdf_stem or norm_stem.startswith(pdf_stem + "_"):
                rows = pdf_map[pdf_stem]
                if len(rows) == 1:
                    identity = rows[0]
                else:
                    # multi-facility PDF: suffix is the Place ID
                    place_id_suffix = norm_stem[len(pdf_stem) + 1:]
                    matching = [r for r in rows if r["Place ID"] == place_id_suffix]
                    identity = matching[0] if matching else rows[0]
                break
        if not identity:
            # No row in site_data_relevant claims this document (a superseded order, or a place
            # since filtered out). Without a Place ID it cannot be attributed to any facility,
            # and keeping it would collapse into a phantom facility with no identity at all.
            unmatched_files.append(filename)
            continue
        result.update(identity)
        results.append(result)

    raw_df = pd.DataFrame(results)[ID_COLS + columns]
    for c in columns:
        raw_df[c] = raw_df[c].map(parse_status)
    print(f"Matched {len(results)} documents ({len(unmatched_files)} unmatched files skipped)")

    # Collapse only the current permit. Unioning across cycles reads a process out of a
    # superseded order as if the facility still ran it.
    has_content = raw_df[columns].isin(STATUS_TOKENS).any(axis=1)
    current = current_permit_mask(raw_df, content=has_content)
    print(f"Current-permit documents: {int(current.sum())} of {len(raw_df)} "
          f"({int((~current).sum())} superseded rows excluded from the facility collapse)")
    collapsed = collapse_facility_processes(
        raw_df[current],
        key_cols=["Place ID"],
        meta_cols=[c for c in ID_COLS if c != "Place ID"],
    )
    collapsed = add_county_and_sort(collapsed, "Facility Name", place_id_col="Place ID", wdid_col="WDID")
    collapsed.to_csv(FACILITY_LLM_CSV, index=False)
    print(f"Collapsed {int(current.sum())} PDF rows → {len(collapsed)} facilities → unit_processes_by_facility_llm.csv")
    add_county_and_sort(raw_df, "Facility Name", place_id_col="Place ID", wdid_col="WDID").to_csv(PDF_LLM_CSV, index=False)
    if unmatched_files:
        print(f"\nNo facility match found in site_data_relevant for {len(unmatched_files)} file(s):")
        for f in sorted(unmatched_files):
            print(f"  {f}")


def build_model_comparison():
    """Manual-vs-prediction long table built in memory from raw JSONs + manual CSV.

    Benchmark facilities = the manual CSV's Place IDs; predictions are regenerated
    fresh from every model dir. Replaces the model_comparison_all.csv intermediate."""
    manual_rows = pd.read_csv(MANUAL_CSV, dtype=str)
    up_columns = [c for c in manual_rows.columns if c not in {"Method", "Model", "PDF_File"}]
    manual_rows["Method"] = "Manual Read"
    # Descriptive facility columns vs unit-process status columns
    meta_cols = [c for c in ["Agency", "Facility Name", "Place ID", "NPDES No."] if c in up_columns]
    proc_columns = [c for c in up_columns if c not in meta_cols]
    # The comparison only covers the benchmark (manually-read) facilities. The default
    # ontology-based_gpt-5-mini folder also holds the full CA set, so filter each model to these.
    benchmark_pids = set(manual_rows["Place ID"])
    # Some facilities have multiple permit-document JSONs (an original + later
    # modifications) sharing one Place ID; pick the document the manual labels
    # were actually read from (see select_json_per_place_id).
    pdf_stem_by_place_id = manual_rows.set_index("Place ID")["PDF_File"].to_dict()

    prediction_rows = []
    for dir_path, method_label, model_label in model_run_dirs():
        postprocess_dir = dir_path / POSTPROCESS_DIR_NAME
        for place_id, json_file in select_json_per_place_id(dir_path, benchmark_pids, pdf_stem_by_place_id).items():
            with open(json_file) as f:
                json_data = json.load(f)

            if method_label == "Ontology":
                unit_process_result = process_json_to_unit_process_dict(
                    json_data, output_json_path=postprocess_dir / json_file.name
                )
            else:
                unit_process_result = process_list_based_json(json_data)
            row = {"Method": method_label, "Model": model_label, "Place ID": place_id}
            for col in proc_columns:
                val = unit_process_result.get(col, "")
                row[col] = val if val else float("nan")
            prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows, columns=["Method", "Model"] + up_columns + ["PDF_File"])

    # Backfill the remaining facility metadata (and PDF_File) from manual rows by Place ID
    backfill_cols = [c for c in meta_cols if c != "Place ID"] + ["PDF_File"]
    meta_by_pid = manual_rows.set_index("Place ID")[backfill_cols]
    pred_df[backfill_cols] = meta_by_pid.reindex(pred_df["Place ID"]).values

    combined_df = pd.concat([manual_rows, pred_df], ignore_index=True)
    return combined_df


if __name__ == "__main__":
    main()
