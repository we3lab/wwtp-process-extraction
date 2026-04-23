import pandas as pd
import json
import os
import re
import argparse
from pathlib import Path
from collections import Counter
from rdflib import Graph, Namespace, RDFS

WATR = Namespace("urn:nawi-water-ontology#")
ontology = Graph()

# GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/main/water"
GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"

# TODO: handle secondary category
# Currently doesn't flag ontology "mistakes" e.g. identification of Sedimentation Basin without Process:Sedimentation

DEFAULT_INPUT_DIR = 'npdes_permits/output/2026-2-18/llm_extraction'
DEFAULT_OUTPUT_CSV = 'npdes_permits/output/llm_unit_processes_by_facility.csv'
DEFAULT_SITE_DATA_CSV = 'npdes_permits/output/2026-2-18/site_data.csv'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Postprocess LLM extraction JSON files into facility-level process matrix.'
    )
    parser.add_argument(
        '--input_dir',
        default=DEFAULT_INPUT_DIR,
        help=f'Folder containing LLM extraction JSON files (default: {DEFAULT_INPUT_DIR}).',
    )
    parser.add_argument(
        '--site_data_csv',
        default=DEFAULT_SITE_DATA_CSV,
        help=f'Site data CSV path with PDF/facility mapping (default: {DEFAULT_SITE_DATA_CSV}).',
    )
    parser.add_argument(
        '--output_csv',
        default=DEFAULT_OUTPUT_CSV,
        help=f'Output CSV path (default: {DEFAULT_OUTPUT_CSV}).',
    )
    return parser.parse_args()


args = parse_args()
input_dir = Path(args.input_dir)
output_csv = Path(args.output_csv)

with open('npdes_permits/data/unitprocess_keywords.json') as f:
    keywords = json.load(f)


def extract_leaves_with_context(processes_dict, top_category=None, group_id=None):
    leaves = []
    for name, details in processes_dict.items():
        if not isinstance(details, dict):
            continue
        current_top = name if top_category is None else top_category
        if 'alt_names' in details:
            leaves.append((name, details, group_id, current_top))
        else:
            leaves.extend(extract_leaves_with_context(details, current_top, name))
    return leaves


leaves = extract_leaves_with_context(keywords)
columns = [name for name, _, _, _ in leaves]
group_to_columns = {}
top_category_to_columns = {}
column_priority = {}
column_global_priority = {}
column_exclude_if_any = {}
column_secondary_categories = {}
column_trigger_clauses = {}
for name, _, group_id, top_category in leaves:
    if group_id:
        group_to_columns.setdefault(group_id, []).append(name)
    top_category_to_columns.setdefault(top_category, []).append(name)
for name, details, _, _ in leaves:
    column_priority[name] = details.get('priority', 1)
    column_global_priority[name] = details.get('global_priority', 1)
    exclude_tokens = details.get('exclude_if_any', [])
    if exclude_tokens and isinstance(exclude_tokens, list):
        column_exclude_if_any[name] = exclude_tokens
    secondary_categories = details.get('secondary_category', [])
    if secondary_categories and isinstance(secondary_categories, list):
        column_secondary_categories[name] = secondary_categories
    trigger = details.get('ontology_triggers')
    if trigger:
        column_trigger_clauses[name] = [trigger] if isinstance(trigger[0], str) else trigger

# ontology_triggers rules sorted by priority
trigger_rules = []
for name, details, group_id, _ in leaves:
    trigger = details.get('ontology_triggers')
    if not trigger:
        continue
    # Normalize: single list → list-of-lists
    clauses = [trigger] if isinstance(trigger[0], str) else trigger
    priority = details.get('priority', 1)
    trigger_rules.append((name, clauses, priority, group_id))
trigger_rules.sort(key=lambda r: r[2])

# ontology_triggers_multi rules: sorted by priority
multi_rules = []
for name, details, group_id, _ in leaves:
    me = details.get('ontology_triggers_multi')
    if me:
        for rule in (me if isinstance(me, list) else [me]):
            multi_rules.append((name, rule, group_id))
multi_rules.sort(key=lambda r: r[1]['priority'])

# Load ontology from GitHub
for filename in ["ontology.ttl", "equipment.ttl", "processtypes.ttl", "enumerationkinds.ttl", "substances.ttl"]:
    try:
        ontology.parse(f"{GITHUB_BASE}/{filename}", format="turtle")
    except Exception as e:
        print(f"Could not download {filename} from GitHub")

site_df = pd.read_csv(args.site_data_csv, dtype=str).fillna('')


def slugify(text):
    slug = re.sub(r'[^A-Za-z0-9]+', '_', str(text or '').strip())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug or 'facility'


def normalize_pdf_name(value):
    text = str(value or '').strip()
    if not text:
        return ''
    if text.lower().endswith('.pdf'):
        return text
    return f"{text}.pdf"


def parse_extraction_filename(filename):
    stem = Path(filename).stem
    # Expected step3 pattern: <pdf_stem>__<row_index_1based_4digits>__<facility_slug>.json
    match = re.match(r'^(?P<pdf_stem>.+)__(?P<row_idx>\d{4})__(?P<facility_slug>.+)$', stem)
    if not match:
        return {
            'pdf_file': normalize_pdf_name(stem),
            'row_idx_zero_based': None,
            'facility_slug': '',
        }
    return {
        'pdf_file': normalize_pdf_name(match.group('pdf_stem')),
        'row_idx_zero_based': int(match.group('row_idx')) - 1,
        'facility_slug': match.group('facility_slug'),
    }


def get_identity_from_row(row):
    return {
        'PERMIT_NUMBER': row.get('NPDES_No', '') or row.get('PERMIT_NUMBER', ''),
        'Agency': row.get('Agency', ''),
        'Facility_Name': row.get('Facility_Name', '') or row.get('facility_name', ''),
    }


site_records = site_df.to_dict(orient='records')
site_by_pdf_and_facility = {}
site_by_pdf = {}
for rec in site_records:
    pdf_name = normalize_pdf_name(rec.get('PDF_File', ''))
    facility = str(rec.get('Facility_Name', '')).strip()
    key = (pdf_name.lower(), facility.lower())
    if pdf_name:
        site_by_pdf_and_facility[key] = rec
        site_by_pdf.setdefault(pdf_name.lower(), rec)

results = []


def normalize_records(json_data):
    if not isinstance(json_data, dict):
        return []
    items = json_data.get('items')
    if isinstance(items, list):
        return items
    return []


def get_field(item, field_name):
    return (
        item.get(field_name)
        or item.get(field_name.lower())
        or item.get(field_name.upper())
    )


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


def normalize_location(value):
    text = str(value or '').strip().lower()
    if text in {'on-site', 'off-site', 'off_site', 'third-party'}:
        return text.replace('-', '_')
    return ''


def normalize_implementation(value, location=None):
    text = str(value or '').strip().lower()
    if text in {'off-site', 'off_site'}:
        return 'off_site'
    if text == 'present':
        location_text = normalize_location(location)
        if location_text in {'off_site', 'third-party'}:
            return 'off_site'
        return 'present'
    if text == 'third-party':
        return 'off_site'
    if text == 'planned':
        return 'future'
    if text == 'past':
        return 'past'
    return ''


def merge_implementation(existing_value, new_value):
    rank = {'': 0, 'past': 1, 'future': 2, 'present': 3, 'off_site': 4}
    existing = normalize_implementation(existing_value)
    new = normalize_implementation(new_value)
    return new if rank.get(new, 0) > rank.get(existing, 0) else existing


def normalize_component_name(component_type, name):
    text = str(name or '').strip()
    if not text:
        return ''
    prefix = f"{component_type}-"
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def ontology_labels(component_type, name, include_descendants=False):
    clean_name = normalize_component_name(component_type, name)
    if not clean_name:
        return set()

    uri_candidates = []
    if component_type == 'Process':
        uri_candidates.append(WATR[f"Process-{clean_name}"])
    if component_type == 'Substance':
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


def resolve_identity_for_extraction(filename):
    # 1) Parse step3 extraction filename pattern for exact row matching.
    parsed = parse_extraction_filename(filename)
    row_idx = parsed.get('row_idx_zero_based')
    if row_idx is not None and 0 <= row_idx < len(site_records):
        rec = site_records[row_idx]
        return get_identity_from_row(rec)

    # 2) Fallback by parsed PDF + facility slug.
    pdf_name = parsed.get('pdf_file', '')
    facility_slug = parsed.get('facility_slug', '')
    if pdf_name and facility_slug:
        matching = [
            rec for rec in site_records
            if normalize_pdf_name(rec.get('PDF_File', '')).lower() == pdf_name.lower()
            and slugify(rec.get('Facility_Name', '')) == facility_slug
        ]
        if matching:
            return get_identity_from_row(matching[0])

    # 3) Fallback by PDF stem only.
    if pdf_name:
        rec = site_by_pdf.get(pdf_name.lower())
        if rec:
            return get_identity_from_row(rec)

    return {
        'PERMIT_NUMBER': '',
        'Agency': '',
        'Facility_Name': '',
    }


for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue
    with open(input_dir / filename) as f:
        json_data = json.load(f)

    result = {col: '' for col in columns}

    records = normalize_records(json_data)
    items = records

    item_components = []
    for item in items:
        implementation = normalize_implementation(get_field(item, 'Implementation'), get_field(item, 'Location'))
        components = {'Process': set(), 'Role': set(), 'Equipment': set(), 'Substance': set()}
        item_role_counts = Counter()
        for comp_type in components:
            for clean in normalize_values(get_field(item, comp_type)):
                components[comp_type].add(clean)
                if comp_type == 'Role':
                    item_role_counts[clean] += 1

        for component_type in ['Process', 'Equipment', 'Substance']:
            expanded = set()
            for name in components[component_type]:
                expanded.update(
                    ontology_labels(
                        component_type,
                        name,
                        include_descendants=(component_type == 'Substance')
                    )
                )
            components[component_type] = expanded

        item_components.append((components, item_role_counts, implementation))

    # ontology_triggers: list = AND, list-of-lists = OR across clauses
    # within sibling group, higher-priority fires first; skip lower-priority siblings
    def matches_item(item, components):
        for comp_type in components:
            prefix = f"{comp_type}-"
            if item.startswith(prefix):
                return normalize_component_name(comp_type, item) in components[comp_type]
        if item in components.get('Substance', set()):
            return True
        return normalize_component_name('Substance', item) in components.get('Substance', set())

    # Pre-group multi rules once; applied per-item below.
    multi_rules_by_group = {}
    for col, rule, group_id in multi_rules:
        if not group_id:
            continue
        multi_rules_by_group.setdefault(group_id, []).append((col, rule))

    # Evaluate each item independently, then merge item-level positives into facility result.
    for components, role_counts, implementation in item_components:
        item_result = {col: '' for col in columns}

        for proc in components['Process']:
            if proc in item_result:
                item_result[proc] = 'present'

        fired_group_best_priority = {}
        for col, clauses, priority, group_id in trigger_rules:
            if col not in item_result:
                continue
            if group_id and priority > fired_group_best_priority.get(group_id, float('inf')):
                continue
            if any(all(matches_item(token, components) for token in clause) for clause in clauses):
                item_result[col] = 'present'
                if group_id:
                    fired_group_best_priority[group_id] = min(
                        fired_group_best_priority.get(group_id, priority),
                        priority,
                    )

        # Optional keyword-level exclusion rules from unitprocess_keywords.json
        # Example: {"exclude_if_any": ["Equipment-GritChamber"]}
        for col, exclusion_tokens in column_exclude_if_any.items():
            if item_result.get(col) != 'present':
                continue
            if any(matches_item(token, components) for token in exclusion_tokens):
                item_result[col] = ''

        for group_id, grouped_rules in multi_rules_by_group.items():
            match_col = None
            for col, rule in grouped_rules:
                if rule.get('Equipment') and not any(eq in components['Equipment'] for eq in rule['Equipment']):
                    continue
                if rule.get('Role') and not all(r in components['Role'] for r in rule['Role']):
                    continue
                if not all(
                    role_counts.get(r, 0) >= b.get('min', 0) and role_counts.get(r, 0) <= b.get('max', float('inf'))
                    for r, b in rule.get('role_counts', {}).items()
                ):
                    continue
                match_col = col
                break

            if match_col:
                sibling_cols = group_to_columns.get(group_id, [])
                existing_present = [c for c in sibling_cols if item_result.get(c) == 'present']
                if existing_present:
                    best_existing_priority = min(column_priority.get(c, 1) for c in existing_present)
                    if best_existing_priority <= column_priority.get(match_col, 1) and match_col not in existing_present:
                        continue

                for sibling_col in sibling_cols:
                    if sibling_col in item_result:
                        item_result[sibling_col] = ''
                item_result[match_col] = 'present'

        # Keep highest-priority sibling within this item only.
        for group_id, sibling_cols in group_to_columns.items():
            present_cols = [c for c in sibling_cols if item_result.get(c) == 'present']
            if len(present_cols) <= 1:
                continue
            best_priority = min(column_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_priority.get(col, 1) > best_priority:
                    item_result[col] = ''

        # Filtration resolution also applies within-item only.
        filtration_cols = top_category_to_columns.get('Filtration', [])
        present_filtration = [c for c in filtration_cols if item_result.get(c) == 'present']
        if len(present_filtration) > 1:
            best_filtration_priority = min(column_priority.get(c, 1) for c in present_filtration)
            for col in present_filtration:
                if column_priority.get(col, 1) > best_filtration_priority:
                    item_result[col] = ''

        # Global priority resolution (optional per keyword via `global_priority`).
        # Lower value means higher priority. This is used to demote generic
        # processes (e.g., unspecified categories) when a more specific trigger
        # is also present in the same item.
        present_cols = [c for c, value in item_result.items() if value == 'present']
        if len(present_cols) > 1:
            best_global_priority = min(column_global_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_global_priority.get(col, 1) > best_global_priority:
                    item_result[col] = ''

        # secondary_category backfill (best effort): if a triggered process requests
        # one or more secondary categories, try to mark at least one process from
        # each requested category using ontology trigger matching on the same item.
        present_cols = [c for c, value in item_result.items() if value == 'present']
        for source_col in present_cols:
            for secondary_category in column_secondary_categories.get(source_col, []):
                secondary_cols = top_category_to_columns.get(secondary_category, [])
                if not secondary_cols:
                    continue
                if any(item_result.get(c) == 'present' for c in secondary_cols):
                    continue

                matching_secondary_cols = []
                for candidate_col in secondary_cols:
                    clauses = column_trigger_clauses.get(candidate_col)
                    if not clauses:
                        continue
                    if any(all(matches_item(token, components) for token in clause) for clause in clauses):
                        matching_secondary_cols.append(candidate_col)

                if not matching_secondary_cols:
                    fallback_secondary_cols = [c for c in secondary_cols if c in item_result]
                    if not fallback_secondary_cols:
                        continue
                    unspecified_fallback = [c for c in fallback_secondary_cols if 'Unspecified' in c]
                    candidate_pool = unspecified_fallback or fallback_secondary_cols
                    chosen_col = min(
                        candidate_pool,
                        key=lambda c: (
                            column_priority.get(c, 1),
                            column_global_priority.get(c, 1),
                            c,
                        ),
                    )
                else:
                    chosen_col = min(
                        matching_secondary_cols,
                        key=lambda c: (
                            column_priority.get(c, 1),
                            column_global_priority.get(c, 1),
                            c,
                        ),
                    )
                item_result[chosen_col] = 'present'

        for col, value in item_result.items():
            if value == 'present':
                result[col] = merge_implementation(result.get(col, ''), implementation)

    result.update(resolve_identity_for_extraction(filename))
    results.append(result)

df = pd.DataFrame(results)
id_cols = ['PERMIT_NUMBER', 'Agency', 'Facility_Name']
cols = id_cols + [c for c in columns if c in df.columns]
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} facilities")
