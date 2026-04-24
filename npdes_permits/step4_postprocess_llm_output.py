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

GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.utils import parse_status, extract_leaves

input_dir       = Path('npdes_permits/output/2026-2-18/llm_search_ontology')
output_csv      = Path('npdes_permits/output/llm_unit_processes_by_facility.csv')
site_data_csv   = Path('npdes_permits/output/2026-2-18/site_data.csv')
output_json_dir = Path('npdes_permits/output/2026-2-18/llm_postprocess_ontology')
output_json_dir.mkdir(parents=True, exist_ok=True)

with open('npdes_permits/data/unitprocess_keywords.json') as f:
    keywords = json.load(f)

leaves = []
for top_cat, cat_val in keywords.items():
    for name, details, group_id in extract_leaves({top_cat: cat_val}, ignore_disposal=False):
        leaves.append((name, details, group_id, top_cat))
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

site_df = pd.read_csv(site_data_csv, dtype=str).fillna('')
pdf_map = {row['PDF_File'].replace('.pdf', '').lower(): {
    'PERMIT_NUMBER': row['NPDES_No'],
    'Agency': row['Agency'],
    'Facility_Name': row['Facility_Name'],
} for _, row in site_df.iterrows()}


def resolve_identity(filename):
    m = re.match(r'^(.+)__\d{4}__.+$', Path(filename).stem)
    pdf_stem = m.group(1) if m else Path(filename).stem
    if pdf_stem.lower().endswith('.pdf'):  # strip spurious .pdf embedded in stem
        pdf_stem = pdf_stem[:-4]
    return pdf_map.get(pdf_stem.lower(), {})


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
    t = str(value or '').strip().lower().replace('-', '_')
    if t in {'off_site', 'third_party'}:  # third_party in legacy JSON → off_site
        return 'off_site'
    if t == 'on_site':
        return 'on_site'
    return ''


def apply_implementation(existing, impl_value, location=None):
    """Normalize impl+location to canonical status and merge with existing facility-level status."""
    text = str(impl_value or '').strip().lower().replace('-', '_')
    location_text = normalize_location(location)
    is_offsite = location_text == 'off_site'

    if text in {'off_site', 'third_party'}:
        new = 'OFFSITE'
    elif text == 'present':
        new = 'OFFSITE' if is_offsite else 'PRESENT'
    elif text == 'planned':
        new = '' if is_offsite else 'FUTURE'
    elif text == 'past':
        new = '' if is_offsite else 'PAST'
    else:
        new = ''

    rank = {'': 0, 'PAST': 1, 'OFFSITE': 2, 'FUTURE': 3, 'PRESENT': 4}
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


for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue
    with open(input_dir / filename) as f:
        json_data = json.load(f)

    result = {col: '' for col in columns}

    records = normalize_records(json_data)
    output_json_data = {'items': [dict(item) if isinstance(item, dict) else item for item in records]}

    item_components = []
    for item_idx, item in enumerate(records):
        impl_value = get_field(item, 'Implementation')
        impl_location = get_field(item, 'Location')
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

        item_components.append((item_idx, components, item_role_counts, impl_value, impl_location))

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
    for item_idx, components, role_counts, impl_value, impl_location in item_components:
        item_result = {col: '' for col in columns}

        for proc in components['Process']:
            if proc in item_result:
                item_result[proc] = 'PRESENT'

        fired_group_best_priority = {}
        for col, clauses, priority, group_id in trigger_rules:
            if col not in item_result:
                continue
            if group_id and priority > fired_group_best_priority.get(group_id, float('inf')):
                continue
            if any(all(matches_item(token, components) for token in clause) for clause in clauses):
                item_result[col] = 'PRESENT'
                if group_id:
                    fired_group_best_priority[group_id] = min(
                        fired_group_best_priority.get(group_id, priority),
                        priority,
                    )

        # Optional keyword-level exclusion rules from unitprocess_keywords.json
        # Example: {"exclude_if_any": ["Equipment-GritChamber"]}
        for col, exclusion_tokens in column_exclude_if_any.items():
            if item_result.get(col) != 'PRESENT':
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
                existing_present = [c for c in sibling_cols if item_result.get(c) == 'PRESENT']
                if existing_present:
                    best_existing_priority = min(column_priority.get(c, 1) for c in existing_present)
                    if best_existing_priority <= column_priority.get(match_col, 1) and match_col not in existing_present:
                        continue

                for sibling_col in sibling_cols:
                    if sibling_col in item_result:
                        item_result[sibling_col] = ''
                item_result[match_col] = 'PRESENT'

        # Keep highest-priority sibling within this item only.
        for group_id, sibling_cols in group_to_columns.items():
            present_cols = [c for c in sibling_cols if item_result.get(c) == 'PRESENT']
            if len(present_cols) <= 1:
                continue
            best_priority = min(column_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_priority.get(col, 1) > best_priority:
                    item_result[col] = ''

        # Filtration resolution also applies within-item only.
        filtration_cols = top_category_to_columns.get('Filtration', [])
        present_filtration = [c for c in filtration_cols if item_result.get(c) == 'PRESENT']
        if len(present_filtration) > 1:
            best_filtration_priority = min(column_priority.get(c, 1) for c in present_filtration)
            for col in present_filtration:
                if column_priority.get(col, 1) > best_filtration_priority:
                    item_result[col] = ''

        # Global priority resolution (optional per keyword via `global_priority`).
        # Lower value means higher priority. This is used to demote generic
        # processes (e.g., unspecified categories) when a more specific trigger
        # is also present in the same item.
        present_cols = [c for c, value in item_result.items() if value == 'PRESENT']
        if len(present_cols) > 1:
            best_global_priority = min(column_global_priority.get(c, 1) for c in present_cols)
            for col in present_cols:
                if column_global_priority.get(col, 1) > best_global_priority:
                    item_result[col] = ''

        # secondary_category backfill (best effort): if a triggered process requests
        # one or more secondary categories, try to mark at least one process from
        # each requested category using ontology trigger matching on the same item.
        present_cols = [c for c, value in item_result.items() if value == 'PRESENT']
        for source_col in present_cols:
            for secondary_category in column_secondary_categories.get(source_col, []):
                secondary_cols = top_category_to_columns.get(secondary_category, [])
                if not secondary_cols:
                    continue
                if any(item_result.get(c) == 'PRESENT' for c in secondary_cols):
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
                item_result[chosen_col] = 'PRESENT'

        triggered = sorted(col for col, v in item_result.items() if v == 'PRESENT')
        if item_idx < len(output_json_data['items']) and isinstance(output_json_data['items'][item_idx], dict):
            output_json_data['items'][item_idx]['trigger_process'] = triggered

        for col, value in item_result.items():
            if value == 'PRESENT':
                result[col] = apply_implementation(result.get(col, ''), impl_value, impl_location)

    with open(output_json_dir / filename, 'w') as f:
        json.dump(output_json_data, f, indent=2)

    result.update(resolve_identity(filename))
    results.append(result)

df = pd.DataFrame(results)
id_cols = ['PERMIT_NUMBER', 'Agency', 'Facility_Name']
cols = id_cols + [c for c in columns if c in df.columns]
for c in cols:
    if c not in id_cols:
        df[c] = df[c].map(parse_status)
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} facilities")
