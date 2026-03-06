import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter
from rdflib import Graph, Namespace, RDFS

WATR = Namespace("urn:nawi-water-ontology#")
ontology = Graph()

# GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/main/water"
GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"


input_dir = Path('npdes_permits/output/2026-2-18/llm_search')
output_csv = Path('npdes_permits/output/llm_unit_processes_by_facility.csv')
output_json_dir = Path('npdes_permits/output/2026-2-18/llm_search_with_triggers')

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

site_df = pd.read_csv('npdes_permits/output/2026-2-18/site_data.csv', dtype=str).fillna('')
pdf_map = {row['PDF_File'].replace('.pdf', ''): {
    'PERMIT_NUMBER': row['NPDES_No'],
    'Agency': row['Agency'],
    'Facility_Name': row['Facility_Name']
} for _, row in site_df.iterrows()}

results = []
output_json_dir.mkdir(parents=True, exist_ok=True)


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

    if isinstance(json_data, dict) and isinstance(json_data.get('items'), list):
        output_json_data = {'items': [dict(item) if isinstance(item, dict) else item for item in json_data['items']]}
    else:
        output_json_data = {'items': []}

    result = {col: '' for col in columns}

    records = normalize_records(json_data)

    item_components = []
    for item_idx, item in enumerate(records):
        if str(get_field(item, 'Implementation') or '').lower() not in ["present", "planned"]:
            if item_idx < len(output_json_data['items']) and isinstance(output_json_data['items'][item_idx], dict):
                output_json_data['items'][item_idx]['trigger_process'] = []
            continue

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

        item_components.append((item_idx, components, item_role_counts))

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
    for item_idx, components, role_counts in item_components:
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
                for sibling_col in group_to_columns.get(group_id, []):
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

        item_triggers = sorted([col for col, value in item_result.items() if value == 'present'])
        if item_idx < len(output_json_data['items']) and isinstance(output_json_data['items'][item_idx], dict):
            output_json_data['items'][item_idx]['trigger_process'] = item_triggers

        for col, value in item_result.items():
            if value == 'present':
                result[col] = 'present'

    with open(output_json_dir / filename, 'w') as f:
        json.dump(output_json_data, f, indent=2)

    result.update(pdf_map.get(filename.replace('.json', ''), {}))
    results.append(result)

df = pd.DataFrame(results)
id_cols = ['PERMIT_NUMBER', 'Agency', 'Facility_Name']
cols = id_cols + [c for c in columns if c in df.columns]
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} facilities")
