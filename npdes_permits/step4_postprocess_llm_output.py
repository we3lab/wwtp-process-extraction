import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter
from rdflib import Graph, Namespace, RDFS
from helpers.utils import extract_leaves

WATR = Namespace("urn:nawi-water-ontology#")
ontology = Graph()

# GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/main/water"
GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"


input_dir = Path('npdes_permits/LLM_extraction/output/temporary_llm_ontology_results')
output_csv = Path('npdes_permits/output/llm_unit_processes_by_facility.csv')

with open('npdes_permits/data/unitprocess_keywords.json') as f:
    keywords = json.load(f)

leaves = extract_leaves(keywords)
columns = [name for name, _, _ in leaves]

# ontology_triggers rules sorted by priority
trigger_rules = []
for name, details, group_id in leaves:
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
for name, details, _ in leaves:
    me = details.get('ontology_triggers_multi')
    if me:
        for rule in (me if isinstance(me, list) else [me]):
            multi_rules.append((name, rule))
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
for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue
    with open(input_dir / filename) as f:
        json_data = json.load(f)

    result = {col: '' for col in columns}

    items = [i for i in json_data if i['Implementation'] in ["present", "planned"]]
    ontology_components = {'Process': set(), 'Role': set(), 'Equipment': set(), 'Substance': set()}
    role_counts = Counter()

    for item in items:
        for ontology_component in ontology_components.keys():
            for value in item.get(ontology_component) or []:
                clean = value.replace(f"{ontology_component}-", "")
                ontology_components[ontology_component].add(clean)
                if ontology_component == 'Role':
                    role_counts[clean] += 1

    # Expand each component set with ontology parent classes (via rdfs:subClassOf)
    for component_type in ['Process', 'Equipment', 'Substance']:
        uri_prefix = "Process-" if component_type == 'Process' else ""
        expanded = set()
        for name in ontology_components[component_type]:
            for parent_uri in ontology.transitive_objects(WATR[f"{uri_prefix}{name}"], RDFS.subClassOf):
                expanded.add(parent_uri.fragment.removeprefix(uri_prefix))
        ontology_components[component_type] = expanded

    # Mark columns for any directly matched or parent processes
    for proc in ontology_components['Process']:
        if proc in result:
            result[proc] = 'present'

    # ontology_triggers: list = AND, list-of-lists = OR across clauses
    # within sibling group, higher-priority fires first; skip lower-priority siblings
    def matches_item(item):
        for comp_type in ontology_components:
            prefix = f"{comp_type}-"
            if item.startswith(prefix):
                return item[len(prefix):] in ontology_components[comp_type]
        return False

    fired_groups = set()
    for col, clauses, priority, group_id in trigger_rules:
        if col not in result:
            continue
        if group_id and group_id in fired_groups:
            continue
        if any(all(matches_item(item) for item in clause) for clause in clauses):
            result[col] = 'present'
            if group_id:
                fired_groups.add(group_id)

    # ontology_triggers_multi: first matching rule by priority; skip if any column already set
    multi_cols = {name for name, _ in multi_rules}
    if not any(result.get(c) == 'present' for c in multi_cols):
        for col, rule in multi_rules:
            if rule.get('Equipment') and not any(eq in ontology_components['Equipment'] for eq in rule['Equipment']):
                continue
            if rule.get('Role') and not all(r in ontology_components['Role'] for r in rule['Role']):
                continue
            if not all(
                role_counts.get(r, 0) >= b.get('min', 0) and role_counts.get(r, 0) <= b.get('max', float('inf'))
                for r, b in rule.get('role_counts', {}).items()
            ):
                continue
            result[col] = 'present'
            break

    result.update(pdf_map.get(filename.replace('.json', ''), {}))
    results.append(result)

df = pd.DataFrame(results)
id_cols = ['PERMIT_NUMBER', 'Agency', 'Facility_Name']
cols = id_cols + [c for c in columns if c in df.columns]
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} facilities")
