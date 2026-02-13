import pandas as pd
import json
import os
from pathlib import Path
from collections import Counter
from rdflib import Graph, Namespace
from helpers.utils import get_all_keys

WATR = Namespace("urn:nawi-water-ontology#")

# GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/main/water"
GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"

def normalize(name):
    return name.lower().replace('-', '').replace(' ', '').replace('process', '').replace('role', '')


def extract_process_rules(processes_dict):
    """Return dict of leaf process name -> details (for inference_* etc.)."""
    out = {}
    for name, details in processes_dict.items():
        if not isinstance(details, dict):
            continue
        if 'alt_names' in details:
            out[name] = details
        else:
            out.update(extract_process_rules(details))
    return out


input_dir = Path('npdes_permits/LLM_extraction/output/temporary_llm_ontology_results')
output_csv = Path('npdes_permits/output/llm_unit_processes_by_facility.csv')

with open('npdes_permits/data/unitprocess_keywords.json') as f:
    keywords = json.load(f)
    
columns = get_all_keys(keywords)
leaf_details = extract_process_rules(keywords)

inherited_rules = []
me_rules = []
for col, details in leaf_details.items():
    inf = details.get('inference_inherited')
    if inf:
        inherited_rules.append((col, inf.get('trigger_equipment') or [], inf.get('trigger_roles') or []))
    me = details.get('inference_mutually_exclusive')
    if me:
        for rule in (me if isinstance(me, list) else [me]):
            me_rules.append((
                rule['priority'],
                col,
                rule.get('equipment_types') or [],
                rule.get('role_counts') or {}
            ))
me_rules.sort(key=lambda x: x[0])

g = Graph()
for filename in ["ontology.ttl", "equipment.ttl", "processtypes.ttl", "enumerationkinds.ttl"]:
    g.parse(f"{GITHUB_BASE}/{filename}", format="turtle")
ontology = g
print(ontology)

site_df = pd.read_csv('npdes_permits/output/2025-10-31/site_data.csv', dtype=str).fillna('')
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
    col_lookup = {normalize(col): col for col in columns}

    items = [i for i in json_data if i['Implementation'] in ["present", "planned"]]
    processes = set()
    roles = set()
    equipment = set()
    for item in items:
        for p in item['Process'] or []:
            processes.add(p.replace('Process-', ''))
        for r in item['Role'] or []:
            roles.add(r.replace('Role-', ''))
        if item.get('Equipment'):
            equipment.add(item['Equipment'])

    # Secondary-only: items that have Role-Secondary (for role_counts / equipment filter)
    secondary_items = [i for i in items if any((r or '').replace('Role-', '') == 'Secondary' for r in (i['Role'] or []))]
    role_counts = Counter()
    secondary_equipment = set()
    for i in secondary_items:
        for r in i['Role'] or []:
            role_counts[r.replace('Role-', '')] += 1
        if i.get('Equipment'):
            secondary_equipment.add(i['Equipment'])

    # For direct processes and ontology parents
    for proc in processes:
        if normalize(proc) in col_lookup:
            result[col_lookup[normalize(proc)]] = 'present'
        for row in ontology.query(
            "SELECT DISTINCT ?parent WHERE { ?p rdfs:subClassOf+ ?parent }",
            initBindings={"p": WATR[f"Process-{proc}"]}
        ):
            parent = str(row.parent).split("#")[-1].replace("Process-", "")
            if normalize(parent) in col_lookup:
                result[col_lookup[normalize(parent)]] = 'present'

    # For inherited processes: trigger_roles (all required) and trigger_equipment (any)
    for col, trigger_equipment, trigger_roles in inherited_rules:
        if col not in result:
            continue
        if trigger_roles and not all(r in roles for r in trigger_roles):
            continue
        if trigger_equipment and not any(eq in equipment for eq in trigger_equipment):
            continue
        result[col] = 'present'

    # Mutually exclusive (role-count): first matching rule by priority; skip if any BNR already set by direct/ontology
    me_cols = {col for _, col, _, _ in me_rules}
    if not any(result.get(c) == 'present' for c in me_cols):
        for priority, col, equipment_types, role_counts_spec in me_rules:
            if equipment_types and not any(eq in secondary_equipment for eq in equipment_types):
                continue
            match = True
            for role_name, bounds in role_counts_spec.items():
                count = role_counts.get(role_name, 0)
                # must have at least the minimum count of units with this role 
                if 'min' in bounds and count < bounds['min']:
                    match = False
                    break
                # must not exceed the max count of units with this role
                if 'max' in bounds and count > bounds['max']:
                    match = False
                    break
            if match:
                result[col] = 'present'
                break

    # result = get_columns_from_json(json_data, ontology, columns, inherited_rules, me_rules)
    result.update(pdf_map.get(filename.replace('.json', ''), {}))
    results.append(result)

df = pd.DataFrame(results)
id_cols = ['PERMIT_NUMBER', 'Agency', 'Facility_Name']
cols = id_cols + [c for c in columns if c in df.columns]
df[cols].to_csv(output_csv, index=False)
print(f"Saved {len(results)} facilities")
