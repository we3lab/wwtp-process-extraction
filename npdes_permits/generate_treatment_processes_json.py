import json
from pathlib import Path

# Generate treatment_processes.json from unitprocess_keywords.json


def extract_processes(node, ancestors=None):
    """Recursively walk the hierarchy and yield flat process dicts.

    A node is a "leaf" if it contains 'alt_names'.  Otherwise follow childre
    to derive category/subcategory.
    """
    if ancestors is None:
        ancestors = []

    for key, value in node.items():
        if not isinstance(value, dict):
            continue

        if "alt_names" in value:
            # Leaf process
            alt_names = list(value.get("alt_names", []))
            case_sensitive = value.get("alt_names_case_sensitive", [])
            for name in case_sensitive:
                if name not in alt_names:
                    alt_names.append(name)

            category = ancestors[0] if len(ancestors) >= 1 else ""
            subcategory = ancestors[1] if len(ancestors) >= 2 else ""

            yield {
                "generic_name": key,
                "category": category,
                "subcategory": subcategory,
                "alternative_names": alt_names,
            }
        else:
            # Intermediate category node — recurse deeper
            yield from extract_processes(value, ancestors + [key])


OUTPUT_PATH = Path("npdes_permits/LLM_extraction/data/treatment_processes.json")
with open("npdes_permits/data/unitprocess_keywords.json") as f:
    keywords = json.load(f)

processes = list(extract_processes(keywords))

output = {"treatment_processes": processes}

# keep alternative_names arrays on a single line
raw = json.dumps(output, indent=2)
import re
def collapse_array(m):
    items = re.findall(r'"[^"]*"', m.group(0))
    return "[" + ", ".join(items) + "]"
raw = re.sub(r'\[(?:\s*"[^"]*",?\s*)+\]', collapse_array, raw)

with open(OUTPUT_PATH, "w") as f:
    f.write(raw + "\n")