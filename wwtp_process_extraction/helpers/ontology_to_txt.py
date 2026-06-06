from pathlib import Path
from rdflib import Graph, Namespace
from rdflib import Graph, Namespace, RDF, RDFS

WATR = Namespace("urn:nawi-water-ontology#")

# GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/main/water"
GITHUB_BASE = "https://raw.githubusercontent.com/DataDrivenCPS/water-ontology/constance/ontology_to_txt/water"

ontology_txt_file = Path(__file__).resolve().parent.parent / "data" / "llm_extraction" / "input" / "ontology.txt"

#list of equipment to skip : 
skip_equip = [
    "ElectromagneticFieldDevice","SolventExtractionSystem", "DMERecoverySystem",
    "StanderdizedFlowCell"
]
skip_parent_suffixes = ["Sensor", "Valve", "Controller", "Electrode"]


SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


def normalize(name):
    if not isinstance(name, str):
        return name
    if name.startswith("Process-"):
        return name[len("Process-"):]
    if name.startswith("Role-"):
        return name[len("Role-"):]
    return name

def equipment_to_txt (equipment_file):
    # Load ontology
    g = Graph()
    g.parse(GITHUB_BASE + "/" + equipment_file, format="turtle", publicID=equipment_file)


    # Namespaces
    WATR = Namespace("urn:nawi-water-ontology#")
    SH = Namespace("http://www.w3.org/ns/shacl#")

    def local_name(uri):
        return uri.split("#")[-1]

    equipment = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = local_name(str(cls))

        # check for equipment to skip:
        if cls_name in skip_equip:
            continue

        # definition
        definition = None
        for c in g.objects(cls, RDFS.comment):
            definition = str(c)
            break

        # subclasses (ignore UnitProcess, Equipment, and s223 namespace items)
        sub_equips = []
        has_parent_suffix = False
        
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_uri = str(parent)
            parent_name = local_name(parent_uri)
            # Check if any parent name ends with suffix to skip:
            for suffix in skip_parent_suffixes:
                if parent_name.endswith(suffix):
                    has_parent_suffix = True
                    break
            # Only include watr namespace items, exclude s223 namespace
            if (parent_name != "UnitProcess" and 
                parent_name != "Equipment" and
                not parent_uri.startswith("http://data.ashrae.org/standard223#")):
                sub_equips.append(parent_name)

        # Skip this equipment if it has a Sensor parent
        if has_parent_suffix :
            continue

        # unit processes (from sh:hasValue where sh:path == watr:hasProcess)
        unit_processes = []
        for prop in g.objects(cls, SH.property):
            for path in g.objects(prop, SH.path):
                if path == WATR.hasProcess:
                    for val in g.objects(prop, SH.hasValue):
                        unit_processes.append(local_name(str(val)))

        equipment.append({
            "id": cls_name,
            "def": definition,
            "SubEquipmentOf": sub_equips if sub_equips else None,
            "UnitProcess": unit_processes if unit_processes else None
        })
    return equipment

def processtypes_to_txt(process_file):
    # Load ontology
    g = Graph()
    g.parse(GITHUB_BASE + "/" + process_file, format="turtle", publicID=process_file)

    # Namespaces
    WATR = Namespace("urn:nawi-water-ontology#")

    def local_name(uri):
        return uri.split("#")[-1]

    processes = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = local_name(str(cls))

        # definition (using skos:definition)
        definition = None
        for c in g.objects(cls, SKOS.definition):
            definition = str(c)
            break

        # parent processes (ignore ProcessType and Process)
        sub_process_of = []
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_name = local_name(str(parent))
            if parent_name != "ProcessType" and parent_name != "Process":
                sub_process_of.append(parent_name)

        processes.append({
            "id": cls_name,
            "def": definition,
            "subProcessOf": sub_process_of if sub_process_of else None
        })
    return processes

def roles_to_txt(enumerationkinds_file):
    # Load ontology
    g = Graph()
    g.parse(GITHUB_BASE + "/" + enumerationkinds_file, format="turtle", publicID=enumerationkinds_file)

    # Namespaces
    WATR = Namespace("urn:nawi-water-ontology#")
    S223 = Namespace("http://data.ashrae.org/standard223#")

    def local_name(uri):
        return uri.split("#")[-1]

    roles = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = local_name(str(cls))
        
        # Only process Role-* items
        if not cls_name.startswith("Role-"):
            continue

        # parent roles (ignore EnumerationKind-Role)
        sub_role_of = []
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_name = local_name(str(parent))
            if parent_name != "EnumerationKind-Role":
                sub_role_of.append(parent_name)

        roles.append({
            "id": cls_name,
            "SubRoleOf": sub_role_of if sub_role_of else None
        })
    return roles

def substances_to_txt(substances_file):
    # Load ontology
    g = Graph()
    g.parse(GITHUB_BASE + "/" + substances_file, format="turtle", publicID=substances_file)

    # Namespaces
    WATR = Namespace("urn:nawi-water-ontology#")

    def local_name(uri):
        return uri.split("#")[-1]

    # Helper function to check if a class has "Constituent-Salt" in its parent hierarchy
    def has_constituent_salt_ancestor(cls_uri, visited=None):
        if visited is None:
            visited = set()
        
        # Avoid infinite loops
        if cls_uri in visited:
            return False
        visited.add(cls_uri)
        
        # Check all parents
        for parent in g.objects(cls_uri, RDFS.subClassOf):
            parent_name = local_name(str(parent))
            
            # If this parent is Constituent-Salt, return True
            if parent_name == "Constituent-Salt":
                return True
            
            # Recursively check parent's ancestors
            if has_constituent_salt_ancestor(parent, visited):
                return True
        
        return False

    substances = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = local_name(str(cls))

        # Skip Constituent-Salt itself
        if cls_name == "Constituent-Salt":
            continue

        # Skip anything with "Brine" in the name
        if "Brine" in cls_name:
            continue

        # Skip if this substance has Constituent-Salt in its ancestry
        if has_constituent_salt_ancestor(cls):
            continue

        # definition (using rdfs:comment)
        definition = None
        subsubstanceOf = []
        for c in g.objects(cls, RDFS.comment):
            definition = str(c)
            break
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_name = local_name(str(parent))
            if parent_name != "Substance":
                subsubstanceOf.append(parent_name)

        substances.append({
            "id": cls_name,
            "def": definition,
            "SubsubstanceOf": subsubstanceOf if subsubstanceOf else None
        })
    return substances

def ontology_to_txt():
    equipment = equipment_to_txt("equipment.ttl")
    print(f"Equipment count: {len(equipment)}")
    processes = processtypes_to_txt("processtypes.ttl")
    print(f"Process type count: {len(processes)}")
    roles = roles_to_txt("enumerationkinds.ttl")
    print(f"Role count: {len(roles)}")
    substances = substances_to_txt("substances.ttl")
    print(f"Substance count: {len(substances)}")

    # create the output directory if it doesn't exist
    output_dir = Path(ontology_txt_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save to text file in compact format
    with open(ontology_txt_file, "w") as f:
        f.write("EQUIPMENTS:\n")
        for eq in equipment:
            parts = [eq['id']]
            if eq.get('def'):
                parts.append(eq['def'])
            if eq.get('SubEquipmentOf'):
                parts.append(f"SubEquip: {', '.join(name for name in eq['SubEquipmentOf'])}")
            if eq.get('UnitProcess'):
                parts.append(f"Process: {', '.join(name for name in eq['UnitProcess'])}")
            f.write(" | ".join(parts) + "\n")
        
        f.write("\n########################\n")
        f.write("PROCESSES:\n")
        for proc in processes:
            parts = [normalize(proc['id'])]
            if proc.get('def'):
                parts.append(proc['def'])
            if proc.get('subProcessOf'):
                parts.append(f"SubProcess: {', '.join(normalize(name) for name in proc['subProcessOf'])}")
            f.write(" | ".join(parts) + "\n")
        
        f.write("\n########################\n")
        f.write("ROLES:\n")
        for role in roles:
            parts = [normalize(role['id'])]
            if role.get('SubRoleOf'):
                parts.append(f"SubRole: {', '.join(normalize(name) for name in role['SubRoleOf'])}")
            f.write(" | ".join(parts) + "\n")

        f.write("\n########################\n")
        f.write("SUBSTANCES:\n")
        for sub in substances:
            parts = [normalize(sub['id'])]
            if sub.get('def'):
                parts.append(sub['def'])
            if sub.get('SubsubstanceOf'):
                parts.append(f"SubSubstanceOf: {', '.join(normalize(name) for name in sub['SubsubstanceOf'])}")
            f.write(" | ".join(parts) + "\n")

    print(f"\nOutput saved to {ontology_txt_file}")

