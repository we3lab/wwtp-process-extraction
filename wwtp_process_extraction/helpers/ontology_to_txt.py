import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS

# DataDrivenCPS/water-ontology v0.2.0 DOI 10.5281/zenodo.23087793
ZENODO_URL = "https://zenodo.org/api/records/23087794/files/DataDrivenCPS/water-ontology-v0.2.0.zip/content"
ZENODO_MD5 = "eb7833805686fd3f89279489e5de34c3"

WATR = Namespace("https://watermetadata.org/ontology/watr#")
SH = Namespace("http://www.w3.org/ns/shacl#")

# Module names as in the release's ontology/ folder (ontology.ttl was renamed watr.ttl in v0.2.0)
MODULES = ["watr", "equipment", "processtypes", "enumerationkinds", "substances"]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ontology_cache" / "water-ontology-v0.2.0"

# Local edits on top of the release.
# 1. Boiler hasProcess points at Process-Incineration
BOILER_PROCESS_FIX = ("watr:Process-Incineration", "watr:Process-Combustion")
# 2. Drying split into thermal vs air
DRYING_BLOCK_END = "rdfs:subClassOf watr:Process-Dewatering, watr:Process-Evaporation .\n"

PROCESSTYPES_ADDITIONS = """
watr:Process-ThermalDrying a watr:Class, watr:Process-ThermalDrying ;
    rdfs:label "ThermalDrying" ;
    rdfs:comment "Drying of biosolids using applied heat from a fuel-fired or waste-heat dryer." ;
    rdfs:subClassOf watr:Process-Drying .

watr:Process-AirDrying a watr:Class, watr:Process-AirDrying ;
    rdfs:label "AirDrying" ;
    rdfs:comment "Passive drying of biosolids by evaporation in open beds, lagoons or greenhouses, with no applied heat." ;
    rdfs:subClassOf watr:Process-Drying .
"""


def load_ontology(modules=MODULES):
    """Downloads + unzips the release's ontology/*.ttl into CACHE_DIR."""
    if not all((CACHE_DIR / f"{m}.ttl").exists() for m in MODULES):
        data = urllib.request.urlopen(ZENODO_URL).read()
        md5 = hashlib.md5(data).hexdigest()
        if md5 != ZENODO_MD5:
            raise ValueError(f"Zenodo zip md5 {md5} != pinned {ZENODO_MD5}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                # zip root is DataDrivenCPS-water-ontology-<sha>/
                parts = Path(name).parts
                if len(parts) == 3 and parts[1] == "ontology" and name.endswith(".ttl"):
                    (CACHE_DIR / parts[2]).write_bytes(z.read(name))
    g = Graph()
    for m in modules:
        ttl = (CACHE_DIR / f"{m}.ttl").read_text()
        if m == "equipment":
            assert ttl.count(BOILER_PROCESS_FIX[0]) == 1, "Boiler shape changed upstream; recheck BOILER_PROCESS_FIX"
            ttl = ttl.replace(*BOILER_PROCESS_FIX)
        if m == "processtypes":
            assert ttl.count(DRYING_BLOCK_END) == 1, "Drying block changed upstream; update DRYING_BLOCK_END"
            ttl = ttl.replace(DRYING_BLOCK_END, DRYING_BLOCK_END + PROCESSTYPES_ADDITIONS)
        g.parse(data=ttl, format="turtle")
    return g


def hasprocess_fragments(graph, cls):
    """Process fragments declared by a class's own SHACL hasProcess shape(s).

    The ontology declares "this equipment implies this process" two equivalent ways:
    `sh:hasValue watr:Process-X` or `sh:qualifiedValueShape [ sh:class watr:Process-X ]`.
    """
    fragments = set()
    for prop in graph.objects(cls, SH.property):
        for path in graph.objects(prop, SH.path):
            if path != WATR.hasProcess:
                continue
            for val in graph.objects(prop, SH.hasValue):
                if val.fragment:
                    fragments.add(val.fragment)
            for qualified_shape in graph.objects(prop, SH.qualifiedValueShape):
                for val in graph.objects(qualified_shape, SH["class"]):
                    if val.fragment:
                        fragments.add(val.fragment)
    return fragments


ontology_txt_file = Path(__file__).resolve().parent.parent / "data" / "llm_extraction" / "input" / "ontology.txt"

#list of equipment to skip : 
skip_equip = [
    "ElectromagneticFieldDevice",
    "SolventExtractionSystem",
    "DMERecoverySystem",
    "StanderdizedFlowCell"
]
skip_parent_suffixes = ["Sensor", "Valve", "Controller", "Electrode"]


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
    g = load_ontology([equipment_file])

    equipment = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = cls.fragment

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
            parent_name = parent.fragment
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

        # unit processes implied by this equipment's own SHACL hasProcess shape(s)
        unit_processes = sorted(hasprocess_fragments(g, cls))

        equipment.append({
            "id": cls_name,
            "def": definition,
            "SubEquipmentOf": sub_equips if sub_equips else None,
            "UnitProcess": unit_processes if unit_processes else None
        })
    return equipment

def processtypes_to_txt(process_file):
    # Load ontology
    g = load_ontology([process_file])

    processes = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = cls.fragment

        # definition
        definition = None
        for c in g.objects(cls, RDFS.comment):
            definition = str(c)
            break

        # parent processes (ignore ProcessType and Process)
        sub_process_of = []
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_name = parent.fragment
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
    g = load_ontology([enumerationkinds_file])

    roles = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = cls.fragment
        
        # Only process Role-* items
        if not cls_name.startswith("Role-"):
            continue

        # parent roles (ignore EnumerationKind-Role)
        sub_role_of = []
        for parent in g.objects(cls, RDFS.subClassOf):
            parent_name = parent.fragment
            if parent_name != "EnumerationKind-Role":
                sub_role_of.append(parent_name)

        roles.append({
            "id": cls_name,
            "SubRoleOf": sub_role_of if sub_role_of else None
        })
    return roles

def substances_to_txt(substances_file):
    # Load ontology
    g = load_ontology([substances_file])

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
            parent_name = parent.fragment
            
            # If this parent is Constituent-Salt, return True
            if parent_name == "Constituent-Salt":
                return True
            
            # Recursively check parent's ancestors
            if has_constituent_salt_ancestor(parent, visited):
                return True
        
        return False

    substances = []

    for cls in g.subjects(RDF.type, WATR.Class):
        cls_name = cls.fragment

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
            parent_name = parent.fragment
            if parent_name != "Substance":
                subsubstanceOf.append(parent_name)

        substances.append({
            "id": cls_name,
            "def": definition,
            "SubsubstanceOf": subsubstanceOf if subsubstanceOf else None
        })
    return substances

def ontology_to_txt():
    equipment = equipment_to_txt("equipment")
    print(f"Equipment count: {len(equipment)}")
    processes = processtypes_to_txt("processtypes")
    print(f"Process type count: {len(processes)}")
    roles = roles_to_txt("enumerationkinds")
    print(f"Role count: {len(roles)}")
    substances = substances_to_txt("substances")
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
                parts.append(f"SubEquipOf: {', '.join(name for name in eq['SubEquipmentOf'])}")
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
                parts.append(f"SubProcessOf: {', '.join(normalize(name) for name in proc['subProcessOf'])}")
            f.write(" | ".join(parts) + "\n")
        
        f.write("\n########################\n")
        f.write("ROLES:\n")
        for role in roles:
            parts = [normalize(role['id'])]
            if role.get('SubRoleOf'):
                parts.append(f"SubRoleOf: {', '.join(normalize(name) for name in role['SubRoleOf'])}")
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

