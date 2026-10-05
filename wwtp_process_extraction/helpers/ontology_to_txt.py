import io
import urllib.request
import zipfile
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS

# URL for water-ontology release v0.2.0 DOI 10.5281/zenodo.23087793
ZENODO_URL = "https://zenodo.org/api/records/23087794/files/DataDrivenCPS/water-ontology-v0.2.0.zip/content"

# Defining the namespace variables
WATR = Namespace("https://watermetadata.org/ontology/watr#")
SH = Namespace("http://www.w3.org/ns/shacl#")

# Module names in ontology/ folder
MODULES = ["watr", "equipment", "processtypes", "enumerationkinds", "substances"]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "ontology_cache" / "water-ontology-v0.2.0"

ontology_txt_file = Path(__file__).resolve().parent.parent / "data" / "llm_extraction" / "input" / "ontology.txt"

#list of equipment to skip :
skip_equip = [
    "ElectromagneticFieldDevice",
    "SolventExtractionSystem",
    "DMERecoverySystem",
    "StanderdizedFlowCell"
]
skip_parent_suffixes = ("Sensor", "Valve", "Controller", "Electrode")


def load_ontology(modules=MODULES):
    """Downloads + unzips the .ttl files into CACHE_DIR."""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ZENODO_URL).read())) as z:
            for name in z.namelist():
                # zip root is DataDrivenCPS-water-ontology-<sha>/
                parts = Path(name).parts
                if len(parts) == 3 and parts[1] == "ontology" and name.endswith(".ttl"):
                    (CACHE_DIR / parts[2]).write_bytes(z.read(name))
    graph = Graph()

    # Define manual fixes (on top of released version)
    for module in modules:
        ttl = (CACHE_DIR / f"{module}.ttl").read_text()
        if module == "equipment":
            # 1. Replace boiler hasProcess Process-Incineration with Process-Combustion
            ttl = ttl.replace("watr:Process-Incineration", "watr:Process-Combustion")
        if module == "processtypes":
            # 2. unitprocess_keywords.json specifies split thermal vs air drying
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
            ttl = ttl.replace(DRYING_BLOCK_END, DRYING_BLOCK_END + PROCESSTYPES_ADDITIONS)
        graph.parse(data=ttl, format="turtle")
    return graph


def hasprocess_fragments(graph, cls):
    """Returns the names of the processes the ontology says each equipment performs."""
    fragments = set()
    for prop in graph.objects(cls, SH.property):
        # skip rules about other things (connection points, roles)
        if (prop, SH.path, WATR.hasProcess) not in graph:
            continue
        # format 1: sh:hasValue watr:Process-X
        for process in graph.objects(prop, SH.hasValue):
            fragments.add(process.fragment)
        # format 2: sh:qualifiedValueShape [ sh:class watr:Process-X ]
        for shape in graph.objects(prop, SH.qualifiedValueShape):
            for process in graph.objects(shape, SH["class"]):
                fragments.add(process.fragment)
    return fragments


def normalize(name):
    return name.removeprefix("Process-").removeprefix("Role-")


def ontology_to_txt():
    """Saves .ttl files to txt file for LLM prompt"""
    equipment_graph = load_ontology(["equipment"])
    equipment_lines = []
    for cls in equipment_graph.subjects(RDF.type, WATR.Class):
        # check for equipment to skip:
        if cls.fragment in skip_equip:
            continue

        parents = list(equipment_graph.objects(cls, RDFS.subClassOf))
        # Skip sensors, valves, controllers and electrodes
        if any(parent.fragment.endswith(skip_parent_suffixes) for parent in parents):
            continue
        # parent equipment
        sub_equip_of = [
            parent.fragment for parent in parents
            if parent.fragment not in ("UnitProcess", "Equipment") # ignore top-level classes
            and not str(parent).startswith("http://data.ashrae.org/standard223#") # ignore s223 namespace
        ]

        parts = [cls.fragment]
        definition = equipment_graph.value(cls, RDFS.comment)
        if definition:
            parts.append(str(definition))
        if sub_equip_of:
            parts.append(f"SubEquipOf: {', '.join(sub_equip_of)}")
        # unit processes implied by this equipment's own SHACL hasProcess shape(s)
        unit_processes = sorted(hasprocess_fragments(equipment_graph, cls))
        if unit_processes:
            parts.append(f"Process: {', '.join(unit_processes)}")
        equipment_lines.append(" | ".join(parts))
    print(f"Equipment count: {len(equipment_lines)}")

    process_graph = load_ontology(["processtypes"])
    process_lines = []
    for cls in process_graph.subjects(RDF.type, WATR.Class):
        parts = [normalize(cls.fragment)]
        definition = process_graph.value(cls, RDFS.comment)
        if definition:
            parts.append(str(definition))
        # parent processes (ignore ProcessType and Process)
        sub_process_of = [
            normalize(parent.fragment) for parent in process_graph.objects(cls, RDFS.subClassOf)
            if parent.fragment not in ("ProcessType", "Process")
        ]
        if sub_process_of:
            parts.append(f"SubProcessOf: {', '.join(sub_process_of)}")
        process_lines.append(" | ".join(parts))
    print(f"Process type count: {len(process_lines)}")

    role_graph = load_ontology(["enumerationkinds"])
    role_lines = []
    for cls in role_graph.subjects(RDF.type, WATR.Class):
        # Only process Role-* items
        if not cls.fragment.startswith("Role-"):
            continue
        parts = [normalize(cls.fragment)]
        # parent roles (ignore EnumerationKind-Role)
        sub_role_of = [
            normalize(parent.fragment) for parent in role_graph.objects(cls, RDFS.subClassOf)
            if parent.fragment != "EnumerationKind-Role"
        ]
        if sub_role_of:
            parts.append(f"SubRoleOf: {', '.join(sub_role_of)}")
        role_lines.append(" | ".join(parts))
    print(f"Role count: {len(role_lines)}")

    substance_graph = load_ontology(["substances"])
    substance_lines = []
    for cls in substance_graph.subjects(RDF.type, WATR.Class):
        # Skip anything with "Brine" in the name
        if "Brine" in cls.fragment:
            continue
        # Skip Constituent-Salt itself and anything with it in its ancestry
        if any(ancestor.fragment == "Constituent-Salt"
               for ancestor in substance_graph.transitive_objects(cls, RDFS.subClassOf)):
            continue
        parts = [normalize(cls.fragment)]
        definition = substance_graph.value(cls, RDFS.comment)
        if definition:
            parts.append(str(definition))
        sub_substance_of = [
            normalize(parent.fragment) for parent in substance_graph.objects(cls, RDFS.subClassOf)
            if parent.fragment != "Substance"
        ]
        if sub_substance_of:
            parts.append(f"SubSubstanceOf: {', '.join(sub_substance_of)}")
        substance_lines.append(" | ".join(parts))
    print(f"Substance count: {len(substance_lines)}")

    # Save to text file in compact format
    sections = [
        ("EQUIPMENTS", equipment_lines),
        ("PROCESSES", process_lines),
        ("ROLES", role_lines),
        ("SUBSTANCES", substance_lines),
    ]
    with open(ontology_txt_file, "w") as f:
        for i, (header, lines) in enumerate(sections):
            if i > 0:
                f.write("\n########################\n")
            f.write(f"{header}:\n")
            for line in lines:
                f.write(line + "\n")

    print(f"\nOutput saved to {ontology_txt_file}")
