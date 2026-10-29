import io
import urllib.request
import zipfile
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS

from helpers.paths import *

# URL for water-ontology release v0.2.0 DOI 10.5281/zenodo.23087793
ZENODO_URL = "https://zenodo.org/api/records/23087794/files/DataDrivenCPS/water-ontology-v0.2.0.zip/content"

# Defining the namespace variables
WATR = Namespace("https://watermetadata.org/ontology/watr#")
SH = Namespace("http://www.w3.org/ns/shacl#")

# Module names in ontology/ folder
MODULES = ["watr", "equipment", "processtypes", "enumerationkinds", "substances"]

CACHE_DIR = DATA_DIR / "ontology_cache" / "water-ontology-v0.2.0"

# Equipment to skip
SKIP_EQUIPMENT = [
    "ElectromagneticFieldDevice",
    "SolventExtractionSystem",
    "DMERecoverySystem",
    "StanderdizedFlowCell"
]
SKIP_PARENT_SUFFIXES = ("Sensor", "Valve", "Controller", "Electrode")


def load_ontology(modules=MODULES):
    """Parse the WaTr modules (downloaded into CACHE_DIR on first use) with our two local fixes."""
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


def class_line(graph, cls, sub_label, ignored_parents, with_definition=True):
    # "name | definition | SubXOf: parents" line for the LLM prompt
    parts = [normalize(cls.fragment)]
    definition = graph.value(cls, RDFS.comment)
    if with_definition and definition:
        parts.append(str(definition))
    parents = [normalize(p.fragment) for p in graph.objects(cls, RDFS.subClassOf) if p.fragment not in ignored_parents]
    if parents:
        parts.append(f"{sub_label}: {', '.join(parents)}")
    return " | ".join(parts)


def ontology_to_txt():
    """Saves .ttl files to txt file for LLM prompt"""
    equipment_graph = load_ontology(["equipment"])
    equipment_lines = []
    for cls in equipment_graph.subjects(RDF.type, WATR.Class):
        # check for equipment to skip:
        if cls.fragment in SKIP_EQUIPMENT:
            continue

        parents = list(equipment_graph.objects(cls, RDFS.subClassOf))
        # Skip sensors, valves, controllers and electrodes
        if any(parent.fragment.endswith(SKIP_PARENT_SUFFIXES) for parent in parents):
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
    process_lines = [
        class_line(process_graph, cls, "SubProcessOf", ("ProcessType", "Process"))
        for cls in process_graph.subjects(RDF.type, WATR.Class)
    ]
    print(f"Process type count: {len(process_lines)}")

    # Role-* classes only, without definitions
    role_graph = load_ontology(["enumerationkinds"])
    role_lines = [
        class_line(role_graph, cls, "SubRoleOf", ("EnumerationKind-Role",), with_definition=False)
        for cls in role_graph.subjects(RDF.type, WATR.Class)
        if cls.fragment.startswith("Role-")
    ]
    print(f"Role count: {len(role_lines)}")

    # Skip brines and Constituent-Salt (itself or in the ancestry)
    substance_graph = load_ontology(["substances"])
    substance_lines = [
        class_line(substance_graph, cls, "SubSubstanceOf", ("Substance",))
        for cls in substance_graph.subjects(RDF.type, WATR.Class)
        if "Brine" not in cls.fragment
        and not any(a.fragment == "Constituent-Salt" for a in substance_graph.transitive_objects(cls, RDFS.subClassOf))
    ]
    print(f"Substance count: {len(substance_lines)}")

    # Save to text file in compact format
    sections = [
        ("EQUIPMENTS", equipment_lines),
        ("PROCESSES", process_lines),
        ("ROLES", role_lines),
        ("SUBSTANCES", substance_lines),
    ]
    ONTOLOGY_TXT.write_text("\n########################\n".join(
        f"{header}:\n" + "".join(line + "\n" for line in lines) for header, lines in sections
    ))

    print(f"\nOutput saved to {ONTOLOGY_TXT}")
