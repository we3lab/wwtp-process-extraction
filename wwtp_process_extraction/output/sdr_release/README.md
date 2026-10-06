# California WWTP unit processes from discharge permits and CWNS

## unit_processes_by_facility.csv
One row per (facility, source). `source` is `ciwqs` (permit documents, LLM extraction) or
`Clean Watershed Needs Survey`. Each unit-process column holds the status at that facility:
`PRESENT`, `PRESENT_AND_FUTURE`, `FUTURE`, `PAST` or `OFFSITE`; blank means not found.
618 facilities have permit data; 322 of them also have a CWNS row.

## wwtp_ontology_extraction.zip
One Turtle (.ttl) file per permit facility (618 files). This is the ontology-native extraction
the CSV's `ciwqs` rows are built from, for the same facilities and the same current-permit documents.
Extraction used GPT-5 mini with the WaTr ontology in the prompt (ontology-based method).

- Each `wwtp:Facility` has permit documents (`wwtp:hasDocument`); each document has items (`wwtp:hasItem`).
- An item is typed with its WaTr equipment class (`watr:UnitProcess` if no equipment was named) and
  links to WaTr terms with `watr:hasProcess`, `s223:hasRole` and `s223:hasMedium` (substances).
- `wwtp:implementation` (`present`, `future`, `past`) and `wwtp:location` (`on-site`, `off-site`)
  are plain text, since WaTr has no terms for them.
- Terms the LLM returned that do not exist in WaTr were left out (187 of 25544).
- The CSV columns come from these items through the `ontology_triggers` rules in
  `unitprocess_keywords.json` (pipeline step 6).

## Ontology
WaTr (Water Treatment ontology) v0.2.0, https://doi.org/10.5281/zenodo.23087793

Two local changes were applied before extraction:
1. `equipment.ttl`: `watr:Process-Incineration` replaced with `watr:Process-Combustion` (boiler).
2. `processtypes.ttl`: two drying classes added:

```turtle
watr:Process-ThermalDrying a watr:Class, watr:Process-ThermalDrying ;
    rdfs:label "ThermalDrying" ;
    rdfs:comment "Drying of biosolids using applied heat from a fuel-fired or waste-heat dryer." ;
    rdfs:subClassOf watr:Process-Drying .

watr:Process-AirDrying a watr:Class, watr:Process-AirDrying ;
    rdfs:label "AirDrying" ;
    rdfs:comment "Passive drying of biosolids by evaporation in open beds, lagoons or greenhouses, with no applied heat." ;
    rdfs:subClassOf watr:Process-Drying .
```
