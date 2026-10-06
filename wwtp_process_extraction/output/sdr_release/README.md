# California WWTP unit processes from discharge permits and CWNS

This dataset lists the treatment processes at California wastewater treatment plants. We
pulled them from each plant's NPDES or WDR permit and compared them with what the EPA's Clean
Watershed Needs Survey (CWNS) reports. Code: https://github.com/we3lab/wwtp-process-extraction

## unit_processes_by_facility.csv

Each facility gets up to two rows: one from its permit documents (`source` = `ciwqs`) and one
from CWNS (`source` = `Clean Watershed Needs Survey`). The remaining columns are unit processes.
A cell holds `PRESENT`, `PRESENT_AND_FUTURE`, `FUTURE`, `PAST` or `OFFSITE`, and is left blank
if the process wasn't found. There are 618 facilities with permit data, and 322 of
those also have a CWNS row.

## wwtp_ontology_extraction.zip

The permit rows in the CSV are a summary. The zip has the underlying extraction, one Turtle
(.ttl) file per facility (618 files). We ran GPT-5 mini over each current permit with the
WaTr ontology in the prompt, so everything it found is already in WaTr terms.

Each file describes one facility (`wwtp:Facility`). The facility links to its permit documents
(`wwtp:hasDocument`), and each document links to the items the model found (`wwtp:hasItem`). An
item's type is its WaTr equipment class, or `watr:UnitProcess` when the permit didn't name any
equipment. The item points to its processes with `watr:hasProcess`, its roles with
`s223:hasRole`, and any substances with `s223:hasMedium`.

WaTr has no terms for whether a unit is built yet or where it sits, so we recorded those as
plain text: `wwtp:implementation` is `present`, `future` or `past`, and `wwtp:location` is
`on-site` or `off-site`. The model occasionally returned names that aren't in WaTr. We left
those out (187 of 25544 terms).

The CSV columns come from these items through the `ontology_triggers` rules in
`unitprocess_keywords.json` (step 6 of the code).

## Ontology

We used WaTr (Water Treatment ontology) v0.2.0: https://doi.org/10.5281/zenodo.23087793

We made two small changes to it before running the extraction:

1. In `equipment.ttl`, boilers point to `watr:Process-Combustion` instead of `watr:Process-Incineration`.
2. In `processtypes.ttl`, we split drying into heat drying and air drying:

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
