# SDR release: the unit-process CSV, a zip of per-facility TTL files (each extracted item typed with
# WaTr classes, for the CSV's facilities and current-permit documents), and a README

import inspect
import json
import shutil
import textwrap
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS

from helpers.ontology_to_txt import load_ontology, WATR
from helpers.paths import *
from helpers.utils import current_permit_mask, cwns_mapping

RELEASE_DIR = OUTPUT_DIR / "sdr_release"
ZIP_NAME = "wwtp_ontology_extraction.zip"
# Deposit README text; {placeholders} are filled with this run's counts
README_TEMPLATE_MD = RELEASE_DIR / "README_TEMPLATE.md"

S223 = Namespace("http://data.ashrae.org/standard223#")
# Project namespace on the deposit's own purl, so IDs point back to the dataset
WWTP = Namespace("https://purl.stanford.edu/nk282rk7079#")


def main():
    ontology = load_ontology()
    known_terms = set(ontology.subjects())

    facilities = pd.read_csv(FACILITY_CSV, dtype=str, keep_default_na=False)
    permit_rows = facilities[facilities["source"] == "ciwqs"].set_index("CIWQS PLACE_ID")

    # Same documents step6 collapses into the facility CSV: current permits only
    documents = pd.read_csv(PDF_LLM_CSV, dtype=str, keep_default_na=False)
    documents = documents[current_permit_mask(documents)]
    documents = documents[documents["Place ID"].isin(permit_rows.index)]
    missing = set(permit_rows.index) - set(documents["Place ID"])
    if missing:
        raise ValueError(f"No current-permit documents for {len(missing)} CSV facilities: {sorted(missing)[:10]}")

    term_counts = Counter()
    dropped = Counter()
    n_items = 0
    with zipfile.ZipFile(RELEASE_DIR / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for place_id, fac_docs in documents.groupby("Place ID"):
            row = permit_rows.loc[place_id]
            graph = Graph()
            for prefix, ns in (("watr", WATR), ("s223", S223), ("wwtp", WWTP)):
                graph.bind(prefix, ns)

            facility = WWTP[f"facility-{place_id}"]
            graph.add((facility, RDF.type, WWTP.Facility))
            graph.add((facility, RDFS.label, Literal(row["CIWQS Facility Name"])))
            graph.add((facility, WWTP.placeId, Literal(place_id)))
            for prop, value in ((WWTP.npdesNo, fac_docs["NPDES No."].iloc[0]),
                                (WWTP.county, row["County"]),
                                (WWTP.cwnsFacilityId, row["CWNS FACILITY_ID"])):
                if value:
                    graph.add((facility, prop, Literal(value)))

            for doc_index, doc in enumerate(fac_docs.itertuples(index=False), 1):
                document = WWTP[f"document-{place_id}-{doc_index}"]
                graph.add((facility, WWTP.hasDocument, document))
                graph.add((document, RDF.type, WWTP.PermitDocument))
                graph.add((document, WWTP.pdfFile, Literal(doc.PDF_File)))
                if doc.document_order_no:
                    graph.add((document, WWTP.orderNo, Literal(doc.document_order_no)))

                json_path = MAIN_RUN_DIR / f"{Path(doc.PDF_File).stem}_{place_id}.json"
                items = json.loads(json_path.read_text(encoding="utf-8"))["items"]
                for item_index, extracted in enumerate(items, 1):
                    # WaTr URIs: Process-X and Role-X carry a prefix; equipment and substances don't
                    links = []
                    for field, prop, prefix in (("Process", WATR.hasProcess, "Process-"),
                                                ("Role", S223.hasRole, "Role-"),
                                                ("Substance", S223.hasMedium, "")):
                        for name in extracted.get(field) or []:
                            term_counts[field] += 1
                            uri = WATR[prefix + name.removeprefix(prefix)]
                            if uri in known_terms:
                                links.append((prop, uri))
                            else:
                                dropped[(field, name)] += 1
                    equipment = extracted.get("Equipment")
                    item_type = WATR.UnitProcess
                    if equipment:
                        term_counts["Equipment"] += 1
                        if WATR[equipment] in known_terms:
                            item_type = WATR[equipment]
                        else:
                            dropped[("Equipment", equipment)] += 1
                    if item_type == WATR.UnitProcess and not links:
                        continue  # nothing in this item resolves to WaTr

                    item = WWTP[f"item-{place_id}-{doc_index}-{item_index}"]
                    graph.add((document, WWTP.hasItem, item))
                    graph.add((item, RDF.type, item_type))
                    for prop, uri in links:
                        graph.add((item, prop, uri))
                    for prop, field in ((WWTP.implementation, "Implementation"), (WWTP.location, "Location")):
                        if extracted.get(field):
                            graph.add((item, prop, Literal(extracted[field])))
                    n_items += 1

            # file name: Place ID + facility name as letters/digits joined by underscores
            name_slug = "_".join("".join(ch if ch.isalnum() else " " for ch in row["CIWQS Facility Name"]).split())
            zf.writestr(f"{place_id}_{name_slug}.ttl", graph.serialize())

    shutil.copy2(FACILITY_CSV, RELEASE_DIR / FACILITY_CSV.name)

    # Facility counts for the README and the paper
    cwns_ids = set(facilities.loc[facilities["source"] != "ciwqs", "CIWQS PLACE_ID"])
    n_permit, n_both = len(permit_rows), len(cwns_ids & set(permit_rows.index))

    # The drying TTL block, read from load_ontology so it matches what the pipeline used
    source = inspect.getsource(load_ontology)
    start = source.index('PROCESSTYPES_ADDITIONS = """') + len('PROCESSTYPES_ADDITIONS = """')
    drying_block = textwrap.dedent(source[start:source.index('"""', start)]).strip()

    readme = README_TEMPLATE_MD.read_text(encoding="utf-8").format(
        n_permit=n_permit, n_both=n_both, zip_name=ZIP_NAME,
        n_dropped=sum(dropped.values()), n_terms=sum(term_counts.values()), drying_block=drying_block,
    )
    (RELEASE_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Facilities: {n_permit} with permit data, {n_both} of them with a CWNS row "
          f"({len(set(cwns_mapping['Place ID']))} Place IDs mapped to a CWNS ID)")
    print(f"Documents: {len(documents)}, items: {n_items}")
    print(f"Dropped {sum(dropped.values())} of {sum(term_counts.values())} terms not in WaTr; most common:")
    for (field, name), count in dropped.most_common(10):
        print(f"  {field} {name}: {count}")
    print(f"Wrote {RELEASE_DIR}/: README.md, {FACILITY_CSV.name}, {ZIP_NAME}")


if __name__ == "__main__":
    main()
