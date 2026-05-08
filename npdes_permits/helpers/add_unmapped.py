import json
import os
import pandas as pd
from geopy.distance import geodesic

# Append rows to data/ciwqs_to_cwns.csv for unmapped site_data.csv rows

DATE_FOLDER = "2026-4-26"

SITE_DATA = f"npdes_permits/output/{DATE_FOLDER}/site_data.csv"
FACILITIES_JSON = f"npdes_permits/output/{DATE_FOLDER}/facilities.json"
ALL_NPDES = f"npdes_permits/output/{DATE_FOLDER}/all_ca_npdes.csv"
CWNS_TABLE = "npdes_permits/output/cwns_processes_by_facility.csv"
CWNS_FACILITIES = "npdes_permits/data/cwns/2022/FACILITIES.csv"
CWNS_FACILITIES_CONFIRMED = "npdes_permits/data/cwns/2022/FACILITIES_CONFIRMED.csv"
CWNS_TYPES = "npdes_permits/data/cwns/2022/FACILITY_TYPES.csv"
CWNS_PHYSICAL = "npdes_permits/data/cwns/2022/PHYSICAL_LOCATION.csv"
CIWQS_MAP = "npdes_permits/data/ciwqs_to_cwns.csv"

ciwqs_cols = [
    "WDID", "Place ID", "Facility Name", "NPDES No.", "Region",
    "Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS",
    "CWNS_ID", "FACILITY_ID", "CWNS Facility Name",
]

def normalize(s):
    return str(s).strip().upper() if pd.notna(s) else ""


def coalesce_blank(left, right):
    return left.replace("", pd.NA).fillna(right).fillna("")


site = pd.read_csv(SITE_DATA, dtype=str).fillna("")
cwns = pd.read_csv(CWNS_TABLE, dtype=str).fillna("")
ciwqs = pd.read_csv(CIWQS_MAP, dtype=str, keep_default_na=False).fillna("").rename(
    columns={"Latitude": "Latitude_CIWQS", "Longitude": "Longitude_CIWQS"}
)
all_npdes = pd.read_csv(ALL_NPDES, dtype=str).fillna("").rename(
    columns={"Latitude": "Latitude_CIWQS_from_npdes", "Longitude": "Longitude_CIWQS_from_npdes"}
)
cwns_fac = pd.concat([
    pd.read_csv(CWNS_FACILITIES, dtype=str).fillna(""),
    pd.read_csv(CWNS_FACILITIES_CONFIRMED, dtype=str).fillna(""),
]).rename(columns={"FACILITY_NAME": "CWNS Facility Name"})
cwns_phys = pd.read_csv(CWNS_PHYSICAL, dtype=str).fillna("").rename(
    columns={"LATITUDE": "Latitude_CWNS_from_cwns", "LONGITUDE": "Longitude_CWNS_from_cwns"}
)
cwns_types = pd.read_csv(CWNS_TYPES, dtype=str).fillna("")

# Filter CWNS facilities to Treatment Plant type only — coordinates from other types are not meaningful
tp_pairs = cwns_types[cwns_types["FACILITY_TYPE"] == "Treatment Plant"][["CWNS_ID", "FACILITY_ID"]].apply(lambda c: c.str.strip())
cwns_fac[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]] = cwns_fac[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]].apply(lambda c: c.str.strip())
cwns_fac_tp = cwns_fac.merge(tp_pairs, on=["CWNS_ID", "FACILITY_ID"], how="inner")

cwns_phys[["CWNS_ID", "FACILITY_ID"]] = cwns_phys[["CWNS_ID", "FACILITY_ID"]].apply(lambda c: c.str.strip())

cwns_loc_map = (
    cwns_fac_tp[["CWNS_ID", "FACILITY_ID", "CWNS Facility Name"]]
    .drop_duplicates()
    .merge(cwns_phys[["CWNS_ID", "FACILITY_ID", "Latitude_CWNS_from_cwns", "Longitude_CWNS_from_cwns"]].drop_duplicates(), on=["CWNS_ID", "FACILITY_ID"], how="left")
    .drop_duplicates()
)

# CWNS_ID → FACILITY_ID lookup for populating existing mapping rows
cwns_id_to_fac_id = cwns_fac_tp.drop_duplicates("CWNS_ID").set_index("CWNS_ID")["FACILITY_ID"].to_dict()

for col in ["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]:
    site[col] = site[col].str.strip()
site_lookup_cols = ["WDID", "Facility Name", "NPDES No.", "Region", "Place ID"]
site_lookup = site[site_lookup_cols].drop_duplicates()

for col in ["WDID", "Facility Name"]:
    all_npdes[col] = all_npdes[col].str.strip()
ciwqs_lookup = all_npdes[["WDID", "Facility Name", "Latitude_CIWQS_from_npdes", "Longitude_CIWQS_from_npdes"]].drop_duplicates()

site = site.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

ciwqs[["WDID", "Facility Name"]] = ciwqs[["WDID", "Facility Name"]].apply(lambda c: c.str.strip())
# ensure CWNS keys are normalized for merges
for col in ["CWNS_ID", "CWNS Facility Name"]:
    if col in ciwqs.columns:
        ciwqs[col] = ciwqs[col].astype(str).str.strip()

# Populate FACILITY_ID for existing rows that predate this column
if "FACILITY_ID" not in ciwqs.columns:
    ciwqs["FACILITY_ID"] = ""
ciwqs["FACILITY_ID"] = ciwqs["FACILITY_ID"].astype(str).str.strip()
needs_fac_id = ciwqs["FACILITY_ID"].eq("") & ciwqs["CWNS_ID"].ne("") & ciwqs["CWNS_ID"].str.upper().ne("NA")
ciwqs.loc[needs_fac_id, "FACILITY_ID"] = ciwqs.loc[needs_fac_id, "CWNS_ID"].map(cwns_id_to_fac_id).fillna("")

ciwqs = ciwqs.merge(site_lookup, on=["WDID", "Facility Name"], how="left", suffixes=("", "_site"))
ciwqs = ciwqs.merge(ciwqs_lookup, on=["WDID", "Facility Name"], how="left")

for dest, src in [("NPDES No.", "NPDES_No_site"), ("Region", "Region_site")]:
    if src in ciwqs.columns:
        ciwqs[dest] = ciwqs[dest] if dest in ciwqs.columns else ciwqs[src]
        ciwqs[dest] = coalesce_blank(ciwqs[dest], ciwqs[src])
        ciwqs = ciwqs.drop(columns=[src])
for dest, src in [("Latitude_CIWQS", "Latitude_CIWQS_from_npdes"), ("Longitude_CIWQS", "Longitude_CIWQS_from_npdes")]:
    if src in ciwqs.columns:
        ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest] if dest in ciwqs.columns else "")
        ciwqs = ciwqs.drop(columns=[src])

ciwqs["Latitude_CWNS"] = ciwqs.get("Latitude_CWNS", "")
ciwqs["Longitude_CWNS"] = ciwqs.get("Longitude_CWNS", "")

already_mapped = set(ciwqs["Facility Name"].map(normalize))
unmapped = (
    site[~site["Facility Name"].map(normalize).isin(already_mapped)]
    .drop_duplicates("Facility Name")
    .copy()
)
print(f"Unmapped facilities: {len(unmapped)}")

# Match unmapped facilities by name against CA Treatment Plant facilities
cwns_fac_ca = cwns_fac_tp[cwns_fac_tp["STATE_CODE"] == "CA"].copy()
cwns_fac_ca["_name"] = cwns_fac_ca["CWNS Facility Name"].map(normalize)
name_idx = cwns_fac_ca.groupby("_name").apply(lambda g: g.to_dict("records")).to_dict()

new_rows = []

for _, row in unmapped.iterrows():
    fac_name = row["Facility Name"].strip()
    permit = str(row.get("NPDES No.", "")).strip().upper()
    base_entry = {
        "WDID": str(row.get("WDID", "")).strip(),
        "Place ID": str(row.get("Place ID", "")).strip(),
        "Facility Name": fac_name,
        "NPDES No.": permit,
        "Region": str(row.get("Region", "")).strip(),
        "Latitude_CIWQS": str(row.get("Latitude_CIWQS_from_npdes", "")).strip(),
        "Longitude_CIWQS": str(row.get("Longitude_CIWQS_from_npdes", "")).strip(),
        "Latitude_CWNS": "",
        "Longitude_CWNS": "",
    }
    cwns_hits = name_idx.get(normalize(fac_name), [])
    if cwns_hits:
        print(f"  [name] {permit} — {fac_name} → {len(cwns_hits)} CWNS row(s)")
        for hit in cwns_hits:
            new_rows.append({**base_entry, "CWNS_ID": hit["CWNS_ID"], "FACILITY_ID": hit["FACILITY_ID"], "CWNS Facility Name": hit["CWNS Facility Name"]})
    else:
        print(f"  [no match] {permit} — {fac_name}")
        new_rows.append({**base_entry, "CWNS_ID": "", "FACILITY_ID": "", "CWNS Facility Name": ""})

ciwqs = ciwqs.merge(
    cwns_loc_map[["CWNS_ID", "FACILITY_ID", "Latitude_CWNS_from_cwns", "Longitude_CWNS_from_cwns"]],
    on=["CWNS_ID", "FACILITY_ID"], how="left"
)

for dest, src in [("Latitude_CWNS", "Latitude_CWNS_from_cwns"), ("Longitude_CWNS", "Longitude_CWNS_from_cwns")]:
    if src in ciwqs.columns:
        ciwqs[dest] = coalesce_blank(ciwqs[src], ciwqs[dest])
        ciwqs = ciwqs.drop(columns=[src])

for col in ciwqs_cols:
    if col not in ciwqs.columns:
        ciwqs[col] = ""
ciwqs_out = ciwqs[ciwqs_cols]

if new_rows:
    new_df = pd.DataFrame(new_rows, columns=ciwqs_cols)
    combined = pd.concat([ciwqs_out, new_df], ignore_index=True)
    print(f"\nAdded {len(new_rows)} rows. ciwqs_to_cwns.csv now has {len(combined)} rows.")
else:
    combined = ciwqs_out
    print("\nNo new rows to add.")

# Dedupe on Place ID + FACILITY_ID, sorting NPDES empties last (matching build_cwns_facility_processes logic)
combined.sort_values(by="NPDES No.", key=lambda s: s.eq(""), ascending=True).drop_duplicates(subset=["Place ID", "FACILITY_ID"], keep="first").to_csv(CIWQS_MAP, index=False)

coord_cols = ["Latitude_CIWQS", "Longitude_CIWQS", "Latitude_CWNS", "Longitude_CWNS"]
geo = combined[combined[coord_cols].replace("", pd.NA).notna().all(axis=1)].copy()
for col in coord_cols:
    geo[col] = pd.to_numeric(geo[col], errors="coerce")
geo = geo.dropna(subset=coord_cols)

geo["_dist_miles"] = geo.apply(
    lambda r: geodesic((r["Latitude_CIWQS"], r["Longitude_CIWQS"]), (r["Latitude_CWNS"], r["Longitude_CWNS"])).miles,
    axis=1,
)
far = geo[geo["_dist_miles"] > 2].sort_values("_dist_miles", ascending=False)
print(f"\nRows where CWNS and CIWQS coords are >2 miles apart: {len(far)}")
print(far[["Facility Name", "NPDES No.", "CWNS_ID", "FACILITY_ID", "_dist_miles"]].to_string(index=False))
