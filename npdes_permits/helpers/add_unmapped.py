"""
Append rows to data/ciwqs_to_cwns.csv for facilities in site_data.csv that are
not yet mapped.

Matching strategy (in order):
  1. Exact facility name match (strip + upper) against cwns_processes_by_facility FACILITY_NAME
  2. Exact NPDES permit number match against cwns_processes_by_facility NPDES_PERMIT
  3. No match — row added with CWNS columns blank for manual completion

"""

import pandas as pd

DATE_FOLDER = '2026-4-25'

SITE_DATA   = f'npdes_permits/output/{DATE_FOLDER}/site_data.csv'
ALL_NPDES   = f'npdes_permits/output/{DATE_FOLDER}/all_ca_npdes.csv'
CWNS_TABLE  = 'npdes_permits/output/cwns_processes_by_facility.csv'
CIWQS_MAP   = 'npdes_permits/data/ciwqs_to_cwns.csv'


def normalize(s):
    return str(s).strip().upper() if pd.notna(s) else ''


def main():
    site   = pd.read_csv(SITE_DATA,  dtype=str).fillna('')
    npdes  = pd.read_csv(ALL_NPDES,  dtype=str).fillna('')
    cwns   = pd.read_csv(CWNS_TABLE, dtype=str).fillna('')
    ciwqs  = pd.read_csv(CIWQS_MAP,  dtype=str).fillna('')

    site['NPDES_No'] = site['NPDES_No'].str.strip().str.upper()
    ciwqs['NPDES_No'] = ciwqs['NPDES_No'].str.strip().str.upper()

    already_mapped = set(ciwqs['NPDES_No'].unique())
    unmapped = site[~site['NPDES_No'].isin(already_mapped)].copy()
    print(f'Unmapped facilities: {len(unmapped)}')

    # WDID lookup from all_ca_npdes
    npdes['_permit'] = npdes['NPDES No.'].str.strip().str.upper()
    wdid_map = npdes.set_index('_permit')['WDID'].to_dict()

    # CWNS CA subset, normalised keys
    cwns_ca = cwns[cwns['STATE_CODE'] == 'CA'].copy()
    cwns_ca['_name'] = cwns_ca['FACILITY_NAME'].apply(normalize)
    cwns_ca['_permit'] = cwns_ca['NPDES_PERMIT'].str.strip().str.upper()

    name_idx   = cwns_ca.groupby('_name').apply(lambda g: g.to_dict('records')).to_dict()
    permit_idx = cwns_ca.groupby('_permit').apply(lambda g: g.to_dict('records')).to_dict()

    new_rows = []

    for _, row in unmapped.iterrows():
        permit    = row['NPDES_No']
        fac_name  = row['Facility_Name'].strip()
        name_norm = normalize(fac_name)
        wdid      = wdid_map.get(permit, '')

        # Try name match first (skip empty names)
        cwns_hits = name_idx.get(name_norm) if name_norm else None
        match_how = 'name'

        # Fall back to permit match (skip empty permits)
        if not cwns_hits and permit:
            cwns_hits = permit_idx.get(permit)
            match_how = 'permit'

        if cwns_hits:
            for hit in cwns_hits:
                new_rows.append({
                    'WDID':              wdid,
                    'Facility_Name':     fac_name,
                    'NPDES_No':          permit,
                    'CWNS_ID':           hit['CWNS_ID'],
                    'CWNS_Facility_Name': hit['FACILITY_NAME'],
                })
            print(f'  [{match_how}] {permit} — {fac_name} → {len(cwns_hits)} CWNS row(s)')
        else:
            new_rows.append({
                'WDID':              wdid,
                'Facility_Name':     fac_name,
                'NPDES_No':          permit,
                'CWNS_ID':           '',
                'CWNS_Facility_Name': '',
            })
            print(f'  [no match] {permit} — {fac_name}')

    if not new_rows:
        print('Nothing to add.')
        return

    new_df = pd.DataFrame(new_rows, columns=['WDID','Facility_Name','NPDES_No','CWNS_ID','CWNS_Facility_Name'])
    combined = pd.concat([ciwqs, new_df], ignore_index=True)
    combined.to_csv(CIWQS_MAP, index=False)
    print(f'\nAdded {len(new_rows)} rows. ciwqs_to_cwns.csv now has {len(combined)} rows.')


if __name__ == '__main__':
    main()
