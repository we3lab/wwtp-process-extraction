# Adapted from https://github.com/jiananf2/US_WWTP_GHG/tree/main/treatment_train_assignment/input_data by Abigayle Hodson, Abigayle_Hodson@lbl.gov
# Publication: https://eartharxiv.org/repository/view/7980/
# Modified by WE3Lab for California-specific analysis

import pandas as pd
from helpers.utils import (
    extract_leaves, build_secondary_category_lookup, apply_secondary_category_backfill, unitprocess_keywords,
    add_county_and_sort, DATA_DIR, FINAL_DIR, CWNS_TABLE_CSV, CIWQS_TO_CWNS_CSV,
)

# Use local input data from el_abbadi/input_data directory
EL_ABBADI_DATA_DIR = DATA_DIR / "el_abbadi"
CWNS_DATA_DIR = DATA_DIR / "cwns"

ALLOWED_FACILITY_TYPES = {"Treatment Plant", "Honey Bucket Lagoon"}
NON_CONTIGUOUS_STATES = ['PR', 'AK', 'VI', 'HI', 'MP', 'GU', 'AS']
YES_NO_TO_BINARY = {'Y': 1, 'N': 0}


def pad_cwns_id(x):
    s = str(x).strip()
    return '0' + s if len(s) < 11 else s


def get_status(row):
    if row['CHANGE_TYPE'] == 'Abandonment':
        return 'PAST'
    if row['PRES_IND'] == 1 and row['PROJ_IND'] == 1:
        # CHANGE_TYPE may be a comma-separated list; real change if any token isn't "No Change"
        change_type = row['CHANGE_TYPE']
        has_change = isinstance(change_type, str) and any(
            t.strip() and t.strip().lower() != 'no change' for t in change_type.split(',')
        )
        # only flag a future change if an actual change is recorded; otherwise just present
        return 'PRESENT_AND_FUTURE' if has_change else 'PRESENT'
    return 'PRESENT' if row['PRES_IND'] == 1 else 'FUTURE'


def percent_of(value, total):
    return '' if value == '' else round(100 * value / total, 1)


# FROM SOURCE with changes to data path
#create inventory of active wwtps in 2022

#upload facility locations — base for all treatment plants regardless of flow
# change from El Abbadi which only used facilities with reported flow
facilities_2022 = pd.concat([
    pd.read_csv(CWNS_DATA_DIR / '2022' / 'FACILITIES.csv', dtype=str),
    pd.read_csv(CWNS_DATA_DIR / '2022' / 'FACILITIES_CONFIRMED.csv', dtype=str),
])

#upload facility types and filter to treatment plants and honey bucket lagoons
types = pd.read_csv(CWNS_DATA_DIR / '2022' / 'FACILITY_TYPES.csv', dtype=str).rename(columns={'CWNS_ID': 'CWNS_NUM'})
types = types.loc[types['FACILITY_TYPE'].isin(ALLOWED_FACILITY_TYPES)].drop_duplicates(subset = 'CWNS_NUM')
types.reset_index(inplace = True, drop = True)

#start from all treatment plants (inner join on type)
wwtps = facilities_2022[['CWNS_ID','STATE_CODE']].drop_duplicates(subset='CWNS_ID').rename(columns={'CWNS_ID':'CWNS_NUM','STATE_CODE':'STATE'})
wwtps = wwtps.merge(types[['CWNS_NUM','FACILITY_TYPE']], on='CWNS_NUM', how='inner')

#check for facilities with duplicate entries
assert wwtps['CWNS_NUM'].value_counts().max() == 1

#add leading zero to CWNS ids with less than 11 digits to ensure correct merge with other datasets
wwtps['CWNS_NUM'] = wwtps['CWNS_NUM'].apply(pad_cwns_id)

#filter to wwtps in the contiguous United States and reset indexing
wwtps = wwtps.loc[~wwtps['STATE'].isin(NON_CONTIGUOUS_STATES)]
wwtps.reset_index(inplace = True, drop = True)

# read in unit processes from the 2022 CWNS
up2022 = pd.read_csv(CWNS_DATA_DIR / '2022' / 'UNIT_PROCESSES.csv', dtype = {"CWNS_ID" : str})
up2022.rename(columns = {'CWNS_ID':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up2022['CWNS_NUM'] = up2022['CWNS_NUM'].apply(pad_cwns_id)

# change formatting of 2022 unit process names to match that of prior years
# note: 'Biological Treatment, Other' was manually corrected to be more specific. 'Chemical N Removal' was assumed to be roughly the same energy intensity as 'Chemical P removal'
upnames_2022 = pd.read_csv(EL_ABBADI_DATA_DIR / 'UNIT_PROCESS_NAMES_2022.csv')
up2022 = pd.merge(left = up2022, right = upnames_2022, how = 'left', left_on = 'UNIT_PROCESS', right_on = '2022_UNIT_PROCESS_NAME')

#filter to relevant columns and rename to match the formatting of old unit process dataframes
up2022 = up2022[['CWNS_NUM','FINAL_UNIT_PROCESS_NAME','EXISTING_FLAG','PLANNED_FLAG']]
up2022.rename(columns = {'EXISTING_FLAG':'PRES_IND','PLANNED_FLAG':'PROJ_IND'}, inplace = True)
up2022['PRES_IND'] = up2022['PRES_IND'].map(YES_NO_TO_BINARY).fillna(0)
up2022['PROJ_IND'] = up2022['PROJ_IND'].map(YES_NO_TO_BINARY).fillna(0)
up2022['REPORT_YEAR'] = 2022

#read in unit processs reported in the 2004, 2008, and 2012 releases of CWNS
old_up_dtypes = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}
up2012 = pd.read_csv(CWNS_DATA_DIR / '2012' / '2012_SUMMARY_UNIT_PROCESS.csv', dtype = old_up_dtypes, encoding='latin1', on_bad_lines='warn')
up2008 = pd.read_csv(CWNS_DATA_DIR / '2008' / '2008_SUMMARY_UNIT_PROCESS.csv', dtype = old_up_dtypes, encoding='latin1')
up2004 = pd.read_csv(CWNS_DATA_DIR / '2004' / '2004_Unit_Processes.csv', dtype = old_up_dtypes, encoding='latin1', low_memory=False)

#aggregate 2004, 2008, and 2012 unit process lists
up_old = pd.concat([up2012, up2008,up2004], axis = 0)
up_old.drop(['BACKUP_IND','PLANNED_YEAR','ADDITIONAL_NOTES','LAST_UPDATED_TS','BLANK','CHANGE_TYPE_CAT','SORT_SEQUENCE','KEEP_UP_CODE', 'CHGTP_NAME_CAT','TREATMENT_TYPE','Notes'], inplace = True, axis = 1)
up_old.rename(columns = {'CWNS_NUMBER':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up_old['CWNS_NUM'] = up_old['CWNS_NUM'].apply(pad_cwns_id)

#reconcile unit process naming conventions between report years
upnames = pd.read_csv(EL_ABBADI_DATA_DIR / 'UNIT_PROCESS_NAMES.csv', dtype=str)
up_old = pd.merge(left = up_old, right = upnames, how = 'left', left_on = 'UNIT_PROCESS', right_on = 'ORIGINAL_UP_NAME')
up_old.drop(['ORIGINAL_UP_NAME'], inplace = True, axis = 1)

#remove processes listed as both PRES_IND = N and PROJ_IND = N; keep abandonments (classified as PAST)
up_old = up_old.loc[~((up_old['PRES_IND'] == 'N') & (up_old['PROJ_IND'] == 'N'))]
up_old = up_old[['CWNS_NUM','REPORT_YEAR','PRES_IND','PROJ_IND','CHANGE_TYPE','FINAL_UNIT_PROCESS_NAME']]

#change formatting of present and projected indices to binary
up_old['PRES_IND'] = up_old['PRES_IND'].map(YES_NO_TO_BINARY)
up_old['PROJ_IND'] = up_old['PROJ_IND'].map(YES_NO_TO_BINARY)

up_old_raw = up_old.copy()
uplist_all = up_old

#sort by CWNS ID and reporting year
uplist_all.sort_values(by = ['CWNS_NUM','REPORT_YEAR'], ascending = True, inplace = True)

#drop duplicate unit processes and keep most recent entry
uplist_all.drop_duplicates(subset = ['CWNS_NUM', 'FINAL_UNIT_PROCESS_NAME','PRES_IND','PROJ_IND'], inplace = True, keep = 'last')
uplist_recent = uplist_all.reset_index(drop = True)

# WE3LAB NEW ADDITIONS

leaves = extract_leaves(unitprocess_keywords)
all_keys = [name for name, _, _ in leaves]
column_priority = {name: details.get("priority", 1) for name, details, _ in leaves}
top_category_to_columns, column_secondary_categories, column_global_priority = \
    build_secondary_category_lookup(unitprocess_keywords)

cwns_to_taxonomy = {}
for process_name, details, _ in leaves:
    for cwns_name in details["cwns_processes"]:
        cwns_to_taxonomy.setdefault(cwns_name.lower().strip(), []).append(process_name)

active_ups = uplist_recent[(uplist_recent['PRES_IND'] == 1) | (uplist_recent['PROJ_IND'] == 1)].copy()
active_ups = (active_ups.sort_values('REPORT_YEAR', kind='stable')
              .drop_duplicates(subset=['CWNS_NUM', 'FINAL_UNIT_PROCESS_NAME'], keep='last'))
active_ups = active_ups[active_ups['CWNS_NUM'].isin(set(wwtps['CWNS_NUM']))]

active_ups['STATUS'] = active_ups.apply(get_status, axis=1)
active_ups['PROCESS'] = active_ups['FINAL_UNIT_PROCESS_NAME'].str.lower().str.strip().map(cwns_to_taxonomy)

unit_processes_df = (
    active_ups[['CWNS_NUM', 'PROCESS', 'STATUS']]
    .explode('PROCESS')
    .dropna(subset=['PROCESS'])
    .drop_duplicates(subset=['CWNS_NUM', 'PROCESS'])
    .pivot(index='CWNS_NUM', columns='PROCESS', values='STATUS')
    .fillna('0')
    .reset_index()
    .rename(columns={'CWNS_NUM': 'CWNS_ID'})
)
unit_processes_df.columns.name = None

unit_processes_df = unit_processes_df.reindex(
    columns=list(unit_processes_df.columns) + [k for k in all_keys if k not in unit_processes_df.columns],
    fill_value='0',
)

facility_permit = pd.read_csv(CWNS_DATA_DIR / '2022' / 'FACILITY_PERMIT.csv', dtype={'CWNS_ID': str, 'STATE_CODE': str})
facility_permit['CWNS_ID'] = facility_permit['CWNS_ID'].apply(pad_cwns_id)

unit_processes_df = unit_processes_df.merge(
    facility_permit[['CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE']],
    on='CWNS_ID',
    how='left'
)

facility_names = facilities_2022[['CWNS_ID', 'FACILITY_NAME', 'FACILITY_ID']].drop_duplicates(['CWNS_ID', 'FACILITY_ID'])
unit_processes_df = unit_processes_df.merge(facility_names, on='CWNS_ID', how='left')

fac12 = pd.read_csv(CWNS_DATA_DIR / '2012' / 'Facility_Details.csv', dtype=str)
fac12['CWNS Number'] = fac12['CWNS Number'].apply(pad_cwns_id)
fac12_map = fac12.drop_duplicates('CWNS Number').set_index('CWNS Number')['Facility/Project Name']
null_name = unit_processes_df['FACILITY_NAME'].isna()
unit_processes_df.loc[null_name, 'FACILITY_NAME'] = unit_processes_df.loc[null_name, 'CWNS_ID'].map(fac12_map)

npdes_only = (facility_permit[facility_permit['PERMIT_SOURCE'] == 'NPDES']
              [['CWNS_ID', 'PERMIT_NUMBER']]
              .drop_duplicates(subset='CWNS_ID', keep='first')
              .rename(columns={'PERMIT_NUMBER': 'NPDES_PERMIT'}))
unit_processes_df = unit_processes_df.merge(npdes_only, on='CWNS_ID', how='left')

ca_only = unit_processes_df[unit_processes_df['STATE_CODE'] == 'CA'].copy()
ca_consolidated = ca_only.groupby('CWNS_ID', dropna=False, sort=False).first().reset_index()

meta_cols = {'CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE', 'FACILITY_NAME', 'NPDES_PERMIT', 'FACILITY_ID'}
process_columns = [c for c in ca_consolidated.columns if c not in meta_cols]

# CA facilities with an NPDES permit (excluding stormwater CAS permits) or a CIWQS match get a
# placeholder row even when CWNS lists no unit processes for them
ca_permits = facility_permit[
    (facility_permit['STATE_CODE'].str.strip() == 'CA')
    & (facility_permit['PERMIT_SOURCE'] == 'NPDES')
    & (~facility_permit['PERMIT_NUMBER'].astype(str).str.upper().str.startswith('CAS'))
].drop_duplicates('CWNS_ID')
permit_by_cwns = dict(zip(ca_permits['CWNS_ID'], ca_permits['PERMIT_NUMBER'].astype(str).str.strip()))

ciwqs_mapping = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str).fillna('')
mapped_cwns = ciwqs_mapping['CWNS_ID'].str.strip()
ciwqs_rows = ciwqs_mapping[mapped_cwns.ne('') & mapped_cwns.str.upper().ne('NA')].copy()
ciwqs_rows['CWNS_ID'] = ciwqs_rows['CWNS_ID'].map(pad_cwns_id)
ciwqs_by_cwns = (ciwqs_rows.drop_duplicates('CWNS_ID').set_index('CWNS_ID')
                 [['NPDES No.', 'CWNS Facility Name', 'Facility Name']].to_dict('index'))

fac_name_map_2022 = facility_names.set_index('CWNS_ID')['FACILITY_NAME'].to_dict()
fac_id_map_2022 = facility_names.set_index('CWNS_ID')['FACILITY_ID'].to_dict()

required_ids = set(permit_by_cwns) | set(ciwqs_by_cwns)
missing_ids = sorted((required_ids - set(ca_consolidated['CWNS_ID'])) & set(wwtps['CWNS_NUM']))
placeholder_rows = []
for cwns_id in missing_ids:
    mapping_row = ciwqs_by_cwns.get(cwns_id, {})
    permit = permit_by_cwns.get(cwns_id, '')
    npdes = permit or mapping_row.get('NPDES No.', '').strip()
    placeholder_rows.append({
        **{col: '0' for col in process_columns},
        'CWNS_ID': cwns_id,
        'STATE_CODE': 'CA',
        'NPDES_PERMIT': npdes,
        'PERMIT_NUMBER': permit or npdes,
        'FACILITY_ID': fac_id_map_2022.get(cwns_id, ''),
        'FACILITY_NAME': (fac_name_map_2022.get(cwns_id) or fac12_map.get(cwns_id, '')
                          or mapping_row.get('CWNS Facility Name', '').strip()
                          or mapping_row.get('Facility Name', '').strip()),
    })
ca_consolidated = pd.concat(
    [ca_consolidated, pd.DataFrame(placeholder_rows).reindex(columns=ca_consolidated.columns)],
    ignore_index=True,
)
print(f"Added {len(placeholder_rows)} CA CWNS placeholder rows")

proc_cols_backfill = [c for c in ca_consolidated.columns if c in set(all_keys)]
for idx in ca_consolidated.index:
    status_dict = ca_consolidated.loc[idx, proc_cols_backfill].to_dict()
    apply_secondary_category_backfill(
        status_dict, column_secondary_categories, top_category_to_columns,
        column_global_priority, column_priority,
    )
    ca_consolidated.loc[idx, proc_cols_backfill] = pd.Series(status_dict)

cwns_phys = pd.read_csv(CWNS_DATA_DIR / '2022' / 'PHYSICAL_LOCATION.csv', dtype=str).fillna("")
ca_consolidated = ca_consolidated.merge(
    cwns_phys[['CWNS_ID', 'FACILITY_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates(),
    on=['CWNS_ID', 'FACILITY_ID'], how='left'
)
ca_consolidated = add_county_and_sort(ca_consolidated, "FACILITY_NAME", cwns_id_col="CWNS_ID")
ca_consolidated.to_csv(CWNS_TABLE_CSV, index=False)
print(f"Saved CA consolidated CWNS: {len(ca_consolidated)} facilities")

# CWNS survey years: facilities listed, and unit process records, for each year
YEARS = [2004, 2008, 2012, 2022]
us_ids = set(wwtps['CWNS_NUM'])
ca_ids = set(ca_consolidated['CWNS_ID'])
facilities_by_year = {
    2004: set(up_old_raw.loc[up_old_raw['REPORT_YEAR'] == 2004, 'CWNS_NUM']),
    2008: set(pd.read_csv(CWNS_DATA_DIR / '2008' / 'Facility_Details.csv', dtype=str, encoding='latin1')
              ['CWNS Number'].apply(pad_cwns_id)),
    2012: set(fac12['CWNS Number']),
    2022: set(facilities_2022['CWNS_ID'].apply(pad_cwns_id)),
}
ups_by_year = {y: up_old_raw[up_old_raw['REPORT_YEAR'] == y] for y in YEARS[:3]} | {2022: up2022}

# Most recent CWNS survey year with unit process records; 0 = never reported
up2022_active = up2022[(up2022['PRES_IND'] == 1) | (up2022['PROJ_IND'] == 1)]
up_years = pd.concat([up_old_raw[['CWNS_NUM', 'REPORT_YEAR']], up2022_active[['CWNS_NUM', 'REPORT_YEAR']]])
latest_year = up_years.groupby('CWNS_NUM')['REPORT_YEAR'].max()

# Table S1: CWNS survey coverage by report year, CA vs. US-wide
# all percentages are of the 2022 cumulative facility count (CA 389, US 16201)
table_rows = []
for label, ids in [('CA', ca_ids), ('US-wide', us_ids)]:
    rec = pd.Series(sorted(ids)).map(latest_year).fillna(0).astype(int).value_counts()
    cum = set()
    procs_prev = None
    n_new, n_upd, n_recent = {}, {}, {}
    for year in YEARS:
        facs = facilities_by_year[year] & ids
        up_data = ups_by_year[year][ups_by_year[year]['CWNS_NUM'].isin(ids)]
        n_new[year] = len(facs - cum)
        cum |= facs
        procs = up_data.groupby('CWNS_NUM')['FINAL_UNIT_PROCESS_NAME'].apply(set)
        if procs_prev is not None:
            common = procs_prev.index.intersection(procs.index)
            n_upd[year] = sum(procs_prev[c] != procs[c] for c in common)
        else:
            n_upd[year] = ''  # no prior survey to compare against
        n_recent[year] = int(rec.get(year, 0))
        procs_prev = procs

    total = len(cum)
    for metric, counts in [
        ('New facilities added (#)', n_new),
        ('New facilities added (% of 2022 cumulative)', {y: percent_of(n_new[y], total) for y in YEARS}),
        ('Facilities with process updates (#)', n_upd),
        ('Facilities with process updates (% of 2022 cumulative)', {y: percent_of(n_upd[y], total) for y in YEARS}),
        ('Facilities with most recent update in this year (#)', n_recent),
        ('Facilities with most recent update in this year (% of 2022 cumulative)', {y: percent_of(n_recent[y], total) for y in YEARS}),
    ]:
        table_rows.append({'Region': label, 'Metric': metric, **{str(y): counts[y] for y in YEARS}})

table_s1 = pd.DataFrame(table_rows, dtype=object)
table_s1.to_csv(FINAL_DIR / 'table_s1.csv', index=False)
print(f"Saved table_s1.csv (CA n={len(ca_ids)}, US n={len(us_ids)})")
