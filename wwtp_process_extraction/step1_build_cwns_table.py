# Adapted from https://github.com/jiananf2/US_WWTP_GHG/tree/main/treatment_train_assignment/input_data by Abigayle Hodson, Abigayle_Hodson@lbl.gov
# Publication: https://eartharxiv.org/repository/view/7980/
# Modified by WE3Lab for California-specific analysis

import pandas as pd
import os
# WE3Lab additions
from helpers.utils import extract_leaves, build_secondary_category_lookup, apply_secondary_category_backfill, unitprocess_keywords, add_county_and_sort

# Change working directory to `data` folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Use local input data from el_abbadi/input_data directory
EL_ABBADI_DATA_DIR = os.path.join("data", "el_abbadi")
OUTPUT_DATA_DIR = os.path.join("output")  # Save to wwtp_process_extraction/output/ for compare_processes.py

ALLOWED_FACILITY_TYPES = {"Treatment Plant", "Honey Bucket Lagoon"}

# FROM SOURCE with changes to data path
#create inventory of active wwtps in 2022

#upload facility locations — base for all treatment plants regardless of flow
# change from El Abbadi which only used facilities with reported flow
facilities_2022 = pd.concat([
    pd.read_csv('data/cwns/2022/FACILITIES.csv', dtype=str),
    pd.read_csv('data/cwns/2022/FACILITIES_CONFIRMED.csv', dtype=str),
])

#upload facility types and filter to treatment plants and honey bucket lagoons
types = pd.read_csv(f'data/cwns/2022/FACILITY_TYPES.csv', dtype=str).rename(columns={'CWNS_ID': 'CWNS_NUM'})
types = types.loc[types['FACILITY_TYPE'].isin(ALLOWED_FACILITY_TYPES)].drop_duplicates(subset = 'CWNS_NUM')
types.reset_index(inplace = True, drop = True)

#start from all treatment plants (inner join on type), then left-join flow so missing flow → NaN
wwtps = facilities_2022[['CWNS_ID','STATE_CODE']].drop_duplicates(subset='CWNS_ID').rename(columns={'CWNS_ID':'CWNS_NUM','STATE_CODE':'STATE'})
wwtps = wwtps.merge(types[['CWNS_NUM','FACILITY_TYPE']], on='CWNS_NUM', how='inner')

#upload columns indicating nutrient removal in 2012 (note, 2022 CWNS does not include these columns, so we have to rely on outdated information)
nutr_rem = pd.read_excel(f'data/cwns/2012/2012_SUMMARY_EFFLUENT.xlsx', sheet_name = 'SUMMARY_EFFLUENT', dtype = {'CWNS_NUMBER':str})
nutr_rem = nutr_rem[['CWNS_NUMBER','PRES_NITROGEN_REMOVAL','PRES_PHOSPHOROUS_REMOVAL','PRES_AMMONIA_REMOVAL']].rename(columns = {'CWNS_NUMBER':'CWNS_NUM'})
wwtps = wwtps.merge(nutr_rem, on = 'CWNS_NUM', how = 'left')


#check for facilities with duplicate entries
assert wwtps['CWNS_NUM'].value_counts().max() == 1

#add leading zero to CWNS ids with less than 11 digits to ensure correct merge with other datasets
wwtps['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in wwtps['CWNS_NUM']]

#filter to wwtps in the contiguous United States and reset indexing
wwtps = wwtps.loc[(wwtps['STATE'] != 'PR') & (wwtps['STATE'] != 'AK') & (wwtps['STATE'] != 'VI') & (wwtps['STATE'] != 'HI') & (wwtps['STATE'] != 'MP') & (wwtps['STATE'] != 'GU') & (wwtps['STATE'] != 'AS')]
wwtps.reset_index(inplace = True, drop = True)

# read in unit processes from the 2022 CWNS
up2022 = pd.read_csv('data/cwns/2022/UNIT_PROCESSES.csv', dtype = {"CWNS_ID" : str})
up2022.rename(columns = {'CWNS_ID':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up2022['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in up2022['CWNS_NUM']]

# change formatting of 2022 unit process names to match that of prior years
# note: 'Biological Treatment, Other' was manually corrected to be more specific. 'Chemical N Removal' was assumed to be roughly the same energy intensity as 'Chemical P removal'
upnames_2022 = pd.read_csv(f'{EL_ABBADI_DATA_DIR}/UNIT_PROCESS_NAMES_2022.csv')
up2022 = pd.merge(left = up2022, right = upnames_2022, how = 'left', left_on = 'UNIT_PROCESS', right_on = '2022_UNIT_PROCESS_NAME')

#filter to relevant columns and rename to match the formatting of old unit process dataframes
up2022 = up2022[['CWNS_NUM','FINAL_UNIT_PROCESS_NAME','EXISTING_FLAG','PLANNED_FLAG']]
up2022.rename(columns = {'EXISTING_FLAG':'PRES_IND','PLANNED_FLAG':'PROJ_IND'}, inplace = True)
up2022.loc[up2022['PRES_IND'] == 'Y', 'PRES_IND'] = 1
up2022.loc[up2022['PRES_IND'] == 'N', 'PRES_IND'] = 0
up2022.loc[pd.isna(up2022['PRES_IND']), 'PRES_IND'] = 0
up2022.loc[up2022['PROJ_IND'] == 'Y', 'PROJ_IND'] = 1
up2022.loc[up2022['PROJ_IND'] == 'N', 'PROJ_IND'] = 0
up2022.loc[pd.isna(up2022['PROJ_IND']), 'PROJ_IND'] = 0
up2022['REPORT_YEAR'] = 2022

#read in unit processs reported in the 2004, 2008, and 2012 releases of CWNS
up2012 = pd.read_csv(f'data/cwns/2012/2012_SUMMARY_UNIT_PROCESS.csv', dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding='latin1', on_bad_lines='warn')
up2008 = pd.read_csv(f'data/cwns/2008/2008_SUMMARY_UNIT_PROCESS.csv',dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding='latin1')
up2004 = pd.read_csv(f'data/cwns/2004/2004_Unit_Processes.csv', dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding='latin1', low_memory=False)

#aggregate 2004, 2008, and 2012 unit process lists
up_old = pd.concat([up2012, up2008,up2004], axis = 0)
up_old.drop(['BACKUP_IND','PLANNED_YEAR','ADDITIONAL_NOTES','LAST_UPDATED_TS','BLANK','CHANGE_TYPE_CAT','SORT_SEQUENCE','KEEP_UP_CODE', 'CHGTP_NAME_CAT','TREATMENT_TYPE','Notes'], inplace = True, axis = 1)
up_old.rename(columns = {'CWNS_NUMBER':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up_old['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in up_old['CWNS_NUM']]

#reconcile unit process naming conventions between report years
upnames = pd.read_csv(f'{EL_ABBADI_DATA_DIR}/UNIT_PROCESS_NAMES.csv', dtype=str)
up_old = pd.merge(left = up_old, right = upnames, how = 'left', left_on = 'UNIT_PROCESS', right_on = 'ORIGINAL_UP_NAME')
up_old.drop(['ORIGINAL_UP_NAME'], inplace = True, axis = 1)

#remove processes listed as both PRES_IND = N and PROJ_IND = N; keep abandonments (classified as PAST)
up_old = up_old.loc[~((up_old['PRES_IND'] == 'N') & (up_old['PROJ_IND'] == 'N'))]
up_old = up_old[['CWNS_NUM','REPORT_YEAR','PRES_IND','PROJ_IND','CHANGE_TYPE','FINAL_UNIT_PROCESS_NAME']]

#change formatting of present and projected indices to binary
up_old.loc[up_old['PRES_IND'] == 'Y', 'PRES_IND'] = 1
up_old.loc[up_old['PRES_IND'] == 'N', 'PRES_IND'] = 0
up_old.loc[up_old['PROJ_IND'] == 'Y', 'PROJ_IND'] = 1
up_old.loc[up_old['PROJ_IND'] == 'N', 'PROJ_IND'] = 0

#join 2022 unit process list and old unit process list
# uplist_all = pd.concat([up2022, up_old], axis = 0)
up_old_raw = up_old.copy()
uplist_all = up_old

#sort by CWNS ID and reporting year
uplist_all.sort_values(by = ['CWNS_NUM','REPORT_YEAR'], ascending = True, inplace = True)

#drop duplicate unit processes and keep most recent entry
uplist_all.drop_duplicates(subset = ['CWNS_NUM', 'FINAL_UNIT_PROCESS_NAME','PRES_IND','PROJ_IND'], inplace = True, keep = 'last')
uplist_recent = uplist_all.reset_index(drop = True)

#assign key unit processes a code (ie. 'Activated Sludge' is assigned the code 'AS'); note, not all unit processes receive a code
up_eicodes = pd.read_csv(f'{EL_ABBADI_DATA_DIR}/UNIT_PROCESS_EI_CODES_WERF_modified.csv', dtype=str)
uplist_eicodes = uplist_recent.merge(up_eicodes[['FINAL_UNIT_PROCESS_NAME','WERF_CODE','DISPOSAL_CODE']].drop_duplicates(subset = ['FINAL_UNIT_PROCESS_NAME']), how = 'left', on = 'FINAL_UNIT_PROCESS_NAME')

#create column to indicate if a unit process was present in 2022
uplist_eicodes['2022_MIN_IND'] = uplist_eicodes['PRES_IND']

# WE3LAB NEW ADDITIONS

leaves = extract_leaves(unitprocess_keywords)
all_keys = [name for name, _, _ in leaves]
column_priority = {name: details.get("priority", 1) for name, details, _ in leaves if isinstance(details, dict)}
top_category_to_columns, column_secondary_categories, column_global_priority = \
    build_secondary_category_lookup(unitprocess_keywords)

cwns_to_taxonomy = {}
for process_name, details, _ in leaves:
    for cwns_name in details["cwns_processes"]:
        cwns_to_taxonomy.setdefault(cwns_name.lower().strip(), []).append(process_name)

def pad_cwns_id(x):
    s = str(x).strip()
    return '0' + s if len(s) < 11 else s

active_ups = uplist_eicodes[(uplist_eicodes['PRES_IND'] == 1) | (uplist_eicodes['PROJ_IND'] == 1)].copy()
active_ups = (active_ups.sort_values('REPORT_YEAR')
              .drop_duplicates(subset=['CWNS_NUM', 'FINAL_UNIT_PROCESS_NAME'], keep='last'))
active_ups = active_ups[active_ups['CWNS_NUM'].isin(set(wwtps['CWNS_NUM']))]

def has_change(change_type):
    # CHANGE_TYPE may be a comma-separated list; real change if any token isn't "No Change"
    if not isinstance(change_type, str):
        return False
    return any(t.strip() and t.strip().lower() != 'no change' for t in change_type.split(','))

def get_status(row):
    if row.get('CHANGE_TYPE') == 'Abandonment':
        return 'PAST'
    if row['PRES_IND'] == 1 and row['PROJ_IND'] == 1:
        # only flag a future change if an actual change is recorded; otherwise just present
        return 'PRESENT_AND_FUTURE' if has_change(row.get('CHANGE_TYPE')) else 'PRESENT'
    return 'PRESENT' if row['PRES_IND'] == 1 else 'FUTURE'

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

for proc in all_keys:
    if proc not in unit_processes_df.columns:
        unit_processes_df[proc] = '0'

facility_permit = pd.read_csv('data/cwns/2022/FACILITY_PERMIT.csv', dtype={'CWNS_ID': str, 'STATE_CODE': str})
facility_permit['CWNS_ID'] = facility_permit['CWNS_ID'].apply(pad_cwns_id)

unit_processes_df = unit_processes_df.merge(
    facility_permit[['CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE']],
    on='CWNS_ID',
    how='left'
)

facility_names = facilities_2022[['CWNS_ID', 'FACILITY_NAME', 'FACILITY_ID']].drop_duplicates(['CWNS_ID', 'FACILITY_ID'])
unit_processes_df = unit_processes_df.merge(facility_names, on='CWNS_ID', how='left')

fac12 = pd.read_csv('data/cwns/2012/Facility_Details.csv', dtype=str)
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
ids_in_export = set(ca_consolidated['CWNS_ID'].astype(str).str.strip())

ca_permits = facility_permit[
    (facility_permit['STATE_CODE'].astype(str).str.strip() == 'CA')
    & (facility_permit['PERMIT_SOURCE'] == 'NPDES')
    & (~facility_permit['PERMIT_NUMBER'].astype(str).str.upper().str.startswith('CAS'))
]
required_ids = set(ca_permits['CWNS_ID'].astype(str).str.strip())

ciwqs_path = os.path.join('data', 'ciwqs_to_cwns.csv')
ciwqs_mapping = (
    pd.read_csv(ciwqs_path, dtype=str).fillna('')
    if os.path.isfile(ciwqs_path)
    else pd.DataFrame()
)
for cid in ciwqs_mapping.get('CWNS_ID', pd.Series(dtype=str)).astype(str).str.strip():
    if cid and cid.upper() != 'NA':
        required_ids.add(pad_cwns_id(cid))

missing_ids = sorted(required_ids - ids_in_export)
if missing_ids:
    allowed_cwns_ids = set(wwtps['CWNS_NUM'].astype(str).str.strip())
    missing_ids = [cid for cid in missing_ids if cid in allowed_cwns_ids]

    permits_by_cwns = (
        ca_permits.drop_duplicates('CWNS_ID', keep='first')
        .assign(cwns_key=lambda d: d['CWNS_ID'].astype(str).str.strip())
        .set_index('cwns_key')
    )

    stripped_cwns = ciwqs_mapping.get('CWNS_ID', pd.Series(dtype=str)).astype(str).str.strip()
    ciwqs_rows = ciwqs_mapping.loc[stripped_cwns.ne('') & stripped_cwns.str.upper().ne('NA')].copy()
    ciwqs_rows['padded_cwns_id'] = ciwqs_rows['CWNS_ID'].map(pad_cwns_id)
    ciwqs_by_cwns = ciwqs_rows.drop_duplicates('padded_cwns_id', keep='first').set_index('padded_cwns_id')

    fac_name_map_2022 = facility_names.set_index('CWNS_ID')['FACILITY_NAME'].to_dict()
    fac_id_map_2022 = facility_names.set_index('CWNS_ID')['FACILITY_ID'].to_dict()

    placeholder_rows = []
    for cwns_id in missing_ids:
        row = {col: '0' for col in process_columns}
        row['CWNS_ID'] = cwns_id
        row['STATE_CODE'] = 'CA'
        row['NPDES_PERMIT'] = ''
        row['PERMIT_NUMBER'] = ''
        row['FACILITY_ID'] = fac_id_map_2022.get(cwns_id, '')
        row['FACILITY_NAME'] = fac_name_map_2022.get(cwns_id) or fac12_map.get(cwns_id, '')

        if cwns_id in permits_by_cwns.index:
            permit = str(permits_by_cwns.loc[cwns_id, 'PERMIT_NUMBER']).strip()
            row['PERMIT_NUMBER'] = permit
            row['NPDES_PERMIT'] = permit

        if cwns_id in ciwqs_by_cwns.index:
            mapping_row = ciwqs_by_cwns.loc[cwns_id]
            if not row['NPDES_PERMIT']:
                row['NPDES_PERMIT'] = str(mapping_row.get("NPDES No.", '')).strip()
            row['FACILITY_NAME'] = (row['FACILITY_NAME']
                                    or str(mapping_row.get('CWNS Facility Name', '')).strip()
                                    or str(mapping_row.get('Facility Name', '')).strip())

        if not row['PERMIT_NUMBER']:
            row['PERMIT_NUMBER'] = row['NPDES_PERMIT']

        placeholder_rows.append(row)

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

cwns_phys = pd.read_csv('data/cwns/2022/PHYSICAL_LOCATION.csv', dtype=str).fillna("")
ca_consolidated = ca_consolidated.merge(
    cwns_phys[['CWNS_ID', 'FACILITY_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates(),
    on=['CWNS_ID', 'FACILITY_ID'], how='left'
)
ca_consolidated = add_county_and_sort(ca_consolidated, "FACILITY_NAME", cwns_id_col="CWNS_ID")
ca_consolidated.to_csv(os.path.join(OUTPUT_DATA_DIR, "unit_processes_by_facility_cwns.csv"), index=False)
print(f"Saved CA consolidated CWNS: {len(ca_consolidated)} facilities")

# Track CA facilities and unit process changes across CWNS survey years
ca_ids = set(ca_consolidated['CWNS_ID'].astype(str).str.strip())
ca_up_check = up_old_raw[up_old_raw['CWNS_NUM'].astype(str).str.strip().isin(ca_ids)]
ca_up_2022 = up2022[up2022['CWNS_NUM'].astype(str).str.strip().isin(ca_ids)]

def ca_cwns_from_facility_file(path, cwns_col, state_col, state_val):
    df = pd.read_csv(path, dtype=str, encoding='latin1')
    return set(df[df[state_col].str.strip() == state_val][cwns_col].apply(pad_cwns_id))

fac_2004 = set(ca_up_check[ca_up_check['REPORT_YEAR'] == 2004]['CWNS_NUM'].astype(str).str.strip()) & ca_ids
fac_2008 = ca_cwns_from_facility_file('data/cwns/2008/Facility_Details.csv', 'CWNS Number', 'State', 'CA') & ca_ids
fac_2012 = ca_cwns_from_facility_file('data/cwns/2012/Facility_Details.csv', 'CWNS Number', 'State', 'CA') & ca_ids
fac_2022 = set(facilities_2022[facilities_2022['STATE_CODE'].str.strip() == 'CA']['CWNS_ID'].apply(pad_cwns_id)) & ca_ids

print(f"\nCA survey coverage (total in export: {len(ca_consolidated)}):")
cumulative = set()
procs_prev = None
for year, facs, up_data in [
    (2004, fac_2004, ca_up_check[ca_up_check['REPORT_YEAR'] == 2004]),
    (2008, fac_2008, ca_up_check[ca_up_check['REPORT_YEAR'] == 2008]),
    (2012, fac_2012, ca_up_check[ca_up_check['REPORT_YEAR'] == 2012]),
    (2022, fac_2022, ca_up_2022),
]:
    new = facs - cumulative
    cumulative |= facs
    procs = up_data.groupby('CWNS_NUM')['FINAL_UNIT_PROCESS_NAME'].apply(set)
    n_confirmed = len(set(up_data['CWNS_NUM'].astype(str).str.strip()) & ca_ids)
    new_str = f", {len(new)} new" if procs_prev is not None else ""
    if procs_prev is not None:
        common = procs_prev.index.intersection(procs.index)
        n_changed = sum(procs_prev[c] != procs[c] for c in common)
        pct = 100 * n_changed / len(cumulative) if cumulative else 0
        change_str = f", {n_changed}/{len(cumulative)} with process updates ({pct:.0f}%)"
    else:
        change_str = ""
    print(f"  {year}: {len(facs)} facilities ({n_confirmed} with process records{new_str}{change_str})")
    procs_prev = procs