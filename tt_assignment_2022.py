# Adapted from Date: Abigayle Hodson, Abigayle_Hodson@lbl.gov
# https://github.com/jiananf2/US_WWTP_GHG/tree/main/treatment_train_assignment/input_data
# Publication https://eartharxiv.org/repository/view/7980/

import pandas as pd
import numpy as np

# ADDED Base GitHub URL for input data
GITHUB_BASE_URL = "https://raw.githubusercontent.com/jiananf2/US_WWTP_GHG/refs/heads/main/treatment_train_assignment/input_data"

# FROM SOURCE with changes to data path
def create_wwtp_inventory():
  '''
  Function that creates an inventory of active wwtps in the United States for 2022 using CWNS data
    Returns:
      wwtps_all = dataframe containing the active wwtps and relevant characteristics (ie. location, flow rate, and nutrient removal flags) for 2022
  '''
  #upload facility flow rates from 2022 CWNS
  flow = pd.read_csv(f'{GITHUB_BASE_URL}/FLOW_2022.csv', dtype = {'CWNS_ID':str})

  #filter to total flow
  flow = flow.loc[flow['FLOW_TYPE'] == 'Total Flow']

  #filter to facilities that report non-zero, non-nan flow in 2022
  flow = flow.loc[(flow['CURRENT_DESIGN_FLOW'] != 0) & (~pd.isna(flow['CURRENT_DESIGN_FLOW']))]
  flow.reset_index(inplace = True, drop = True)

  #create dataframe of active wwtps based on facilities that report flow in 2022
  wwtps_all = flow[['CWNS_ID', 'CURRENT_DESIGN_FLOW', 'FUTURE_DESIGN_FLOW']].rename(columns = {'CURRENT_DESIGN_FLOW':'FLOW_2022_MGD', 'FUTURE_DESIGN_FLOW':'FLOW_PROJ_MGD'})

  #upload columns indicating nutrient removal in 2012 (note, 2022 CWNS does not include these columns, so we have to rely on outdated information)
  nutr_rem = pd.read_excel(f'{GITHUB_BASE_URL}/2012_SUMMARY_EFFLUENT.xlsx', sheet_name = 'SUMMARY_EFFLUENT', dtype = {'CWNS_NUMBER':str})
  nutr_rem = nutr_rem[['CWNS_NUMBER','PRES_NITROGEN_REMOVAL','PRES_PHOSPHOROUS_REMOVAL','PRES_AMMONIA_REMOVAL']].rename(columns = {'CWNS_NUMBER':'CWNS_ID'})

  #merge nutrient removal information with main dataframe and rename columns
  wwtps_all = wwtps_all.merge(nutr_rem, on = 'CWNS_ID', how = 'left')

  #upload facility locations
  locations = pd.read_csv('data/cwns/2022/FACILITIES.csv', dtype = {'CWNS_ID':str})

  #add state column to main dataframe
  wwtps_all = wwtps_all.merge(locations[['CWNS_ID','STATE_CODE']], on = 'CWNS_ID', how = 'left')
  wwtps_all.rename(columns = {'CWNS_ID':'CWNS_NUM','STATE_CODE':'STATE'}, inplace = True)

  #upload facility types
  types = pd.read_csv(f'{GITHUB_BASE_URL}/2022_FACILITY_TYPES.csv', dtype = {'CWNS_ID':str}).rename(columns= {'CWNS_ID':'CWNS_NUM'})

  #filter to just treatment plants and honey bucket lagoons
  types = types.loc[(types['FACILITY_TYPE'] == 'Treatment Plant') | (types['FACILITY_TYPE'] == 'Honey Bucket Lagoon')].drop_duplicates(subset = 'CWNS_NUM')
  types.reset_index(inplace = True, drop = True)

  #merge facility types with main dataframe to screen out non-treatment plants and honey bucket lagoons from inventory
  wwtps_all = wwtps_all.merge(types[['CWNS_NUM','FACILITY_TYPE']], on = 'CWNS_NUM', how = 'inner')

  #categorize average daily flow rate
  wwtps_all.loc[wwtps_all['FLOW_2022_MGD'] < 2, '2022_FLOW_CAT_MGD'] = 'LESS THAN 2'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 2) & (wwtps_all['FLOW_2022_MGD'] < 4), '2022_FLOW_CAT_MGD'] = '2 TO 4'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 4) & (wwtps_all['FLOW_2022_MGD'] < 7), '2022_FLOW_CAT_MGD'] = '4 TO 7'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 7) & (wwtps_all['FLOW_2022_MGD'] < 16), '2022_FLOW_CAT_MGD'] = '7 TO 16'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 16) & (wwtps_all['FLOW_2022_MGD'] < 46), '2022_FLOW_CAT_MGD'] = '16 TO 46'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 46) & (wwtps_all['FLOW_2022_MGD'] < 100), '2022_FLOW_CAT_MGD'] = '46 TO 100'
  wwtps_all.loc[(wwtps_all['FLOW_2022_MGD'] >= 100), '2022_FLOW_CAT_MGD'] = '100 AND ABOVE'

  #categorize projected flow rate
  wwtps_all.loc[wwtps_all['FLOW_PROJ_MGD'] < 2, 'PROJ_FLOW_CAT_MGD'] = 'LESS THAN 2'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 2) & (wwtps_all['FLOW_PROJ_MGD'] < 4), 'PROJ_FLOW_CAT_MGD'] = '2 TO 4'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 4) & (wwtps_all['FLOW_PROJ_MGD'] < 7), 'PROJ_FLOW_CAT_MGD'] = '4 TO 7'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 7) & (wwtps_all['FLOW_PROJ_MGD'] < 16), 'PROJ_FLOW_CAT_MGD'] = '7 TO 16'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 16) & (wwtps_all['FLOW_PROJ_MGD'] < 46), 'PROJ_FLOW_CAT_MGD'] = '16 TO 46'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 46) & (wwtps_all['FLOW_PROJ_MGD'] < 100), 'PROJ_FLOW_CAT_MGD'] = '46 TO 100'
  wwtps_all.loc[(wwtps_all['FLOW_PROJ_MGD'] >= 100), 'PROJ_FLOW_CAT_MGD'] = '100 AND ABOVE'

  #upload EPA regions by state
  epa_regions = pd.read_csv(f'{GITHUB_BASE_URL}/state_EPA_regions.csv', dtype = {'STATE':str})

  #add column for EPA region
  wwtps_all = wwtps_all.merge(epa_regions, on = 'STATE', how = 'left')

  #check for facilities with duplicate entries
  assert wwtps_all['CWNS_NUM'].value_counts().max() == 1

  return wwtps_all

#create inventory of active wwtps in 2022
wwtps = create_wwtp_inventory()

#add leading zero to CWNS ids with less than 11 digits to ensure correct merge with other datasets
wwtps['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in wwtps['CWNS_NUM']]

#filter to wwtps in the contiguous United States and reset indexing
wwtps = wwtps.loc[(wwtps['STATE'] != 'PR') & (wwtps['STATE'] != 'AK') & (wwtps['STATE'] != 'VI') & (wwtps['STATE'] != 'HI') & (wwtps['STATE'] != 'MP') & (wwtps['STATE'] != 'GU') & (wwtps['STATE'] != 'AS')]
wwtps.reset_index(inplace = True, drop = True)

#read in data from the Water Environment Federation's (WEF) biogas database (https://app.powerbi.com/view?r=eyJrIjoiMGFjZDFjZmItMjQ5Yi00ZTlhLWJmNTQtODFiNjlkYjFlODJjIiwidCI6ImI3ZTk3ODAyLTJhNjktNDc3ZS1iN2QyLWY0ZDE2MWMyMTBjYiIsImMiOjF9) to identify wwtps that utilize biogas for electricity generation; note, data was pulled ~2018 before website switched to MS BI
wef_biogas = pd.read_csv(f'{GITHUB_BASE_URL}/WERF_BIOGAS.csv', dtype = {'CWNS_NUM' : str}, encoding = 'latin1')

#change formatting of WEF data from string to binary
wef_biogas.loc[wef_biogas['Electricity_from_combustion-engine'] != 'yes', 'Electricity_from_combustion-engine'] = 0
wef_biogas.loc[wef_biogas['Electricity_from_combustion-engine'] == 'yes', 'Electricity_from_combustion-engine'] = 1
wef_biogas.loc[wef_biogas['Electricity_from_microturbine'] != 'yes', 'Electricity_from_microturbine'] = 0
wef_biogas.loc[wef_biogas['Electricity_from_microturbine'] == 'yes', 'Electricity_from_microturbine'] = 1
wef_biogas.loc[wef_biogas['Electricity_from_turbine'] != 'yes', 'Electricity_from_turbine'] = 0
wef_biogas.loc[wef_biogas['Electricity_from_turbine'] == 'yes', 'Electricity_from_turbine'] = 1
wef_biogas.loc[wef_biogas['Electricity_from_fuelcell'] != 'yes', 'Electricity_from_fuelcell'] = 0
wef_biogas.loc[wef_biogas['Electricity_from_fuelcell'] == 'yes', 'Electricity_from_fuelcell'] = 1
wef_biogas.loc[wef_biogas['Electricity_supplied_to_grid'] != 'yes', 'Electricity_supplied_to_grid'] = 0
wef_biogas.loc[wef_biogas['Electricity_supplied_to_grid'] == 'yes', 'Electricity_supplied_to_grid'] = 1
wef_biogas.loc[wef_biogas['AD'] == 'yes', 'AD'] = 1

#create column that indicates if biogas is used to generate electricity
wef_biogas['biogas_werf'] = wef_biogas['Electricity_from_combustion-engine'] + wef_biogas['Electricity_from_microturbine'] + wef_biogas['Electricity_from_turbine'] + wef_biogas['Electricity_from_fuelcell'] + wef_biogas['Electricity_supplied_to_grid']
wef_biogas.loc[wef_biogas['biogas_werf'] > 0, 'BIOGASELEC_WERF'] = 1

#drop nan values from biogas dataframe to avoid creating duplicates in wwtps dataframe post-merge
wef_biogas = wef_biogas.dropna(subset = 'CWNS_NUM')

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
wef_biogas['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in wef_biogas['CWNS_NUM']]

#create new dataframe that just indicates whether biogas is utilized for electricity generation, according to WEF
biogas_wef = wef_biogas[['CWNS_NUM','BIOGASELEC_WERF']].dropna()

#merge biogas info from WEF with faciltiies info from CWNS
wwtps = pd.merge(left = wwtps, right = biogas_wef, how = 'left', on = 'CWNS_NUM')

#replace nan values with zeros
wwtps['BIOGASELEC_WERF'] = wwtps['BIOGASELEC_WERF'].fillna(0)

#read in data from the Department of Energy's Combined Heat and Power Installation database (https://doe.icfwebservices.com/chp) to identify wwtps that utilize biogas for electricity generation
doe_biogas = pd.read_csv(f'{GITHUB_BASE_URL}/doe_chpdb-WWTP.csv', dtype = {'CWNS_NUM' : str}, encoding = 'latin1')

#drop duplicate and nan values from biogas dataframe to avoid creating duplicates in wwtps dataframe post-merge
doe_biogas.dropna(subset = 'CWNS_NUM', inplace = True)

#add a leading zero to old CWNS ids with a length less than 11 to ensure proper merge
doe_biogas['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in doe_biogas['CWNS_NUM']]
doe_biogas.drop_duplicates(subset = 'CWNS_NUM', inplace = True)

#create a dataframe that just indicates whether biogas is used for electricity generation, according to DOE
biogas_doe = doe_biogas[['CWNS_NUM','BIOGAS_DOE_2022']]

#merge biogas info from DOE with faciltiies info from CWNS
wwtps = pd.merge(left = wwtps, right = biogas_doe, how = 'left', on = 'CWNS_NUM')

#replace nan values with zeros
wwtps['BIOGAS_DOE_2022'] = wwtps['BIOGAS_DOE_2022'].fillna(0)

#create column that indicate whether biogas was used for electricity generation based on the results of both DOE and WEF
wwtps['BIOGAS_EL_2022'] = 0
wwtps.loc[((wwtps['BIOGASELEC_WERF'] + wwtps['BIOGAS_DOE_2022']) > 0), 'BIOGAS_EL_2022'] = 1

#read in unit processes from the 2022 CWNS
up2022 = pd.read_csv('data/cwns/2022/UNIT_PROCESSES.csv', dtype = {"CWNS_ID" : str})
up2022.rename(columns = {'CWNS_ID':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up2022['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in up2022['CWNS_NUM']]

#change formatting of 2022 unit process names to match that of prior years
#note: 'Biological Treatment, Other' was manually corrected to be more specific. 'Chemical N Removal' was assumed to be roughly the same energy intensity as 'Chemical P removal'
upnames_2022 = pd.read_csv(f'{GITHUB_BASE_URL}/UNIT_PROCESS_NAMES_2022.csv')
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
up2012 = pd.read_csv(f'{GITHUB_BASE_URL}/2012_SUMMARY_UNIT_PROCESS.csv', dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding = 'latin1')
up2008 = pd.read_csv(f'{GITHUB_BASE_URL}/2008_SUMMARY_UNIT_PROCESS.csv',dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding = 'latin1')
up2004 = pd.read_csv(f'{GITHUB_BASE_URL}/2004_Unit_Processes.csv', dtype = {'REPORT_YEAR':int, "CWNS_NUMBER":str, "TREATMENT_TYPE":str,"UNIT_PROCESS":str}, encoding = 'latin1')

#aggregate 2004, 2008, and 2012 unit process lists
up_old = pd.concat([up2012, up2008,up2004], axis = 0)
up_old.drop(['BACKUP_IND','PLANNED_YEAR','ADDITIONAL_NOTES','LAST_UPDATED_TS','BLANK','CHANGE_TYPE_CAT','SORT_SEQUENCE','KEEP_UP_CODE', 'CHGTP_NAME_CAT','TREATMENT_TYPE','Notes'], inplace = True, axis = 1)
up_old.rename(columns = {'CWNS_NUMBER':'CWNS_NUM'}, inplace = True)

#add a leading zero to CWNS ids with a length less than 11 to ensure proper merge
up_old['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in up_old['CWNS_NUM']]

#reconcile unit process naming conventions between report years
upnames = pd.read_csv(f'{GITHUB_BASE_URL}/UNIT_PROCESS_NAMES.csv')
up_old = pd.merge(left = up_old, right = upnames, how = 'left', left_on = 'UNIT_PROCESS', right_on = 'ORIGINAL_UP_NAME')
up_old.drop(['ORIGINAL_UP_NAME'], inplace = True, axis = 1)

#remove processes listed for abandoment in 2004, 2008, or 2012 and processes listed as both PRES_IND = N and PROJ_IND = N
up_old = up_old.loc[up_old['CHANGE_TYPE'] != 'Abandonment']
up_old = up_old.loc[~((up_old['PRES_IND'] == 'N') & (up_old['PRES_IND'] == 'N'))]
up_old = up_old[['CWNS_NUM','REPORT_YEAR','PRES_IND','PROJ_IND','FINAL_UNIT_PROCESS_NAME']]

#change formatting of present and projected indices to binary
up_old.loc[up_old['PRES_IND'] == 'Y', 'PRES_IND'] = 1
up_old.loc[up_old['PRES_IND'] == 'N', 'PRES_IND'] = 0
up_old.loc[up_old['PROJ_IND'] == 'Y', 'PROJ_IND'] = 1
up_old.loc[up_old['PROJ_IND'] == 'N', 'PROJ_IND'] = 0

#join 2022 unit process list and old unit process list
uplist_all = pd.concat([up2022, up_old], axis = 0)

#sort by CWNS ID and reporting year
uplist_all.sort_values(by = ['CWNS_NUM','REPORT_YEAR'], ascending = True, inplace = True)

#drop duplicate unit processes and keep most recent entry
uplist_all.drop_duplicates(subset = ['CWNS_NUM', 'FINAL_UNIT_PROCESS_NAME','PRES_IND','PROJ_IND'], inplace = True, keep = 'last')
uplist_recent = uplist_all.reset_index(drop = True)

#use supplementary biogas databases to add anaerobic digestion to unit process list for facilities flagged as using biogas for electricity production
wef_biogas_ad = wef_biogas[['CWNS_NUM', 'AD']].drop_duplicates(subset = 'CWNS_NUM')
wef_biogas_ad.loc[wef_biogas_ad['AD'] == 1, 'FINAL_UNIT_PROCESS_NAME'] = 'Biosolids Anaerobic Digestion, Other'
wef_biogas_ad['REPORT_YEAR'] = 2013
wef_biogas_ad['PRES_IND'] = 1
wef_biogas_ad['PROJ_IND'] = 1
wef_biogas_ad = wef_biogas_ad[['CWNS_NUM','FINAL_UNIT_PROCESS_NAME','REPORT_YEAR','PRES_IND','PROJ_IND']]

doe_biogas_ad = doe_biogas.loc[doe_biogas['BIOGAS_DOE_2022'] == 1][['CWNS_NUM','Last Verified']]
doe_biogas_ad.rename(columns = {'Last Verified':'REPORT_YEAR'}, inplace = True)
doe_biogas_ad['FINAL_UNIT_PROCESS_NAME'] = 'Biosolids Anaerobic Digestion, Other'
doe_biogas_ad['PRES_IND'] = 1
doe_biogas_ad['PROJ_IND'] = 1

#merge additional anaerobic digestion processes with cumulative unit process list
uplist_recent = pd.concat([uplist_recent, wef_biogas_ad, doe_biogas_ad], axis = 0, ignore_index = False)

#assign key unit processes a code (ie. 'Activated Sludge' is assigned the code 'AS'); note, not all unit processes receive a code
up_eicodes = pd.read_csv(f'{GITHUB_BASE_URL}/UNIT_PROCESS_EI_CODES_WERF.csv')
uplist_eicodes = uplist_recent.merge(up_eicodes[['FINAL_UNIT_PROCESS_NAME','WERF_CODE','DISPOSAL_CODE']].drop_duplicates(subset = ['FINAL_UNIT_PROCESS_NAME']), how = 'left', on = 'FINAL_UNIT_PROCESS_NAME')

#create column to indicate if a unit process was present in 2022
uplist_eicodes['2022_MIN_IND'] = uplist_eicodes['PRES_IND']

#manual corrections to cumulative unit process list (Christina Polcuch, 2023) for large facilities that were initially assigned multiple treatment trains
#here, we conducted manual verification of unit processes for selected facilities using publicly available information. Relevant unit processes were updated accordingly in the unit process list, and the note "Corrected based on manual check of large facilities with multiple treatment train assignments (2023)" was added in the UP_ID_NOTE column.

# REMOVED manual edits to the list 
# #fix Lewiston, ME; no nutrient removal
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '23000011001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '23000011001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '23000011001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix Brockton, MA; no TF, no incineration
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'PROJ_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'MHI')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'MHI')), 'PROJ_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '25000024001') & ((uplist_eicodes['WERF_CODE'] == 'MHI')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix GLWA plants; no phosphorus or nutrient removal
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000569001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000569001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000569001') & ((uplist_eicodes['WERF_CODE'] == 'AS-A2O')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix WY WWTP; no trickling filter
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000334001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000334001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '26000334001') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix Mcalpine Creek WWTP; no BNIT, LAGOON_AER, NIT, or TF; add biogas utilization
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'BNIT')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'BNIT')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'BNIT')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'LAGOON_AER')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'LAGOON_AER')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'LAGOON_AER')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'NIT')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'NIT')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'NIT')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'TF')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '37006001002') & ((uplist_eicodes['WERF_CODE'] == 'TF')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# wwtps.loc[wwtps['CWNS_NUM'] == '37006001002', 'BIOGAS_EL_2022'] = 1

# #fix NEORSD Westerly WWTP; no AND, AS, or CHEM-P
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AND')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AND')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AND')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AS')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AS')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'AS')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'CHEM-P')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'CHEM-P')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '39001666003') & ((uplist_eicodes['WERF_CODE'] == 'CHEM-P')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix Hopewell Regional WWTP; no FBI, add MHI
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '51000238001') & ((uplist_eicodes['WERF_CODE'] == 'FBI')), '2022_MIN_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '51000238001') & ((uplist_eicodes['WERF_CODE'] == 'FBI')), 'PRES_IND'] = 0
# uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == '51000238001') & ((uplist_eicodes['WERF_CODE'] == 'FBI')), 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'
# hopewell_idx = uplist_eicodes[uplist_eicodes['CWNS_NUM'] == '51000238001'].index.max()
# hopewell_add = pd.Series({'CWNS_NUM': '51000238001', 'WERF_CODE': 'MHI', '2022_MIN_IND': 1, 'PRES_IND': 1, 'FINAL_UNIT_PROCESS_NAME': 'Biosolids Incineration, Multiple Hearth', 'UP_ID_NOTE': 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'}).to_frame().T
# uplist_eicodes = pd.concat([uplist_eicodes.iloc[:hopewell_idx], hopewell_add, uplist_eicodes.iloc[hopewell_idx:]], ignore_index=True)

# #fix Arlington, CO WPCP; add LIME
# arlington_idx = uplist_eicodes[uplist_eicodes['CWNS_NUM'] == '51000319001'].index.max()
# arlington_add = pd.Series({'CWNS_NUM': '51000319001', 'WERF_CODE': 'LIME', '2022_MIN_IND': 1, 'PRES_IND': 1, 'FINAL_UNIT_PROCESS_NAME': 'Biosolids Lime Stabilization', 'UP_ID_NOTE': 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'}).to_frame().T
# uplist_eicodes = pd.concat([uplist_eicodes.iloc[:arlington_idx], arlington_add, uplist_eicodes.iloc[arlington_idx:]], ignore_index=True)

#manual updates to add and remove lagoons (Christina Polcuch, 2023)
#here, we confirmed the presence of lagoons at facilities larger than 10 MGD using publicly available information

#import list of lagoons to add from EPA lagoon inventory and manual checks
lagoon_add = pd.read_csv(f'{GITHUB_BASE_URL}/cwns_lagoon_add_ttrains_info.csv', dtype = {'CWNS_NUM':str})

#add leading zeros to CWNS ids to ensure proper merge with other datasets
lagoon_add['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in lagoon_add['CWNS_NUM']]

#add in columns for concatenation with main uplist_eicodes dataframe
lagoon_add['REPORT_YEAR'] = 2022
lagoon_add['UP_ID_NOTE'] = 'Assigned using updated data on the presence of lagoons from EPA (2022)'
lagoon_add['2022_MIN_IND'] = 1
lagoon_add['PRES_IND'] = 1
lagoon_add.rename(columns = {'LAGOON_CODE':'WERF_CODE','LAGOON_NAME':'FINAL_UNIT_PROCESS_NAME'}, inplace = True)

#concatenate dataframe which contains additional lagoons found in EPA survey/manual checks and main unit process list dataframe
uplist_eicodes = pd.concat([uplist_eicodes, lagoon_add], axis = 0)

#assume lagoons that were added are projected to remain
uplist_eicodes['PROJ_IND'] = uplist_eicodes['PROJ_IND'].fillna(1)

#drop duplicates unit processes to ensure most recent lagoons are kept in final unit process list
uplist_eicodes.sort_values(by = ['CWNS_NUM','REPORT_YEAR'], ascending = True, inplace = True)
uplist_eicodes.drop_duplicates(subset = ['CWNS_NUM','WERF_CODE','DISPOSAL_CODE','PRES_IND','PROJ_IND'], inplace = True, keep = 'last')
uplist_eicodes.reset_index(inplace = True, drop = True)

#import list of lagoons to remove based on manual checks
lagoon_removed = pd.read_csv(f'{GITHUB_BASE_URL}/cwns_lagoon_remove.csv', dtype = {'CWNS_NUM': str})

#add leading zeros to CWNS ids to ensure proper match with uplist_eicodes
lagoon_removed['CWNS_NUM'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in lagoon_removed['CWNS_NUM']]

#loop through lagoons that need to be removed
for index, row in lagoon_removed.iterrows():
    #get the CWNS number and lagoon code to remove
    lagoon_removed_CWNS = row['CWNS_NUM']
    lagoon_removed_code = row['REMOVE']

    #search for the row in uplist_eicodes where the CWNS number matches the CWNS number in lagoon_removed
    CWNS_match_row = uplist_eicodes.loc[(uplist_eicodes['CWNS_NUM'] == lagoon_removed_CWNS) & (uplist_eicodes['WERF_CODE'].str.contains(lagoon_removed_code))]

    #if a matching row is found, set '2022_MIN_IND', '2022_MAX_IND', and 'PROJ_IND' equal to 0 and update 'UP_ID_NOTE'
    if not CWNS_match_row.empty:
        uplist_eicodes.loc[CWNS_match_row.index, ['2022_MIN_IND','PROJ_IND']] = 0
        uplist_eicodes.loc[CWNS_match_row.index, 'UP_ID_NOTE'] = 'Corrected based on manual check of large lagoons (2023/2024)'


#identify plants that have FBI, MHI, LAND_APP, or LANDFILL as a biosolids disposal method in 2022 and export for use in biosolids emissions calculations
disposal_2022 = uplist_eicodes[['CWNS_NUM','REPORT_YEAR','PRES_IND','DISPOSAL_CODE']]
disposal_2022 = disposal_2022.loc[(disposal_2022['PRES_IND'] == 1)]
disposal_2022 = disposal_2022.dropna(subset = ['DISPOSAL_CODE'])
disposal_2022.sort_values(by = ['CWNS_NUM','REPORT_YEAR'], ascending = False, inplace = True, ignore_index = True)
disposal_2022.drop_duplicates(subset = 'CWNS_NUM', inplace = True, ignore_index = False, keep = "first")
# disposal_2022_pvt = pd.pivot_table(disposal_2022, index = 'CWNS_NUM', values = 'PRES_IND',  columns = 'DISPOSAL_CODE', aggfunc = np.sum, fill_value = 0)
# disposal_2022_pvt.to_csv('disposal_2022.csv')
                         
#drop unit processes that do not have an associated WERF code, as these are not necessary to form treatment train assignments
uplist_eicodes.dropna(subset = 'WERF_CODE', inplace = True)

#manual check for nutrient removal (Christina Polcuch, 2023)
#for facilities where the cumulative unit process list was revised based on publicly available information, correct the nutrient removal flags in the main wwtps dataframe to remain consistent with uplist_eicodes

# #fix Lewiston, ME; remove nutrient removal
# wwtps.loc[wwtps['CWNS_NUM'] == '23000011001', 'PRES_NITROGEN_REMOVAL'] = 0
# wwtps.loc[wwtps['CWNS_NUM'] == '23000011001', 'PRES_PHOSPHOROUS_REMOVAL'] = 0
# wwtps.loc[wwtps['CWNS_NUM'] == '23000011001', 'PRES_AMMONIA_REMOVAL'] = 0
# uplist_eicodes.loc[uplist_eicodes['CWNS_NUM'] == '23000011001', 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

# #fix GLWA plants; no phosphorus or nutrient removal yet
# wwtps.loc[wwtps['CWNS_NUM'] == '26000569001', 'PRES_NITROGEN_REMOVAL'] = 0
# wwtps.loc[wwtps['CWNS_NUM'] == '26000569001', 'PRES_PHOSPHOROUS_REMOVAL'] = 0
# wwtps.loc[wwtps['CWNS_NUM'] == '26000569001', 'PRES_AMMONIA_REMOVAL'] = 0
# uplist_eicodes.loc[uplist_eicodes['CWNS_NUM'] == '26000569001', 'UP_ID_NOTE'] = 'Corrected based on manual check of large facilities with multiple treatment train assignments (2023)'

#create table with manually corrected wwtps to later add UP_ID_NOTE column to treatment train assignment dataframe
manual_check_ups = uplist_eicodes[['CWNS_NUM','UP_ID_NOTE']]
manual_check_ups = manual_check_ups.dropna()
manual_check_ups = manual_check_ups.drop_duplicates(subset = 'CWNS_NUM')

# NEW ADDITIONS
# Create DataFrame with CWNS_NUM and all unit processes
# Filter to only present unit processes (PRES_IND = 1)
present_ups = uplist_eicodes[uplist_eicodes['PRES_IND'] == 1]

# Create pivot table with CWNS_NUM as index and unit processes as columns
unit_processes_df = pd.pivot_table(
    present_ups, 
    index='CWNS_NUM', 
    columns='WERF_CODE', 
    values='PRES_IND', 
    aggfunc='sum', 
    fill_value=0
)

# Reset index to make CWNS_NUM a column
unit_processes_df = unit_processes_df.reset_index()

# Load facility permit data to get NPDES permit numbers and state codes
facility_permit = pd.read_csv('data/cwns/2022/FACILITY_PERMIT.csv', dtype={'CWNS_ID': str, 'STATE_CODE': str})

# Add leading zero to CWNS ids with length less than 11 to ensure proper merge
facility_permit['CWNS_ID'] = ['0' + str(cwns) if len(str(cwns)) < 11 else str(cwns) for cwns in facility_permit['CWNS_ID']]

# Merge permit data with unit processes DataFrame
unit_processes_df = unit_processes_df.merge(
    facility_permit[['CWNS_ID', 'PERMIT_NUMBER', 'STATE_CODE']], 
    left_on='CWNS_NUM', 
    right_on='CWNS_ID', 
    how='left'
)

# Drop the duplicate CWNS_ID column and rename CWNS_NUM to CWNS_ID for consistency
unit_processes_df = unit_processes_df.drop('CWNS_ID', axis=1)
unit_processes_df = unit_processes_df.rename(columns={'CWNS_NUM': 'CWNS_ID'})

# Save the DataFrame
unit_processes_df.to_csv('output/unit_processes_by_facility.csv', index=False)