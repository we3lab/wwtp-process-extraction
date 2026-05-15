# load unmatched_kw_no_cwns.csv

import pandas as pd
unmatched_kw_no_cwns = pd.read_csv('npdes_permits/output/2026-4-26/unmatched_kw_no_cwns.csv')

# convert the FACILITY_NAME column to a list of strings

facility_names = unmatched_kw_no_cwns['FACILITY_NAME'].tolist()

# print(facility_names)

# print plain text, comma-separated, without "" around strings

print(', '.join(facility_names))