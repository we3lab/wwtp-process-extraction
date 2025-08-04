import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

with open('data/unitprocess_keywords.json', 'r') as f:
    data = json.load(f)

def search_processes(processes_dict, results, parent_name=None):
    """Recursively search through processes and their sub-categories"""
    sub_category_found = False
    
    for process_name, details in processes_dict.items():
        if isinstance(details, dict) and 'alt_names' in details:
            # Initialize unit process key to 0
            if process_name not in results:
                results[process_name] = 0
            
            # Check each alt_name for each process
            for i, alt_name in enumerate(details['alt_names']):
                case_sensitive = details['alt_names_case_sensitive'][i] if i < len(details['alt_names_case_sensitive']) else "N"
                
                if case_sensitive == "Y":
                    found = alt_name in example_sentence
                else:
                    found = alt_name.lower() in example_sentence.lower()
                
                if found:
                    results[process_name] = 1
                    sub_category_found = True
                    break  # Found one alt_name, don't need to check others
            
            # Search sub-categories (if they exist) and recursively add to results
            if 'sub_categories' in details:
                sub_found = search_processes(details['sub_categories'], results, process_name)
                if sub_found:
                    sub_category_found = True
    
    # If this is a parent category and any sub-category was found, set parent to 1
    if parent_name and sub_category_found:
        results[parent_name] = 1
    
    return sub_category_found

# Get all process names to create consistent column headers

def extract_processes(processes_dict):
    """Function to recursively extract list of process details from json"""
    for process_name, details in processes_dict.items():
        if isinstance(details, dict) and 'alt_names' in details:
            process_names.append(process_name)
            # Recursively extract sub-categories
            if 'sub_categories' in details:
                extract_processes(details['sub_categories'])

process_names = []
for category, processes in data.items():
    if isinstance(processes, dict):
        extract_processes(processes)
    
# Example sentence search
example_sentence = """The facility has a headworks with grit and FOG removal, followed by a primary clarifier. 
The secondary treatment includes four activated sludge basins and clarifiers. 
The secondary effluent then passes through the chlorine contact tank before discharge."""
example_results = {}
for category, processes in data.items():
    if isinstance(processes, dict):
        search_processes(processes, example_results, None)

# Create DataFrame for example sentence with all process columns
example_scraped_data = pd.DataFrame([example_results])
# print(example_scraped_data.head())

# Load facility data from tt_assignment_2022 output
cwns_data = pd.read_csv('output/unit_processes_by_facility.csv')
ca_facilities = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()

# Load WERF code mapping from GitHub CSV
def extract_werf_mapping(data_dict, parent_key=None):
    """Recursively extract WERF code mappings from the JSON structure"""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            if 'EL_ABBADI_2024_CODE' in value and value['EL_ABBADI_2024_CODE'] != "None":
                werf_code = value['EL_ABBADI_2024_CODE']
                process_name = parent_key if parent_key else key
                werf_to_process_mapping[werf_code] = process_name
            
            # Recursively check sub-categories
            if 'sub_categories' in value:
                for sub_key, sub_value in value['sub_categories'].items():
                    if isinstance(sub_value, dict) and 'EL_ABBADI_2024_CODE' in sub_value and sub_value['EL_ABBADI_2024_CODE'] != "None":
                        werf_code = sub_value['EL_ABBADI_2024_CODE']
                        werf_to_process_mapping[werf_code] = sub_key
            else:
                # Continue recursion for nested structures
                extract_werf_mapping(value, key)
werf_mapping_url = "https://raw.githubusercontent.com/jiananf2/US_WWTP_GHG/refs/heads/main/treatment_train_assignment/input_data/UNIT_PROCESS_NAMES_2022.csv"
werf_mapping_df = pd.read_csv(werf_mapping_url)

# Create mapping from WERF codes to process names by extracting from unitprocess_keywords.json
werf_to_process_mapping = {}
for category, processes in data.items():
    if isinstance(processes, dict):
        extract_werf_mapping(processes)

# Count California facilities with each process
facility_results_df = pd.DataFrame(0, index=[0], columns=process_names)
for werf_code, process_name in werf_to_process_mapping.items():
    if werf_code in ca_facilities.columns:
        count = (ca_facilities[werf_code] > 0).sum()
        if process_name in facility_results_df.columns:
            facility_results_df.loc[0, process_name] = count

# DUMMY DATA - Duplicate example scraped data for every CA facility
ca_cwns_ids = ca_facilities['CWNS_ID'].tolist()
example_scraped_data_expanded = pd.DataFrame()
for cwns_id in ca_cwns_ids:
    facility_data = example_scraped_data.copy()    
    facility_data['CWNS_ID'] = cwns_id
    facility_data['STATE_CODE'] = 'CA'    
    example_scraped_data_expanded = pd.concat([example_scraped_data_expanded, facility_data], ignore_index=True)

# Get secondary category processes (parent categories)
secondary_processes = []
if 'secondary' in data:
    for process_name, details in data['secondary'].items():
        if isinstance(details, dict) and 'alt_names' in details:
            secondary_processes.append(process_name)
            # Note: Not adding sub-categories to keep only parent categories

# Create bar plot for secondary processes
fig, ax = plt.subplots(figsize=(15, 8))

# Get counts from both DataFrames using all secondary processes
# Count how many facilities have each process in the expanded dummy data
dummy_scraped_counts = []
for p in secondary_processes:
    if p in example_scraped_data_expanded.columns:
        count = (example_scraped_data_expanded[p] > 0).sum()
    else:
        count = 0
    dummy_scraped_counts.append(count)

cwns_counts = [facility_results_df.loc[0, p] if p in facility_results_df.columns else 0 for p in secondary_processes]

# Create bars
width = 0.35
ax.bar([i - width/2 for i in range(len(secondary_processes))], dummy_scraped_counts, width, label='Dummy Scraped Data', alpha=0.8)
ax.bar([i + width/2 for i in range(len(secondary_processes))], cwns_counts, width, label='CWNS Data', alpha=0.8)

ax.set_xlabel('Secondary Unit Processes')
ax.set_ylabel('Count')
ax.set_xticks(range(len(secondary_processes)))
ax.set_xticklabels(secondary_processes, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/secondary_processes_comparison.png', dpi=300, bbox_inches='tight')