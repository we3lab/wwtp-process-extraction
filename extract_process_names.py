import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

with open('data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

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

# Example sentence search
example_sentence = """The facility has a headworks with grit and FOG removal, followed by a primary clarifier. 
The secondary treatment includes four activated sludge basins and clarifiers. 
The secondary effluent then passes through the chlorine contact tank before discharge."""

# Load facility data from tt_assignment_2022 output
cwns_data = pd.read_csv('output/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()

# Extract process names for column headers then search for process in text
process_names = []
example_results = {}
for category, processes in unitprocess_keywords.items():
    if isinstance(processes, dict):
        extract_processes(processes)
        search_processes(processes, example_results, None)

# Count California facilities with each process
def get_el_abbadi_code_mapping(process_name, unitprocess_keywords):
    """Get EL_ABBADI_2024_CODE for a process name as needed"""
    for category, processes in unitprocess_keywords.items():
        if isinstance(processes, dict):
            # Check parent categories then sub-categories
            if process_name in processes and 'EL_ABBADI_2024_CODE' in processes[process_name]:
                return processes[process_name]['EL_ABBADI_2024_CODE']
            for parent_name, parent_details in processes.items():
                if 'sub_categories' in parent_details and process_name in parent_details['sub_categories']:
                    if 'EL_ABBADI_2024_CODE' in parent_details['sub_categories'][process_name]:
                        return parent_details['sub_categories'][process_name]['EL_ABBADI_2024_CODE']
    return None
ca_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)
for process_name in process_names:
    el_abbadi_code = get_el_abbadi_code_mapping(process_name, unitprocess_keywords)
    if el_abbadi_code and el_abbadi_code in ca_cwns_data.columns:
        count = (ca_cwns_data[el_abbadi_code] > 0).sum()
        ca_cwns_results_df.loc[0, process_name] = count

# Create dummy scraped data DataFrame for all CA facilities
example_scraped_data = pd.DataFrame([example_results])
example_scraped_data_expanded = pd.concat([example_scraped_data] * len(ca_cwns_data), ignore_index=True)
example_scraped_data_expanded['CWNS_ID'] = ca_cwns_data['CWNS_ID'].values
example_scraped_data_expanded['STATE_CODE'] = 'CA'

# Get secondary category processes (parent categories)
secondary_processes = []
if 'secondary' in unitprocess_keywords:
    for process_name, details in unitprocess_keywords['secondary'].items():
        if isinstance(details, dict) and 'alt_names' in details:
            secondary_processes.append(process_name)
            # Note: Not adding sub-categories to keep only parent categories

# Create df for bar plot
plot_data = pd.DataFrame({
    'Process': secondary_processes,
    'Dummy_Scraped': [(example_scraped_data_expanded[p] > 0).sum() if p in example_scraped_data_expanded.columns else 0 for p in secondary_processes],
    'CWNS_Data': [ca_cwns_results_df.loc[0, p] if p in ca_cwns_results_df.columns else 0 for p in secondary_processes]
})

# Create bar plot
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.35
ax.bar([i - width/2 for i in range(len(plot_data))], plot_data['Dummy_Scraped'], width, label='Dummy Scraped Data', alpha=0.8)
ax.bar([i + width/2 for i in range(len(plot_data))], plot_data['CWNS_Data'], width, label='CWNS Data', alpha=0.8)

ax.set_xlabel('Secondary Unit Processes')
ax.set_ylabel('Count')
ax.set_xticks(range(len(plot_data)))
ax.set_xticklabels(plot_data['Process'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/secondary_processes_comparison.png', dpi=300, bbox_inches='tight')