import json
import matplotlib.pyplot as plt
import pandas as pd

with open('data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

def search_processes_in_text(processes_dict, results, parent_name=None):
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
                sub_found = search_processes_in_text(details['sub_categories'], results, process_name)
                if sub_found:
                    sub_category_found = True
    
    # If this is a parent category and any sub-category was found, set parent to 1
    if parent_name and sub_category_found:
        results[parent_name] = 1
    
    return sub_category_found

# Get all process names, to become column names
def extract_process_names(processes_dict):
    """Function to recursively extract list of process details from json"""
    for process_name, details in processes_dict.items():
        if isinstance(details, dict) and 'alt_names' in details:
            process_names.append(process_name)
            # Recursively extract sub-categories
            if 'sub_categories' in details:
                extract_process_names(details['sub_categories'])

# 1. LOAD AND ORGANIZE CWNS DATA
cwns_data = pd.read_csv('output/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()

# Extract all process names for column headers
process_names = []
for category, processes in unitprocess_keywords.items():
    if isinstance(processes, dict):
        extract_process_names(processes)

# Count California facilities with each process
def get_el_abbadi_code_mapping(process_name, unitprocess_keywords):
    """Get EL_ABBADI_2024_CODE for a process name as needed"""
    for category, processes in unitprocess_keywords.items():
        if isinstance(processes, dict):
            # Check parent categories then sub-categories
            if process_name in processes: # if it's a parent category
                return processes[process_name]['EL_ABBADI_2024_CODE']
            for parent_name, parent_details in processes.items():
                if 'sub_categories' in parent_details and process_name in parent_details['sub_categories']:
                    if 'EL_ABBADI_2024_CODE' in parent_details['sub_categories'][process_name]:
                        return parent_details['sub_categories'][process_name]['EL_ABBADI_2024_CODE']
    return None

ca_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)
# First, count individual processes (both parent and sub-categories)
for process_name in process_names:
    el_abbadi_code = get_el_abbadi_code_mapping(process_name, unitprocess_keywords)
    if el_abbadi_code and el_abbadi_code in ca_cwns_data.columns:
        count = (ca_cwns_data[el_abbadi_code] > 0).sum()
        ca_cwns_results_df.loc[0, process_name] = count
# Then, update parent categories to include sum of sub-categories
for parent_name, parent_details in unitprocess_keywords['secondary'].items():
    if isinstance(parent_details, dict) and 'alt_names' in parent_details:
        has_sub_categories = 'sub_categories' in parent_details and any(
            isinstance(sub_details, dict) and 'alt_names' in sub_details 
            for sub_details in parent_details['sub_categories'].values()
        )
        if has_sub_categories and parent_name in ca_cwns_results_df.columns:
            sub_category_sum = 0
            for sub_name, sub_details in parent_details['sub_categories'].items():
                if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                    if sub_name in ca_cwns_results_df.columns:
                        sub_category_sum += ca_cwns_results_df.loc[0, sub_name]
            # Add the parent's own count (if it has one) plus the sum of sub-categories
            parent_own_count = ca_cwns_results_df.loc[0, parent_name]
            ca_cwns_results_df.loc[0, parent_name] = parent_own_count + sub_category_sum


# 2. CREATE EXAMPLE SCRAPED DATA
example_sentence = """The facility has a headworks with grit and FOG removal, followed by a primary clarifier. 
The secondary treatment includes four activated sludge basins and clarifiers. 
The secondary effluent then passes through the chlorine contact tank before discharge."""

# Create dummy scraped data DataFrame for all CA facilities
example_results = {}
for category, processes in unitprocess_keywords.items():
    if isinstance(processes, dict):
        search_processes_in_text(processes, example_results, None)
example_scraped_data = pd.DataFrame([example_results])
example_scraped_data_expanded = pd.concat([example_scraped_data] * len(ca_cwns_data), ignore_index=True)
example_scraped_data_expanded['CWNS_ID'] = ca_cwns_data['CWNS_ID'].values
example_scraped_data_expanded['STATE_CODE'] = 'CA'


# PLOT
# Create plot data
plot_data = []
current_pos = 0

for parent_name, parent_details in unitprocess_keywords['secondary'].items():
    if isinstance(parent_details, dict) and 'alt_names' in parent_details:
        has_sub_categories = 'sub_categories' in parent_details and any(
            isinstance(sub_details, dict) and 'alt_names' in sub_details 
            for sub_details in parent_details['sub_categories'].values()
        )
        
        if has_sub_categories:  # Add total first, then sub-categories
            # Add parent category total first
            plot_data.append({
                'Category': 'Total',
                'Parent': parent_name,
                'Process': f'{parent_name} (Total)',
                'Dummy_Scraped': (example_scraped_data_expanded[parent_name] > 0).sum() if parent_name in example_scraped_data_expanded.columns else 0,
                'CWNS_Data': ca_cwns_results_df.loc[0, parent_name] if parent_name in ca_cwns_results_df.columns else 0,
                'Position': current_pos
            })
            current_pos += 1
            
            # Then add sub-categories
            for sub_name, sub_details in parent_details['sub_categories'].items():
                if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                    plot_data.append({
                        'Category': 'Sub-Category',
                        'Parent': parent_name,
                        'Process': sub_name,
                        'Dummy_Scraped': (example_scraped_data_expanded[sub_name] > 0).sum() if sub_name in example_scraped_data_expanded.columns else 0,
                        'CWNS_Data': ca_cwns_results_df.loc[0, sub_name] if sub_name in ca_cwns_results_df.columns else 0,
                        'Position': current_pos
                    })
                    current_pos += 1
        else:  # Parent category without sub-categories - just add total
            plot_data.append({
                'Category': 'Total',
                'Parent': parent_name,
                'Process': f'{parent_name} (Total)',
                'Dummy_Scraped': (example_scraped_data_expanded[parent_name] > 0).sum() if parent_name in example_scraped_data_expanded.columns else 0,
                'CWNS_Data': ca_cwns_results_df.loc[0, parent_name] if parent_name in ca_cwns_results_df.columns else 0,
                'Position': current_pos
            })
            current_pos += 1
        
        current_pos += 0.5  # spacing

plot_df = pd.DataFrame(plot_data)

# Create bar plot
fig, ax = plt.subplots(figsize=(16, 6))
width = 0.35

# Plot bars with different colors for parent vs sub-categories
for category in ['Total', 'Sub-Category']:
    mask = plot_df['Category'] == category
    if mask.any():
        if category == 'Sub-Category':
            ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Dummy_Scraped'], width, 
                   color='lightblue', alpha=0.6, label='Dummy Scraped (Sub-Category)' if category == 'Sub-Category' else "")
            ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width, 
                   color='lightcoral', alpha=0.6, label='CWNS Data (Sub-Category)' if category == 'Sub-Category' else "")
        else:  # Totals
            ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Dummy_Scraped'], width, 
                   color='blue', alpha=0.8, label='Dummy Scraped (Total)' if category == 'Total' else "")
            ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width, 
                   color='coral', alpha=0.8, label='CWNS Data (Total)' if category == 'Total' else "")

# Set x-axis labels for sub-categories and totals
all_positions = plot_df['Position'].tolist()
all_labels = plot_df['Process'].tolist()

# Make "(Total)" labels bold
bold_labels = []
for label in all_labels:
    if "(Total)" in label:
        bold_labels.append(f"$\\bf{{{label}}}$")  # LaTeX bold formatting
    else:
        bold_labels.append(label)

ax.set_xticks(all_positions)
ax.set_xticklabels(bold_labels, rotation=45, ha='right', fontsize=10)

ax.set_ylabel('WWTP Count', fontsize=12)
ax.legend()
plt.subplots_adjust(bottom=0.25)  # Reduce bottom margin
plt.savefig('output/secondary_processes_comparison.png', dpi=300, bbox_inches='tight')