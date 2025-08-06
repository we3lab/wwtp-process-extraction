import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from utils import *

COLORS = {
    'cwns': 'coral',
    'npdes': 'blue'
}  # can change to any hex code or matplotlib default

with open('data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

# 1. LOAD TEST RESULTS DATA FROM PDF SCRAPING
test_results = pd.read_csv('output/test_results.csv')  # TODO: update with final output file
test_permit_numbers = test_results['PERMIT_NUMBER'].unique()
# Create aggregated test results (sum across all facilities for each process)
test_results_aggregated = test_results.drop(['Agency', 'PERMIT_NUMBER'], axis=1).sum().to_dict()
test_results_df = pd.DataFrame([test_results_aggregated])


# 2. LOAD AND ORGANIZE CWNS DATA FROM EL ABBADI 2024
cwns_data = pd.read_csv('output/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()
matching_cwns_data = ca_cwns_data[ca_cwns_data['PERMIT_NUMBER'].isin(test_permit_numbers)].copy()

# Extract all process names and parent-child mapping
process_names = get_all_keys(unitprocess_keywords['secondary'])
parent_child_mapping = get_parent_child_mapping(unitprocess_keywords['secondary'])

# Count matching CWNS facilities with each process
matching_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)

# First, count individual processes (both parent and sub-categories)
for process_name in process_names:
    el_abbadi_codes = get_el_abbadi_code_mapping(process_name, unitprocess_keywords)
    if el_abbadi_codes:
        # Check if ANY code in the list matches a column in the CWNS data
        count = 0
        for code in el_abbadi_codes:
            if code in matching_cwns_data.columns:
                count += (matching_cwns_data[code] > 0).sum()
        matching_cwns_results_df.loc[0, process_name] = count

# Create parent category columns by summing their child processes
for parent_name, child_processes in parent_child_mapping.items():
    # Sum child processes for test results
    test_sub_category_sum = sum(test_results_df.loc[0, child] for child in child_processes 
                               if child in test_results_df.columns)
    
    # Sum child processes for CWNS results  
    cwns_sub_category_sum = sum(matching_cwns_results_df.loc[0, child] for child in child_processes 
                               if child in matching_cwns_results_df.columns)
    
    # Create parent columns and set values
    test_results_df[parent_name] = test_sub_category_sum
    matching_cwns_results_df[parent_name] = cwns_sub_category_sum

# 3. CREATE COMPARISON PLOTS FOR SECONDARY PROCESSES
test_processes = [process for process, count in test_results_aggregated.items() if count > 0]

# Create plot data for processes that exist in test results
plot_data = []
current_pos = 0

for parent_name, child_processes in parent_child_mapping.items():
    # Find child processes that exist in test results
    existing_child_processes = [child for child in child_processes if child in test_processes]
    
    # Only add to plot if parent or any child process exists in test results
    parent_in_tests = parent_name in test_processes
    if not (parent_in_tests or existing_child_processes):
        continue
    
    # Add parent category total
    plot_data.append({
        'Category': 'Total',
        'Parent': parent_name,
        'Process': f'{parent_name} (Total)',
        'Test_Results': test_results_df.loc[0, parent_name],
        'CWNS_Data': matching_cwns_results_df.loc[0, parent_name],
        'Position': current_pos
    })
    current_pos += 1
    
    # Add child processes
    for process_name in existing_child_processes:
        plot_data.append({
            'Category': 'Process',
            'Parent': parent_name,
            'Process': process_name,
            'Test_Results': test_results_df.loc[0, process_name],
            'CWNS_Data': matching_cwns_results_df.loc[0, process_name],
            'Position': current_pos
        })
        current_pos += 1
    
    current_pos += 0.5  # spacing

plot_df = pd.DataFrame(plot_data)

fig, ax = plt.subplots(figsize=(16, 6))
width = 0.35

# Plot bars for each category
for category in ['Total', 'Process']:
    mask = plot_df['Category'] == category
    
    alpha = 1.0 # bar transparency
    if category == 'Process':
        alpha = 0.5
    
    # Plot NPDES data
    ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Test_Results'], width,
           color=COLORS['npdes'], alpha=alpha)
    
    # Plot CWNS data
    ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width,
           color=COLORS['cwns'], alpha=alpha)

# Set x-axis labels for sub-categories and totals
all_positions = plot_df['Position'].tolist()
all_labels = plot_df['Process'].tolist()

# Make parent categories bold
bold_labels = [f"$\\bf{{{label}}}$" if is_parent_category(label, unitprocess_keywords) 
                else label for label in all_labels]

ax.set_xticks(all_positions)
ax.set_xticklabels(bold_labels, rotation=45, ha='right', fontsize=14)

# Set y-axis to integers
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax.set_ylabel('WWTP Count', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)

# Create custom legend
legend_elements = [
    Patch(facecolor=COLORS['cwns'], label='CWNS'),
    Patch(facecolor=COLORS['npdes'], label='NPDES'),
    Patch(facecolor='black', label='Process Category Total'),
    Patch(facecolor='grey', label='Process')
]
legend = ax.legend(handles=legend_elements, loc='upper right')
# Make the third legend entry bold
legend.get_texts()[2].set_weight('bold')

plt.subplots_adjust(bottom=0.25)  # Reduce bottom margin
plt.savefig('output/test_results_vs_cwns_comparison.png', dpi=300, bbox_inches='tight')