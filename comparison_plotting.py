import json
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {
    'test_results_total': 'blue',
    'cwns_data_total': 'coral',
    'test_results_subcategory': 'lightblue',
    'cwns_data_subcategory': 'lightcoral'
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
def extract_process_names(processes_dict):
    """Function to recursively extract list of process details from json"""
    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            # Check if this is a process with alt_names (lowest level)
            if 'alt_names' in details and details['alt_names']:
                process_names.append(process_name)
            else:  # Recursively extract children of parent category
                for nested_key, nested_details in details.items():
                    if isinstance(nested_details, dict) and 'alt_names' in nested_details:
                        process_names.append(nested_key)

def get_el_abbadi_code_mapping(process_name, unitprocess_keywords):
    """Get EL_ABBADI_2024_CODE for a process name as needed"""
    for category, processes in unitprocess_keywords.items():
        if isinstance(processes, dict):
            # Check if this is a process with alt_names (lowest level)
            if process_name in processes and 'EL_ABBADI_2024_CODE' in processes[process_name]:
                return processes[process_name]['EL_ABBADI_2024_CODE']
            # Check parent categories then their children
            for parent_name, parent_details in processes.items():
                if isinstance(parent_details, dict) and process_name in parent_details:
                    if 'EL_ABBADI_2024_CODE' in parent_details[process_name]:
                        return parent_details[process_name]['EL_ABBADI_2024_CODE']
    return None

def has_sub_processes(process_details):
    """Check if a process has sub-processes (is a parent category)"""
    if not isinstance(process_details, dict):
        return False
    # If it has alt_names, it's a leaf process
    if 'alt_names' in process_details:
        return False
    # If it has other dict items that have alt_names, it's a parent
    return any(isinstance(v, dict) and 'alt_names' in v for v in process_details.values())

# Load CWNS data and filter to include matching permit numbers from PDF scraping
cwns_data = pd.read_csv('output/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()
matching_cwns_data = ca_cwns_data[ca_cwns_data['PERMIT_NUMBER'].isin(test_permit_numbers)].copy()

# Extract process names from keywords
process_names = []
for category, processes in unitprocess_keywords.items():
    if isinstance(processes, dict):
        extract_process_names(processes)

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
for parent_name, parent_details in unitprocess_keywords['secondary'].items():
    if has_sub_processes(parent_details):
        # Sum child processes for test results
        test_sub_category_sum = 0
        for sub_name, sub_details in parent_details.items():
            if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                if sub_name in test_results_df.columns:
                    test_sub_category_sum += test_results_df.loc[0, sub_name]

        # Sum child processes for CWNS results
        cwns_sub_category_sum = 0
        for sub_name, sub_details in parent_details.items():
            if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                if sub_name in matching_cwns_results_df.columns:
                    cwns_sub_category_sum += matching_cwns_results_df.loc[0, sub_name]

        # Create parent columns if they don't exist
        if parent_name not in test_results_df.columns:
            test_results_df[parent_name] = 0
        if parent_name not in matching_cwns_results_df.columns:
            matching_cwns_results_df[parent_name] = 0

        # Set the parent values
        test_results_df.loc[0, parent_name] = test_sub_category_sum
        matching_cwns_results_df.loc[0, parent_name] = cwns_sub_category_sum

# 3. CREATE COMPARISON PLOTS FOR SECONDARY PROCESSES
# Get processes that exist in test results (have non-zero values)
test_processes = [process for process, count in test_results_aggregated.items() if count > 0]

# Create plot data for processes that exist in test results
plot_data = []
current_pos = 0

for parent_name, parent_details in unitprocess_keywords['secondary'].items():
    if has_sub_processes(parent_details):
        # Check if this parent or any of its sub-categories exist in test results
        parent_in_tests = parent_name in test_processes
        sub_categories_in_tests = []
        for sub_name, sub_details in parent_details.items():
            if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                if sub_name in test_processes:
                    sub_categories_in_tests.append(sub_name)

        # Only add to plot if parent or any sub-category exists in test results
        if parent_in_tests or sub_categories_in_tests:
            # Always add parent category total first (even if it's 0)
            plot_data.append({
                'Category': 'Total',
                'Parent': parent_name,
                'Process': f'{parent_name} (Total)',
                'Test_Results': test_results_df.loc[0, parent_name] if parent_name in test_results_df.columns else 0,
                'CWNS_Data': matching_cwns_results_df.loc[0, parent_name] if parent_name in matching_cwns_results_df.columns else 0,
                'Position': current_pos
            })
            current_pos += 1

            # Then add sub-categories that exist in test results
            for sub_name, sub_details in parent_details.items():
                if isinstance(sub_details, dict) and 'alt_names' in sub_details:
                    if sub_name in test_processes:
                        plot_data.append({
                            'Category': 'Sub-Category',
                            'Parent': parent_name,
                            'Process': sub_name,
                            'Test_Results': test_results_df.loc[0, sub_name] if sub_name in test_results_df.columns else 0,
                            'CWNS_Data': matching_cwns_results_df.loc[0, sub_name] if sub_name in matching_cwns_results_df.columns else 0,
                            'Position': current_pos
                        })
                        current_pos += 1

            current_pos += 0.5  # spacing

plot_df = pd.DataFrame(plot_data)

# Plot bars with different colors for parent vs sub-categories
fig, ax = plt.subplots(figsize=(16, 6))
width = 0.35

for category in ['Total', 'Sub-Category']:
    mask = plot_df['Category'] == category
    if mask.any():
        if category == 'Sub-Category':
            ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Test_Results'], width,
                   color=COLORS['test_results_subcategory'], alpha=0.6, label='Test Results (Sub-Category)' if category == 'Sub-Category' else "")
            ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width,
                   color=COLORS['cwns_data_subcategory'], alpha=0.6, label='CWNS Data (Sub-Category)' if category == 'Sub-Category' else "")
        else:  # Totals
            ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Test_Results'], width,
                   color=COLORS['test_results_total'], alpha=0.8, label='Test Results (Total)' if category == 'Total' else "")
            ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width,
                   color=COLORS['cwns_data_total'], alpha=0.8, label='CWNS Data (Total)' if category == 'Total' else "")

# Set x-axis labels for sub-categories and totals
all_positions = plot_df['Position'].tolist()
all_labels = plot_df['Process'].tolist()

# Make parent categories bold
bold_labels = []
for label in all_labels:
    # Check if this is a parent category (has "(Total)" or is a standalone parent)
    is_parent = False
    if "(Total)" in label:
        is_parent = True
    else:
        # Check if this is a standalone parent category in secondary
        for parent_name, parent_details in unitprocess_keywords['secondary'].items():
            if label == parent_name and has_sub_processes(parent_details):
                is_parent = True
                break

    if is_parent:
        bold_labels.append(f"$\\bf{{{label}}}$")  # LaTeX bold formatting
    else:
        bold_labels.append(label)

ax.set_xticks(all_positions)
ax.set_xticklabels(bold_labels, rotation=45, ha='right', fontsize=10)

ax.set_ylabel('WWTP Count', fontsize=12)
ax.legend()
plt.subplots_adjust(bottom=0.25)  # Reduce bottom margin
plt.savefig('output/test_results_vs_cwns_comparison.png', dpi=300, bbox_inches='tight')