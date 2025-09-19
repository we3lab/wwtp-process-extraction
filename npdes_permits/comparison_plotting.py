import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from utils import *

DATE_FOLDER = '2025-9-19'

COLORS = {
    'cwns': '#8cd23c', 
    'npdes': '#1482a5ff'
}

def append_plot_data(plot_data, test_results_df, matching_cwns_results_df, parent_child_mapping, current_pos):
    """Append plot data for a category to the given plot_data list"""
for parent_name, child_processes in parent_child_mapping.items():
    plot_data.append({
        'Category': 'Total',
        'Parent': parent_name,
        'Process': f'{parent_name} (Total)',
            'Test_Results': test_results_df.loc[0, parent_name],
        'CWNS_Data': matching_cwns_results_df.loc[0, parent_name],
        'Position': current_pos
    })
    current_pos += 1
    
        for process_name in child_processes:
            test_count = test_results_df.loc[0, process_name] if process_name in test_results_df.columns else 0
            cwns_count = matching_cwns_results_df.loc[0, process_name] if process_name in matching_cwns_results_df.columns else 0
            
        plot_data.append({
            'Category': 'Process',
            'Parent': parent_name,
            'Process': process_name,
                'Test_Results': test_count,
                'CWNS_Data': cwns_count,
            'Position': current_pos
        })
        current_pos += 1
    
        current_pos += 0.5
    
    return current_pos

def create_plot_data(test_results_df, matching_cwns_results_df, parent_child_mapping, category_name):
    """Create plot data for a specific category"""
    plot_data = []
    current_pos = 0
    current_pos = append_plot_data(plot_data, test_results_df, matching_cwns_results_df, parent_child_mapping, current_pos)
    return plot_data

def create_plot(plot_data, category_name, title_suffix="", figsize=(16, 6), fontsize=14, save_path=None):
    """Create and save a comparison plot"""
    if not plot_data:
        print(f"No processes found for '{category_name}'")
        return

plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=figsize)
width = 0.35

    for data_category in ['Total', 'Process']:
        mask = plot_df['Category'] == data_category
    
        alpha = 1.0
        if data_category == 'Process':
        alpha = 0.5
    
        ax.bar(plot_df[mask]['Position'] - width/2, plot_df[mask]['Test_Results'], width,
           color=COLORS['npdes'], alpha=alpha)
    
    ax.bar(plot_df[mask]['Position'] + width/2, plot_df[mask]['CWNS_Data'], width,
           color=COLORS['cwns'], alpha=alpha)

all_positions = plot_df['Position'].tolist()
all_labels = plot_df['Process'].tolist()

bold_labels = [f"$\\bf{{{label}}}$" if is_parent_category(label, unitprocess_keywords) 
                else label for label in all_labels]

ax.set_xticks(all_positions)
    ax.set_xticklabels(bold_labels, rotation=45, ha='right', fontsize=fontsize)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plot_title = f'{category_name.replace("_", " ").title()} Treatment Process Comparison{title_suffix}'
    ax.set_title(plot_title, fontsize=18)
ax.set_ylabel('WWTP Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

legend_elements = [
    Patch(facecolor=COLORS['cwns'], label='CWNS'),
    Patch(facecolor=COLORS['npdes'], label='NPDES'),
    Patch(facecolor='black', label='Process Category Total'),
    Patch(facecolor='grey', label='Process')
]
legend = ax.legend(handles=legend_elements, loc='upper right')
legend.get_texts()[2].set_weight('bold')

    plt.subplots_adjust(bottom=0.25)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def count_facilities_with_processes(processes, data_source, process_names=None, unitprocess_keywords=None, matching_cwns_data=None):
    """Count facilities that have any of the given processes"""
    facilities_with_processes = set()
    
    for process in processes:
        if process_names and process in process_names:
            # For CWNS data - need to map to El Abbadi codes
            el_abbadi_codes = get_el_abbadi_code_mapping(process, unitprocess_keywords)
            if el_abbadi_codes:
                for code in el_abbadi_codes:
                    if code in matching_cwns_data.columns:
                        facilities_with_process = matching_cwns_data[matching_cwns_data[code] > 0]['PERMIT_NUMBER'].tolist()
                        facilities_with_processes.update(facilities_with_process)
    
    return len(facilities_with_processes)


def append_to_all_categories_plot_data(all_categories_plot_data, test_results_df, matching_cwns_results_df, parent_child_mapping, current_pos):
    """Append data for a category to the comprehensive plot data"""
    return append_plot_data(all_categories_plot_data, test_results_df, matching_cwns_results_df, parent_child_mapping, current_pos)

def process_category_data(category, unitprocess_keywords, unit_process_results, matching_cwns_data):
    """Process data for a specific category and return DataFrames"""
    category_keywords = unitprocess_keywords[category]
    process_names = get_all_keys(category_keywords)
    parent_child_mapping = get_parent_child_mapping(category_keywords)

    # Create test results DataFrame
    test_results_filtered = unit_process_results[unit_process_results.columns.intersection(process_names)]
    test_results_aggregated = test_results_filtered.sum().to_dict()
    test_results_df = pd.DataFrame([test_results_aggregated])

    # Count individual processes for CWNS data
    matching_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)
    for process_name in process_names:
        el_abbadi_codes = get_el_abbadi_code_mapping(process_name, unitprocess_keywords)
        if el_abbadi_codes:
            count = 0
            for code in el_abbadi_codes:
                if code in matching_cwns_data.columns:
                    count += (matching_cwns_data[code] > 0).sum()
            matching_cwns_results_df.loc[0, process_name] = count

    # Count parent categories for both test results and CWNS data
    for parent_name, child_processes in parent_child_mapping.items():
        test_parent_count = count_facilities_with_processes(
            child_processes, unit_process_results
        )
        
        cwns_parent_count = count_facilities_with_processes(
            child_processes, None, process_names, unitprocess_keywords, matching_cwns_data
        )
        
        test_results_df[parent_name] = test_parent_count
        matching_cwns_results_df[parent_name] = cwns_parent_count
    
    return test_results_df, matching_cwns_results_df, parent_child_mapping, process_names

# Load all required data
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

# Get all categories from the JSON file
categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load test results data from the specified date folder
unit_process_results = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv')
test_permit_numbers = unit_process_results['PERMIT_NUMBER'].unique()

# Load and organize CWNS data from the specified date folder
cwns_data = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()
matching_cwns_data = ca_cwns_data[ca_cwns_data['PERMIT_NUMBER'].isin(test_permit_numbers)].copy()

# Initialize data structures for "all categories" plot
all_categories_plot_data = []
all_categories_current_pos = 0
all_parent_child_mappings = {}
all_process_names = set()

for category in categories_to_plot:
    # Process data for this category
    test_results_df, matching_cwns_results_df, parent_child_mapping, process_names = process_category_data(
        category, unitprocess_keywords, unit_process_results, matching_cwns_data
    )
    
    all_process_names.update(process_names)
    all_parent_child_mappings.update(parent_child_mapping)

    plot_data = create_plot_data(test_results_df, matching_cwns_results_df, parent_child_mapping, category)
    create_plot(plot_data, category, save_path=f'npdes_permits/output/{DATE_FOLDER}/{category}_comparison.png')
    
    all_categories_current_pos = append_to_all_categories_plot_data(
        all_categories_plot_data, test_results_df, matching_cwns_results_df, parent_child_mapping, all_categories_current_pos
    )

# Create comprehensive plot with all categories
create_plot(
    all_categories_plot_data, 
    'all_categories', 
    title_suffix=' - All Categories',
    figsize=(24, 8), 
    fontsize=10,
    save_path=f'npdes_permits/output/{DATE_FOLDER}/all_categories_comparison.png'
)
print("Comprehensive plot saved as 'all_categories_comparison.png'")