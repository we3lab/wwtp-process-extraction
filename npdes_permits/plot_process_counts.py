import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os

from utils import *

DATE_FOLDER = '2025-10-13'

COLORS = {
    'cwns': '#8cd23c', 
    'npdes': '#1482a5ff',
    'npdes_present': '#1482a5ff',
    'npdes_future': '#87CEEB'  # Light blue for future/planned
}


def append_plot_data(plot_data, npdes_results_df, matching_cwns_results_df, parent_child_mapping, current_pos):
    """Append plot data for a category to the given plot_data list"""
    for parent_name, child_processes in parent_child_mapping.items():
        plot_data.append({
            'Category': 'Total',
            'Parent': parent_name,
            'Process': f'{parent_name} (Total)',
            'Test_Results': npdes_results_df.loc[0, parent_name],
            'CWNS_Data': matching_cwns_results_df.loc[0, parent_name],
            'Position': current_pos
        })
        current_pos += 1
        
        for process_name in child_processes:
            test_count = npdes_results_df.loc[0, process_name] if process_name in npdes_results_df.columns else 0
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


def count_facilities_with_processes(processes, data_source, process_names=None, unitprocess_keywords=None, matching_cwns_data=None, status_filter=None):
    """
    Count facilities that have any of the given processes
    
    Args:
        processes: List of process names to count
        data_source: DataFrame with NPDES data (if applicable)
        process_names: List of all process names (for CWNS mapping)
        unitprocess_keywords: Keywords dictionary for mapping
        matching_cwns_data: CWNS data DataFrame
        status_filter: Filter for status column ('present', 'future', 'present_and_future', or None for binary)
    """
    facilities_with_processes = set()
    
    if data_source is not None:
        # NPDES data - use new _status and _binary columns
        for process in processes:
            if status_filter:
                # Count based on status
                status_col = f'{process}_status'
                if status_col in data_source.columns:
                    if status_filter == 'any':
                        # Count present, future, or present_and_future
                        mask = data_source[status_col].isin(['present', 'future', 'present_and_future'])
                    else:
                        # Count specific status
                        mask = data_source[status_col] == status_filter
                    facilities = data_source[mask]['PERMIT_NUMBER'].tolist()
                    facilities_with_processes.update(facilities)
            else:
                # Count based on binary column
                binary_col = f'{process}_binary'
                if binary_col in data_source.columns:
                    facilities = data_source[data_source[binary_col] == 1]['PERMIT_NUMBER'].tolist()
                    facilities_with_processes.update(facilities)
    else:
        # CWNS data
        for process in processes:
            if process_names and process in process_names:
                el_abbadi_codes = get_el_abbadi_code_mapping(process, unitprocess_keywords)
                if el_abbadi_codes:
                    for code in el_abbadi_codes:
                        if code in matching_cwns_data.columns:
                            facilities_with_process = matching_cwns_data[matching_cwns_data[code] > 0]['PERMIT_NUMBER'].tolist()
                            facilities_with_processes.update(facilities_with_process)
    
    return len(facilities_with_processes)


def process_category_data(category, unitprocess_keywords, unit_process_results, matching_cwns_data, status_filter='any'):
    """
    Process data for a specific category and return DataFrames
    
    Args:
        category: Category name from keywords
        unitprocess_keywords: Keywords dictionary
        unit_process_results: NPDES results DataFrame (with _status and _binary columns)
        matching_cwns_data: CWNS data DataFrame
        status_filter: 'present', 'future', 'present_and_future', or 'any' (default)
    """
    category_keywords = unitprocess_keywords[category]
    process_names = get_all_keys(category_keywords)
    parent_child_mapping = get_parent_child_mapping(category_keywords)

    # Create NPDES results DataFrame - count based on binary columns
    npdes_results_aggregated = {}
    for process_name in process_names:
        binary_col = f'{process_name}_binary'
        if binary_col in unit_process_results.columns:
            # Count facilities where binary = 1
            count = (unit_process_results[binary_col] == 1).sum()
            npdes_results_aggregated[process_name] = count
        else:
            npdes_results_aggregated[process_name] = 0
    
    npdes_results_df = pd.DataFrame([npdes_results_aggregated])

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

    # Count parent categories for both NPDES and CWNS data
    for parent_name, child_processes in parent_child_mapping.items():
        # NPDES parent count using binary columns
        npdes_parent_count = count_facilities_with_processes(
            child_processes, 
            unit_process_results,
            status_filter=None  # Use binary for overall count
        )
        
        # CWNS parent count
        cwns_parent_count = count_facilities_with_processes(
            child_processes, 
            None, 
            process_names, 
            unitprocess_keywords, 
            matching_cwns_data
        )
        
        npdes_results_df[parent_name] = npdes_parent_count
        matching_cwns_results_df[parent_name] = cwns_parent_count
    
    return npdes_results_df, matching_cwns_results_df, parent_child_mapping, process_names


def create_status_breakdown_plot(unit_process_results, process_names, category_name, save_path=None):
    """
    Create a plot showing breakdown of present vs future treatments
    
    Args:
        unit_process_results: DataFrame with _status columns
        process_names: List of process names to plot
        category_name: Name of the category for title
        save_path: Path to save the plot
    """
    status_counts = {
        'present': [],
        'future': [],
        'present_and_future': []
    }
    plot_processes = []
    
    for process_name in process_names:
        status_col = f'{process_name}_status'
        if status_col in unit_process_results.columns:
            present = (unit_process_results[status_col] == 'present').sum()
            future = (unit_process_results[status_col] == 'future').sum()
            both = (unit_process_results[status_col] == 'present_and_future').sum()
            
            # Only include processes that have at least one facility
            if present + future + both > 0:
                status_counts['present'].append(present)
                status_counts['future'].append(future)
                status_counts['present_and_future'].append(both)
                plot_processes.append(process_name)
    
    if not plot_processes:
        print(f"No status data found for '{category_name}'")
        return
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x_pos = range(len(plot_processes))
    
    # Stack the bars
    ax.bar(x_pos, status_counts['present'], label='Present', color=COLORS['npdes_present'])
    ax.bar(x_pos, status_counts['future'], bottom=status_counts['present'], 
           label='Future/Planned', color=COLORS['npdes_future'])
    ax.bar(x_pos, status_counts['present_and_future'], 
           bottom=[p + f for p, f in zip(status_counts['present'], status_counts['future'])],
           label='Present & Future', color='#FFD700')  # Gold color
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_processes, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Facility Count', fontsize=14)
    ax.set_title(f'{category_name.replace("_", " ").title()} - Present vs Future Treatments', fontsize=16)
    ax.legend(loc='upper right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.subplots_adjust(bottom=0.25)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# Load all required data
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

# Get all categories from the JSON file
categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load NPDES results data from the specified date folder (NEW FORMAT)
unit_process_results = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv')
print(f"Loaded {len(unit_process_results)} NPDES facilities")
print(f"Columns sample: {unit_process_results.columns[:10].tolist()}")

test_permit_numbers = unit_process_results['PERMIT_NUMBER'].unique()

# Load and organize CWNS data from the specified date folder
cwns_data = pd.read_csv(f'npdes_permits/output/unit_processes_by_facility.csv')
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()
matching_cwns_data = ca_cwns_data[ca_cwns_data['PERMIT_NUMBER'].isin(test_permit_numbers)].copy()
print(f"Matched {len(matching_cwns_data)} CWNS facilities")

figures_dir = f'npdes_permits/output/{DATE_FOLDER}/figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Initialize data structures for "all categories" plot
all_categories_plot_data = []
all_categories_current_pos = 0
all_parent_child_mappings = {}
all_process_names = set()

for category in categories_to_plot:
    print(f"\nProcessing category: {category}")
    
    # Process data for this category
    npdes_results_df, matching_cwns_results_df, parent_child_mapping, process_names = process_category_data(
        category, unitprocess_keywords, unit_process_results, matching_cwns_data
    )
    
    all_process_names.update(process_names)
    all_parent_child_mappings.update(parent_child_mapping)

    # Create comparison plot (NPDES vs CWNS)
    plot_data = []
    current_pos = 0
    current_pos = append_plot_data(plot_data, npdes_results_df, matching_cwns_results_df, parent_child_mapping, current_pos)

    create_plot(
        plot_data,
        category,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{category}_comparison.png'
    )
    print(f"  ✓ Saved {category}_comparison.png")
    
    # Create status breakdown plot (Present vs Future)
    create_status_breakdown_plot(
        unit_process_results,
        process_names,
        category,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{category}_status_breakdown.png'
    )
    print(f"  ✓ Saved {category}_status_breakdown.png")
    
    # Add to all categories plot
    all_categories_current_pos = append_plot_data(
        all_categories_plot_data, 
        npdes_results_df, 
        matching_cwns_results_df, 
        parent_child_mapping, 
        all_categories_current_pos
    )

# Create comprehensive plot with all categories
create_plot(
    all_categories_plot_data, 
    'all_categories', 
    title_suffix=' - All Categories',
    figsize=(24, 8), 
    fontsize=10,
    save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/all_categories_comparison.png'
)
print("\n✓ Comprehensive plot saved as 'all_categories_comparison.png'")

# Create overall status summary
print("\n" + "="*80)
print("OVERALL STATUS SUMMARY")
print("="*80)

total_present = 0
total_future = 0
total_both = 0

for category in categories_to_plot:
    category_keywords = unitprocess_keywords[category]
    process_names = get_all_keys(category_keywords)
    
    for process_name in process_names:
        status_col = f'{process_name}_status'
        if status_col in unit_process_results.columns:
            total_present += (unit_process_results[status_col] == 'present').sum()
            total_future += (unit_process_results[status_col] == 'future').sum()
            total_both += (unit_process_results[status_col] == 'present_and_future').sum()

print(f"Total process instances marked as 'present': {total_present}")
print(f"Total process instances marked as 'future': {total_future}")
print(f"Total process instances marked as 'present_and_future': {total_both}")
print(f"Grand total: {total_present + total_future + total_both}")
print("="*80)