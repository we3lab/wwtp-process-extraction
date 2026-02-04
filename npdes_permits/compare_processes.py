import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os

from utils import *

DATE_FOLDER = '2025-10-31'

COLORS = {
    'cwns': '#FFD700',           # Gold for CWNS
    'npdes': '#1482a5ff',        # Dark blue for NPDES present
}

# Hatch patterns for status values
HATCH_PATTERNS = {
    'present': '',               # Solid fill
    'future': '///',             # Diagonal lines
    'present_and_future': 'xxx'  # Cross-hatch
}


def build_status_mask(df, process_name, unitprocess_keywords, status_filter):
    if process_name in df.columns:
        series = df[process_name].astype(str).str.lower()
        if status_filter == 'any':
            return series.isin(['present', 'future', 'present_and_future'])
        return series == status_filter
    return None


def build_binary_mask(df, process_name, unitprocess_keywords):
    if process_name in df.columns:
        return build_status_mask(df, process_name, unitprocess_keywords, 'any')
    return None


def build_cwns_presence_mask(series):
    """Return boolean mask for CWNS presence values (numeric or string)."""
    s = series.astype(str).str.lower()
    numeric = pd.to_numeric(series, errors='coerce')
    return (numeric > 0) | s.isin({'present', 'planned', 'present_and_future', 'present_and_planned'}) | s.str.startswith('present')


def get_status_counts(process_name, unit_process_results):
    """Extract status breakdown for a process"""
    status_data = {'present': 0, 'future': 0, 'present_and_future': 0}
    if process_name in unit_process_results.columns:
        status_series = unit_process_results[process_name].astype(str).str.lower()
        status_data['present'] = (status_series == 'present').sum()
        status_data['future'] = (status_series == 'future').sum()
        status_data['present_and_future'] = (status_series == 'present_and_future').sum()
    return status_data


def plot_npdes_status_bars(ax, pos, width, status_data, alpha=1.0):
    """Plot stacked NPDES bars with status-based hatching"""
    bottom = 0
    for status in ['present', 'future', 'present_and_future']:
        if status_data[status] > 0:
            ax.bar(pos - width/2, status_data[status], width,
                   bottom=bottom,
                   color=COLORS['npdes'],
                   hatch=HATCH_PATTERNS[status],
                   alpha=alpha,
                   edgecolor='black',
                   linewidth=0.5)
            bottom += status_data[status]


def create_status_legend(include_cwns=True):
    """Create legend with status and data source information"""
    list = [
        Patch(facecolor=COLORS['npdes'], label='NPDES (Present)'),
        Patch(facecolor=COLORS['npdes'], hatch=HATCH_PATTERNS['future'], label='NPDES (Future)'),
        Patch(facecolor=COLORS['npdes'], hatch=HATCH_PATTERNS['present_and_future'], label='NPDES (Present & Future)'),
        ]
    if include_cwns:
        list.extend([Patch(facecolor=COLORS['cwns'], label='CWNS')])
    return list


def append_plot_data(plot_data, npdes_results_df, matching_cwns_results_df, parent_child_mapping, current_pos, process_names=None):
    """Append plot data for a category to the given plot_data list"""
    child_process_set = set()
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
            child_process_set.add(process_name)
        
        current_pos += 0.5

    if process_names:
        for process_name in process_names:
            if process_name in child_process_set:
                continue
            test_count = npdes_results_df.loc[0, process_name] if process_name in npdes_results_df.columns else 0
            cwns_count = matching_cwns_results_df.loc[0, process_name] if process_name in matching_cwns_results_df.columns else 0

            plot_data.append({
                'Category': 'Process',
                'Parent': None,
                'Process': process_name,
                'Test_Results': test_count,
                'CWNS_Data': cwns_count,
                'Position': current_pos
            })
            current_pos += 1

        current_pos += 0.5
        
    return current_pos


def create_status_plot(plot_data, unit_process_results, category_name, 
                       include_cwns=True, include_legend=True,
                       title_suffix="", figsize=(16, 6), fontsize=14, save_path=None):
    if not plot_data:
        print(f"No processes found for '{category_name}'")
        return

    plot_df = pd.DataFrame(plot_data)
    
    # Add NPDES totals and sort descending
    plot_df['npdes_total'] = plot_df['Process'].apply(
        lambda p: sum(get_status_counts(p, unit_process_results).values())
    )

    plot_df = plot_df[plot_df['npdes_total'] > 0]
    if include_cwns:
        plot_df = plot_df[(plot_df['CWNS_Data'] > 0)]

    if plot_df.empty:
        print(f"No non-zero data to plot for '{category_name}'")
        return

    plot_df = plot_df.sort_values('npdes_total', ascending=False).reset_index(drop=True)
    plot_df['Position'] = range(len(plot_df))
    
    fig, ax = plt.subplots(figsize=figsize)
    width = 0.35 if include_cwns else 0.6

    # Plot bars
    for idx, row in plot_df.iterrows():
        status_data = get_status_counts(row['Process'], unit_process_results)
        alpha = 0.5 if row['Category'] == 'Process' else 1.0
        
        plot_npdes_status_bars(ax, row['Position'], width, status_data, alpha)
        if include_cwns:
            ax.bar(row['Position'] + width/2, row['CWNS_Data'], width,
                   color=COLORS['cwns'], alpha=alpha, edgecolor='black', linewidth=0.5)

    # Format axes
    ax.set_xticks(plot_df['Position'])
    ax.set_xticklabels(plot_df['Process'], rotation=45, ha='right', fontsize=fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Titles and labels
    if include_cwns:
        plot_title = f'{category_name.replace("_", " ").title()} Treatment Process Comparison {title_suffix}'
        ax.set_title(plot_title, fontsize=18)
    
    ax.set_ylabel('WWTP Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if include_legend:
        ax.legend(handles=create_status_legend(), loc='upper right', fontsize=11)

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
                mask = build_status_mask(data_source, process, unitprocess_keywords, status_filter)
                if mask is not None:
                    facilities = data_source[mask]['PERMIT_NUMBER'].tolist()
                    facilities_with_processes.update(facilities)
            else:
                mask = build_binary_mask(data_source, process, unitprocess_keywords)
                if mask is not None:
                    facilities = data_source[mask]['PERMIT_NUMBER'].tolist()
                    facilities_with_processes.update(facilities)
    else:
        # CWNS data
        for process in processes:
            if process_names and process in process_names:
                process_details = find_process_details(process, unitprocess_keywords)
                if process_details:
                    cwns_names = get_cwns_unit_process_names(process, process_details)
                    for cwns_name in cwns_names:
                        if cwns_name in matching_cwns_data.columns:
                            mask = build_cwns_presence_mask(matching_cwns_data[cwns_name])
                            facilities_with_process = matching_cwns_data[mask]['PERMIT_NUMBER'].tolist()
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
    process_names = get_process_names_for_category(category, category_keywords)
    parent_child_mapping = get_parent_child_mapping(category_keywords)

    # Create NPDES results DataFrame - count based on binary columns
    npdes_results_aggregated = {}
    for process_name in process_names:
        binary_mask = build_binary_mask(unit_process_results, process_name, unitprocess_keywords)
        if binary_mask is not None:
            npdes_results_aggregated[process_name] = int(binary_mask.sum())
        else:
            npdes_results_aggregated[process_name] = 0
    
    npdes_results_df = pd.DataFrame([npdes_results_aggregated])

    # Count individual processes for CWNS data
    matching_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)
    for process_name in process_names:
        process_details = find_process_details(process_name, unitprocess_keywords)
        if process_details:
            cwns_names = get_cwns_unit_process_names(process_name, process_details)
            count = sum(build_cwns_presence_mask(matching_cwns_data[cwns_name]).sum()
                       for cwns_name in cwns_names 
                       if cwns_name in matching_cwns_data.columns)
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




# Load all required data
with open('npdes_permits/data/unitprocess_keywords.json', 'r') as f:
    unitprocess_keywords = json.load(f)

# Get all categories from the JSON file
categories_to_plot = list(unitprocess_keywords.keys())
print(f"Categories: {categories_to_plot}")

# Load NPDES results data from the specified date folder
unit_process_results = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/unit_processes.csv')

print(f"\n{'='*80}")
print("DATA LOADING SUMMARY")
print(f"{'='*80}")
print(f"NPDES data: Loaded {len(unit_process_results)} facilities")
print(f"Columns sample: {unit_process_results.columns[:10].tolist()}")

# Get unique NPDES permit numbers
npdes_permit_numbers = set(unit_process_results['PERMIT_NUMBER'].unique())
print(f"Unique NPDES permit numbers: {len(npdes_permit_numbers)}")

# Load and organize CWNS data
cwns_data = pd.read_csv(f'npdes_permits/output/unit_processes_by_facility.csv')
print(f"\nCWNS data: Total facilities: {len(cwns_data)}")

# Filter for California only
ca_cwns_data = cwns_data[cwns_data['STATE_CODE'] == 'CA'].copy()
print(f"CWNS data: California facilities: {len(ca_cwns_data)}")

# Get unique CWNS permit numbers (California only)
cwns_permit_numbers = set(ca_cwns_data['PERMIT_NUMBER'].unique())
print(f"Unique CWNS CA permit numbers: {len(cwns_permit_numbers)}")

# Find matching permit numbers
matching_permit_numbers = npdes_permit_numbers.intersection(cwns_permit_numbers)
print(f"\n{'='*80}")
print("PERMIT NUMBER MATCHING")
print(f"{'='*80}")
print(f"Facilities in BOTH datasets: {len(matching_permit_numbers)}")
print(f"Facilities ONLY in NPDES: {len(npdes_permit_numbers - cwns_permit_numbers)}")
print(f"Facilities ONLY in CWNS: {len(cwns_permit_numbers - npdes_permit_numbers)}")

# Filter both datasets to only matching facilities
unit_process_results_matched = unit_process_results[
    unit_process_results['PERMIT_NUMBER'].isin(matching_permit_numbers)
].copy()
matching_cwns_data = ca_cwns_data[
    ca_cwns_data['PERMIT_NUMBER'].isin(matching_permit_numbers)
].copy()

print(f"\nFiltered NPDES data: {len(unit_process_results_matched)} facilities")
print(f"Filtered CWNS data: {len(matching_cwns_data)} facilities")

# Show some example matching permit numbers
if matching_permit_numbers:
    examples = list(matching_permit_numbers)[:5]
    print(f"\nExample matching permit numbers: {examples}")

# Show some examples of non-matching
npdes_only = npdes_permit_numbers - cwns_permit_numbers
cwns_only = cwns_permit_numbers - npdes_permit_numbers
if npdes_only:
    print(f"\nExample NPDES-only permits: {list(npdes_only)[:5]}")
if cwns_only:
    print(f"Example CWNS-only permits: {list(cwns_only)[:5]}")

print(f"{'='*80}\n")

# Use matched data for the rest of the analysis
unit_process_results = unit_process_results_matched

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
    safe_category = category.replace("/", "_").replace(os.sep, "_")
    
    # Process data for this category
    npdes_results_df, matching_cwns_results_df, parent_child_mapping, process_names = process_category_data(
        category, unitprocess_keywords, unit_process_results, matching_cwns_data
    )
    
    all_process_names.update(process_names)
    all_parent_child_mappings.update(parent_child_mapping)

    plot_data = []  # for this category
    current_pos = 0
    current_pos = append_plot_data(plot_data, npdes_results_df, matching_cwns_results_df, parent_child_mapping, current_pos, process_names)

    # Create comparison plot with CWNS bars
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        include_cwns=True,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{safe_category}_comparison_with_status.png'
    )
    print(f"  Saved {safe_category}_comparison_with_status.png")
    
    # Create NPDES-only status breakdown plot
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        include_cwns=False,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{safe_category}_status_breakdown.png'
    )
    print(f"  Saved {safe_category}_status_breakdown.png")
    
    # Add to all categories plot
    all_categories_current_pos = append_plot_data(
        all_categories_plot_data, 
        npdes_results_df, 
        matching_cwns_results_df, 
        parent_child_mapping, 
        all_categories_current_pos,
        process_names
    )

# Create comprehensive plot with all categories (with status hatching)
all_plot_df = pd.DataFrame(all_categories_plot_data)
create_status_plot(
    all_categories_plot_data,
    unit_process_results,
    'all_categories',
    include_cwns=True,
    title_suffix=' - All Categories',
    figsize=(24, 8),
    fontsize=10,
    save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/all_categories_comparison_with_status.png'
)
print("\n Comprehensive plot saved as 'all_categories_comparison_with_status.png'")


# Create matching statistics report
print("\n" + "="*80)
print("FACILITY MATCHING REPORT")
print("="*80)
print(f"Total California CWNS facilities: {len(ca_cwns_data)}")
print(f"Total NPDES facilities analyzed: {len(unit_process_results_matched)}")
print(f"Facilities matched and compared: {len(matching_permit_numbers)}")
print(f"Match rate: {len(matching_permit_numbers) / len(npdes_permit_numbers) * 100:.1f}% of NPDES facilities")
print(f"Coverage: {len(matching_permit_numbers) / len(ca_cwns_data) * 100:.1f}% of CA CWNS facilities")

# Create overall status summary
print("\n" + "="*80)
print("OVERALL STATUS SUMMARY")
print("="*80)

total_present = 0
total_future = 0
total_both = 0

for category in categories_to_plot:
    category_keywords = unitprocess_keywords[category]
    process_names = get_process_names_for_category(category, category_keywords)
    
    for process_name in process_names:
        if process_name in unit_process_results.columns:
            total_present += (unit_process_results[process_name] == 'present').sum()
            total_future += (unit_process_results[process_name] == 'future').sum()
            total_both += (unit_process_results[process_name] == 'present_and_future').sum()

print(f"Total process instances marked as 'present': {total_present}")
print(f"Total process instances marked as 'future': {total_future}")
print(f"Total process instances marked as 'present_and_future': {total_both}")
print(f"Grand total: {total_present + total_future + total_both}")