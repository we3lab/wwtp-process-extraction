import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os

from helpers.utils import *
from helpers.plotting import COLORS, HATCH_PATTERNS, plot_status_bars

DATE_FOLDER = '2026-2-18'


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



def get_process_names_for_category(category_name, category_keywords):
    """Return process names for a category, including leaf categories with alt_names."""
    if isinstance(category_keywords, dict) and 'alt_names' in category_keywords:
        return [category_name]
    return [name for name, _, _ in extract_leaves(category_keywords)]

    
def get_status_counts(process_name, unit_process_results):
    """Extract status breakdown for a process"""
    status_data = {'present': 0, 'future': 0, 'present_and_future': 0}
    if process_name in unit_process_results.columns:
        status_series = unit_process_results[process_name].astype(str).str.lower()
        status_data['present'] = (status_series == 'present').sum()
        status_data['future'] = (status_series == 'future').sum()
        status_data['present_and_future'] = (status_series == 'present_and_future').sum()
    return status_data



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
                       matching_cwns_data=None, unit_full_data=None, keyword_data=None,
                       match_only=True, include_cwns=True, include_legend=True,
                       title_suffix="", figsize=(12, 4), fontsize=14, save_path=None):
    """
    match_only=True  → LLM results matched to CWNS + CWNS bars (comparison plots)
    match_only=False → LLM all CA + optional Keyword all CA side-by-side (breakdown plots)

    keyword_data: optional second NPDES source (keyword-based); only used when match_only=False
    """
    if not plot_data:
        print(f"No processes found for '{category_name}'")
        return

    # LLM source: matched data for comparison, full CA for breakdown
    llm_data = unit_process_results if (match_only or unit_full_data is None) else unit_full_data
    has_keyword = not match_only and keyword_data is not None

    plot_df = pd.DataFrame(plot_data)
    plot_df['npdes_total'] = plot_df['Process'].apply(
        lambda p: sum(get_status_counts(p, llm_data).values())
    )
    plot_df = plot_df[plot_df['npdes_total'] > 0]
    if include_cwns:
        plot_df = plot_df[plot_df['CWNS_Data'] > 0]

    if plot_df.empty:
        print(f"No non-zero data to plot for '{category_name}'")
        return

    plot_df = plot_df.sort_values('npdes_total', ascending=False).reset_index(drop=True)
    plot_df['Position'] = range(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)

    # Bar layout: comparison = LLM + CWNS; breakdown = LLM [+ Keyword]
    if include_cwns:
        width, llm_offset, cwns_offset = 0.35, -0.175, 0.175
    elif has_keyword:
        width, llm_offset, kw_offset = 0.35, -0.175, 0.175
    else:
        width, llm_offset = 0.6, 0.0

    for idx, row in plot_df.iterrows():
        pos = row['Position']
        alpha = 0.5 if row['Category'] == 'Process' else 1.0

        llm_status = get_status_counts(row['Process'], llm_data)
        plot_status_bars(ax, pos + llm_offset, width, llm_status, alpha, color_key='npdes_total')

        if has_keyword:
            kw_status = get_status_counts(row['Process'], keyword_data)
            plot_status_bars(ax, pos + kw_offset, width, kw_status, alpha, color_key='npdes')

        if include_cwns:
            if matching_cwns_data is not None:
                cwns_status = get_status_counts(row['Process'], matching_cwns_data)
                plot_status_bars(ax, pos + cwns_offset, width, cwns_status, alpha, color_key='cwns')
            else:
                ax.bar(pos + cwns_offset, row['CWNS_Data'], width,
                       color=COLORS['cwns'], alpha=alpha, edgecolor='black', linewidth=0.5)

    ax.set_xticks(plot_df['Position'])
    ax.set_xticklabels(plot_df['Process'], rotation=45, ha='right', fontsize=fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if include_cwns:
        ax.set_title(f'{category_name.replace("_", " ").title()} Treatment Process Comparison {title_suffix}',
                     fontsize=18)

    ax.set_ylabel('WWTP Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if include_legend:
        llm_label = '  NPDES LLM (matched)' if match_only else '  NPDES LLM'
        legend_handles = [
            Patch(color='none', label='Data Source'),
            Patch(facecolor=COLORS['npdes_total'], edgecolor='black', linewidth=0.5, label=llm_label),
        ]
        if has_keyword:
            legend_handles.append(Patch(facecolor=COLORS['npdes'], edgecolor='black', linewidth=0.5, label='  NPDES Keyword'))
        if include_cwns:
            legend_handles.append(Patch(facecolor=COLORS['cwns'], edgecolor='black', linewidth=0.5, label='  CWNS'))
        n_sources = len(legend_handles)
        legend_handles += [
            Patch(color='none', label='Status'),
            Patch(facecolor='gray', edgecolor='black', linewidth=0.5, label='  Present'),
            Patch(facecolor='gray', hatch=HATCH_PATTERNS['future'], edgecolor='black', linewidth=0.5, label='  Future (Planned)'),
            Patch(facecolor='gray', hatch=HATCH_PATTERNS['present_and_future'], edgecolor='black', linewidth=0.5, label='  Present & Future'),
        ]
        leg = ax.legend(handles=legend_handles, loc='upper right', fontsize=11)
        header_indices = {0, n_sources}
        for i, (handle, text) in enumerate(zip(leg.legend_handles, leg.get_texts())):
            if i in header_indices:
                handle.set_visible(False)
                text.set_fontweight('bold')

    plt.subplots_adjust(bottom=0.25)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def count_npdes_facilities(processes, df):
    """Count unique NPDES facilities (by FACILITY_KEY) with any of the given processes."""
    facilities = set()
    for process in processes:
        mask = build_binary_mask(df, process, None)
        if mask is not None:
            facilities.update(df.loc[mask, 'FACILITY_KEY'])
    return len(facilities)


def count_cwns_facilities(processes, df):
    """Count unique CWNS facilities (by CWNS_ID) with any of the given processes.

    Expects df columns to be taxonomy names (as produced by step0).
    """
    facilities = set()
    for process in processes:
        if process in df.columns:
            mask = build_cwns_presence_mask(df[process])
            facilities.update(df.loc[mask, 'CWNS_ID'])
    return len(facilities)


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
    parent_child_mapping = {}
    for parent_name, parent_details in category_keywords.items():
        if isinstance(parent_details, dict) and 'alt_names' not in parent_details:
            # This is a parent category
            children = [child_name for child_name, child_details in parent_details.items()
                       if isinstance(child_details, dict) and 'alt_names' in child_details]
            if children:
                parent_child_mapping[parent_name] = children

    # Create NPDES results DataFrame - count based on binary columns
    npdes_results_aggregated = {}
    for process_name in process_names:
        binary_mask = build_binary_mask(unit_process_results, process_name, unitprocess_keywords)
        if binary_mask is not None:
            npdes_results_aggregated[process_name] = int(binary_mask.sum())
        else:
            npdes_results_aggregated[process_name] = 0
    
    npdes_results_df = pd.DataFrame([npdes_results_aggregated])

    # Count individual processes for CWNS data (columns are taxonomy names, as built by step0)
    matching_cwns_results_df = pd.DataFrame(0, index=[0], columns=process_names)
    for process_name in process_names:
        if process_name in matching_cwns_data.columns:
            matching_cwns_results_df.loc[0, process_name] = int(
                build_cwns_presence_mask(matching_cwns_data[process_name]).sum()
            )

    # Count parent categories for both NPDES and CWNS data
    for parent_name, child_processes in parent_child_mapping.items():
        npdes_parent_count = count_npdes_facilities(child_processes, unit_process_results)
        cwns_parent_count = count_cwns_facilities(child_processes, matching_cwns_data)
        
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

# Extracting CA permit numbers from PDF filename?
nan_permit_mask = unit_process_results['PERMIT_NUMBER'].isna()
if nan_permit_mask.any():
    extracted = unit_process_results.loc[nan_permit_mask, 'PDF_File'].apply(
        lambda f: m.group(0).upper() if (m := re.search(r'CA\d{7}', str(f), re.IGNORECASE)) else None
    )
    filled = extracted.notna().sum()
    unit_process_results.loc[extracted.index, 'PERMIT_NUMBER'] = extracted
    print(f"Resolved {filled} of {nan_permit_mask.sum()} NaN PERMIT_NUMBERs from PDF filenames")

unit_process_results['FACILITY_KEY'] = (
    unit_process_results['PERMIT_NUMBER'].fillna('') + '||' +
    unit_process_results['FACILITY_NAME'].fillna('')
)

print(f"NPDES data: Loaded {len(unit_process_results)} rows, "
      f"{unit_process_results['FACILITY_KEY'].nunique()} unique facilities")

# Get unique NPDES permit numbers (exclude NaN)
npdes_permit_numbers = set(unit_process_results['PERMIT_NUMBER'].dropna().unique())
print(f"Unique NPDES permit numbers: {len(npdes_permit_numbers)}")

# Load and consolidate CWNS data with facility names and clean NPDES permits
cwns_data = pd.read_csv(f'npdes_permits/output/unit_processes_by_facility.csv',
                         low_memory=False, dtype={'CWNS_ID': str})
print(f"\nCWNS data: Total rows: {len(cwns_data)}")

ca_cwns_data = prepare_cwns_ca(
    cwns_data,
    'npdes_permits/data/cwns/cwns_permits_match_manual.csv',
    'npdes_permits/data/cwns/cwns_facility_name_match_manual.csv',
)
print(f"CWNS CA facilities (consolidated): {len(ca_cwns_data)}")

# Build name→permit lookup for exact normalized name matching (Tier 3)
# Combine LLM results and keyword-based all_ca_npdes so both data sources inform CWNS matching
all_ca_npdes = pd.read_csv(f'npdes_permits/output/{DATE_FOLDER}/all_ca_npdes.csv', dtype=str)
all_ca_npdes_names = (all_ca_npdes[all_ca_npdes['NPDES No.'].notna()]
                      [['NPDES No.', 'Facility Name']]
                      .rename(columns={'NPDES No.': 'PERMIT_NUMBER', 'Facility Name': 'FACILITY_NAME'}))
llm_names = (unit_process_results[unit_process_results['PERMIT_NUMBER'].notna()]
             [['PERMIT_NUMBER', 'FACILITY_NAME']])
# LLM names take priority (keep='first') when permit numbers conflict
combined_names = pd.concat([llm_names, all_ca_npdes_names]).drop_duplicates(
    subset=['PERMIT_NUMBER', 'FACILITY_NAME'], keep='first')
npdes_name_to_permit = (combined_names.drop_duplicates(subset='FACILITY_NAME', keep='first')
                        .set_index('FACILITY_NAME')['PERMIT_NUMBER']
                        .to_dict())

# Match CWNS to NPDES permits
ca_cwns_data = match_cwns_to_npdes(ca_cwns_data, npdes_permit_numbers,
                                    npdes_name_to_permit=npdes_name_to_permit)
matching_cwns_data = ca_cwns_data[ca_cwns_data['matched']].copy()

# Filter NPDES to only facilities with matching CWNS data
matching_permit_numbers = set(matching_cwns_data['linking_permit'].dropna())
unit_process_results_matched = unit_process_results[
    unit_process_results['PERMIT_NUMBER'].isin(matching_permit_numbers)
].copy()

print(f"CWNS facilities matched: {len(matching_cwns_data)}")
print(f"NPDES rows matched: {len(unit_process_results_matched)}")
print(f"NPDES permits matched: {len(matching_permit_numbers)}")
npdes_unmatched = npdes_permit_numbers - matching_permit_numbers
print(f"NPDES permits unmatched: {len(npdes_unmatched)}")
print(f"CWNS facilities unmatched: {len(ca_cwns_data) - len(matching_cwns_data)}")

# Save unmatched CWNS facilities
unmatched_cwns = ca_cwns_data[~ca_cwns_data['matched']]
unmatched_cwns[['CWNS_ID', 'FACILITY_NAME', 'raw_permit_list', 'NPDES_PERMIT']].to_csv(
    f'npdes_permits/output/{DATE_FOLDER}/unmatched_cwns_facilities.csv', index=False)

# Use matched data for the rest of the analysis
unit_full = unit_process_results.copy()  # all-CA keyword results
unit_process_results = unit_process_results_matched

# Load LLM results for breakdown comparison
llm_results_path = 'npdes_permits/output/llm_unit_processes_by_facility.csv'
llm_results = pd.read_csv(llm_results_path) if os.path.exists(llm_results_path) else None
if llm_results is not None:
    print(f"LLM results: {len(llm_results)} facilities")
else:
    print(f"LLM results not found at {llm_results_path}; breakdown plots will show keyword only")

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

    # Comparison plot: matched NPDES (keyword) + CWNS
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        matching_cwns_data=matching_cwns_data,
        match_only=True,
        include_cwns=True,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{safe_category}_source_comparison.png'
    )
    print(f"  Saved {safe_category}_source_comparison.png")

    # Status breakdown: all CA LLM vs keyword side-by-side
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        unit_full_data=llm_results,
        keyword_data=unit_full if llm_results is not None else None,
        match_only=False,
        include_cwns=False,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{safe_category}_npdes_method_comparison.png'
    )
    print(f"  Saved {safe_category}_npdes_method_comparison.png")
    
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
create_status_plot(
    all_categories_plot_data,
    unit_process_results,
    'all_categories',
    matching_cwns_data=matching_cwns_data,
    match_only=True,
    include_cwns=True,
    title_suffix=' - All Categories',
    figsize=(20, 6),
    fontsize=10,
    save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/all_categories_source_comparison.png'
)


# Create matching statistics report
print(f"Total California CWNS facilities (consolidated): {len(ca_cwns_data)}")
print(f"Total NPDES rows analyzed: {len(unit_process_results)}")
print(f"CWNS facilities matched: {len(matching_cwns_data)}")
print(f"NPDES permits matched: {len(matching_permit_numbers)}")
print(f"Match rate: {len(matching_permit_numbers) / len(npdes_permit_numbers) * 100:.1f}% of NPDES permits")
print(f"Coverage: {len(matching_cwns_data) / len(ca_cwns_data) * 100:.1f}% of CA CWNS facilities")

# Create overall status summary

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


