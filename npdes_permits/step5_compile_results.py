import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import os
from collections import defaultdict

from helpers.utils import *
from helpers.plotting import COLORS, COLORS_GT, HATCH_PATTERNS
from helpers.load_google_sheet import load_google_sheet_csv

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


def build_cwns_presence_mask(series):
    """Return boolean mask for CWNS presence values (present, future, or present_and_future)."""
    s = series.astype(str).str.lower()
    return s.isin({'present', 'future', 'present_and_future'})


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


def plot_npdes_status_bars(ax, center, width, status_data, alpha=1.0, color_key='npdes'):
    """Plot stacked NPDES bars with status-based hatching at the given center x position."""
    bottom = 0
    for status in ['present', 'future', 'present_and_future']:
        if status_data[status] > 0:
            ax.bar(center, status_data[status], width,
                   bottom=bottom,
                   color=COLORS[color_key],
                   hatch=HATCH_PATTERNS[status],
                   alpha=alpha,
                   edgecolor='black',
                   linewidth=0.5)
            bottom += status_data[status]


def get_cwns_status_counts(process_name, cwns_df):
    """Extract status breakdown for a CWNS process column."""
    status_data = {'present': 0, 'future': 0, 'present_and_future': 0}
    if process_name in cwns_df.columns:
        s = cwns_df[process_name].astype(str).str.lower()
        status_data['present'] = int((s == 'present').sum())
        status_data['future'] = int((s == 'future').sum())
        status_data['present_and_future'] = int((s == 'present_and_future').sum())
    return status_data


def plot_cwns_status_bars(ax, center, width, status_data, alpha=1.0):
    """Plot stacked CWNS bars with status-based hatching at the given center x position."""
    bottom = 0
    for status in ['present', 'future', 'present_and_future']:
        if status_data[status] > 0:
            ax.bar(center, status_data[status], width,
                   bottom=bottom,
                   color=COLORS['cwns'],
                   hatch=HATCH_PATTERNS[status],
                   alpha=alpha,
                   edgecolor='black',
                   linewidth=0.5)
            bottom += status_data[status]


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
        plot_npdes_status_bars(ax, pos + llm_offset, width, llm_status, alpha, color_key='npdes_total')

        if has_keyword:
            kw_status = get_status_counts(row['Process'], keyword_data)
            plot_npdes_status_bars(ax, pos + kw_offset, width, kw_status, alpha, color_key='npdes')

        if include_cwns:
            if matching_cwns_data is not None:
                cwns_status = get_cwns_status_counts(row['Process'], matching_cwns_data)
                plot_cwns_status_bars(ax, pos + cwns_offset, width, cwns_status, alpha)
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
unit_full = unit_process_results.copy()
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

    # Comparison plot: matched NPDES (keyword) + CWNS
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        matching_cwns_data=matching_cwns_data,
        match_only=True,
        include_cwns=True,
        save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/{safe_category}_comparison_with_status.png'
    )
    print(f"  Saved {safe_category}_comparison_with_status.png")

    # Status breakdown: all CA LLM + Keyword (when available)
    create_status_plot(
        plot_data,
        unit_process_results,
        category,
        unit_full_data=unit_full,
        keyword_data=None,  # TODO: pass keyword_results once available
        match_only=False,
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
    save_path=f'npdes_permits/output/{DATE_FOLDER}/figures/all_categories_comparison_with_status.png'
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


# GROUND TRUTH COMPARISON: GroundTruth vs NPDES Text vs CWNS

GOOGLE_SHEET_ID = '18U4IlfAiNH1UNdUYH5fF35fX99ll9SciKYRUuHUdT8w'


def is_yes(val):
    """Check if a cell value means the process is present (YES or PLANNED)."""
    return str(val).strip().upper() in ('YES', 'PLANNED')


def count_yes(series):
    """Count YES/PLANNED values in a sheet column (case-insensitive)."""
    return series.fillna('').apply(is_yes).sum()


def create_ground_truth_plot(gt_rows, n_facilities, save_path):
    """
    Stacked error magnitude chart: FN (solid, bottom) + FP (hatched x, top) per source.
    Secondary y-axis shows % of total GT facilities (N=n_facilities).

    gt_rows: list of dicts with keys 'Process_Category', 'GroundTruth',
             'NPDES_FP', 'NPDES_FN', 'CWNS_FP', 'CWNS_FN'
    n_facilities: total number of ground truth facilities (denominator for % axis)
    """
    fontsize = 12
    df = pd.DataFrame(gt_rows)
    df = df[df['GroundTruth'] > 0].copy()
    if df.empty:
        print("No populated processes to plot.")
        return
    df = df.sort_values('GroundTruth', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    w = 0.35

    for i, row in df.iterrows():
        ax.bar(i - w / 2, row['NPDES_FN'], w,
               color=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i - w / 2, row['NPDES_FP'], w, bottom=row['NPDES_FN'],
               color=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5, hatch='++')
        ax.bar(i + w / 2, row['CWNS_FN'], w,
               color=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5, hatch='--')
        ax.bar(i + w / 2, row['CWNS_FP'], w, bottom=row['CWNS_FN'],
               color=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5, hatch='++')

    ax.set_xticks(list(x))
    ax.set_xticklabels(df['Process_Category'], rotation=45, ha='right', fontsize=fontsize)
    ax.set_ylabel('Facility Count (error magnitude)', fontsize=16)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Secondary y-axis: % of total GT facilities
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo / n_facilities * 100, hi / n_facilities * 100)
    ax2.set_ylabel(f'Error % (N={n_facilities})', fontsize=16)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax2.tick_params(axis='y', labelsize=fontsize)

    # Custom legend: colors (top) + hatch types (bottom), top-left
    legend_handles = [
        Patch(facecolor=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5,
              label='NPDES Text'),
        Patch(facecolor=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5,
              label='CWNS'),
        Patch(color='none', label=''),  # spacer
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='--',
              label='False Negative (missed)'),
        Patch(facecolor='white', edgecolor='black', linewidth=0.5, hatch='++',
              label='False Positive (extra)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=11,
              handlelength=2, handleheight=1.2)

    plt.subplots_adjust(bottom=0.35)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {os.path.basename(save_path)}")
    plt.close(fig)


# Load the two Google Sheet tabs
ground_truth_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - Ground Truth')
npdes_text_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - From NPDES Text')
bacwa_ground_truth_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'BACWA - Ground Truth')
bacwa_npdes_text_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'BACWA - From NPDES Text')

print(f"GroundTruth sheet: {len(ground_truth_df)} facilities")
print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")
print(f"BACWA Ground Truth sheet: {len(bacwa_ground_truth_df)} facilities")
print(f"BACWA NPDES Text sheet: {len(bacwa_npdes_text_df)} facilities")

# Determine the process columns (everything after the first 4 metadata cols)
meta_cols = ['Agency', 'Facility_Name', 'NPDES_No', 'PDF_File', "Ground Truth Sources"]
ground_truth_process_cols = [c for c in ground_truth_df.columns if c not in meta_cols]
npdes_text_process_cols = [c for c in npdes_text_df.columns if c not in meta_cols]
bacwa_ground_truth_process_cols = [c for c in bacwa_ground_truth_df.columns if c not in meta_cols]
bacwa_npdes_text_process_cols = [c for c in bacwa_npdes_text_df.columns if c not in meta_cols]

# Use GroundTruth process columns as the canonical list (union with NPDES text if needed)
all_sheet_process_cols = list(dict.fromkeys(ground_truth_process_cols + npdes_text_process_cols))

# Filter to only facilities present in all three datasets (GroundTruth, NPDES text, and CWNS)
ground_truth_permits = set(ground_truth_df['NPDES_No'].dropna().str.strip())
text_permits = set(npdes_text_df['NPDES_No'].dropna().str.strip())
cwns_permits_gt = set(ca_cwns_data['linking_permit'].dropna().str.strip())
common_permits = ground_truth_permits & text_permits & cwns_permits_gt

print(f"Facilities in all 3 sources: {len(common_permits)}")

# Filter each source to common facilities
ground_truth_common = ground_truth_df[ground_truth_df['NPDES_No'].str.strip().isin(common_permits)].copy()
text_common = npdes_text_df[npdes_text_df['NPDES_No'].str.strip().isin(common_permits)].copy()
cwns_common = ca_cwns_data[ca_cwns_data['linking_permit'].str.strip().isin(common_permits)].copy()

# GROUND TRUTH COMPARISON: +/- % vs GroundTruth grouped by JSON category

# Build mapping from leaf process name -> top-level JSON category
leaf_to_category = {}
for cat_name, cat_value in unitprocess_keywords.items():
    if isinstance(cat_value, dict) and 'alt_names' in cat_value:
        # Leaf-level category (e.g. Comminution, Grit Removal)
        leaf_to_category[cat_name] = cat_name
    else:
        for leaf_name, _, _ in extract_leaves(cat_value):
            leaf_to_category[leaf_name] = cat_name

# Aggregate facility-level counts per category using set unions to avoid double-counting
cat_ground_truth_facilities = defaultdict(set)
cat_npdes_facilities = defaultdict(set)
cat_cwns_facilities = defaultdict(set)

for col in all_sheet_process_cols:
    category = leaf_to_category.get(col, col)
    # GroundTruth
    if col in ground_truth_common.columns:
        for _, row in ground_truth_common.iterrows():
            if is_yes(row.get(col, '')):
                cat_ground_truth_facilities[category].add(row['NPDES_No'])
    # NPDES text
    if col in text_common.columns:
        for _, row in text_common.iterrows():
            if is_yes(row.get(col, '')):
                cat_npdes_facilities[category].add(row['NPDES_No'])
    # CWNS
    if col in cwns_common.columns:
        mask = build_cwns_presence_mask(cwns_common[col])
        for permit in cwns_common.loc[mask, 'linking_permit']:
            cat_cwns_facilities[category].add(permit)

# Build summary rows
all_cats = sorted(set(list(cat_ground_truth_facilities.keys()) + list(cat_npdes_facilities.keys()) + list(cat_cwns_facilities.keys())))
gt_simple_rows = []
for cat in all_cats:
    ground_truth = len(cat_ground_truth_facilities[cat])
    npdes = len(cat_npdes_facilities[cat])
    cwns = len(cat_cwns_facilities[cat])
    if ground_truth == 0 and npdes == 0 and cwns == 0:
        continue
    if ground_truth > 0:
        npdes_str = f"{(npdes - ground_truth) / ground_truth * 100:+.0f}%"
        cwns_str = f"{(cwns - ground_truth) / ground_truth * 100:+.0f}%"
    else:
        npdes_str = f"+{npdes}" if npdes > 0 else "0"
        cwns_str = f"+{cwns}" if cwns > 0 else "0"
    # FP/FN restricted to common_permits where all three sources have data
    gt_p    = cat_ground_truth_facilities[cat] & common_permits
    npdes_p = cat_npdes_facilities[cat]        & common_permits
    cwns_p  = cat_cwns_facilities[cat]         & common_permits
    gt_simple_rows.append({
        'Process_Category': cat,
        'GroundTruth': ground_truth,
        'NPDES_Manual': npdes,
        'NPDES_vs_GT': npdes_str,
        'NPDES_FP': len(npdes_p - gt_p),
        'NPDES_FN': len(gt_p - npdes_p),
        'CWNS': cwns,
        'CWNS_vs_GT': cwns_str,
        'CWNS_FP': len(cwns_p - gt_p),
        'CWNS_FN': len(gt_p - cwns_p),
    })

gt_simple_df = pd.DataFrame(gt_simple_rows)
gt_simple_df = gt_simple_df.sort_values('GroundTruth', ascending=False).reset_index(drop=True)
print(gt_simple_df.to_string(index=False))

create_ground_truth_plot(
    gt_simple_rows,
    n_facilities=len(common_permits),
    save_path=f'{figures_dir}/ground_truth_ground_truth_vs_npdes_text_vs_cwns.png',
)

# Per-facility comparison table
print("FACILITY-LEVEL COMPARISON TO GROUND TRUTH (GroundTruth)")

facility_rows = []
for permit in sorted(common_permits):
    ground_truth_row = ground_truth_common[ground_truth_common['NPDES_No'].str.strip() == permit].iloc[0]
    text_row = text_common[text_common['NPDES_No'].str.strip() == permit].iloc[0]
    cwns_row = cwns_common[cwns_common['linking_permit'].str.strip() == permit].iloc[0]

    facility_name = ground_truth_row.get('Facility_Name', permit)

    # Build ground-truth set from GroundTruth
    gt_set = {col for col in all_sheet_process_cols
              if col in ground_truth_common.columns and is_yes(ground_truth_row.get(col, ''))}

    # Build predicted set from NPDES text
    npdes_set = {col for col in all_sheet_process_cols
                 if col in text_common.columns and is_yes(text_row.get(col, ''))}

    # Build CWNS set (direct column match only)
    cwns_set = set()
    for col in all_sheet_process_cols:
        if col in cwns_common.columns:
            val = cwns_row.get(col, '')
            s = str(val).strip().lower()
            try:
                if float(val) > 0:
                    cwns_set.add(col)
                    continue
            except (ValueError, TypeError):
                pass
            if s in ('present', 'planned', 'present_and_future', 'present_and_planned') or s.startswith('present'):
                cwns_set.add(col)

    # Compute metrics vs ground truth
    npdes_tp = len(gt_set & npdes_set)
    npdes_fp = len(npdes_set - gt_set)
    npdes_fn = len(gt_set - npdes_set)
    npdes_precision = npdes_tp / (npdes_tp + npdes_fp) if (npdes_tp + npdes_fp) else 0
    npdes_recall = npdes_tp / (npdes_tp + npdes_fn) if (npdes_tp + npdes_fn) else 0
    npdes_f1 = (2 * npdes_precision * npdes_recall / (npdes_precision + npdes_recall)
                if (npdes_precision + npdes_recall) else 0)

    cwns_tp = len(gt_set & cwns_set)
    cwns_fp = len(cwns_set - gt_set)
    cwns_fn = len(gt_set - cwns_set)
    cwns_precision = cwns_tp / (cwns_tp + cwns_fp) if (cwns_tp + cwns_fp) else 0
    cwns_recall = cwns_tp / (cwns_tp + cwns_fn) if (cwns_tp + cwns_fn) else 0
    cwns_f1 = (2 * cwns_precision * cwns_recall / (cwns_precision + cwns_recall)
               if (cwns_precision + cwns_recall) else 0)

    facility_rows.append({
        'NPDES_No': permit,
        'Facility_Name': facility_name,
        'GT_Count': len(gt_set),
        'NPDES_TP': npdes_tp, 'NPDES_FP': npdes_fp, 'NPDES_FN': npdes_fn,
        'NPDES_Precision': npdes_precision, 'NPDES_Recall': npdes_recall, 'NPDES_F1': npdes_f1,
        'NPDES_Missed': '|'.join(sorted(gt_set - npdes_set)),
        'NPDES_Extra': '|'.join(sorted(npdes_set - gt_set)),
        'CWNS_TP': cwns_tp, 'CWNS_FP': cwns_fp, 'CWNS_FN': cwns_fn,
        'CWNS_Precision': cwns_precision, 'CWNS_Recall': cwns_recall, 'CWNS_F1': cwns_f1,
        'CWNS_Missed': '|'.join(sorted(gt_set - cwns_set)),
        'CWNS_Extra': '|'.join(sorted(cwns_set - gt_set)),
    })

gt_comparison_df = pd.DataFrame(facility_rows)
gt_comparison_csv = f'npdes_permits/output/{DATE_FOLDER}/ground_truth_comparison_by_facility.csv'
gt_comparison_df.to_csv(gt_comparison_csv, index=False)
print(f"Saved facility-level comparison: {os.path.basename(gt_comparison_csv)}")