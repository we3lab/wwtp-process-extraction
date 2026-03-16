import json
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
    """Return boolean mask for CWNS presence values (numeric or string)."""
    s = series.astype(str).str.lower()
    numeric = pd.to_numeric(series, errors='coerce')
    return (numeric > 0) | s.isin({'present', 'planned', 'present_and_future', 'present_and_planned'}) | s.str.startswith('present')


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
        list = [
            Patch(facecolor=COLORS['npdes'], label='NPDES (Present)'),
            Patch(facecolor=COLORS['npdes'], hatch=HATCH_PATTERNS['future'], label='NPDES (Future)'),
            Patch(facecolor=COLORS['npdes'], hatch=HATCH_PATTERNS['present_and_future'], label='NPDES (Present & Future)'),
            ]
        if include_cwns:
            list.extend([Patch(facecolor=COLORS['cwns'], label='CWNS')])
        ax.legend(handles=list, loc='upper right', fontsize=11)

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


def count_cwns_facilities(processes, df, unitprocess_keywords):
    """Count unique CWNS facilities (by CWNS_ID) with any of the given processes."""
    facilities = set()
    for process in processes:
        details = find_process_details(process, unitprocess_keywords)
        if details:
            for cwns_name in get_cwns_unit_process_names(process, details):
                if cwns_name in df.columns:
                    mask = build_cwns_presence_mask(df[cwns_name])
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
        npdes_parent_count = count_npdes_facilities(child_processes, unit_process_results)
        cwns_parent_count = count_cwns_facilities(child_processes, matching_cwns_data, unitprocess_keywords)
        
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
unit_process_results['FACILITY_KEY'] = (
    unit_process_results['PERMIT_NUMBER'].fillna('') + '||' +
    unit_process_results['FACILITY_NAME'].fillna('')
)

print(f"\n{'='*80}")
print("DATA LOADING SUMMARY")
print(f"{'='*80}")
print(f"NPDES data: Loaded {len(unit_process_results)} rows, "
      f"{unit_process_results['FACILITY_KEY'].nunique()} unique facilities")

# Get unique NPDES permit numbers
npdes_permit_numbers = set(unit_process_results['PERMIT_NUMBER'].unique())
print(f"Unique NPDES permit numbers: {len(npdes_permit_numbers)}")

# Load and consolidate CWNS data with facility names and clean NPDES permits
cwns_data = pd.read_csv(f'npdes_permits/output/unit_processes_by_facility.csv')
print(f"\nCWNS data: Total rows: {len(cwns_data)}")

ca_cwns_data = prepare_cwns_ca(
    cwns_data,
    'npdes_permits/data/cwns/2022/2022_FACILITIES.csv',
    'npdes_permits/data/cwns/2022/FACILITY_PERMIT.csv',
    'npdes_permits/data/cwns/cwns_facilities_match_manual.csv',
)
print(f"CWNS CA facilities (consolidated): {len(ca_cwns_data)}")

# Match CWNS to NPDES permits
ca_cwns_data = match_cwns_to_npdes(ca_cwns_data, npdes_permit_numbers)
matching_cwns_data = ca_cwns_data[ca_cwns_data['matched']].copy()

# Filter NPDES to only facilities with matching CWNS data
matching_permit_numbers = set(matching_cwns_data['linking_permit'].dropna())
unit_process_results_matched = unit_process_results[
    unit_process_results['PERMIT_NUMBER'].isin(matching_permit_numbers)
].copy()

print(f"\n{'='*80}")
print("FACILITY MATCHING")
print(f"{'='*80}")
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
print(f"\nSaved {len(unmatched_cwns)} unmatched CWNS facilities to unmatched_cwns_facilities.csv")

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
print(f"Total California CWNS facilities (consolidated): {len(ca_cwns_data)}")
print(f"Total NPDES rows analyzed: {len(unit_process_results_matched)}")
print(f"CWNS facilities matched: {len(matching_cwns_data)}")
print(f"NPDES permits matched: {len(matching_permit_numbers)}")
print(f"Match rate: {len(matching_permit_numbers) / len(npdes_permit_numbers) * 100:.1f}% of NPDES permits")
print(f"Coverage: {len(matching_cwns_data) / len(ca_cwns_data) * 100:.1f}% of CA CWNS facilities")

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


# GROUND TRUTH COMPARISON: PFD vs NPDES Text vs CWNS

GOOGLE_SHEET_ID = '18U4IlfAiNH1UNdUYH5fF35fX99ll9SciKYRUuHUdT8w'


def count_yes(series):
    """Count YES/PLANNED values in a sheet column (case-insensitive)."""
    s = series.fillna('').astype(str).str.strip().str.upper()
    return (s.isin(['YES', 'PLANNED'])).sum()


def create_ground_truth_plot(process_counts, save_path):
    """
    Create a grouped bar chart comparing PFD, NPDES Text, and CWNS counts.

    process_counts: list of dicts with keys 'process', 'pfd', 'npdes_text', 'cwns'
    """
    fontsize = 12
    df = pd.DataFrame(process_counts)
    # Keep only processes where at least one source has a count > 0
    df = df[(df['pfd'] > 0) | (df['npdes_text'] > 0) | (df['cwns'] > 0)]
    if df.empty:
        print("No populated processes to plot.")
        return
    # Sort by PFD count descending
    df = df.sort_values('pfd', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(20, 7))
    x = range(len(df))
    width = 0.25

    ax.bar([i - width for i in x], df['pfd'], width,
           color=COLORS_GT['pfd'], edgecolor='black', linewidth=0.5,
           label='Process Flow Diagrams (Ground Truth)')
    ax.bar(list(x), df['npdes_text'], width,
           color=COLORS_GT['npdes_text'], edgecolor='black', linewidth=0.5,
           label='NPDES Text (Manual)')
    ax.bar([i + width for i in x], df['cwns'], width,
           color=COLORS_GT['cwns'], edgecolor='black', linewidth=0.5,
           label='CWNS')

    ax.set_xticks(list(x))
    ax.set_xticklabels(df['process'], rotation=45, ha='right', fontsize=fontsize)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylabel('WWTP Count', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(loc='upper right', fontsize=11)

    plt.subplots_adjust(bottom=0.30)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved {os.path.basename(save_path)}")
    plt.close(fig)


# Load the two Google Sheet tabs
pfd_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - Ground Truth')
npdes_text_df = load_google_sheet_csv(GOOGLE_SHEET_ID, 'Train - From NPDES Text')

print(f"PFD sheet: {len(pfd_df)} facilities")
print(f"NPDES Text sheet: {len(npdes_text_df)} facilities")

# Determine the process columns (everything after the first 4 metadata cols)
meta_cols = ['Agency', 'Facility_Name', 'NPDES_No', 'PDF_File', "Ground Truth Sources"]
pfd_process_cols = [c for c in pfd_df.columns if c not in meta_cols]
npdes_text_process_cols = [c for c in npdes_text_df.columns if c not in meta_cols]

# Use PFD process columns as the canonical list (union with NPDES text if needed)
all_sheet_process_cols = list(dict.fromkeys(pfd_process_cols + npdes_text_process_cols))

# Filter to only facilities present in all three datasets (PFD, NPDES text, and CWNS)
pfd_permits = set(pfd_df['NPDES_No'].dropna().str.strip())
text_permits = set(npdes_text_df['NPDES_No'].dropna().str.strip())
cwns_permits_gt = set(ca_cwns_data['linking_permit'].dropna().str.strip())
common_permits = pfd_permits & text_permits & cwns_permits_gt

print(f"Facilities in all 3 sources: {len(common_permits)}")

# Filter each source to common facilities
pfd_common = pfd_df[pfd_df['NPDES_No'].str.strip().isin(common_permits)].copy()
text_common = npdes_text_df[npdes_text_df['NPDES_No'].str.strip().isin(common_permits)].copy()
cwns_common = ca_cwns_data[ca_cwns_data['linking_permit'].str.strip().isin(common_permits)].copy()

# Deduplicate (some sheets may have duplicate NPDES_No rows)
pfd_common = pfd_common.drop_duplicates(subset='NPDES_No', keep='first')
text_common = text_common.drop_duplicates(subset='NPDES_No', keep='first')
cwns_common = cwns_common.drop_duplicates(subset='CWNS_ID', keep='first')

print(f"After dedup - PFD: {len(pfd_common)}, NPDES Text: {len(text_common)}, CWNS: {len(cwns_common)}")

# Build counts for each process
process_counts = []
for col in all_sheet_process_cols:
    pfd_count = count_yes(pfd_common[col]) if col in pfd_common.columns else 0
    text_count = count_yes(text_common[col]) if col in text_common.columns else 0

    # Map sheet column to CWNS column name
    # cwns_col = SHEET_TO_CWNS_COL.get(col, col)
    # if cwns_col in cwns_common.columns:
    print(col)
    # print(cwns_common.keys())
    cwns_count = int(build_cwns_presence_mask(cwns_common[col]).sum())
    process_counts.append({
        'process': col,
        'pfd': pfd_count,
        'npdes_text': text_count,
        'cwns': cwns_count,
    })

# Create the comparison bar chart
create_ground_truth_plot(
    process_counts,
    save_path=f'{figures_dir}/ground_truth_pfd_vs_npdes_text_vs_cwns.png',
)


def is_yes(val):
    """Check if a cell value means the process is present."""
    return str(val).strip().upper() in ('YES', 'PLANNED')

# GROUND TRUTH COMPARISON: +/- % vs PFD grouped by JSON category

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
cat_pfd_facilities = defaultdict(set)
cat_npdes_facilities = defaultdict(set)
cat_cwns_facilities = defaultdict(set)

for col in all_sheet_process_cols:
    category = leaf_to_category.get(col, col)
    # PFD
    if col in pfd_common.columns:
        for _, row in pfd_common.iterrows():
            if is_yes(row.get(col, '')):
                cat_pfd_facilities[category].add(row['NPDES_No'])
    # NPDES text
    if col in text_common.columns:
        for _, row in text_common.iterrows():
            if is_yes(row.get(col, '')):
                cat_npdes_facilities[category].add(row['NPDES_No'])
    # CWNS
    if col in cwns_common.columns:
        mask = build_cwns_presence_mask(cwns_common[col])
        for cwns_id in cwns_common.loc[mask, 'CWNS_ID']:
            cat_cwns_facilities[category].add(cwns_id)

# Build summary rows
all_cats = sorted(set(list(cat_pfd_facilities.keys()) + list(cat_npdes_facilities.keys()) + list(cat_cwns_facilities.keys())))
gt_simple_rows = []
for cat in all_cats:
    pfd = len(cat_pfd_facilities[cat])
    npdes = len(cat_npdes_facilities[cat])
    cwns = len(cat_cwns_facilities[cat])
    if pfd == 0 and npdes == 0 and cwns == 0:
        continue
    if pfd > 0:
        npdes_str = f"{(npdes - pfd) / pfd * 100:+.0f}%"
        cwns_str = f"{(cwns - pfd) / pfd * 100:+.0f}%"
    else:
        npdes_str = f"+{npdes}" if npdes > 0 else "0"
        cwns_str = f"+{cwns}" if cwns > 0 else "0"
    gt_simple_rows.append({
        'Process_Category': cat,
        'PFD_Ground_Truth': pfd,
        'NPDES_Manual': npdes,
        'NPDES_vs_GT': npdes_str,
        'CWNS': cwns,
        'CWNS_vs_GT': cwns_str,
    })

gt_simple_df = pd.DataFrame(gt_simple_rows)
gt_simple_df = gt_simple_df.sort_values('PFD_Ground_Truth', ascending=False).reset_index(drop=True)
gt_simple_csv = f'npdes_permits/output/{DATE_FOLDER}/ground_truth_summary.csv'
gt_simple_df.to_csv(gt_simple_csv, index=False)
print(f"\nSaved simplified ground truth comparison: {os.path.basename(gt_simple_csv)}")
print(gt_simple_df.to_string(index=False))


# Per-facility comparison table
print("FACILITY-LEVEL COMPARISON TO GROUND TRUTH (PFD)")

facility_rows = []
for permit in sorted(common_permits):
    pfd_row = pfd_common[pfd_common['NPDES_No'].str.strip() == permit].iloc[0]
    text_row = text_common[text_common['NPDES_No'].str.strip() == permit].iloc[0]
    cwns_row = cwns_common[cwns_common['linking_permit'].str.strip() == permit].iloc[0]

    facility_name = pfd_row.get('Facility_Name', permit)

    # Build ground-truth set from PFD
    gt_set = {col for col in all_sheet_process_cols
              if col in pfd_common.columns and is_yes(pfd_row.get(col, ''))}

    # Build predicted set from NPDES text
    npdes_set = {col for col in all_sheet_process_cols
                 if col in text_common.columns and is_yes(text_row.get(col, ''))}

    # Build CWNS set (map sheet column names to CWNS column names)
    cwns_set = set()
    for col in all_sheet_process_cols:
        # cwns_col = SHEET_TO_CWNS_COL.get(col, col)
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