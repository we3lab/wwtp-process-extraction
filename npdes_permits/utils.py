def get_all_keys(processes_dict):
    """Extract all leaf processes (those with alt_names) from the JSON structure"""
    processes = []
    for process_name, details in processes_dict.items():
        if isinstance(details, dict):
            if 'alt_names' in details and details['alt_names']:
                processes.append(process_name)
            else:  # Parent category, recursively get children
                processes.extend(get_all_keys(details))
    return processes

def get_process_names_for_category(category_name, category_keywords):
    """Return process names for a category, including leaf categories with alt_names."""
    if isinstance(category_keywords, dict) and 'alt_names' in category_keywords:
        return [category_name]
    return get_all_keys(category_keywords)


def get_parent_child_mapping(processes_dict):
    """Create mapping of parent processes to their child processes"""
    mapping = {}
    for parent_name, parent_details in processes_dict.items():
        if isinstance(parent_details, dict) and 'alt_names' not in parent_details:
            # This is a parent category
            children = [child_name for child_name, child_details in parent_details.items()
                       if isinstance(child_details, dict) and 'alt_names' in child_details]
            if children:
                mapping[parent_name] = children
    return mapping


def get_cwns_unit_process_names(process_name, process_details):
    """Get CWNS unit process names for a given process from keywords"""
    if isinstance(process_details, dict) and 'cwns_processes' in process_details:
        names = process_details['cwns_processes']
        return names if isinstance(names, list) else [names]
    return []


def find_process_details(process_name, unitprocess_keywords):
    """Find process details in keywords hierarchy, searching both top-level and nested"""
    for category, cat_keywords in unitprocess_keywords.items():
        if not isinstance(cat_keywords, dict):
            continue
        # Check top level
        if process_name in cat_keywords:
            return cat_keywords[process_name]
        # Check in parent categories
        for parent_name, parent_details in cat_keywords.items():
            if isinstance(parent_details, dict) and process_name in parent_details:
                return parent_details[process_name]
    return None


def get_werf_codes_for_cwns_process(cwns_process_name):
    """
    Get WERF codes for a CWNS unit process name (for mapping back to El Abbadi codes).
    This requires loading the CWNS data which has FINAL_UNIT_PROCESS_NAME -> WERF_CODE mappings.
    Use this for future comparisons with El Abbadi treatment train assignments.
    
    Returns a list of WERF codes that correspond to this CWNS process name.
    """
    try:
        import pandas as pd
        import os
        # Load the mapping from CWNS data if available
        el_abbadi_dir = os.path.join(os.path.dirname(__file__), 'data', 'el_abbadi', 'input')
        werf_codes_df = pd.read_csv(os.path.join(el_abbadi_dir, 'UNIT_PROCESS_EI_CODES_WERF_modified.csv'))
        matching = werf_codes_df[werf_codes_df['FINAL_UNIT_PROCESS_NAME'] == cwns_process_name]
        return matching['WERF_CODE'].unique().tolist() if not matching.empty else []
    except Exception:
        # If loading fails, return empty list
        return []