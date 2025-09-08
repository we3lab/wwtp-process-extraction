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

def get_el_abbadi_code_mapping(process_name, unitprocess_keywords):
    """Get EL_ABBADI_2024_CODE for a process name"""
    for category, processes in unitprocess_keywords.items():
        if isinstance(processes, dict):
            if process_name in processes and 'EL_ABBADI_2024_CODE' in processes[process_name]:
                return processes[process_name]['EL_ABBADI_2024_CODE']
            for parent_name, parent_details in processes.items():
                if isinstance(parent_details, dict) and process_name in parent_details:
                    if 'EL_ABBADI_2024_CODE' in parent_details[process_name]:
                        return parent_details[process_name]['EL_ABBADI_2024_CODE']
    return None

def is_parent_category(label, unitprocess_keywords):
    """Check if a label represents a parent category"""
    if "(Total)" in label:
        return True
    # Check if this is a standalone parent category in secondary
    for parent_name, parent_details in unitprocess_keywords['secondary'].items():
        if (label == parent_name and 
            any(isinstance(v, dict) and 'alt_names' in v 
                for v in parent_details.values())):
            return True
    return False