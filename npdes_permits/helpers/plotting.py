
COLORS_GT = {
    'ground_truth': '#2ca02c',         # Green for Process Flow Diagrams (ground truth)
    'npdes_text': '#1482a5ff',         # Dark blue for NPDES Text (manual)
    'llm': '#ff7f0e',                  # Orange for LLM extraction
    'cwns': '#FFD700',                 # Gold for CWNS
}

COLORS = {
    'cwns': '#FFD700',           # Gold for CWNS
    'npdes': '#1482a5ff',        # Dark blue for NPDES matched
    'npdes_total': '#ff7f0e',    # Orange for total LLM results (all CA)
}

# Hatch patterns for status values
HATCH_PATTERNS = {
    'present': '',               # Solid fill
    'future': '///',             # Diagonal lines
    'present_and_future': 'xxx'  # Cross-hatch
}