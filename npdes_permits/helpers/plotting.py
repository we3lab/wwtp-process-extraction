
COLORS_GT = {
    'pfd': '#2ca02c',        # Green for Process Flow Diagrams (ground truth)
    'npdes_text': '#1482a5ff',  # Dark blue for NPDES Text (manual)
    'cwns': '#FFD700',       # Gold for CWNS
}

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