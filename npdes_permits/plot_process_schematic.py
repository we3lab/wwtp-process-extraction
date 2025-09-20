import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

with open('data/unitprocess_keywords.json', 'r') as f:
    data = json.load(f)
    # Create schematic diagram
G = nx.DiGraph()

edges = [
    ('primary', 'secondary'),
    ('secondary', 'tertiary'),
    ('tertiary', 'disinfection'),
    ('disinfection', 'advanced'),
    ('primary', 'solids'),
    ('secondary', 'solids'),
    ('solids', 'primary'),  # recycle back
    ('solids', 'cogeneration'),
]

# Add nodes and edges
for category in data.keys():
    G.add_node(category)
for edge in edges:
    G.add_edge(edge[0], edge[1])

plt.figure(figsize=(14, 10))

# primary -> secondary -> tertiary in one row
pos = {
    # top row
    'primary': (0, 0.5),
    'secondary': (0.6, 0.5), 
    'tertiary': (1.2, 0.5),
    'disinfection': (1.8, 0.5),
    # lower rows
    'advanced': (1.8, 0),
    'solids': (0.6, -0.6),
    'cogeneration': (1.2, -0.6)
}

# Draw nodes as rectangles with labels
for node in G.nodes():
    x, y = pos[node]
    rect = plt.Rectangle((x-0.2, y-0.15), 0.4, 0.3,
                        facecolor='lightblue', edgecolor='black', linewidth=2)
    plt.gca().add_patch(rect)
    plt.text(x, y+0.08, node.upper(), ha='center', va='center', 
             fontsize=12, fontweight='bold')  # category name
    
    # Add process examples
    if node in data:
        unit_processes = []
        for process_name, details in data[node].items():
            if isinstance(details, dict) and 'alt_names' in details:
                unit_processes.append(process_name)
                if len(unit_processes) >= 5:
                    break
        if unit_processes:
            process_text = "\n".join(unit_processes) + "\n..."
        else:
            process_text = "(no unit processes)"
        plt.text(x, y-0.02, process_text, ha='center', va='center', fontsize=9, fontweight='normal')
    else:
        plt.text(x, y-0.02, "(no processes)", ha='center', va='center', fontsize=9, fontweight='normal')

# Draw edges with arrows
nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=30,
                        arrowstyle='->', width=3,
                        node_size=0, min_source_margin=50, min_target_margin=50)
# Save plot
plt.axis('off')
plt.tight_layout()
plt.savefig('npdes_permits/output/process_schematic.png')