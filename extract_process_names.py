import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

with open('unitprocess_keywords.json', 'r') as f:
    data = json.load(f)

example_sentence = """The facility has a headworks with grit and FOG removal, followed by a primary clarifier. 
The secondary treatment includes four activated sludge basins and clarifiers. 
The secondary effluent then passes through the chlorine contact tank before discharge."""

# Test search process
results = {}
for category, processes in data.items():
    if isinstance(processes, dict):
        for process_name, details in processes.items():
            if isinstance(details, dict) and 'alt_names' in details:
                # Initialize unit process key to 0
                if process_name not in results:
                    results[process_name] = 0
                
                # Check each alt_name for each process
                for i, alt_name in enumerate(details['alt_names']):
                    case_sensitive = details['alt_names_case_sensitive'][i] if i < len(details['alt_names_case_sensitive']) else "N"
                    
                    if case_sensitive == "Y":
                        found = alt_name in example_sentence
                    else:
                        found = alt_name.lower() in example_sentence.lower()
                    
                    if found:
                        results[process_name] = 1
                        break  # Found one alt_name, don't need to check others

# Save in DataFrame
df = pd.DataFrame([results])
print(df.head())
