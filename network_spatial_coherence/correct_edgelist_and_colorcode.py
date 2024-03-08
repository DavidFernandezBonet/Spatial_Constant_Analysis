import pandas as pd
from structure_and_args import GraphArgs

# This is code to solve the fact that UMI A and UMI B have sometimes the same names (Shuai...)
args = GraphArgs()
edge_list_folder = args.directory_map["edge_lists"]
color_folder = args.directory_map["colorfolder"]

df1 = pd.read_csv(f"{edge_list_folder}/weinstein_data_january.csv")
df2 = pd.read_csv(f"{color_folder}/weinstein_color.csv")

# Step 1: Renaming elements in DataFrame 1
unique_sources = df1['source'].unique()
unique_targets = df1['target'].unique()

# Generating mappings
source_mapping = {val: idx for idx, val in enumerate(unique_sources)}
last_source_id = max(source_mapping.values()) + 1
target_mapping = {val: idx + last_source_id for idx, val in enumerate(unique_targets)}

# Applying mappings
df1['source'] = df1['source'].map(source_mapping)
df1['target'] = df1['target'].map(target_mapping)


df2['node_ID'] = df2['node_ID'].map(target_mapping)

df1.to_csv("weinstein_data_january_corrected.csv", index=False)
df2.to_csv("weinstein_color_corrected.csv", index=False)
