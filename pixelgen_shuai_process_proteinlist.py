import pandas as pd
import os
from structure_and_args import GraphArgs

# 1st  step - Load
args = GraphArgs()
title_sample = 'unstimulated_RCVCMP0000133_neigbours_s_proteinlist'
data_folder = args.directory_map['pixelgen_data']
data_path = f"{data_folder}/{title_sample}.csv"
edge_list_folder = args.directory_map['edge_lists']
df = pd.read_csv(data_path)

print(df)
filter_threshold_event = 0
filtered_df = df[df["n_events"] > filter_threshold_event]
# Combine and Unique-ify the Elements
unique_elements = set(filtered_df["node1"]).union(set(filtered_df["node2"]))
# Create a Mapping
element_to_int = {element: idx for idx, element in enumerate(unique_elements)}
# Transform the DataFrame
df_transformed = pd.DataFrame({
    "source": filtered_df["node1"].map(element_to_int),
    "target": filtered_df["node2"].map(element_to_int)
})
df_transformed.to_csv(f'{edge_list_folder}/shuai_protein_edgelist_{title_sample}_thresh={filter_threshold_event}.csv', index=False)