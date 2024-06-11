import pandas as pd
import os
from structure_and_args import GraphArgs


### Old code for loading slidetag data (basically outputting graph in my format with source and target
args = GraphArgs()
title_sample = 'edge_list_filtered_by_bed_n_connections_thresholds_2-16.csv'    ## SRR11, SRR07, edge_list_abundant_beads_cut_beadsum_thresholds_8_256_SRR11.csv
data_folder = args.directory_map['slidetag_data']
data_path = f"{data_folder}/{title_sample}"
df = pd.read_csv(data_path)


filtered_df = df
# Integer edge list
# Extract unique values from 'upia' and 'upib'
unique_upia = filtered_df['cell_bc_10x'].unique()
unique_upib = filtered_df['bead_bc'].unique()

upia_mapping = {value: i for i, value in enumerate(unique_upia, start=0)}
upib_mapping = {value: i + len(upia_mapping) for i, value in enumerate(unique_upib, start=0)}

# Replace the 'upia' and 'upib' values in the DataFrame with their corresponding integers
filtered_df.loc[:, 'source'] = filtered_df['cell_bc_10x'].map(upia_mapping)
filtered_df.loc[:, 'target'] = filtered_df['bead_bc'].map(upib_mapping)


# Create new DataFrames with 'source' and 'target' based on 'upia' and 'upib'
df_source = pd.DataFrame({'source': filtered_df['source'], 'target': filtered_df['target'], 'weight': filtered_df['nUMI']})


# Write to CSV
filename = f'slidetag_processed_edgelist_{os.path.splitext(title_sample)[0]}.csv'
edge_list_folder = args.directory_map['edge_lists']
df_source.to_csv(f'{edge_list_folder}/{filename}', index=False)
print(f'Written: {filename}')


# Store the mapping

# Combine the mappings
combined_mapping = {**upia_mapping, **upib_mapping}

# Convert the combined mapping to a DataFrame
mapping_df = pd.DataFrame(list(combined_mapping.items()), columns=['OG SEQUENCE', 'MAPPED INT'])

# Export the DataFrame to a CSV file
mapping_df.to_csv(f'{data_folder}/{title_sample[:-4]}_mapping.csv', index=False)
