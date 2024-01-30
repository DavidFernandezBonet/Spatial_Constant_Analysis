import pandas as pd
import os
from structure_and_args import GraphArgs

# This 0th step is now already handled by "quality_control_pixelgen_data" script

# ### Before anything, get the data from their website. It will be a .pxl file that contains an edge list with .parquet extension. Use that
# ### https://software.pixelgen.com/datasets
# edge_list_name = 'edgelist_CD3.parquet' #edge_list_human_1k.parquet  # edgelist_CD3.parquet
# df = pd.read_parquet(edge_list_name)
# pd.set_option('display.max_columns', None)
# df.to_csv('edge_list_pixelgen.csv')
# print(df.head())

# 1st  step - Load
args = GraphArgs()
title_sample = 'shuai_RCVCMP0000073_cd3'
data_folder = args.directory_map['pixelgen_data']
data_path = f"{data_folder}/filtered_edge_list_{title_sample}.csv"
df = pd.read_csv(data_path)

# 2nd step - Get 3 components (cells)
unique_component_count = df['component'].nunique()
print("Number of unique components/cells:", unique_component_count)
# Identify the first three unique components
first_three_components = df['component'].drop_duplicates().head(3).tolist()

# 3rd step - Write them into an edge list
# Separate and write each DataFrame to CSV
for i, component in enumerate(first_three_components):
    # Filter the DataFrame for each component
    filtered_df = df[df['component'] == component]

    # Integer edge list
    # Extract unique values from 'upia' and 'upib'
    unique_upia = filtered_df['upia'].unique()
    unique_upib = filtered_df['upib'].unique()

    upia_mapping = {value: i for i, value in enumerate(unique_upia, start=0)}
    upib_mapping = {value: i + len(upia_mapping) for i, value in enumerate(unique_upib, start=0)}

    # Replace the 'upia' and 'upib' values in the DataFrame with their corresponding integers
    filtered_df.loc[:, 'source'] = filtered_df['upia'].map(upia_mapping)
    filtered_df.loc[:, 'target'] = filtered_df['upib'].map(upib_mapping)

    # Create new DataFrames with 'source' and 'target' based on 'upia' and 'upib'
    df_source = pd.DataFrame({'source': filtered_df['source'], 'target': filtered_df['target']})


    # Write to CSV
    filename = f'pixelgen_processed_edgelist_{os.path.splitext(title_sample)[0]}_cell_{i+1}_{component}.csv'
    edge_list_folder = args.directory_map['edge_lists']
    df_source.to_csv(f'{edge_list_folder}/{filename}', index=False)
    print(f'Written: {filename}')
