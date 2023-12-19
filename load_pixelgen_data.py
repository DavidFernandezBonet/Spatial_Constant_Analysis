import pandas as pd
import os

### Before anything, get the data from their website. It will be a .pxl file that contains an edge list with .parquet extension. Use that
### https://software.pixelgen.com/datasets
edge_list_name = 'edgelist_CD3.parquet' #edge_list_human_1k.parquet  # edgelist_CD3.parquet
df = pd.read_parquet(edge_list_name)
pd.set_option('display.max_columns', None)
df.to_csv('edge_list_pixelgen.csv')
print(df.head())

# Keep only the unique values in 'component' column
unique_components = df.drop_duplicates(subset='component')

# Select the first three unique components
first_three_unique = unique_components.head(3)

# Display the result
print(first_three_unique)

unique_component_count = df['component'].nunique()

# Display the result
print("Number of unique components/cells:", unique_component_count)


### Only graph 3 cells
# Identify the first three unique components
first_three_components = df['component'].drop_duplicates().head(3).tolist()

# Separate and write each DataFrame to CSV
for i, component in enumerate(first_three_components):
    # Filter the DataFrame for each component
    filtered_df = df[df['component'] == component]

    # Integer edge list
    # Extract unique values from 'upia' and 'upib'
    unique_upia = filtered_df['upia'].unique()
    unique_upib = filtered_df['upib'].unique()

    # Create a mapping for each unique value to an integer
    upia_mapping = {value: i for i, value in enumerate(unique_upia, start=0)}
    upib_mapping = {value: i + len(upia_mapping) for i, value in enumerate(unique_upib, start=0)}

    # Replace the 'upia' and 'upib' values in the DataFrame with their corresponding integers
    filtered_df.loc[:, 'source'] = filtered_df['upia'].map(upia_mapping)
    filtered_df.loc[:, 'target'] = filtered_df['upib'].map(upib_mapping)

    # Create new DataFrames with 'source' and 'target' based on 'upia' and 'upib'
    df_source = pd.DataFrame({'source': filtered_df['source'], 'target': filtered_df['target']})


    # Write to CSV
    filename = f'pixelgen_{os.path.splitext(edge_list_name)[0]}_cell_{i+1}_{component}.csv'
    df_source.to_csv(filename, index=False)
    print(f'Written: {filename}')
