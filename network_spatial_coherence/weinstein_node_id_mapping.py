from structure_and_args import GraphArgs
import pandas as pd

args = GraphArgs()
id_mapping_folder = args.directory_map['colorfolder']
reconstructed_positions_folder = args.directory_map['original_positions']

mapping_df = pd.read_csv(f'{id_mapping_folder}/node_id_mapping_from_same_beacontarget_to_different.csv')
color_df = pd.read_csv(f'{id_mapping_folder}/weinstein_colorcode_february.csv')
position_df = pd.read_csv(f'{reconstructed_positions_folder}/reconstructed_positions_weinstein_febraury.csv')


def remap_node_ids(color_df, position_df, mapping_df):
    # Separate the mappings for beacons and targets
    beacon_mapping = mapping_df[mapping_df['b_or_t'] == 0][['old_index', 'new_index']]
    target_mapping = mapping_df[mapping_df['b_or_t'] == 1][['old_index', 'new_index']]

    # Convert to dictionary for faster lookup
    beacon_dict = pd.Series(beacon_mapping.new_index.values, index=beacon_mapping.old_index).to_dict()
    target_dict = pd.Series(target_mapping.new_index.values, index=target_mapping.old_index).to_dict()

    # Apply mapping based on 'color' column
    # For beacons (color == -1)
    beacon_indices = color_df['color'] == -1
    color_df.loc[beacon_indices, 'node_ID'] = color_df.loc[beacon_indices, 'node_ID'].map(beacon_dict)

    # For targets (color in [0, 1, 2])
    target_indices = color_df['color'].isin([0, 1, 2])
    color_df.loc[target_indices, 'node_ID'] = color_df.loc[target_indices, 'node_ID'].map(target_dict)

    position_df['node_ID'] = position_df['node_ID'].astype(mapping_df['old_index'].dtype)  # Ensure correct dtype
    position_df.loc[beacon_indices, 'node_ID'] = position_df.loc[beacon_indices, 'node_ID'].map(beacon_dict)
    position_df.loc[target_indices, 'node_ID'] = position_df.loc[target_indices, 'node_ID'].map(target_dict)
    return color_df, position_df

# if beacon is set to color

new_color_df, new_position_df = remap_node_ids(color_df, position_df, mapping_df)


new_color_df.to_csv(f'{id_mapping_folder}/weinstein_colorcode_february_corrected.csv')
new_position_df.to_csv(f'{reconstructed_positions_folder}/reconstructed_positions_weinstein_febraury_corrected.csv')