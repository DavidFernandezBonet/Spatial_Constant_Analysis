import pandas as pd

from structure_and_args import GraphArgs


args = GraphArgs()
edge_folder = args.directory_map['edge_lists']

title1 = 'old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering_feb23.csv'
title2 = 'edge_list_nbead_7_filtering_feb23.csv'   # edge_list_distance_150_filtering_simon.csv, # edge_list_nbead_7_filtering_feb23.csv
df1 = pd.read_csv(f'{edge_folder}/{title1}')
df2 = pd.read_csv(f'{edge_folder}/{title2}')


def are_edges_of_A_in_B(dataframeA, dataframeB):
    # Convert dataframe edges to tuples and handle symmetry by sorting the node pairs
    setA = {tuple(sorted(edge)) for edge in dataframeA.to_records(index=False)}
    setB = {tuple(sorted(edge)) for edge in dataframeB.to_records(index=False)}

    # Check if setA is a subset of setB
    is_contained = setA.issubset(setB)

    return is_contained


print(are_edges_of_A_in_B(df1, df2))