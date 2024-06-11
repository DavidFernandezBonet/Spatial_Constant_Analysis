import os
import pandas as pd
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from structure_and_args import GraphArgs
import re

### Used to process slidetag subgraphs and store them in a folder. This is done to make boxplots, one box for each filtering power
def read_and_process_network(file_path):
    df = pd.read_csv(file_path)
    G = nx.from_pandas_edgelist(df, 'source', 'target')

    base_path = Path(file_path)
    folder_name = base_path.stem

    short_folder_name = re.search(r'nbead_\d+', folder_name)
    if short_folder_name:
        short_folder_name = short_folder_name.group(0)
    else:
        short_folder_name = folder_name  # Fallback to full name if pattern not found


    output_dir = base_path.parent / folder_name
    output_dir.mkdir(exist_ok=True)

    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    stats = []

    for idx, sg in enumerate(subgraphs, 1):
        subgraph_df = nx.to_pandas_edgelist(sg)
        output_file_path = output_dir / f"{folder_name}_subgraph_{idx}.csv"
        subgraph_df.to_csv(output_file_path, index=False)

        # Collect stats for the current subgraph
        stats.append({
            'Filename': short_folder_name,
            'Subgraph Index': idx,
            'Number of Nodes': sg.number_of_nodes(),
            'Number of Edges': sg.number_of_edges()
        })

    return stats


def process_multiple_slidtag_networks(filenames):
    base_path = Path(filenames[0])
    output_dir = base_path.parent
    all_stats = []

    for filename in filenames:
        file_stats = read_and_process_network(filename)
        all_stats.extend(file_stats)

    # Convert stats to DataFrame
    stats_df = pd.DataFrame(all_stats)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=stats_df, x='Filename', y='Number of Nodes', inner='point', scale='width')
    plt.title('Distribution of Nodes Across Subgraphs')
    plt.xlabel('File')
    plt.ylabel('Number of Nodes')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.savefig(f'{output_dir}/slidetag_node_distribution.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=stats_df, x='Filename', y='Number of Edges', inner='point', scale='width')
    plt.title('Distribution of Edges Across Subgraphs')
    plt.xlabel('File')
    plt.ylabel('Number of Edges')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.savefig(f'{output_dir}/slidetag_edge_distribution.png')

    plt.show()


args = GraphArgs()
data_folder = args.directory_map['slidetag_data']
filenames = [f'{data_folder}/edge_list_nbead_{i}_filtering.csv' for i in range(1, 10)]
process_multiple_slidtag_networks(filenames)