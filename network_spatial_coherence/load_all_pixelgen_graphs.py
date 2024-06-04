import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pixelator import read, simple_aggregate
import seaborn as sns

from structure_and_args import GraphArgs


def read_mpx_data(file_path):
    data = read(file_path)
    return data


def filter_components(pg_data, edge_lower_threshold, edge_upper_threshold, tau_type):
    # Identify components where edge counts are within the specified thresholds and of the specific tau_type
    components_to_keep = pg_data.adata[
        (pg_data.adata.obs["edges"] >= edge_lower_threshold)
        & (pg_data.adata.obs["edges"] <= edge_upper_threshold)
        & (pg_data.adata.obs["tau_type"] == tau_type)
        ].obs.index

    # Return the filtered pg_data with components that meet the criteria
    return pg_data.filter(components=components_to_keep)



def plot_component_edges(pg_data, edge_lower_threshold, edge_upper_threshold, tau_type):
    # Filter the data for the specified tau_type
    filtered_data = pg_data.adata.obs[pg_data.adata.obs["tau_type"] == tau_type]

    # Sort the data by 'edges' in descending order to rank components
    filtered_data = filtered_data.sort_values(by='edges', ascending=False).reset_index(drop=True)

    # Creating the log-log plot
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data.index + 1, filtered_data['edges'], marker='o', linestyle='-',
             color='blue')  # +1 to adjust rank starting from 1

    # Plotting the edge threshold as a horizontal line
    plt.axhline(y=edge_upper_threshold, color='red', linestyle='--', linewidth=2)
    plt.text(1, edge_upper_threshold, f' Edge Threshold: {edge_upper_threshold}', verticalalignment='bottom', color='red')
    plt.axhline(y=edge_lower_threshold, color='red', linestyle='--', linewidth=2)
    plt.text(1, edge_lower_threshold, f' Edge Threshold: {edge_lower_threshold}', verticalalignment='bottom', color='red')

    # Setting log scale for both axes
    plt.yscale('log')
    plt.xscale('log')

    # Adding titles and labels
    plt.title('Log-Log Plot of Number of Edges per Component')
    plt.xlabel('Component Rank (log scale)')
    plt.ylabel('Number of Edges (log scale)')
    plt.show()
def write_subgraphs_edge_lists(pg_data, base_directory, dataset_name, edge_upper_threshold, edge_lower_threshold,
                               lower_limit=500, upper_limit=3000):
    dataset_dir = Path(base_directory) / f'{dataset_name}_edge_t={edge_lower_threshold}-{edge_upper_threshold}'
    dataset_dir.mkdir(exist_ok=True)
    node_sizes = []
    edge_sizes = []
    for i, (component, data) in enumerate(pg_data.edgelist.groupby('component')):
        print(i, component)
        unique_edges = data[['upia', 'upib']].drop_duplicates()
        unique_nodes = pd.concat([unique_edges['upia'], unique_edges['upib']]).unique()
        num_unique_nodes = len(unique_nodes)
        num_edges = len(unique_edges)

        node_sizes.append(num_unique_nodes)
        edge_sizes.append(num_edges)

        if not (lower_limit <= num_unique_nodes <= upper_limit):
            print(f"Skipping component {component} with {num_unique_nodes} nodes. Does not meet node count limits [{lower_limit}, {upper_limit}].")
            continue

        node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
        unique_edges['upia'] = unique_edges['upia'].map(node_mapping).astype(int)
        unique_edges['upib'] = unique_edges['upib'].map(node_mapping).astype(int)
        unique_edges.rename(columns={'upia': 'source', 'upib': 'target'}, inplace=True)
        edge_list_path = dataset_dir / f"{dataset_name}_component_{component}_edgelist.csv"
        unique_edges.to_csv(edge_list_path, index=False, header=True)
        print(f"Edge list for component {component} written to {edge_list_path}")

    # Plot the node and edge size distribution
    plt.figure(figsize=(12, 6))

    # Creating a subplot for node sizes
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.violinplot(data=node_sizes, inner="point", linewidth=1.5)
    plt.title('Node Sizes Distribution')
    plt.ylabel('Node Count')
    plt.xlabel('Nodes')

    # Creating a subplot for edge sizes
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.violinplot(data=edge_sizes, inner="point", linewidth=1.5)
    plt.title('Edge Sizes Distribution')
    plt.ylabel('Edge Count')
    plt.xlabel('Edges')

    # Saving the figure
    plt.savefig(f"{base_directory}/{dataset_name}_size_distribution_violins.png")
    plt.close()


if __name__ == "__main__":
    title_file = "Sample01_human_pbmcs_unstimulated.dataset.pxl"  # Sample03_Raji_control.dataset.pxl  # "Uropod_control.dataset.pxl"  # Sample01_human_pbmcs_unstimulated.dataset.pxl
    args = GraphArgs()
    data_directory = args.directory_map['pixelgen_data']
    data_file = f"{data_directory}/{title_file}"
    dataset_name = Path(Path(title_file).stem).stem
    edge_upper_threshold = 8000
    edge_lower_threshold = 2000

    pg_data = read_mpx_data(data_file)
    plot_component_edges(pg_data=pg_data, edge_lower_threshold=edge_lower_threshold, edge_upper_threshold=edge_upper_threshold,
                         tau_type="normal")
    pg_data_filtered = filter_components(pg_data=pg_data,  edge_lower_threshold=edge_lower_threshold,
                                         edge_upper_threshold=edge_upper_threshold,  tau_type="normal")
    # plot_histogram_with_threshold(pg_data=pg_data_filtered, edge_threshold=edge_threshold, tau_type="normal")

    write_subgraphs_edge_lists(pg_data_filtered, data_directory, dataset_name, edge_lower_threshold=edge_lower_threshold, edge_upper_threshold=edge_upper_threshold)  # TODO: make new directory for each datasset (with the dataset name)
