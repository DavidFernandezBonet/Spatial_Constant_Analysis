import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import random
from plots import plot_original_or_reconstructed_image

import create_proximity_graph
from structure_and_args import GraphArgs
from utils import load_graph
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd


np.random.seed(42)
random.seed(42)

def compute_and_color_nodes(pos_df, sparse_graph):
    # Convert sparse matrix to NetworkX graph
    G = nx.from_scipy_sparse_array(sparse_graph)

    # Compute the shortest path distances from all nodes to all others
    # This step can be computationally expensive for very large graphs
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    # Identify the most central node based on closeness centrality
    centrality = nx.closeness_centrality(G)
    most_central_node = max(centrality, key=centrality.get)

    # Get distances from the most central node to all others
    distances_from_central = shortest_paths[most_central_node]

    # Create a colormap based on distances
    max_distance = max(distances_from_central.values())
    node_colors = [distances_from_central[node] / max_distance for node in G.nodes()]

    # Plotting
    pos = {node: (pos_df.loc[node, 'x'], pos_df.loc[node, 'y']) for node in G.nodes()}
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=20, cmap='viridis', with_labels=False)
    plt.title('Nodes colored by distance from the most central node')
    # plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=max_distance)),
    #              label='Distance to most central node')
    plt.show()
    return G


def plot_shortest_path_distance_matrix_heatmap(G):
    """
    Computes the shortest path distance matrix for a graph G and plots it as a heatmap.

    Parameters:
    - G: A NetworkX graph object.
    """
    # Ensure the graph is connected to compute a meaningful distance matrix
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected to compute a full distance matrix.")

    # Compute the shortest path length distance matrix
    length = dict(nx.all_pairs_shortest_path_length(G))
    n = len(G.nodes())
    distance_matrix = np.zeros((n, n))

    for i, node_i in enumerate(G.nodes()):
        for j, node_j in enumerate(G.nodes()):
            distance_matrix[i, j] = length[node_i][node_j]

    # Plot the distance matrix as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Shortest path distance')
    plt.title('Shortest Path Distance Matrix Heatmap')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()


# def create_and_visualize_network():
#     G = nx.Graph()
#
#     # Manually add edges based on the pattern provided
#     # Central node connections
#     # G.add_edges_from([(1, 2), (1, 3), (1, 4)])
#     # # Connections for Node 2, 3, ...
#     # G.add_edges_from([(2, 7), (2, 5), (2, 8), (3, 7), (3, 6), (3, 9)])
#     # # Assuming a similar pattern for the remaining nodes, adjusting connections to avoid edge crossings
#     # G.add_edges_from([(4, 11), (4,5), (4,6), (5,12), (5,2), (5,16), (5,15), (6,10), (6, 14), (6,3), (7,9), (7,8),
#     #                   (15,12), (15, 8), (12, 16), (11,16), (11, 10), (10,14), (14,9), (9, 13), (13, 8)])
#
#     G.add_edges_from([(1, 2), (1, 3), (1, 4), (1,5),
#                       (3,6), (3,2), (6,4) , (4,5), (5,7), (7,2), (3,4), (2,5) ])
#     # Visualize the graph
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(G, seed=42)  # For a more organized layout
#     nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='k')
#     plt.title("Special Graph Visualization")
#     # plt.show()
#     return G
#
#
# def plot_distance_matrix_heatmap(G):
#     # Ensure the graph is connected to compute a meaningful distance matrix
#     if not nx.is_connected(G):
#         raise ValueError("Graph must be connected to compute a full distance matrix.")
#
#     # Get the number of nodes
#     n = len(G.nodes())
#
#     # Initialize an empty distance matrix
#     distance_matrix = np.zeros((n, n))
#
#     # Compute the shortest path lengths between all pairs of nodes
#     all_pairs_shortest_path_length = dict(nx.all_pairs_shortest_path_length(G))
#
#     for i, node_i in enumerate(G.nodes()):
#         for j, node_j in enumerate(G.nodes()):
#             distance_matrix[i, j] = all_pairs_shortest_path_length[node_i][node_j]
#
#     # Plot the distance matrix as a heatmap
#     plt.figure(figsize=(8, 6))
#     plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.title('Distance Matrix Heatmap')
#     plt.xlabel('Node Index')
#     plt.ylabel('Node Index')
#     plt.show()
#
#
def sort_distance_matrix(G):
    """
    Sorts the distance matrix of a graph based on hierarchical clustering.

    Parameters:
    - G: A networkx graph.

    Returns:
    - A sorted distance matrix and the order of nodes.
    """
    # Check if the graph is connected
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected.")

    # Compute the shortest path length distance matrix
    path_length = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(G.nodes())
    dist_matrix = np.zeros((len(nodes), len(nodes)))

    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            dist_matrix[i, j] = path_length[node_i][node_j]

    # Perform hierarchical clustering
    condensed_dist_matrix = squareform(dist_matrix, checks=False)
    linkage_matrix = hierarchy.linkage(condensed_dist_matrix, method='average')

    # Get the order of nodes based on the dendrogram
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']

    # Sort the distance matrix
    sorted_dist_matrix = dist_matrix[order, :][:, order]

    return sorted_dist_matrix, order
#
def plot_sorted_distance_matrix_heatmap(sorted_dist_matrix):
    """
    Plots a heatmap of the sorted distance matrix.

    Parameters:
    - sorted_dist_matrix: The sorted distance matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(sorted_dist_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Sorted Distance Matrix Heatmap')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()


def create_bfs_subgraphs_and_highlight(G, pos_df, sizes=[500, 1000]):
    """
    Performs BFS to create subgraphs of specified sizes and highlights them on plots.

    Parameters:
    - G: A NetworkX graph.
    - pos: A dictionary mapping node indices to positions (x, y).
    - sizes: A list of sizes for the subgraphs to be created.
    """
    # Identify the most central node for the starting point of BFS
    centrality = nx.closeness_centrality(G)
    start_node = max(centrality, key=centrality.get)

    # Perform BFS and get nodes in the order they are visited
    bfs_nodes = list(nx.bfs_edges(G, start_node))
    flat_list = [item for sublist in bfs_nodes for item in sublist]
    unique_nodes_in_bfs_order = []
    [unique_nodes_in_bfs_order.append(item) for item in flat_list if item not in unique_nodes_in_bfs_order]
    pos = {node: (pos_df.loc[node, 'x'], pos_df.loc[node, 'y']) for node in G.nodes()}
    for size in sizes:
        # Create subgraphs based on the first N nodes encountered in BFS
        subgraph_nodes = unique_nodes_in_bfs_order[:size]
        H = G.subgraph(subgraph_nodes)

        # Plot the original graph
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, node_color='lightgray', with_labels=False, node_size=20)

        # Highlight the subgraph
        nx.draw(H, {n: pos[n] for n in H.nodes()}, node_color='red', with_labels=False, node_size=50)

        plt.title(f'Subgraph with {size} Nodes Highlighted')
        plt.show()


def create_bfs_subgraphs_and_color_regions(G, pos_df, sizes=[500, 1000]):
    """
    Performs BFS to create subgraphs of specified sizes and colors the regions corresponding to these subgraphs.

    Parameters:
    - G: A NetworkX graph.
    - pos_df: DataFrame containing node positions with 'x' and 'y' columns.
    - sizes: A list of sizes for the subgraphs to be created.
    """
    # Identify the most central node for the starting point of BFS
    centrality = nx.closeness_centrality(G)
    start_node = max(centrality, key=centrality.get)

    # Perform BFS and get nodes in the order they are visited
    bfs_nodes = list(nx.bfs_edges(G, start_node))
    flat_list = [item for sublist in bfs_nodes for item in sublist]
    unique_nodes_in_bfs_order = []
    [unique_nodes_in_bfs_order.append(item) for item in flat_list if item not in unique_nodes_in_bfs_order]

    pos = {node: (pos_df.loc[node, 'x'], pos_df.loc[node, 'y']) for node in G.nodes()}

    plt.figure(figsize=(10, 8))


    if len(sizes) == 3:
        colors = ['red', 'orange', 'yellow']
    else:
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(sizes))]
    colors, sizes = reversed(colors), reversed(sizes)
    for size, color in zip(sizes, colors):
        # Create subgraphs based on the first N nodes encountered in BFS
        subgraph_nodes = unique_nodes_in_bfs_order[:size]
        subgraph_pos = np.array([pos[n] for n in subgraph_nodes])

        # Compute the convex hull of the subgraph positions
        if len(subgraph_pos) > 2:  # ConvexHull needs at least 3 points to compute
            hull = ConvexHull(subgraph_pos)
            hull_points = subgraph_pos[hull.vertices]

            # Plot the convex hull as a filled region
            plt.fill(*zip(*hull_points), color=color, alpha=1, label=f'Size {size}')

    nx.draw(G, pos, node_color='lightgray', with_labels=False, node_size=20)

    plt.legend()
    plt.show()


def plot_subgraphs_with_avg_distance_scalebars(G, sizes=[500, 1000]):
    """
    Plots the distance matrices for subgraphs of specified sizes as subplots with a shared color scale,
    and includes an additional subplot for scale bars indicating average distances.

    Parameters:
    - G: A NetworkX graph.
    - sizes: List of sizes for the subgraphs to be created.
    """
    centrality = nx.closeness_centrality(G)
    start_node = max(centrality, key=centrality.get)
    bfs_edges = list(nx.bfs_edges(G, start_node))
    bfs_nodes = [start_node] + [v for u, v in bfs_edges]

    max_dist = 0
    avg_distances = []
    distance_matrices = []

    # Compute distance matrices and average distances
    for size in sizes:
        subgraph_nodes = bfs_nodes[:size]
        H = G.subgraph(subgraph_nodes)
        path_length = dict(nx.all_pairs_shortest_path_length(H))
        distance_matrix = np.zeros((size, size))

        for i, node_i in enumerate(subgraph_nodes):
            for j, node_j in enumerate(subgraph_nodes):
                distance_matrix[i, j] = path_length[node_i].get(node_j, np.inf)

        finite_distances = distance_matrix[np.isfinite(distance_matrix)]
        if finite_distances.size > 0:
            max_dist = max(max_dist, np.max(finite_distances))
            avg_distances.append(np.mean(finite_distances))
        else:
            avg_distances.append(0)

        distance_matrices.append(distance_matrix)

    for i, distance_matrix in enumerate(distance_matrices):
        distance_matrix[np.isinf(distance_matrix)] = max_dist + 1  # Adjust infinities

    # Plotting setup
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(20, 6), gridspec_kw={'width_ratios': [1] * len(sizes) + [0.5]})

    # Plot distance matrices
    for i, size in enumerate(sizes):
        im = axes[i].imshow(distance_matrices[i], cmap='viridis', interpolation='nearest', vmin=0, vmax=max_dist + 1)
        axes[i].set_title(f'Subgraph Size {size}')
        axes[i].set_xlabel('Node Index')
        axes[i].set_ylabel('Node Index')

    fig.colorbar(im, ax=axes[:-1], orientation='vertical', label='Shortest path distance')

    # Add scale bars subplot
    axes[-1].barh(range(len(avg_distances)), avg_distances, color='skyblue')
    axes[-1].set_yticks(range(len(avg_distances)))
    axes[-1].set_yticklabels([f'Size {size}' for size in sizes])
    axes[-1].invert_yaxis()  # Keep the largest size on top
    axes[-1].set_xlabel('Average Distance')
    axes[-1].set_title('Average Distances')

    plt.tight_layout()
    plt.show()

args = GraphArgs()
args.dim = 2
args.proximity_mode = "delaunay_corrected"
args.num_points = 2000
args.false_edges_count = 0

# # # 1 Simulation
create_proximity_graph.write_proximity_graph(args, point_mode="circle", order_indices=False)
sparse_graph = load_graph(args, load_mode='sparse')
# plot_original_or_reconstructed_image(args, image_type='original')
original_position_folder = args.directory_map["original_positions"]
pos_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

G = compute_and_color_nodes(pos_df, sparse_graph)
sorted_dist_matrix, order = sort_distance_matrix(G)
# plot_sorted_distance_matrix_heatmap(sorted_dist_matrix)
# create_bfs_subgraphs_and_highlight(G, pos_df, sizes=[500, 1000])

# list(np.linspace(100, 1000, 10).astype(int))
sizes = [200, 500, 1000, 1500, 2000]
create_bfs_subgraphs_and_color_regions(G, pos_df, sizes=sizes)
plot_subgraphs_with_avg_distance_scalebars(G, sizes=sizes)

