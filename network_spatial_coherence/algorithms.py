import igraph as ig
import numpy as np
import pandas as pd
import random
from nodevectors import GGVec

import umap
from sklearn import manifold
import multiprocessing

import pecanpy
import tempfile
from pecanpy import pecanpy as node2vec
import torch
import pymde
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.csgraph import connected_components
from scipy.sparse import issparse

from scipy.sparse.csgraph import breadth_first_order
import networkx as nx
from sklearn.manifold import MDS

from scipy.sparse import coo_matrix, csr_matrix

from scipy.spatial import KDTree



def get_mean_shortest_path(igraph_graph, return_all_paths=False, args=None):
    # Check if args is provided and it has the attribute 'mean_shortest_paths'
    if args is not None and args.mean_shortest_paths is not None and return_all_paths is False:
        mean_shortest_path = args.mean_shortest_paths
        return mean_shortest_path
    else:
        if not isinstance(igraph_graph, ig.Graph):
            raise ValueError("Graph is not of igraph type")
        shortest_paths = igraph_graph.shortest_paths()
        path_lengths = [path for row in shortest_paths for path in row if path > 0]
        mean_shortest_path = np.mean(path_lengths)
        if args is not None:
            args.mean_shortest_paths = mean_shortest_path
        if return_all_paths:
            return mean_shortest_path, path_lengths
        else:
            return mean_shortest_path

def get_local_clustering_coefficients(igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    G = igraph_graph
    # Compute the local clustering coefficient for each node
    clustering_coefficients = G.transitivity_local_undirected()
    mean_clustering_coefficient = np.mean(clustering_coefficients)

    return clustering_coefficients


def bipartite_clustering_coefficient(igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    if not igraph_graph.is_bipartite():
        raise ValueError("Graph is not bipartite")

    G = igraph_graph
    clustering_coefficients = []

    for u in G.vs:
        second_order_neighbors = set()
        for neighbor in u.neighbors():
            second_order_neighbors.update(neighbor.neighbors())
        if u in second_order_neighbors:
            second_order_neighbors.remove(u)

        c_uv_sum = 0
        for v in second_order_neighbors:
            intersection = set(u.neighbors()).intersection(set(v.neighbors()))
            union = set(u.neighbors()).union(set(v.neighbors()))
            c_uv = len(intersection) / len(union) if len(union) > 0 else 0
            c_uv_sum += c_uv

        c_u = c_uv_sum / len(second_order_neighbors) if second_order_neighbors else 0
        clustering_coefficients.append(c_u)

    mean_clustering_coefficient = np.mean(clustering_coefficients)

    return clustering_coefficients, mean_clustering_coefficient


def bipartite_clustering_coefficient_optimized(args, igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    # Check if the graph is bipartite and get the types
    # is_bipartite, types = igraph_graph.is_bipartite(return_types=True)
    if not args.is_bipartite:
        raise ValueError("Graph is not bipartite")

    G = igraph_graph
    G.vs['type'] = args.bipartite_sets  # Assigning the types as attributes


    ## manual computation (deprecated)
    # G = assign_bipartite_sets(igraph_graph)  # Distinguish set1 and set2
    # # Identify the two sets
    type_attribute = G.vs['type'][0]  # Assumes 'type' attribute is used to distinguish the sets
    set1 = [v for v in G.vs if v['type'] == type_attribute]
    set2 = [v for v in G.vs if v['type'] != type_attribute]

    # Cache for neighbors and pairwise coefficients
    neighbors_cache = {}
    pairwise_coeff_cache = {}

    def get_neighbors(v):
        if v.index not in neighbors_cache:
            neighbors_cache[v.index] = set(v.neighbors())
        return neighbors_cache[v.index]

    def pairwise_coeff(u, v):
        if (u.index, v.index) in pairwise_coeff_cache:
            return pairwise_coeff_cache[(u.index, v.index)]
        if (v.index, u.index) in pairwise_coeff_cache:
            return pairwise_coeff_cache[(v.index, u.index)]

        neighbors_u = get_neighbors(u)
        neighbors_v = get_neighbors(v)
        intersection = neighbors_u.intersection(neighbors_v)
        union = neighbors_u.union(neighbors_v)
        c_uv = len(intersection) / len(union) if len(union) > 0 else 0

        pairwise_coeff_cache[(u.index, v.index)] = c_uv
        return c_uv

    # Function to calculate clustering coefficients for a set
    def calc_clustering_for_set(node_set):
        coefficients = []
        for u in node_set:
            second_order_neighbors = set()
            for neighbor in get_neighbors(u):
                second_order_neighbors.update(get_neighbors(neighbor))
            if u in second_order_neighbors:
                second_order_neighbors.remove(u)

            c_uv_sum = sum(pairwise_coeff(u, v) for v in second_order_neighbors)
            c_u = c_uv_sum / len(second_order_neighbors) if second_order_neighbors else 0
            coefficients.append(c_u)
        return coefficients

    clustering_coefficients_set1 = calc_clustering_for_set(set1)
    clustering_coefficients_set2 = calc_clustering_for_set(set2)

    mean_clustering_coefficient_set1 = np.mean(clustering_coefficients_set1)
    mean_clustering_coefficient_set2 = np.mean(clustering_coefficients_set2)

    return clustering_coefficients_set1, mean_clustering_coefficient_set1, \
           clustering_coefficients_set2, mean_clustering_coefficient_set2


def assign_bipartite_sets(igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    if not igraph_graph.is_bipartite():
        raise ValueError("Graph is not bipartite")

    G = igraph_graph
    visited = [False] * len(G.vs)
    type_attribute = [None] * len(G.vs)

    def bfs(start_vertex):
        queue = [start_vertex]
        visited[start_vertex] = True
        type_attribute[start_vertex] = 0  # Assign to set 1

        while queue:
            vertex = queue.pop(0)
            current_set = type_attribute[vertex]
            next_set = 1 if current_set == 0 else 0

            for neighbor in G.vs[vertex].neighbors():
                if not visited[neighbor.index]:
                    visited[neighbor.index] = True
                    type_attribute[neighbor.index] = next_set
                    queue.append(neighbor.index)

    # Start BFS from the first unvisited node
    for v in range(len(G.vs)):
        if not visited[v]:
            bfs(v)

    G.vs['type'] = type_attribute
    return G

def get_degree_distribution(igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    G = igraph_graph
    # Get the degree of each node
    degrees = G.degree()
    # Create a frequency distribution of the degrees
    max_degree = max(degrees)
    degree_distribution = [0] * (max_degree + 1)
    for degree in degrees:
        degree_distribution[degree] += 1
    return degree_distribution


def get_bipartite_degree_distribution(args, igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")

    if not args.is_bipartite:
        raise ValueError("Graph is not bipartite")

    G = igraph_graph
    G.vs['type'] = args.bipartite_sets  # Assigning the types as attributes


    # Separate the vertices into two sets based on type
    set1_indices = [v.index for v in G.vs if v['type']]
    set2_indices = [v.index for v in G.vs if not v['type']]

    # Function to calculate degree distribution for a set
    def calc_degree_distribution(node_indices):
        degrees = [G.degree(v) for v in node_indices]
        max_degree = max(degrees)
        degree_distribution = [0] * (max_degree + 1)
        for degree in degrees:
            degree_distribution[degree] += 1
        return degree_distribution

    # Get degree distribution for each set
    degree_distribution_set1 = calc_degree_distribution(set1_indices)
    degree_distribution_set2 = calc_degree_distribution(set2_indices)

    return degree_distribution_set1, degree_distribution_set2

def add_random_edges_igraph(args, graph, num_edges_to_add):
    possible_edges = [(i, j) for i in range(graph.vcount()) for j in range(i + 1, graph.vcount())]
    possible_edges = [edge for edge in possible_edges if not graph.are_connected(edge[0], edge[1])]
    random_edges = random.sample(possible_edges, num_edges_to_add)
    graph.add_edges(random_edges)
    return graph


def grow_graph_bfs(G, nodes_start, nodes_finish, n_graphs):
    if nodes_start > G.vcount() or nodes_finish > G.vcount():
        raise ValueError("nodes_start and nodes_finish must be less than or equal to the number of nodes in G")

    # Generate an array of node counts for each subgraph
    node_counts = np.linspace(nodes_start, nodes_finish, n_graphs, dtype=int)
    subgraphs = []
    for count in node_counts:
        visited = set()
        queue = [0]  # Start BFS from node 0 (this can be randomized or parameterized)
        while len(visited) < count:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(neigh for neigh in G.neighbors(current) if neigh not in visited)

        # Create subgraph from visited nodes
        subgraph = G.subgraph(visited)
        subgraphs.append(subgraph)

    return subgraphs

def get_bfs_samples(G, n_graphs, min_nodes):
    if min_nodes > G.vcount():
        raise ValueError("min_nodes must be less than or equal to the number of nodes in G")
    subgraphs = []
    for _ in range(n_graphs):
        start_node = random.randint(0, G.vcount() - 1)  # Randomize the start node for each subgraph
        visited = set()
        queue = [start_node]

        while len(visited) < min_nodes:
            if not queue:  # If the queue is empty, break the loop
                break
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(neigh for neigh in G.neighbors(current, mode="ALL") if neigh not in visited)

        # Create subgraph from visited nodes
        if visited:
            subgraph = G.subgraph(visited)
            subgraphs.append(subgraph)
    return subgraphs


def get_bfs_samples_by_depth(G, n_graphs, bfs_depth, node_ids=None):
    if bfs_depth < 0:
        raise ValueError("bfs_depth must be non-negative")


    subgraphs = []

    for i in range(n_graphs):
        if node_ids is not None:
            start_node = node_ids[i]
        else:
            start_node = random.randint(0, G.vcount() - 1)  # Randomize the start node for each subgraph


        visited = {start_node}
        queue = [(start_node, 0)]  # Queue stores tuples (node, depth)

        while queue:
            current, depth = queue.pop(0)
            if depth < bfs_depth:
                neighbors = [neigh for neigh in G.neighbors(current, mode="ALL") if neigh not in visited]
                visited.update(neighbors)
                queue.extend((neigh, depth + 1) for neigh in neighbors)

        if visited:
            subgraph = G.subgraph(visited)
            subgraphs.append(subgraph)

    return subgraphs


def get_one_bfs_sample(G, sample_size):
    if sample_size > G.vcount():
        raise ValueError("sample_size must be less than or equal to the number of nodes in G")

    start_node = random.randint(0, G.vcount() - 1)  # Randomize the start node for each subgraph
    visited = set()
    queue = [start_node]

    while len(visited) < sample_size:
        if not queue:  # If the queue is empty, break the loop
            break
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            queue.extend(neigh for neigh in G.neighbors(current, mode="ALL") if neigh not in visited)

    # Create subgraph from visited nodes
    if visited:
        subgraph = G.subgraph(visited)
    return subgraph



    # df_new = pd.DataFrame([results])
    #
    # csv_path = path_sp_results+ "spatial_constant_data.csv"
    # if os.path.exists(csv_path):
    #     df_existing = pd.read_csv(csv_path)
    #     df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    # else:
    #     df_combined = df_new
    #
    # # Sort by 'proximity_mode' and save
    # df_combined = df_combined.sort_values(by=['proximity_mode', 'dim', 'num_nodes'])
    # df_combined = df_combined.round(4)
    # df_combined.to_csv(csv_path, index=False)
    #
    # # Plotting the shortest path length distribution
    # plt.figure()
    # plt.hist(path_lengths, bins=20, edgecolor='k', alpha=0.7)
    # plt.axvline(mean_shortest_path, color='r', linestyle='dashed', linewidth=1)
    # plt.annotate(f'Average SPL: {mean_shortest_path:.2f}\nNumber of Nodes: {G.vcount()}',
    #              xy=(0.6, 0.7), xycoords='axes fraction',
    #              bbox=dict(facecolor='white', alpha=0.5))
    #
    # plt.xlabel('Shortest Path Length')
    # plt.ylabel('Number of Pairs')
    # plt.title('Distribution of Shortest Path Lengths')
    # plt.savefig(path_plot + "shortest_path_distribution.png")
    #
    # return mean_shortest_path


def get_minimum_spanning_tree_igraph(igraph_graph, weighted=False):
    if weighted:
        mst = igraph_graph.spanning_tree(weights=igraph_graph.es['weight'], return_tree=True)
    else:
        mst = igraph_graph.spanning_tree(return_tree=True)
    return mst

def compute_mean_std_per_group(dataframe, group_column, value_column):
    """
    When we repeat a simulation in the same settings to get statistical power.
    "group_column" should be the x variable (e.g. size)
    "value column" should be the y variable (e.g. spatial constant)
    This groups the results and extracts the mean and std
    """

    unique_groups = dataframe[group_column].unique()
    means = []
    std_devs = []
    groups = []

    # Calculate mean and standard deviation for each group
    for group in unique_groups:
        subset = dataframe[dataframe[group_column] == group]
        mean = subset[value_column].mean()
        std = subset[value_column].std()
        means.append(mean)
        std_devs.append(std)
        groups.append(group)
    # usage:
    # sizes, means, std_devs = compute_mean_std_per_group(df, 'intended_size', 'S_general')
    return np.array(groups), np.array(means), np.array(std_devs)

class ImageReconstruction:
    """
    A class for reconstructing the spatial structure of graphs using various node embedding and
    manifold learning techniques. It supports converting graphs into a format suitable for
    embeddings, applying dimensionality reduction, and optionally adjusting the embedding method
    for weighted graphs.

    Attributes:
        graph (sparse matrix): The graph to be reconstructed, expected to be in a sparse matrix format.
        dim (int): The target dimension for the reconstruction (typically 2 or 3).
        node_embedding_mode (str): The method used for generating node embeddings. Supported modes include
                                   'ggvec', 'STRND', 'landmark_isomap', 'PyMDE', and others.
        manifold_learning_mode (str): The manifold learning technique applied for dimensionality reduction,
                                      such as 'UMAP'.
        node_embedding_components (int): The number of components (dimensions) to use for node embeddings.
        manifold_learning_neighbors (int): The number of neighbors to consider in manifold learning techniques.

    Methods:
        detect_and_adjust_for_weighted_graph(): Adjusts the node embedding mode if the graph is detected to be weighted.
        compute_embeddings(args=None): Computes node embeddings based on the specified `node_embedding_mode`.
        reduce_dimensions(embeddings): Applies dimensionality reduction to the computed embeddings to achieve the target dimension.
        write_positions(args, np_positions, output_path, old_indices=False): Saves the reconstructed positions to a CSV file.
        reconstruct(do_write_positions=False, args=None): Performs the complete reconstruction process, including embedding computation and dimensionality reduction.
        landmark_isomap(): A specific embedding technique that uses Isomap based on landmarks for dimensionality reduction.
    """

    def __init__(self, graph, shortest_path_matrix, dim=2, node_embedding_mode="ggvec", manifold_learning_mode="UMAP",
                 node_embedding_components=64, manifold_learning_neighbors=15):
        """
        Initialize the ImageReconstruction object.

        :param graph: graph of type sparse #TODO: include more types
        :param dim: Target dimension for the UMAP reduction (2 or 3).
        """
        self.graph = graph
        self.shortest_path_matrix = shortest_path_matrix
        self.dim = dim
        self.node_embedding_components = node_embedding_components
        self.manifold_learning_neighbors = manifold_learning_neighbors
        self.node_embedding_mode = node_embedding_mode  # node2vec, ggvec, landmark_isomap, PyMDE
        self.manifold_learning_mode = manifold_learning_mode
        # Detect if the graph is weighted and adjust embedding mode if necessary
        self.detect_and_adjust_for_weighted_graph()
    def detect_and_adjust_for_weighted_graph(self):
        """
        Detects if the graph is weighted. If it is, adjusts the node_embedding_mode to 'PyMDE_weighted'.
        """
        if self.graph is not None:
            # Assuming the graph is in a format that allows checking for weights
            sparse_matrix_coo = self.graph.tocoo() if not sp.isspmatrix_coo(self.graph) else self.graph
            weights = sparse_matrix_coo.data

            # Graph is considered weighted if any weight is different from 1
            is_weighted = not np.all(weights == 1)
            if is_weighted and (self.node_embedding_mode != "PyMDE" and self.node_embedding_mode != "MDS"):
                if is_weighted and (self.node_embedding_mode == "landmark_isomap"):
                    self.node_embedding_mode = "landmark_isomap_weighted"
                    print("Detected a weighted graph. Switching node_embedding_mode to 'landmark_isomap_weighted'.")
                elif is_weighted and (self.node_embedding_mode == "STRND"):
                    print("Detected a weighted graph. Let's try STRND for weighted graphs.")
                else:
                    self.node_embedding_mode = "PyMDE_weighted"
                    print("Detected a weighted graph. Switching node_embedding_mode to 'PyMDE_weighted'.")


    def compute_embeddings(self, args=None):
        """
        Compute node embeddings using ggvec.
        """
        # graph = ig.Graph.TupleList(self.edge_list, directed=False)
        if self.node_embedding_mode == 'ggvec':
            ggvec_model = GGVec(n_components=self.node_embedding_components)
            node_embeddings = ggvec_model.fit_transform(self.graph)
        elif self.node_embedding_mode == "STRND":
            # TODO: add node_embedding_components
            # raise ValueError("Not implemented yet")
            ### nodevectors
            # node2vec_model = Node2Vec(n_components=self.node_embedding_components)
            # node_embeddings = node2vec_model.fit_transform(self.graph)
            ### pecanpy
            edge_list_folder = args.directory_map['edge_lists']
            edge_list_path = f'{edge_list_folder}/{args.edge_list_title}'

            # Temporary file without header
            delimiter = ","
            edge_df = pd.read_csv(edge_list_path, delimiter=delimiter)
            if args.weighted:
                edge_df = edge_df[['source', 'target', 'weight']]
            else:
                edge_df = edge_df[['source', 'target']]
            # with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
            #     with open(edge_list_path, 'r') as f:
            #         next(f)  # Skip the header line
            #         for line in f:
            #             parts = line.strip().split(delimiter)
            #             # Ensure only node1, node2, and weight are written to the new file
            #             if len(parts) >= 3:
            #                 # Write only the first three columns to the new file
            #                 tmp_file.write(f'{parts[0]}{delimiter}{parts[1]}{delimiter}{parts[3]}\n')  # weight is in the 4th column
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
                edge_df.to_csv(tmp_file.name, index=False, header=False)

            g = node2vec.PreComp(p=1, q=1, workers=4, verbose=True, )
            # load graph from temporary edgelist file
            if args.weighted and args.weight_to_distance:
                ### print the weights
                # tmp = pd.read_csv(tmp_file.name, header=None, names=['source', 'target', 'weight'])
                # print("weights", tmp['weight'])

                g.read_edg(tmp_file.name, weighted=True, directed=False, delimiter=',')
            else:
                g.read_edg(tmp_file.name, weighted=False, directed=False, delimiter=',')
            g.preprocess_transition_probs()
            node_embeddings = g.embed(dim=self.node_embedding_components)

            # Reorder nodes so they match the embeddings
            node_ids = g.nodes  # Get the list of node IDs
            idx_to_id = {idx: int(node_id) for idx, node_id in enumerate(node_ids)}
            reordered_array = np.empty_like(node_embeddings)

            # Iterate over the existing array and reorder
            for node_id, row in enumerate(node_embeddings):
                index = idx_to_id[node_id]  # Find the new index for this row
                reordered_array[index] = row  # Place the row in the new position
            node_embeddings = reordered_array


        elif self.node_embedding_mode == "landmark_isomap":

            node_embeddings = self.landmark_isomap()
        elif self.node_embedding_mode == "landmark_isomap_weighted":
            node_embeddings = self.landmark_isomap_weighted()

        elif self.node_embedding_mode == "spring_relaxation":
            G = nx.from_scipy_sparse_array(self.graph)
            pos = nx.spring_layout(G)
            pos_array = np.array([pos[i] for i in range(len(pos))])
            node_embeddings = pos_array

        elif self.node_embedding_mode == "MDS":
            if args.shortest_path_matrix is not None:
                sp_matrix = args.shortest_path_matrix
            else:
                sp_matrix = compute_shortest_path_matrix_sparse_graph(self.graph, args=args)

            mds = MDS(n_components=args.dim, dissimilarity='precomputed', random_state=42)

            positions = mds.fit_transform(sp_matrix)
            node_embeddings = positions



        elif self.node_embedding_mode == "PyMDE":
            retain_fraction = 1

            # Load graph in pymde format
            sparse_matrix_coo = self.graph.tocoo() if not sp.isspmatrix_coo(self.graph) else self.graph
            rows = sparse_matrix_coo.row
            cols = sparse_matrix_coo.col
            edges_np = np.vstack([rows, cols]).T
            edges = torch.tensor(edges_np,  dtype=torch.int64)
            graph = pymde.Graph.from_edges(edges, weights=None)

            # Reconstructions work best in the "dense" form, that is, computing the shortest path distances
            shortest_paths_graph = pymde.preprocess.graph.shortest_paths(graph,
                                                                         retain_fraction=retain_fraction)  # Takes a while for large graphs
            full_edges = shortest_paths_graph.edges.to("cpu")
            deviations = shortest_paths_graph.distances.to("cpu")

            # Create the MDE problem, dense edges, deviations
            mde = pymde.MDE(args.num_points, self.node_embedding_components, full_edges,
                            # pymde.penalties.Quadratic(weights=deviations),  # For weights (similarities)
                            pymde.losses.Quadratic(deviations=deviations),  # For deviations (dissimilarities)
                            pymde.constraints.Standardized(),
                            # pymde.constraints.Centered())
                            # TODO: ANCHOR?
                            #pymde.constraints.Anchored(anchors=torch.tensor([0, 1, 2, 3]), values=original_points[:4]))
                            )

            node_embeddings = mde.embed()

        elif self.node_embedding_mode == "PyMDE_weighted":
            retain_fraction = 1

            # Load graph in pymde format
            sparse_matrix_coo = self.graph.tocoo() if not sp.isspmatrix_coo(self.graph) else self.graph
            rows = sparse_matrix_coo.row
            cols = sparse_matrix_coo.col
            weights = sparse_matrix_coo.data  # Extract weights from the sparse matrix

            # inferred_distances = np.exp((-weights + np.mean(weights))/(np.max(weights) - np.min(weights)))  # TODO: here I apply the weight-to-distance transformation (in this case log)

            # TODO: for now just inverse power mapping, but an exponential with the right parameters might make more sense
            inferred_distances = np.power(weights.astype(float), -1)
            print("inferred distances from weights", inferred_distances)
            edges_np = np.vstack([rows, cols]).T

            # ### Using GPU  (not working)
            # device = torch.device("hip:0")  # For AMD GPUs using ROCm
            # edges = torch.tensor(edges_np, dtype=torch.int64).to(device)
            # inferred_distances = torch.tensor(inferred_distances, dtype=torch.float32).to(device)

            ## Using CPU
            edges = torch.tensor(edges_np, dtype=torch.int64)
            inferred_distances = torch.tensor(inferred_distances, dtype=torch.float32)  # Make sure weights are in the correct format


            graph = pymde.Graph.from_edges(edges, weights=inferred_distances)

            # Reconstructions work best in the "dense" form, that is, computing the shortest path distances

            shortest_paths_graph = pymde.preprocess.graph.shortest_paths(graph,
                                                                         retain_fraction=retain_fraction)


            full_edges = shortest_paths_graph.edges.to("cpu")
            deviations = shortest_paths_graph.distances.to("cpu")
            # Create the MDE problem, dense edges, deviations
            mde = pymde.MDE(args.num_points, self.node_embedding_components, full_edges,
                            pymde.losses.Quadratic(deviations=deviations),  # Use deviations for dissimilarities
                            pymde.constraints.Standardized()
                            )

            print("proceeding to PyMDE embedding")

            node_embeddings = mde.embed()
            print("finished PyMDE embedding")

        else:
            raise ValueError('Please input a valid node embedding mode')

        return node_embeddings

    def reduce_dimensions(self, embeddings):
        """
        Reduce the dimensionality of embeddings using UMAP.

        :param embeddings: High-dimensional embeddings of nodes.
        """

        if self.manifold_learning_mode == 'UMAP':
            umap_model = umap.UMAP(n_components=self.dim, n_neighbors=self.manifold_learning_neighbors, min_dist=1)
            reduced_embeddings = umap_model.fit_transform(embeddings)
        else:
            raise ValueError('Please input a valid manifold learning mode')

        return reduced_embeddings

    def write_positions(self, args, np_positions, output_path, old_indices=True):
        # Write standard dataframe format:
        if args.dim == 2:
            positions_df = pd.DataFrame(np_positions, columns=['x', 'y'])
        elif args.dim == 3:
            positions_df = pd.DataFrame(np_positions, columns=['x', 'y', 'z'])
        else:
            raise ValueError("Please input a valid dimension")

        if old_indices and args.node_ids_map_old_to_new:  # Preserve old indicies, this will make it so it doesn't go from 0 to N-1
            node_ids_map_new_to_old = {new: old for old, new in args.node_ids_map_old_to_new.items()}

            print(args.node_ids_map_old_to_new)
            node_ids = [node_ids_map_new_to_old[new_index] for new_index in sorted(node_ids_map_new_to_old)]
            positions_df['node_ID'] = node_ids
            # Define the output file path
            title = args.args_title
            output_file_path = f"{output_path}/positions_old_index_{title}.csv"
            positions_df.to_csv(output_file_path, index=False)

        node_ids = range(args.num_points)   # from 0 to N-1
        positions_df['node_ID'] = node_ids
        title = args.args_title
        output_file_path = f"{output_path}/positions_{title}.csv"

        # Write the DataFrame to a CSV file
        positions_df.to_csv(output_file_path, index=False)
    def reconstruct(self, do_write_positions=False, args=None):
        """
        Performs the complete graph reconstruction process, including computing embeddings, applying dimensionality
        reduction, and optionally writing the reconstructed positions to a CSV file.

        Args:
            do_write_positions (bool): Whether to save the reconstructed positions to a CSV file. Defaults to False.
            args (Optional[object]): Additional arguments required for certain operations, such as writing positions.

        Returns:
            numpy.ndarray: The reconstructed positions of the nodes.
        """
        embeddings = self.compute_embeddings(args)

        # TODO: Modes that do not require dimensionality reduction  (careful, the list has to be updated if I add new modes)
        excluded_modes = [
            'landmark_isomap',
            'landmark_isomap_weighted',
            'spring_relaxation',
            'PyMDE',
            'MDS',
            'PyMDE_weighted',
        ]
        # Check if the current mode is not in the list of excluded modes
        if self.node_embedding_mode not in excluded_modes:
            reconstructed_points = self.reduce_dimensions(embeddings)
        else:  # in landmark isomap the result is already the reconstructed points
            reconstructed_points = embeddings

        if do_write_positions:
            if args == None:
                raise ValueError("Pass args to the function please")

            if args.handle_all_subgraphs:
                output_path = args.directory_map["rec_positions_subgraphs"]
                self.write_positions(args, np_positions=np.array(reconstructed_points), output_path=output_path, old_indices=True)

            output_path = args.directory_map["reconstructed_positions"]
            self.write_positions(args, np_positions=np.array(reconstructed_points), output_path=output_path,
                                 old_indices=True)

        return reconstructed_points


    def landmark_isomap(self):
        def from_edge_list_to_dict(edge_list):
            import collections
            dict_graph = collections.defaultdict(set)
            for edge in edge_list:
                i, j = edge[0], edge[1]
                dict_graph[i].add(j)
                dict_graph[j].add(i)
            return dict_graph

        def bfs_single_source(graph, source):
            from collections import deque
            # Initialize distance dictionary with infinite distance for all nodes except source
            distances = {node: float('inf') for node in graph}
            distances[source] = 0

            # Initialize queue with source node
            queue = deque([source])

            # Traverse graph using BFS
            while queue:  # while there are nodes in the queue
                node = queue.popleft()
                # Visit all neighbors of current node
                for neighbor in graph[node]:
                    # Update distance and add to queue if not already visited
                    if distances[neighbor] == float('inf'):  # (if not visited before)
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            return distances

        def sparse_matrix_to_edge_list(sparse_matrix):
            rows, cols = sparse_matrix.nonzero()
            edge_list = np.column_stack((rows, cols))
            return edge_list
        def symmetrize(a):
            """
            Return a symmetrized version of NumPy array a.

            Values 0 are replaced by the array value at the symmetric
            position (with respect to the diagonal), i.e. if a_ij = 0,
            then the returned array a' is such that a'_ij = a_ji.

            Diagonal values are left untouched.

            a -- square NumPy array, such that a_ij = 0 or a_ji = 0,
            for i != j.
            """
            return a + a.T - np.diag(a.diagonal())

        # np_edge_list = np.array(get_edge_list_as_df(self.args))
        np_edge_list = sparse_matrix_to_edge_list(self.graph)

        # np_edge_list = np.unique(
        #     np.genfromtxt(self.args.title_edge_list, dtype=int), axis=0) - 1

        dict_graph = from_edge_list_to_dict(np_edge_list)
        N = len(dict_graph)

        # Select random landmarks
        selected_landmarks = np.random.choice(np.arange(N), self.node_embedding_components, replace=False)

        # Initialize distance from every node to every landmark (NxD matrix)
        all_distances_to_landmarks = np.empty((N, self.node_embedding_components))
        # Single source BFS using landmarks as sources
        for j, landmark in enumerate(selected_landmarks):
            short_path = bfs_single_source(dict_graph, landmark)
            for sp_node_id, sp_length in short_path.items():
                all_distances_to_landmarks[sp_node_id][j] = sp_length

        # Landmark DxD distance matrix (symmetric positive)
        landmark_distance_matrix = all_distances_to_landmarks[selected_landmarks]
        landmark_distance_matrix = symmetrize(landmark_distance_matrix)

        # np.set_printoptions(threshold=sys.maxsize)
        # print("LANDMARK DISTANCE DXD", landmark_distance_matrix)
        # print("L2 DXD", all_distances_to_landmarks2[selected_landmarks])
        def landmark_MDS(diss_matrix_landmarks, all_distance_to_landmarks):
            """
            1. Apply MDS to position landmark nodes
            2. Use landmark positions eigenvalues (moore penrose inverse) to position the rest of the nodes
            """
            mds = manifold.MDS(n_components=self.dim, metric=True, random_state=2,
                               dissimilarity="precomputed")

            L = np.array(mds.fit_transform(diss_matrix_landmarks))  # landmark_coordinates --> good results

            # Triangulate all points
            D2 = diss_matrix_landmarks ** 2
            D2_all = all_distance_to_landmarks ** 2
            mean_column = D2.mean(axis=0)
            L_slash = np.linalg.pinv(L)
            recovered_positions = np.transpose(0.5 * L_slash.dot(np.transpose(mean_column - D2_all)))
            return recovered_positions

        recovered_positions = landmark_MDS(landmark_distance_matrix, all_distances_to_landmarks)
        vectors = recovered_positions
        return vectors

    def landmark_isomap_weighted(self):
        def select_landmarks(num_landmarks, num_nodes):
            return np.random.choice(np.arange(num_nodes), num_landmarks, replace=False)

        def symmetrize(a):
            """
            """
            return np.maximum(a, a.T)

        def landmark_MDS(diss_matrix_landmarks, all_distance_to_landmarks):
            """
            1. Apply MDS to position landmark nodes
            2. Use landmark positions eigenvalues (Moore-Penrose inverse) to position the rest of the nodes
            """
            mds = manifold.MDS(n_components=self.dim, metric=True, random_state=2,
                               dissimilarity="precomputed")

            L = np.array(mds.fit_transform(diss_matrix_landmarks))  # landmark_coordinates --> good results

            # Triangulate all points
            D2 = diss_matrix_landmarks ** 2
            D2_all = all_distance_to_landmarks ** 2
            mean_column = D2.mean(axis=0)
            L_slash = np.linalg.pinv(L)
            recovered_positions = np.transpose(0.5 * L_slash.dot(np.transpose(mean_column - D2_all)))
            return recovered_positions

        # The shortest path matrix (NxN) is provided as input
        shortest_path_matrix = self.shortest_path_matrix
        N = shortest_path_matrix.shape[0]

        # Select random landmarks
        selected_landmarks = select_landmarks(self.node_embedding_components, N)
        all_distances_to_landmarks = shortest_path_matrix[:, selected_landmarks]

        # Landmark DxD distance matrix (symmetric positive)
        landmark_distance_matrix = all_distances_to_landmarks[selected_landmarks]
        landmark_distance_matrix = symmetrize(landmark_distance_matrix)

        # Apply landmark MDS to get node embeddings
        recovered_positions = landmark_MDS(landmark_distance_matrix, all_distances_to_landmarks)
        vectors = recovered_positions
        return vectors


def compute_distance_matrix(coords):
    """
    Compute the Euclidean distance matrix for a set of 2D or 3D points. (numpy array)

    :param coords: A Nx2 or Nx3 numpy array of coordinates, where each row is a point.
    :return: NxN numpy array representing the distance matrix.
    """
    expanded_coords = np.expand_dims(coords, axis=1)
    differences = coords - expanded_coords
    squared_distances = np.sum(differences**2, axis=-1)
    distance_matrix = np.sqrt(squared_distances)
    return distance_matrix


def global_efficiency(graph):
    """ global efficiency for igraph graphs"""
    distances = graph.shortest_paths_dijkstra()
    inv_distances = [[1/x if x != 0 else 0 for x in row] for row in distances]
    sum_inv_distances = sum(sum(row) for row in inv_distances)
    n = len(distances)
    return sum_inv_distances / (n * (n - 1))

def local_efficiency(graph):
    """ local efficiency for igraph graphs"""
    local_effs = []
    for vertex in range(graph.vcount()):
        subgraph = graph.subgraph(graph.neighborhood(vertex, order=1))
        if subgraph.vcount() > 1:
            local_effs.append(global_efficiency(subgraph))
        else:
            local_effs.append(0)
    return sum(local_effs) / len(local_effs)


def select_false_edges(graph, max_false_edges):
    """
    Selects false edge candidates for an igraph
    """
    possible_edges = [(i, j) for i in range(graph.vcount()) for j in range(i + 1, graph.vcount())]
    possible_edges = [edge for edge in possible_edges if not graph.are_connected(edge[0], edge[1])]
    random_false_edges = random.sample(possible_edges, min(len(possible_edges), max_false_edges))
    return random_false_edges

def select_false_edges_csr(graph, max_false_edges, args=None):
    """
    Selects false edge candidates for a CSR graph (scipy.sparse.csr_matrix).
    Usage:         all_random_false_edges = select_false_edges_csr(sparse_graph, max_false_edges)

        for num_edges in false_edges_list:
            modified_graph = add_specific_random_edges_to_csrgraph(sparse_graph.copy(), all_random_false_edges,
                                                                   num_edges)
    """
    num_nodes = graph.shape[0]
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]

    # Filter out existing edges
    existing_edges = set(zip(*graph.nonzero()))
    possible_false_edges = [edge for edge in possible_edges if edge not in existing_edges]

    # Randomly select false edges up to the specified limit
    random_false_edges = random.sample(possible_false_edges, min(len(possible_false_edges), max_false_edges))

    if args:
        args.false_edge_ids = random_false_edges

    return random_false_edges

def add_specific_random_edges_igraph(graph, false_edges_ids, num_edges_to_add):
    # Add only the first num_edges_to_add edges from the false_edges list
    edges_to_add = false_edges_ids[:num_edges_to_add]
    graph.add_edges(edges_to_add)
    return graph


def compute_eigenvalues_laplacian_csgraph(graph):
    # Convert the adjacency matrix to a sparse graph Laplacian
    L = sp.csgraph.laplacian(graph, normed=True)

    # Compute the second smallest eigenvalue
    # We use which='SM' to request the smallest magnitude eigenvalues and k=2 since we need the second smallest
    eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=True)

    # The second smallest eigenvalue
    second_smallest_eigenvalue = eigenvalues[1]
    return second_smallest_eigenvalue


def find_central_nodes(distance_matrix, num_central_nodes=1):
    """
    Finds nodes that are closer to every other node based on the average shortest path.

    Args:
        distance_matrix (numpy.ndarray): A 2D numpy array representing the pairwise shortest path distances between nodes.
        num_central_nodes (int): The number of central nodes to return.

    Returns:
        numpy.ndarray: An array of indices representing the central nodes, sorted from most central to less central.
    """
    # Calculate the average shortest path for each node
    average_shortest_paths = distance_matrix.mean(axis=1)
    central_node_indices = np.argsort(average_shortest_paths)[:num_central_nodes]
    if len(central_node_indices) == 1:
        central_node_indices = central_node_indices[0]
    return central_node_indices



def find_geometric_central_node(positions_df):
    """ Find central node fore Euclidean coordinates"""
    centroid_x = positions_df['x'].mean()
    centroid_y = positions_df['y'].mean()

    distances_to_centroid = np.sqrt((positions_df['x'] - centroid_x) ** 2 + (positions_df['y'] - centroid_y) ** 2)
    closest_node_index = distances_to_centroid.idxmin()
    central_node_ID = positions_df.iloc[closest_node_index]['node_ID']
    return central_node_ID


def compute_shortest_path_mapping_from_central_node(central_node_ID, positions_df, shortest_path_matrix):
    central_node_index = positions_df.index[positions_df['node_ID'] == central_node_ID].tolist()[0]
    central_node_distances = shortest_path_matrix[central_node_index]
    node_ID_to_shortest_path_mapping = dict(zip(positions_df['node_ID'], central_node_distances))
    return node_ID_to_shortest_path_mapping

def compute_shortest_path_matrix_sparse_graph(sparse_graph, args=None, force_recompute=False):
    """
    Computes the shortest path matrix for a given sparse graph. If `args` is provided and contains a precomputed
    shortest path matrix, that matrix is returned instead of recomputing it. Otherwise, the shortest path matrix
    is computed from the sparse graph, and if `args` is provided, the computed matrix and its mean are stored
    in `args`.

    The function supports both the computation of shortest paths in the absence of the `args` object and the
    utilization of precomputed values within `args` to avoid redundant computations.

    Args:
        sparse_graph: A sparse graph representation for which the shortest path matrix will be computed. The graph
                      should be compatible with the `shortest_path` function requirements from scipy's csgraph module.
        args (Optional[object]): An optional object that may contain the precomputed shortest path matrix and can
                                 store the computed shortest path matrix and its mean. This object should have
                                 `shortest_path_matrix` and `mean_shortest_path` attributes if utilized.

    Returns:
        numpy.ndarray: A numpy array representing the shortest path matrix of the given sparse graph.

    Side Effects:
        If `args` is provided and does not contain a precomputed shortest path matrix, the computed shortest path
        matrix and its mean are stored in `args.shortest_path_matrix` and `args.mean_shortest_path`, respectively.

    Note:
        - The function relies on `convert_graph_type` from a utils module to ensure the sparse graph is in the
          desired format for computation.
        - The shortest path computation is performed using the `shortest_path` function from scipy's csgraph module,
          assuming an undirected graph.
    """
    # TODO: this doesn't force to recompute the shortest path matrix
    from utils import convert_graph_type
    sparse_graph = convert_graph_type(args, graph=sparse_graph, desired_type="sparse")
    is_weighted = np.any(sparse_graph.data != 1) and issparse(sparse_graph)

    if args is None:
        sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False, unweighted=not is_weighted))
        if np.any((sp_matrix == 0) & (~np.eye(sp_matrix.shape[0], dtype=bool))):
            raise ValueError("Shortest path matrix contains zero values off the diagonal, which is invalid for a shortest path matrix.")
        return sp_matrix
    elif (args is not None) and (args.shortest_path_matrix is not None) and (not force_recompute):
        return args.shortest_path_matrix
    else:
        sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False, unweighted=not is_weighted))
        if np.any((sp_matrix == 0) & (~np.eye(sp_matrix.shape[0], dtype=bool))):
            raise ValueError("Shortest path matrix contains zero values off the diagonal, which is invalid for a shortest path matrix.")
        args.shortest_path_matrix = sp_matrix
        args.mean_shortest_path = sp_matrix.mean()
        return sp_matrix






def safely_remove_edges_sparse(csgraph, percentage=10):
    """
    Randomly removes a percentage of edges from a sparse graph (csgraph) without disconnecting it.
    Might be not computationally efficient, as it is checking the connected components every time
    """
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100.")

    graph = csgraph.tolil()
    rows, cols = graph.nonzero()
    edges = list(zip(rows, cols))
    np.random.shuffle(edges)

    total_edges = len(edges)
    edges_to_remove = int((percentage / 100) * total_edges)
    removed_edges = 0

    for edge in edges:
        if removed_edges >= edges_to_remove:
            break
        graph[edge[0], edge[1]] = 0
        graph[edge[1], edge[0]] = 0
        # Check if graph is connected
        num_components, labels = connected_components(csgraph=graph, directed=False)
        if num_components == 1:
            removed_edges += 1  # Permanently remove the edge
        else:

            graph[edge[0], edge[1]] = 1
            graph[edge[1], edge[0]] = 1

    return graph.tocsr()


def custom_bfs_csgraph(csgraph, min_nodes=3000):
    """
    Perform a custom BFS on a csgraph starting from a given node until
    at least min_nodes are visited or the entire graph is traversed.

    Parameters:
    - csgraph: CSR matrix representing the graph.
    - start_node: The starting node for BFS.
    - min_nodes: Minimum number of nodes to visit.

    Returns:
    - visited_nodes: Indices of the nodes visited during BFS.
    """
    start_node = 0  # random start node  --> #TODO: best would be should actually choose a central one
    n_nodes = csgraph.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    queue = [start_node]
    visited_nodes = []
    while queue and len(visited_nodes) < min_nodes:
        current_node = queue.pop(0)

        if not visited[current_node]:
            visited[current_node] = True
            visited_nodes.append(current_node)

            # Get neighbors of the current node
            neighbors = csgraph.indices[csgraph.indptr[current_node]:csgraph.indptr[current_node + 1]]

            for neighbor in neighbors:
                if not visited[neighbor]:
                    queue.append(neighbor)

    return visited_nodes







def sample_csgraph_subgraph(args, csgraph, min_nodes=3000):
    """
    Sample a subgraph from a csgraph using a custom BFS to visit at least min_nodes,
    write the edge list of the sampled subgraph, and maintain a mapping to original indices.
    """
    from utils import custom_bfs_csgraph  # Ensure this is implemented and imported correctly

    visited_nodes = custom_bfs_csgraph(csgraph, min_nodes=min_nodes)
    mask = np.zeros(csgraph.shape[0], dtype=bool)
    mask[visited_nodes] = True
    subgraph = csgraph[mask, :][:, mask]
    sorted_visited_nodes = sorted(visited_nodes)

    args.num_points = len(visited_nodes)
    rows, cols = subgraph.nonzero()
    args.sparse_graph = subgraph
    args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(subgraph, None)
    args.mean_shortest_path = args.shortest_path_matrix.mean()

    filtered_edges = [(i, j) for i, j in zip(rows, cols) if i < j]
    edge_df = pd.DataFrame(filtered_edges, columns=['source', 'target'])

    edge_list_folder = args.directory_map["edge_lists"]
    edge_list_title = f"edge_list_{args.args_title}_subgraph.csv"  # Assuming 'args.title' exists
    args.edge_list_title = edge_list_title  # update the edge list title
    edge_df.to_csv(f"{edge_list_folder}/{edge_list_title}", index=False)

    if args.node_ids_map_old_to_new is not None:  # TODO: case where we have originally a disconnected subgraph. Not sure if this works, and it is relevant for experimental data that is disconnected
        # Reverse the mapping to get from new (csgraph) indices to old (original) indices
        new_to_old_index_mapping = {new: old for old, new in args.node_ids_map_old_to_new.items()}
        original_indices = [new_to_old_index_mapping.get(idx, idx) for idx in sorted_visited_nodes]
    else:
        original_indices = sorted_visited_nodes


        # Extract edges from the subgraph and map indices back to original
    rows, cols = subgraph.nonzero()

    filtered_edges = [(original_indices[i], original_indices[j]) for i, j in zip(rows, cols) if i < j]
    edge_list_df = pd.DataFrame(filtered_edges, columns=['source', 'target'])


    # Define the file path and write the DataFrame to a CSV file
    edge_list_folder = args.directory_map["edge_lists"]
    edge_list_title = f"subgraph_edge_list_with_old_indices_{args.args_title}.csv"
    edge_list_df.to_csv(f"{edge_list_folder}/{edge_list_title}", index=False)

    # Update args with a new mapping from subgraph BFS indices back to original graph indices
    # This step might be adjusted based on your needs for using this mapping later

    args.node_ids_map_old_to_new = {original: idx for idx, original in enumerate(original_indices)}

    return subgraph


def edge_list_to_sparse_graph(edge_list):
    """
    Convert an edge list to a sparse adjacency matrix.
    Edge list is a list of tuples (node1, node2)
    """

    # Flatten the edge list and get unique nodes
    nodes = np.unique(np.array(list(edge_list)).flatten())
    # Map node IDs to matrix indices
    node_index = {node: i for i, node in enumerate(nodes)}

    # Prepare data for the COO format matrix
    row_indices = [node_index[edge[0]] for edge in edge_list]
    col_indices = [node_index[edge[1]] for edge in edge_list]
    data = np.ones(len(edge_list), dtype=int)

    # Create a symmetric matrix by doubling the edge list (for undirected graph)
    # and ensuring (i, j) and (j, i) entries are both filled
    row_indices += col_indices
    col_indices += [node_index[edge[0]] for edge in edge_list]  # Original row_indices
    data = np.concatenate([data, data])  # Double the data

    # Number of nodes
    n = len(nodes)

    # Create the COO-format matrix and convert to CSR
    adjacency_matrix_coo = coo_matrix((data, (row_indices, col_indices)), shape=(n, n))
    adjacency_matrix_csr = adjacency_matrix_coo.tocsr()

    # Ensure the matrix is symmetric
    adjacency_matrix_csr = (adjacency_matrix_csr + adjacency_matrix_csr.T) / 2

    return adjacency_matrix_csr


def generate_lattice_graph(N, k, dim=2):
    # Step 1: Generate N points uniformly distributed in a square [0,1] x [0,1]

    points = np.random.rand(N, dim)  # N points in 2 dimensions

    # Step 2: Create a graph using KNN
    tree = KDTree(points)  # Create a KD-tree for fast k-nearest neighbor lookup
    edges = []
    for i in range(N):
        distances, indices = tree.query(points[i], k=k + 1)  # Query includes the point itself
        for j in range(1, k + 1):  # Start from 1 to skip the point itself
            edges.append((i, indices[j]))

    # Step 3: Create and return the graph object using igraph
    g = ig.Graph(edges=edges, directed=False)
    g.vs['x'], g.vs['y'] = points[:, 0], points[:, 1]  # Store coordinates for plotting or further use

    return g

def compute_largeworldness(args, sparse_graph):
    # Convert CSR matrix to igraph

    N = sparse_graph.shape[0]
    num_edges = sparse_graph.nnz // 2
    k_avg = 2 * num_edges / N if N != 0 else 0

    # Analytical mean shortest path for a random graph
    mean_shortest_path_random = (np.log(N) - 0.5772) / np.log(k_avg) + 0.5
    lattice_graph = generate_lattice_graph(N, int(k_avg), dim=args.dim)

    mean_shortest_path_lattice_1d = lattice_graph.average_path_length()

    mean_shortest_path_score = (args.mean_shortest_path - mean_shortest_path_random) / (mean_shortest_path_lattice_1d - mean_shortest_path_random)
    print(f"Mean shortest path score: {mean_shortest_path_score}")
    return mean_shortest_path_score


def ensure_connected(csr_graph):
    """ Check if the graph is fully connected """
    n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
    # print(f"Number of connected components: {n_components}")
    return n_components == 1



def randomly_delete_edges(args, csr_graph, delete_ratio=0.1):
    """
    Randomly delete edges from a sparse CSR matrix graph, ensuring the graph remains fully connected.
    Keeps the graph symmetric throughout the process.

    Parameters:
    args (object): Object that carries various properties and configurations.
    csr_graph (sp.csr_matrix): The input graph in CSR format.
    delete_ratio (float): The fraction of existing edges to attempt to delete.

    Returns:
    object, sp.csr_matrix: Updated args and the modified CSR graph.
    """
    # Ensure the graph is initially connected and symmetric
    if not ensure_connected(csr_graph):
        raise ValueError("Initial graph must be connected")

    # Work with the upper triangle to avoid redundant operations
    # csr_graph = sp.triu(csr_graph, format='csr')
    rows, cols = csr_graph.nonzero()
    if args.false_edge_ids:
        # Create a set of tuples from the edges not to be deleted
        protected_edges = set(tuple(edge) for edge in args.false_edge_ids)
    else:
        protected_edges = set()

    all_edges = [(i, j) for i, j in zip(rows, cols) if
                 i < j and (i, j) not in protected_edges and (j, i) not in protected_edges]

    # Number of edges to attempt to delete
    num_deletable_edges = len(all_edges)
    edges_to_delete = int(num_deletable_edges * delete_ratio)


    # Randomly select edges to delete
    random.shuffle(all_edges)  # Shuffle the list of deletable edges

    lil_graph = csr_graph.tolil()  # Convert to LIL format for easier manipulation

    # print("number of edges to delete", edges_to_delete)
    # print("total number of edges", len(all_edges))
    deletions = 0
    while deletions < edges_to_delete:
        if not all_edges:
            raise ValueError("No more edges to delete! The graph cannot be any more sparse without disconnecting it")
            break  # Exit if there are no more edges to consider

        # Select a random edge
        edge_index = random.randint(0, len(all_edges) - 1)
        row, col = all_edges.pop(edge_index)  # Remove the edge from the list

        # Temporarily remove the edge
        original_value = lil_graph[row, col]
        lil_graph[row, col] = 0
        lil_graph[col, row] = 0  # Ensure symmetry

        # Check if the graph is still connected
        if not ensure_connected(lil_graph) or lil_graph[row, :].nnz == 0 or lil_graph[:, col].nnz == 0:
            # If not connected, revert the deletion
            lil_graph[row, col] = original_value
            lil_graph[col, row] = original_value
        else:
            deletions += 1


    # Convert back to CSR, removing explicit zeros
    csr_graph = lil_graph.tocsr()

    rows, cols = csr_graph.nonzero()
    filtered_edges = [(i, j) for i, j in zip(rows, cols) if i > j]
    # print("number of edges deleted", edges_to_delete)
    # print("AFTER number of edges", len(filtered_edges))

    if len(filtered_edges) - len(protected_edges)!= num_deletable_edges - edges_to_delete:
        raise ValueError("Incorrect number of edges:", len(filtered_edges)- len(protected_edges),
                         "Should be:", num_deletable_edges - edges_to_delete)

    edge_df = pd.DataFrame(filtered_edges, columns=['source', 'target'])
    edge_list_folder = args.directory_map["edge_lists"]
    args.args_title = args.args_title + f"_edge_del_ratio_{delete_ratio}"
    edge_list_title = f"edge_list_{args.args_title}.csv"
    edge_df.to_csv(f"{edge_list_folder}/{edge_list_title}", index=False)



    args.edge_list_title = edge_list_title
    args.sparse_graph = csr_graph


    args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(csr_graph, None)
    args.mean_shortest_path = args.shortest_path_matrix.mean()
    args.num_edges = args.sparse_graph.nnz // 2
    print("Mean shortest path:", args.mean_shortest_path)



    return args, csr_graph


def replace_infinities_sparse(sparse_matrix):
    """
    Replace all infinity values in a sparse matrix with 1.
    """
    coo_matrix = sparse_matrix.tocoo()
    coo_matrix.data[np.isinf(coo_matrix.data)] = 1
    if isinstance(sparse_matrix, sp.csr_matrix):
        return sp.csr_matrix((coo_matrix.data, (coo_matrix.row, coo_matrix.col)), shape=sparse_matrix.shape)
    elif isinstance(sparse_matrix, sp.csc_matrix):
        return sp.csc_matrix((coo_matrix.data, (coo_matrix.row, coo_matrix.col)), shape=sparse_matrix.shape)
    else:
        raise ValueError("Unsupported sparse matrix format")

def rank_based_combination(community_scores, betweenness_scores):
    """
    Takes as input 2 score dictionaries and returns a combined score dictionary based on ranks (relative rather than absolute numbers)
    """
    edges = list(community_scores.keys())
    community_values = np.array([community_scores[edge] for edge in edges])
    betweenness_values = np.array([betweenness_scores[edge] for edge in edges])
    community_ranks = np.argsort(-community_values)
    betweenness_ranks = np.argsort(betweenness_values)
    community_ranks = community_ranks + 1
    betweenness_ranks = betweenness_ranks + 1
    combined_ranks = (community_ranks + betweenness_ranks) / 2.0

    combined_scores = {edge: rank for edge, rank in zip(edges, combined_ranks)}
    return combined_scores