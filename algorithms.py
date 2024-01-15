import igraph as ig
import numpy as np
import pandas as pd
import random
from nodevectors import GGVec
from nodevectors import Node2Vec
import umap
from sklearn import manifold
import multiprocessing

import pecanpy
import tempfile
from pecanpy import pecanpy as node2vec


def get_mean_shortest_path(igraph_graph, return_all_paths=False):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    G = igraph_graph
    # Compute the shortest paths for all pairs of nodes
    shortest_paths = G.shortest_paths()
    # Flatten the matrix to get a list of shortest path lengths
    path_lengths = [path for row in shortest_paths for path in row if path > 0]
    # Compute metrics
    mean_shortest_path = np.mean(path_lengths)

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

def add_random_edges_igraph(graph, num_edges_to_add):
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
    def __init__(self, graph, dim=2, node_embedding_mode="ggvec", manifold_learning_mode="UMAP",
                 node_embedding_components=64, manifold_learning_neighbors=15):
        """
        Initialize the ImageReconstruction object.

        :param graph: graph of type igraph #TODO: include more types
        :param dim: Target dimension for the UMAP reduction (2 or 3).
        """
        self.graph = graph
        self.dim = dim
        self.node_embedding_components = node_embedding_components
        self.manifold_learning_neighbors = manifold_learning_neighbors
        self.node_embedding_mode = node_embedding_mode
        self.manifold_learning_mode = manifold_learning_mode

    def compute_embeddings(self, args=None):
        """
        Compute node embeddings using ggvec.
        """
        # graph = ig.Graph.TupleList(self.edge_list, directed=False)
        if self.node_embedding_mode == 'ggvec':
            ggvec_model = GGVec(n_components=self.node_embedding_components)
            node_embeddings = ggvec_model.fit_transform(self.graph)

        # TODO: implement node2vec compatible with python 3.10 (problem with how nodevectors calls gensim, update nodevectors)
        # TODO: or try pecanpy again
        elif self.node_embedding_mode == "node2vec":
            # raise ValueError("Not implemented yet")
            ### nodevectors
            # node2vec_model = Node2Vec(n_components=self.node_embedding_components)
            # node_embeddings = node2vec_model.fit_transform(self.graph)

            ### pecanpy

            edge_list_folder = args.directory_map['edge_lists']
            edge_list_path = f'{edge_list_folder}/{args.edge_list_title}'
            # adj_mat = np.array(self.graph.toarray())
            # print("adj mat", adj_mat)
            # node_ids = [str(i) for i in range(adj_mat.shape[0])]
            # g = pecanpy.graph.SparseGraph.from_mat(self.graph, node_ids)
            #
            # indptr, indices, data = g.to_csr()  # convert to csr
            #
            # dense_mat = g.to_dense()  # convert to dense adjacency matrix
            #
            # g.save(edg_outpath)  # save the graph to an edge list file

            # initialize node2vec object, similarly for SparseOTF and DenseOTF

            # Temporary file without header
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
                with open(edge_list_path, 'r') as f:
                    next(f)  # Skip the header line
                    for line in f:
                        tmp_file.write(line)

            g = node2vec.PreComp(p=1, q=1, workers=4, verbose=True)

            # load graph from temporary edgelist file
            g.read_edg(tmp_file.name, weighted=False, directed=False, delimiter=',')
            g.preprocess_transition_probs()
            node_embeddings = g.embed()

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

    def write_positions(self, args, np_positions, output_path):
        # Write standard dataframe format:
        if args.dim == 2:
            positions_df = pd.DataFrame(np_positions, columns=['x', 'y'])
        elif args.dim == 3:
            positions_df = pd.DataFrame(np_positions, columns=['x', 'y', 'z'])
        else:
            raise ValueError("Please input a valid dimension")
        node_ids = range(args.num_points)
        positions_df['node_ID'] = node_ids
        # Define the output file path
        title = args.args_title
        output_file_path = f"{output_path}/positions_{title}.csv"

        # Write the DataFrame to a CSV file
        positions_df.to_csv(output_file_path, index=True)
    def reconstruct(self, do_write_positions=False, args=None):
        """
        Perform the entire reconstruction process and return the reconstructed points.
        """
        embeddings = self.compute_embeddings(args)

        if self.node_embedding_mode != 'landmark_isomap':
            reconstructed_points = self.reduce_dimensions(embeddings)
        else:  # in landmark isomap the result is already the reconstructed points
            reconstructed_points = embeddings

        if do_write_positions:
            if args == None:
                raise ValueError("Pass args to the function please")
            output_path = args.directory_map["reconstructed_positions"]
            self.write_positions(args, np_positions=np.array(reconstructed_points), output_path=output_path)
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


