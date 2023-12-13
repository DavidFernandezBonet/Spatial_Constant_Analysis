import igraph as ig
import numpy as np
import pandas as pd
import random
from nodevectors import GGVec
import umap
import multiprocessing

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

class ImageReconstruction:
    def __init__(self, graph, dim=2):
        """
        Initialize the ImageReconstruction object.

        :param graph: graph of type igraph #TODO: include more types
        :param dim: Target dimension for the UMAP reduction (2 or 3).
        """
        self.graph = graph
        self.dim = dim

    def compute_embeddings(self):
        """
        Compute node embeddings using ggvec.
        """
        # graph = ig.Graph.TupleList(self.edge_list, directed=False)
        ggvec_model = GGVec()
        node_embeddings = ggvec_model.fit_transform(self.graph)

        return node_embeddings

    def reduce_dimensions(self, embeddings):
        """
        Reduce the dimensionality of embeddings using UMAP.

        :param embeddings: High-dimensional embeddings of nodes.
        """
        umap_model = umap.UMAP(n_components=self.dim)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        return reduced_embeddings

    def reconstruct(self):
        """
        Perform the entire reconstruction process and return the reconstructed points.
        """
        embeddings = self.compute_embeddings()
        reconstructed_points = self.reduce_dimensions(embeddings)
        return reconstructed_points