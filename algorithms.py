import igraph as ig
import numpy as np
import pandas as pd
import random
from nodevectors import GGVec
import umap
def get_mean_shortest_path(igraph_graph):
    if not isinstance(igraph_graph, ig.Graph):
        raise ValueError("Graph is not of igraph type")
    G = igraph_graph
    # Compute the shortest paths for all pairs of nodes
    shortest_paths = G.shortest_paths()
    # Flatten the matrix to get a list of shortest path lengths
    path_lengths = [path for row in shortest_paths for path in row if path > 0]
    # Compute metrics
    mean_shortest_path = np.mean(path_lengths)
    return mean_shortest_path

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