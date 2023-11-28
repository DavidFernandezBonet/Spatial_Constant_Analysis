from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import KDTree
import pandas as pd
from collections import defaultdict
from scipy.spatial import Delaunay
import networkx as nx
import igraph as ig



class QualityMetrics:
    def __init__(self, original_points, reconstructed_points, k=15, threshold=1000):
        """
        :param k: the "k" of the knn metric
        :param threshold: the cpd threshold, mainly due to memory restrictions
        """
        self.k = k
        self.threshold = threshold
        
        self.original_points = original_points
        self.reconstructed_points = reconstructed_points


    def knn_metric(self, visualize=False):
        num_points = len(self.original_points)
        original_tree = KDTree(self.original_points)
        reconstructed_tree = KDTree(self.reconstructed_points)
        original_neighbors = original_tree.query(self.original_points, self.k + 1)[1][:, 1:]
        reconstructed_neighbors = reconstructed_tree.query(self.reconstructed_points, self.k + 1)[1][:, 1:]
        shared_neighbors = sum([len(set(original).intersection(set(reconstructed))) for original, reconstructed in zip(original_neighbors, reconstructed_neighbors)])

        # TODO: delete this or get in another function
        point_index = 50  # dummy index
        if visualize and point_index < num_points:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original points and neighbors
            axes[0].scatter(self.original_points[:, 0], self.original_points[:, 1], c='grey')
            axes[0].scatter(self.original_points[point_index, 0], self.original_points[point_index, 1], c='red')
            axes[0].scatter(self.original_points[original_neighbors[point_index], 0],
                            self.original_points[original_neighbors[point_index], 1], c='blue')
            axes[0].text(self.original_points[point_index, 0], self.original_points[point_index, 1], str(point_index),
                         fontsize=12, color='red')

            for neighbor_index in original_neighbors[point_index]:
                axes[0].text(self.original_points[neighbor_index, 0], self.original_points[neighbor_index, 1],
                             str(neighbor_index), fontsize=12, color='blue')

            axes[0].set_title('Original Point and Neighbors')

            # Reconstructed points and neighbors
            axes[1].scatter(self.reconstructed_points[:, 0], self.reconstructed_points[:, 1], c='grey')
            axes[1].scatter(self.reconstructed_points[point_index, 0], self.reconstructed_points[point_index, 1], c='red')
            axes[1].scatter(self.reconstructed_points[reconstructed_neighbors[point_index], 0],
                            self.reconstructed_points[reconstructed_neighbors[point_index], 1], c='blue')
            axes[1].text(self.reconstructed_points[point_index, 0], self.reconstructed_points[point_index, 1], str(point_index),
                         fontsize=12, color='red')

            for neighbor_index in reconstructed_neighbors[point_index]:
                axes[1].text(self.reconstructed_points[neighbor_index, 0], self.reconstructed_points[neighbor_index, 1],
                             str(neighbor_index), fontsize=12, color='blue')

            axes[1].set_title('Reconstructed Point and Neighbors')

            plt.show()
        return shared_neighbors / (self.k * num_points)

    def cpd_metric(self):
        num_points = len(self.original_points)
        if num_points > self.threshold:
            indices = np.random.choice(num_points, self.threshold, replace=False)
            self.original_points = self.original_points[indices]
            self.reconstructed_points = self.reconstructed_points[indices]
        original_distances = pdist(self.original_points)
        reconstructed_distances = pdist(self.reconstructed_points)
        correlation, _ = pearsonr(original_distances, reconstructed_distances)
        return correlation

    def compute_distortion(self, original_positions, reconstructed_positions):
        """
        Compute the distortion between the original and reconstructed 2D point clouds.

        :param original_positions: NumPy array of shape (N, 2) representing the original points
        :param reconstructed_positions: NumPy array of shape (N, 2) representing the reconstructed points
        :return: The distortion, a median value of the distances between original and reconstructed points
        """
        distances = [np.linalg.norm(original - reconstructed) for original, reconstructed in
                     zip(original_positions, reconstructed_positions)]
        distortion = np.median(distances)
        return distortion

    def evaluate_metrics(self,  distortion=False):
        knn_result = self.knn_metric()
        cpd_result = self.cpd_metric()
        quality_metrics = {'KNN': knn_result, 'CPD': cpd_result}
        if distortion:
            distortion_result = self.compute_distortion()
            quality_metrics.update({"Distortion": distortion_result})
        print(quality_metrics)
        return quality_metrics



class GTA_Quality_Metrics:
    # TODO: edge list as part of the input in init
    # TODO: GTA CPD
    # TODO: individual GTA CPD
    # TODO: metric comparison with original metrics
    def __init__(self, edge_list, reconstructed_points, k=15, threshold=1000):
        """
        :param k: the "k" of the knn metric
        :param threshold: the cpd threshold, mainly due to memory restrictions
        :param edge_list: DataFrame with two columns, each row representing an edge.
        """
        self.k = k
        self.threshold = threshold
        
        self.edge_list = edge_list
        self.reconstructed_points = reconstructed_points

        self.gta_knn_metric = None
        self.gta_cpd_metric = None

    def _extract_neighbors_from_edge_list(self):
        """
        Extracts the neighbors for each node from an edge list.

        """
        # If edge_list is a DataFrame, convert to list of tuples
        if isinstance(self.edge_list, pd.DataFrame):
            self.edge_list = list(self.edge_list.itertuples(index=False, name=None))
        neighbors_dict = defaultdict(set)
        for edge in self.edge_list:
            neighbors_dict[edge[0]].add(edge[1])
            neighbors_dict[edge[1]].add(edge[0])
        # Convert the neighbors dictionary to a list of neighbors
        max_index = max(neighbors_dict.keys())
        neighbors_list = [list(neighbors_dict.get(i, set())) for i in range(max_index + 1)]
        return neighbors_list

    def get_gta_knn(self, visualize=False):
        # Extract neighbors from edge list
        original_neighbors = self._extract_neighbors_from_edge_list()
        # Use KDTree for the reconstructed points
        reconstructed_tree = KDTree(self.reconstructed_points)
        reconstructed_neighbors = reconstructed_tree.query(self.reconstructed_points, self.k + 1)[1][:, 1:]
        individual_agt_knn = []
        for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):
            n = len(original)
            # TODO: this probably raises error if len(reconstructed) < len(original)
            individual_agt_knn.append(len(set(original).intersection(set(reconstructed[:n]))) / n)

        # Calculate the average agt_knn
        gta_knn = sum(individual_agt_knn) / len(individual_agt_knn)

        # Visualize the distribution if needed
        if visualize:
            plt.boxplot(individual_agt_knn)
            plt.title("Distribution of Individual GTA_KNN")
            plt.ylabel("Shared Neighbors Fraction")
            plt.savefig("individual_agt_knn_boxplot.png")

        return individual_agt_knn, gta_knn


    def get_gta_cpd(self):
        # Create an igraph graph
        igraph_graph = ig.Graph.TupleList(self.edge_list, directed=False)

        num_points = igraph_graph.vcount()

        # Determine if sampling is needed
        if num_points > self.threshold:
            indices = np.random.choice(num_points, self.threshold, replace=False)
            sampled_reconstructed_points = self.reconstructed_points[indices]
        else:
            indices = range(num_points)
            sampled_reconstructed_points = self.reconstructed_points

        # Necessary to get the proper ordering with the node IDs
        indices_as_indices = [igraph_graph.vs.find(name=n).index for n in indices]
        # Compute shortest paths for the sampled nodes
        graph_distances = igraph_graph.shortest_paths_dijkstra(source=indices_as_indices, target=indices_as_indices)

        graph_distances_flat = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                graph_distances_flat.append(graph_distances[i][j])
        graph_distances_flat = np.array(graph_distances_flat)

        # Distance matrix for sampled reconstructed points
        reconstructed_distances = pdist(sampled_reconstructed_points)
        print(reconstructed_distances[0])
        print(len(graph_distances_flat), len(reconstructed_distances))

        # Calculate Pearson correlation
        correlation, _ = pearsonr(graph_distances_flat, reconstructed_distances)
        return correlation
    
    def evaluate_metrics(self):
        _, self.gta_knn_metric = self.get_gta_knn()
        self.gta_cpd_metric = self.get_gta_cpd()
        quality_metrics = {'GTA_KNN': self.gta_knn_metric, 'GTA_CPD': self.gta_cpd_metric}

        print(quality_metrics)
        return quality_metrics
    # Usage

    #
    # metrics = QualityMetrics()
    # results = metrics.evaluate_metrics(points, all_predicted_positions)