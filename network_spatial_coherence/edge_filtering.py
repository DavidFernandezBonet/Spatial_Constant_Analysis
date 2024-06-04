
import numpy as np
import networkx as nx
from community import community_louvain
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import matplotlib.lines as mlines
import re
from utils import reload_graph_from_edge_list

def find_threshold_by_percentile(score_dict, percentile):
    """
    Computes the threshold score at the specified percentile from a score dictionary.

    :param score_dict: Dictionary where keys are edges (tuples) and values are scores (floats).
    :param percentile: Float representing the desired percentile (between 0 and 100).
    :return: Float value of the score at the given percentile.
    """
    scores = list(score_dict.values())
    threshold = np.percentile(scores, percentile)
    return threshold







class FilteringStrategy:
    """Common interface for all filtering strategies."""
    def apply(self, *args, **kwargs):
        raise NotImplementedError("Each strategy must implement the apply method.")


class CommunityFiltering(FilteringStrategy):
    def __init__(self, graph, args, n_community_iterations):
        self.graph = graph
        self.args = args
        self.n_community_iterations = n_community_iterations
        self.edges_within_communities = None
        self.edges_between_communities = None
        self.communities = None
        self.modularities = None
    def detect_communities(self, n_runs=1):
        csgraph = self.graph
        all_communities = []
        all_modularities = []
        G = nx.from_scipy_sparse_array(csgraph)
        for _ in range(n_runs):
            partition = community_louvain.best_partition(G)
            communities = np.zeros(G.number_of_nodes())
            node_label_to_position = {node: i for i, node in enumerate(G.nodes())}
            for node, community in partition.items():
                node_position = node_label_to_position[node]
                communities[node_position] = community
            modularity = community_louvain.modularity(partition, G)
            all_communities.append(communities)
            all_modularities.append(modularity)
        if n_runs == 1:
            return all_communities[0], all_modularities[0]
        else:
            self.communities = all_communities
            self.modularities = all_modularities
            return all_communities, all_modularities

    def count_false_edges_within_communities(self, args, communities):
        # Convert sparse graph to NetworkX graph
        G = nx.from_scipy_sparse_array(args.sparse_graph)
        node_to_community = {node: int(communities[i]) for i, node in enumerate(G.nodes())}

        false_edges_within_communities_count = 0
        edges_within_communities = set()
        edges_between_communities = set()

        # Iterate over all edges to categorize them
        for i, (node1, node2) in enumerate(G.edges()):
            # Check if both nodes are in the same community
            if node_to_community[node1] == node_to_community[node2]:
                edges_within_communities.add((node1, node2))
                # If this edge is also listed as a false edge, increase the count
                if (node1, node2) in args.false_edge_ids:
                    false_edges_within_communities_count += 1
            else:
                edges_between_communities.add((node1, node2))
        print("false edges within communities", false_edges_within_communities_count)
        return edges_within_communities, edges_between_communities

    def identify_consistent_edge_classifications(self, args, all_communities, within_threshold_ratio=0.3):
        # within_threshold_ratio = 0.4 by default
        G = nx.from_scipy_sparse_array(args.sparse_graph)

        # Initialize containers for tracking edge classifications across runs
        edges_within_communities_runs = [set() for _ in range(len(all_communities))]
        edges_between_communities_runs = [set() for _ in range(len(all_communities))]


        for run_idx, communities in enumerate(all_communities):
            node_to_community = {node: int(communities[i]) for i, node in enumerate(G.nodes())}

            # Classify edges for this run
            for edge in G.edges():
                node1, node2 = edge
                if node_to_community[node1] == node_to_community[node2]:
                    edges_within_communities_runs[run_idx].add(edge)
                else:
                    edges_between_communities_runs[run_idx].add(edge)

        ## This sets the edges that are more than 50% inside a community as probably true edges
        within_counter = Counter(edge for run in edges_within_communities_runs for edge in run)
        between_counter = Counter(edge for run in edges_between_communities_runs for edge in run)

        # Determine the predominant classification for each edge
        total_runs = len(edges_within_communities_runs)  # Assuming equal number of runs for both classifications
        consistently_within = set()
        consistently_between = set()

        edge_score_dict = {}
        for edge in set(within_counter) | set(between_counter):  # Union of all edges seen
            within_count = within_counter.get(edge, 0)
            between_count = between_counter.get(edge, 0)

            total_count = within_count + between_count
            score = within_count / total_count  # confidence we have on the edge
            edge_score_dict[edge] = score

            # Adjust the condition to use the within_threshold_ratio
            if score > within_threshold_ratio:
                consistently_within.add(edge)
            elif score <= within_threshold_ratio and between_count > 0:
                consistently_between.add(edge)

        likely_true_edges = consistently_within
        likely_false_edges = consistently_between
        return likely_true_edges, likely_false_edges, edge_score_dict
    def apply(self, graph):
        # Runs 1) community, 2) betweenness, and 3) rank filtering
        communities, modularity = self.detect_communities(n_runs=self.n_community_iterations)
        if self.n_community_iterations == 1:
            edges_within_communities, edges_between_communities = \
                self.count_false_edges_within_communities(self.args, communities)
            community_score_dict = None
        else:
            all_communities = communities  # several runs of the algorithm
            self.edges_within_communities, self.edges_between_communities, community_score_dict = (
                self.identify_consistent_edge_classifications(self.args, all_communities))
            self.identify_consistent_edge_classifications(self.args, all_communities, within_threshold_ratio=0.3)
        return community_score_dict

class BetweennessFiltering(FilteringStrategy):
    def apply(self, sparse_matrix):
        """
        Calculate and return the edge betweenness centrality for the graph,
        where the graph is constructed from a given sparse matrix.
        """
        # TODO: faster betweenness centrality maybe not using networkx
        G = nx.from_scipy_sparse_array(sparse_matrix)
        centrality = nx.edge_betweenness_centrality(G, normalized=True)
        return centrality

class RankFiltering(FilteringStrategy):
    def apply(self, community_scores, betweenness_scores):
        """
        Takes as input 2 score dictionaries and returns a combined score dictionary based on ranks (relative rather than absolute numbers)
        Careful because while community scores are good in increasing order, betweenness scores are good in decreasing order
        """
        edges = list(community_scores.keys())
        community_values = np.array([community_scores[edge] for edge in edges])
        betweenness_values = np.array([betweenness_scores[edge] for edge in edges])
        sorted_community = np.argsort(community_values)
        sorted_betweenness = np.argsort(-betweenness_values)

        # Create a rank array initialized with zeros
        community_ranks = np.zeros_like(community_values)
        betweenness_ranks = np.zeros_like(betweenness_values)

        community_ranks[sorted_community] = np.arange(len(community_values)) + 1
        betweenness_ranks[sorted_betweenness] = np.arange(len(betweenness_values)) + 1

        community_ranks = community_ranks + 1
        betweenness_ranks = betweenness_ranks + 1
        combined_ranks = community_ranks * (50 / 100) + betweenness_ranks * (50 / 100)

        combined_scores = {edge: rank for edge, rank in zip(edges, combined_ranks)}
        return combined_scores







class Thresholds:
    def __init__(self, score_dict):
        self.score_dict = score_dict
        self.score_threshold = None

    def find_threshold_by_percentile(self, percentile, top=True):
        """
        Finds the threshold at the given percentile.
        Set top=True for top percentile, False for bottom.
        """
        scores = np.array(list(self.score_dict.values()))
        if top:
            self.score_threshold = np.percentile(scores, 100 - percentile)
            return self.score_threshold
        else:
            self.score_threshold = np.percentile(scores, percentile)
            return self.score_threshold

    def find_threshold_by_user_defined(self, threshold):
        self.score_threshold = threshold
        return threshold

    def find_threshold_by_otsu(self):
        # TODO: other ways of finding threshold, this one seems to not work super well
        scores = np.array(list(self.score_dict.values()))
        hist, bin_edges = np.histogram(scores, bins=256, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Otsu's method: Between-class variance maximization
        total_weight = hist.sum()
        current_max, threshold = 0, 0
        sum_total, sum_foreground, weight_foreground = 0, 0, 0

        for i, value in enumerate(hist):
            sum_total += value * bin_centers[i]

        for i, value in enumerate(hist):
            weight_foreground += value
            if weight_foreground == 0:
                continue
            weight_background = total_weight - weight_foreground
            if weight_background == 0:
                break

            sum_foreground += value * bin_centers[i]
            mean_foreground = sum_foreground / weight_foreground
            mean_background = (sum_total - sum_foreground) / weight_background

            # Calculate between-class variance
            var_between = weight_foreground * weight_background
            var_between *= (mean_foreground - mean_background) ** 2

            # Check if new maximum found
            if var_between > current_max:
                current_max = var_between
                threshold = bin_centers[i]
        self.score_threshold = threshold
        return threshold

    def filter_edges_by_threshold(self, score_threshold):
        """
        Filters edges based on the provided score threshold.
        """
        passed_edges = [edge for edge, score in self.score_dict.items() if score > score_threshold]
        return passed_edges

    def visualize_threshold(self):
        """
        Visualizes the score distribution and the threshold.
        """
        plt.close('all')

        if self.score_threshold == None:
            raise ValueError("Threshold not found. Call find_threshold_by_otsu or find_threshold_by_percentile first.")
        scores = list(self.score_dict.values())
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, color='blue', alpha=0.7, label='Scores')

        # Highlighting the threshold
        plt.axvline(x=self.score_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.score_threshold:.2f}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution with Threshold')
        plt.legend()
        plt.show()


class Visualization:
    def __init__(self, graph_analysis, show_plots):
        self.graph_analysis = graph_analysis
        self.show_plots = show_plots

    def visualize_communities_positions(self, args, communities, modularity, edges_within_communities,
                                        edges_between_communities,
                                        edge_score_dict=None, betweenness_dict=None, rank_score_dict=None):
        import networkx as nx
        """
        Visualize the network with community colors, false edges, and modularity score.

        :param df: DataFrame with columns ['node_id', 'x', 'y'] for node positions.
        :param communities: Array or list of community labels corresponding to df['node_id'].
        :param false_edges: List of tuples (node_id_start, node_id_end) representing false edges.
        :param modularity: Modularity score of the community partition.
        """
        plt.close()
        plt.rcdefaults()

        def read_position_df(args, return_df=False):
            if hasattr(args, 'reconstruction_mode') and args.reconstruction_mode in args.args_title:
                old_args_title = args.args_title.replace(f"_{args.reconstruction_mode}",
                                                         "")
            else:
                old_args_title = args.args_title

            if args.proximity_mode == "experimental" and args.original_positions_available:
                filename = args.original_edge_list_title
                print("FILENAME", filename)
                match = re.search(r"edge_list_(.*?)\.csv", filename)
                if match:
                    extracted_part = match.group(1)
                    old_args_title = extracted_part
                else:
                    old_args_title = filename[:-4]

            original_points_path = f"{args.directory_map['original_positions']}/positions_{old_args_title}.csv"
            original_points_df = pd.read_csv(original_points_path)
            # Choose columns based on the dimension specified in args.dim
            if args.dim == 2:
                columns_to_read = ['x', 'y']
            elif args.dim == 3:
                columns_to_read = ['x', 'y', 'z']
            else:
                raise ValueError("Invalid dimension specified. Choose '2D' or '3D'.")

            # Read the specified columns from the DataFrame
            original_points_array = np.array(original_points_df[columns_to_read])

            if return_df:
                return original_points_df
            else:
                return original_points_array

        if args.node_ids_map_old_to_new is not None:
            original_points = read_position_df(args, return_df=True)
            original_points['node_ID'] = original_points['node_ID'].map(args.node_ids_map_old_to_new)
            original_points = original_points.dropna()
            original_points['node_ID'] = original_points['node_ID'].astype(int)
            original_points_df = original_points.sort_values(by='node_ID')
            original_points = original_points_df[['x', 'y']].to_numpy()
        else:
            original_position_folder = args.directory_map["original_positions"]
            original_points_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

            # original_points_df = read_position_df(args, return_df=True)

        # ### Plotting
        # Load positions DataFrame
        positions_df = original_points_df
        # node_ids_map_new_to_old = {v: k for k, v in args.node_ids_map_old_to_new.items()}
        # Load edges DataFrame
        edge_list_folder = args.directory_map["edge_lists"]
        edges_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")

        # TODO: this plots everything to do with community detection and false edges

        # # This one plots histograms depending on distances
        # plot_edge_communities_and_distances(edges_within_communities, edges_between_communities, original_positions_df=positions_df)
        if edge_score_dict is not None:
            optimal_comm_thresh = self.plot_edge_scores_vs_distances(args, edge_score_dict, original_positions_df=positions_df,
                                          score_interpretation='negative')

        if betweenness_dict is not None:
            optimal_betw_thresh = self.plot_edge_scores_vs_distances(args, betweenness_dict, original_positions_df=positions_df,
                                          score_interpretation='positive')

        if rank_score_dict is not None:
            optimal_rank_thresh = self.plot_edge_scores_vs_distances(args, rank_score_dict, original_positions_df=positions_df,
                                          score_interpretation='negative')

        self.plot_false_true_edges_percentages(args, edges_within_communities, edges_between_communities)
        self.plot_edge_categories_and_distances(edges_within_communities, edges_between_communities, args.false_edge_ids,
                                           original_positions_df=positions_df)
        # Convert positions_df to a dictionary for efficient access
        positions_dict = positions_df.set_index('node_ID')[['x', 'y']].T.to_dict('list')

        # Map communities to colors
        if isinstance(communities[0], (np.ndarray, np.generic)):
            communities = communities[0]

        communities_dict = {i: community for i, community in enumerate(communities)}
        unique_communities = set(communities_dict.values())
        # # community_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))  # Adjust color map as needed
        # # community_to_color = {community: color for community, color in zip(unique_communities, community_colors)}
        community_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
        community_to_color = {community: color for community, color in zip(unique_communities, community_colors)}

        plt.figure(figsize=(20, 30))

        # Plot original points with community colors
        for node_ID, (x, y) in positions_dict.items():
            community_id = communities_dict[node_ID]
            plt.plot(x, y, 'o', color=community_to_color[community_id])

        # Plot edges
        for _, row in edges_df.iterrows():
            source_new_id = row['source']
            target_new_id = row['target']

            if source_new_id in positions_dict and target_new_id in positions_dict:
                source_pos = positions_dict[source_new_id]
                target_pos = positions_dict[target_new_id]

                # Check if the edge is a false edge
                is_false_edge = (row['source'], row['target']) in args.false_edge_ids or (
                    row['target'], row['source']) in args.false_edge_ids

                edge_linewidth = 0.5
                edge = tuple(sorted((row['source'], row['target'])))
                if edge in edges_within_communities:
                    edge_color = "blue"
                elif edge in edges_between_communities:
                    edge_color = "green"
                    edge_linewidth = 3
                else:
                    raise ValueError(f"Edge {edge} not found in edges_within_communities or edges_between_communities")

                if is_false_edge:
                    edge_color = "red"
                    edge_linewidth = 1
                edge_alpha = 0.1 if is_false_edge else 1

                # distance = np.sqrt((source_pos[0] - target_pos[0]) ** 2 + (source_pos[1] - target_pos[1]) ** 2)
                # if distance > 0.5:
                #     if (row['source'], row['target']) in edges_within_communities:
                #         edge_linewidth += 2  # Increase by 0.5 or any other value you see fit

                plt.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], color=edge_color,
                         alpha=edge_alpha,
                         linewidth=edge_linewidth)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Modularity: {modularity}')

        # Assuming the rest of your plotting code is in place and you've already created 'community_to_color'

        # Create a list of proxy artists for the legend
        legend_handles = [mlines.Line2D([], [], color=community_to_color[community], marker='o', linestyle='None',
                                        markersize=10, label=f'Community {community}') for community in
                          unique_communities]

        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Make sure the plot is displayed properly with the legend outside
        plt.tight_layout()

        print("Modularity: ", modularity)
        if self.show_plots:
            plt.show()
        return optimal_comm_thresh, optimal_betw_thresh, optimal_rank_thresh

    def plot_edge_communities_and_distances(self, edges_within_communities, edges_between_communities, original_positions_df):
        # Function to compute Euclidean distance between two points
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Initialize lists to store distances
        distances_within = []
        distances_between = []

        # Create a dictionary for quick node ID to position lookup
        positions_dict = original_positions_df.set_index('node_ID').to_dict('index')

        # Compute distances for edges within communities
        for edge in edges_within_communities:
            node1, node2 = edge
            pos1 = positions_dict[node1]
            pos2 = positions_dict[node2]
            distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
            distances_within.append(distance)

        # Compute distances for edges between communities
        for edge in edges_between_communities:
            node1, node2 = edge
            pos1 = positions_dict[node1]
            pos2 = positions_dict[node2]
            distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
            distances_between.append(distance)

        # Plotting the histograms
        plt.figure(figsize=(10, 6))
        plt.hist(distances_within, bins=20, alpha=0.5, label='Within Communities')
        plt.hist(distances_between, bins=20, alpha=0.5, label='Between Communities')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Distances')
        plt.legend(loc='upper right')
        if self.show_plots:
            plt.show()

    def plot_edge_scores_vs_distances(self, args, edge_score_dict, original_positions_df, score_interpretation='positive'):
        # Function to compute Euclidean distance between two points
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Create a dictionary for quick node ID to position lookup
        positions_dict = original_positions_df.set_index('node_ID').to_dict('index')

        # Initialize lists to store scores and distances
        scores = []
        distances = []
        colors = []
        labels = []  # This will hold the binary labels needed for ROC computation

        # Compute distances and retrieve scores for edges
        for edge, score in edge_score_dict.items():
            node1, node2 = edge
            pos1 = positions_dict[node1]
            pos2 = positions_dict[node2]
            distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])

            if edge in args.false_edge_ids:
                colors.append('red')
                labels.append(1)  # False edge
            else:
                colors.append('blue')
                labels.append(0)  # True edge

            distances.append(distance)
            if score_interpretation == 'negative':
                # Transform score if higher scores indicate a false edge
                score = max(edge_score_dict.values()) + 1 - score
            scores.append(score)

        # Set up the figure and the subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Now three subplots

        # Scatter plot of scores vs distances
        axs[0].scatter(distances, scores, alpha=0.5, color=colors)
        axs[0].set_xlabel('Distance')
        axs[0].set_ylabel('Score')
        axs[0].set_title('Plot of Score vs. Distance for Edges')

        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        axs[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        axs[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_ylim([0.0, 1.05])
        axs[1].set_xlabel('False Positive Rate')
        axs[1].set_ylabel('True Positive Rate')
        axs[1].set_title('Receiver Operating Characteristic')
        axs[1].legend(loc="lower right")

        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        axs[2].plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        axs[2].set_xlabel('Recall')
        axs[2].set_ylabel('Precision')
        axs[2].set_title('Precision-Recall Curve')
        axs[2].legend(loc="best")

        # Display the plot
        plt.tight_layout()
        if self.show_plots:
            plt.show()


        f_scores = (2 * precision * recall) / (np.maximum(precision + recall, 1e-8))
        optimal_idx = np.argmax(f_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]

        #### Additional plotting for a specific threshold of the PR curve
        # Plotting the Precision-Recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {auc(recall, precision):.2f})')
        plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, edgecolor='k', zorder=5)
        plt.text(recall[optimal_idx], precision[optimal_idx], f'  Threshold={optimal_threshold:.4f}',
                 verticalalignment='bottom', horizontalalignment='right')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Optimal Threshold Highlighted')
        plt.legend()

        if self.show_plots:
            plt.show()

        predictions = [1 if x >= optimal_threshold else 0 for x in scores]
        conf_matrix = confusion_matrix(labels, predictions)

        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix at Optimal Threshold: {:.2f}'.format(optimal_threshold))
        plt.xticks([0.5, 1.5], ['True Edge', 'False Edge'], rotation=0)
        plt.yticks([0.5, 1.5], ['True Edge', 'False Edge'], rotation=0)
        if self.show_plots:
            plt.show()

        if score_interpretation == 'negative':
            optimal_threshold = max(edge_score_dict.values()) + 1 - optimal_threshold
        return optimal_threshold

    def plot_false_true_edges_percentages(self, args, edges_within_communities, edges_between_communities):
        false_edges = set(args.false_edge_ids)  # Assuming this is already in the form of tuples (node1, node2)

        false_within = sum(edge in false_edges for edge in edges_within_communities)
        true_within = len(edges_within_communities) - false_within
        false_between = sum(edge in false_edges for edge in edges_between_communities)
        true_between = len(edges_between_communities) - false_between

        total_false_edges = len(false_edges)

        # Calculate the percentages for the main plot
        total_within = len(edges_within_communities)
        total_between = len(edges_between_communities)
        percentages_within = [false_within / total_within * 100, true_within / total_within * 100]
        percentages_between = [false_between / total_between * 100, true_between / total_between * 100]

        # Calculate the percentages of false edges that are 'within' and 'between'
        false_within_percentage = (false_within / total_false_edges) * 100 if total_false_edges > 0 else 0
        false_between_percentage = (false_between / total_false_edges) * 100 if total_false_edges > 0 else 0

        # Setup for subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Main plot with stacked bars
        categories = ['Within Communities', 'Between Communities']
        axs[0].bar(categories, [percentages_within[0], percentages_between[0]], label='False Edges', color='r')
        axs[0].bar(categories, [percentages_within[1], percentages_between[1]],
                   bottom=[percentages_within[0], percentages_between[0]], label='True Edges', color='g')
        axs[0].set_ylabel('Percentage')
        axs[0].set_title('Percentage of False and True Edges')
        axs[0].legend()

        # Subplot for percentage of false edges that are within/between
        axs[1].bar(['False Edges Distribution'], [false_within_percentage], label='Within', color='blue')
        axs[1].bar(['False Edges Distribution'], [false_between_percentage], bottom=[false_within_percentage],
                   label='Between', color='orange')
        axs[1].set_ylabel('Percentage')
        axs[1].set_title('Distribution of False Edges')
        axs[1].legend()

        plt.tight_layout()
        if self.show_plots:
            plt.show()

    def plot_edge_categories_and_distances(self, edges_within_communities, edges_between_communities,
                                           false_edges, original_positions_df):
        def euclidean_distance(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        positions_dict = original_positions_df.set_index('node_ID').to_dict('index')
        categories = ['False Within', 'True Within', 'False Between', 'True Between']
        distances = {category: [] for category in categories}

        for edge in edges_within_communities.union(edges_between_communities):
            node1, node2 = edge
            pos1, pos2 = positions_dict[node1], positions_dict[node2]
            distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
            if edge in edges_within_communities:
                if edge in false_edges:
                    distances['False Within'].append(distance)
                else:
                    distances['True Within'].append(distance)
            else:
                if edge in false_edges:
                    distances['False Between'].append(distance)
                else:
                    distances['True Between'].append(distance)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()

        for i, category in enumerate(categories):
            axs[i].hist(distances[category], bins=20, alpha=0.7, label=category)
            axs[i].set_title(category)
            axs[i].set_xlabel('Distance')
            axs[i].set_ylabel('Frequency')
            axs[i].legend()

        plt.tight_layout()
        if self.show_plots:
            plt.show()


class EdgeFiltering:
    def __init__(self, graph, args, show_plots=False, n_community_iterations=100):
        self.graph = graph
        self.args = args
        self.community_detection_filtering = CommunityFiltering(self.graph, self.args,
                                                                n_community_iterations=n_community_iterations)
        self.betweeness_filtering = BetweennessFiltering()
        self.rank_filtering = RankFiltering()
        self.visualization = Visualization(self.community_detection_filtering, show_plots)
        self.gt_available = args.proximity_mode != 'experimental'
        self.show_plots = show_plots


    def run_analysis(self):

        community_score_dict = self.community_detection_filtering.apply(self.graph)
        betweenness_score_dict = self.betweeness_filtering.apply(self.graph)
        rank_score_dict = self.rank_filtering.apply(community_score_dict, betweenness_score_dict)

        print("HOLA 1")
        print("len noised edges", len(rank_score_dict))
        if self.gt_available:
            edges_within_communities = self.community_detection_filtering.edges_within_communities
            edges_between_communities = self.community_detection_filtering.edges_between_communities
            communities = self.community_detection_filtering.communities
            modularity = self.community_detection_filtering.modularities
            optimal_comm_thresh, optimal_betw_thresh, optimal_rank_thresh =(
                self.visualization.visualize_communities_positions(self.args, communities, modularity,
                                                               edges_within_communities, edges_between_communities,
                                        community_score_dict, betweenness_score_dict, rank_score_dict))


        print("HOLAAAA")
        rank_score_threshold = Thresholds(rank_score_dict)

        if self.gt_available:
            threshold = rank_score_threshold.find_threshold_by_user_defined(threshold=optimal_rank_thresh)
        else:
            threshold = rank_score_threshold.find_threshold_by_percentile(percentile=99)
        # threshold = rank_score_threshold.find_threshold_by_otsu()


        # rank_score_threshold.visualize_threshold()

        print("threshold: ", threshold)

        denoised_edges = rank_score_threshold.filter_edges_by_threshold(score_threshold=threshold)
        print("len noised edges", len(rank_score_threshold.score_dict))
        print("denoised edges", denoised_edges)
        print("len denoised edges", len(denoised_edges))

        G = nx.Graph()
        G.add_edges_from(denoised_edges)
        component_lengths = [len(component) for component in nx.connected_components(G)]
        print("Lengths of all components:", component_lengths)

        largest_component = max(nx.connected_components(G), key=len)

        # Create a subgraph of the largest component
        subgraph = G.subgraph(largest_component)

        # Get the edge list of the largest component
        largest_component_edges = list(subgraph.edges())

        # print("BEFORE", self.args.node_ids_map_old_to_new)
        # self.args.node_ids_map_old_to_new = {old: new for old, new in self.args.node_ids_map_old_to_new.items() if new in largest_component}
        # self.args.num_points = len(largest_component)
        #
        # print("AFTER", self.args.node_ids_map_old_to_new)
        # print(len(self.args.node_ids_map_old_to_new))

        # returns what they should be true edges
        return largest_component_edges, self.args
        # return denoised_edges


# Usage
if __name__ == "__main__":
    csgraph = nx.to_scipy_sparse_matrix(nx.gnp_random_graph(10, 0.5))
    edge_filtering = EdgeFiltering(csgraph)
    edge_filtering.run_analysis()