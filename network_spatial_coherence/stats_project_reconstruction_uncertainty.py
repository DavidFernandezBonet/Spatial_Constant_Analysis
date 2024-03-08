from spatial_constant_analysis import *
import pandas as pd
import numpy as np
from collections import defaultdict
import numpy.linalg as la
from scipy.spatial import ConvexHull
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib



class EdgeList:
    def __init__(self):
        self.edges = {}  # Stores edge data

    def add_edge(self, node1, node2, mean_distance, std_distance):
        edge_key = tuple(sorted((node1, node2)))
        self.edges[edge_key] = {'mean_distance': mean_distance, 'std_distance': std_distance}

    def get_edge_data(self, node1, node2):
        edge_key = tuple(sorted((node1, node2)))
        return self.edges.get(edge_key, None)

class NodeTable:
    def __init__(self):
        self.nodes = {}  # Stores node data

    def add_node(self, node_id, position):
        # Node attributes
        self.nodes[node_id] = {'position': position, 'neighbors': set(), 'neighbor_stress': {},
                               'neighbor_stress_std': {}, 'total_stress': None,
                               'hull_area': None, 'knn_individual': None, 'gta_knn_individual': None, 'std_sum': None}

    def add_neighbor(self, node_id, neighbor_id):
        if node_id in self.nodes:
            self.nodes[node_id]['neighbors'].add(neighbor_id)

    def set_neighbor_stress(self, node_id, neighbor_id, stress):
        if node_id in self.nodes and neighbor_id in self.nodes[node_id]['neighbors']:
            self.nodes[node_id]['neighbor_stress'][neighbor_id] = stress

    def set_total_stress(self, node_id, total_stress):
        if node_id in self.nodes:
            self.nodes[node_id]['total_stress'] = total_stress


    def get_node_data(self, node_id):
        return self.nodes.get(node_id, None)

def euclidean_distance(pos1, pos2):
    return la.norm(pos1 - pos2)




# 1 - Reconstruct many times and gather data
# Store neighbor distances (best data structure for this?)
# Get mean and std for each negibhor


def run_reconstructions_and_calculate_stats(args, n_reconstructions, edges_df, sparse_graph, node_embedding_mode):
    # Dictionary to hold all distances for each edge across reconstructions
    all_distances = defaultdict(list)

    for _ in range(n_reconstructions):
        reconstructed_points, metrics = run_reconstruction(args, sparse_graph=sparse_graph, ground_truth_available=False,
                                                  node_embedding_mode=node_embedding_mode)
        reconstructed_points = standardize_point_cloud(reconstructed_points)

        for _, row in edges_df.iterrows():
            node1, node2 = row['source'], row['target']
            pos1, pos2 = reconstructed_points[node1], reconstructed_points[node2]
            distance = euclidean_distance(pos1, pos2)
            edge_key = tuple(sorted((node1, node2)))
            all_distances[edge_key].append(distance)

    # Calculate mean and std for each edge
    mean_and_std_distances = {edge: (np.mean(distances), np.std(distances)) for edge, distances in all_distances.items()}
    return mean_and_std_distances

def preprocess_neighbor_info(mean_and_std_distances):
    neighbor_info = defaultdict(dict)
    for (node1, node2), stats in mean_and_std_distances.items():
        neighbor_info[node1][node2] = stats
        neighbor_info[node2][node1] = stats  # for undirected graph
    return neighbor_info

def construct_data_structures(reconstructed_points, mean_and_std_distances):
    edge_list = EdgeList()
    node_table = NodeTable()

    # TODO: any discrepancy here with the node ids?
    # Add nodes
    for node_id, position in enumerate(reconstructed_points):
        node_table.add_node(node_id, position)

    # Add edges and neighbors
    for (node1, node2), (mean_distance, std_distance) in mean_and_std_distances.items():
        # print("edge", (node1, node2))
        edge_list.add_edge(node1, node2, mean_distance, std_distance)
        node_table.add_neighbor(node1, node2)
        node_table.add_neighbor(node2, node1)

    return edge_list, node_table

def calculate_stress(node_table, edge_list):
    for node_id, node_data in node_table.nodes.items():
        node_pos = node_data['position']
        stress_accumulator = np.zeros_like(node_pos)

        std_sum = 0
        for neighbor_id in node_data['neighbors']:
            neighbor_pos = node_table.get_node_data(neighbor_id)['position']
            direction = neighbor_pos - node_pos


            # Check if the direction vector is zero
            norm = np.linalg.norm(direction)
            if norm == 0:
                unit_direction = np.array([0, 0])
            else:
                unit_direction = direction / norm

            rec_dist = norm
            mean_distance = edge_list.get_edge_data(node_id, neighbor_id)['mean_distance']
            std_distance = edge_list.get_edge_data(node_id, neighbor_id)['std_distance']
            std_sum += std_distance

            neighbor_stress_std = std_distance
            node_table.nodes[node_id]['neighbor_stress_std'][neighbor_id] = neighbor_stress_std
            neighbor_stress = (rec_dist - mean_distance) * unit_direction
            node_table.set_neighbor_stress(node_id, neighbor_id, neighbor_stress)

            stress_accumulator += neighbor_stress

        total_stress_scalar = np.linalg.norm(stress_accumulator)  # If it is interesting I could also get it in vector form
        node_table.nodes[node_id]['std_sum'] = std_sum
        node_table.set_total_stress(node_id, total_stress_scalar)



def plot_graph_with_stresses(node_table, plot_folder, plot_ellipses=True):
    plt.figure(figsize=(10, 10))
    plt.gca().set_facecolor('black')  # Set background to black

    # Plot nodes and edges
    for node_id, node_data in node_table.nodes.items():
        x, y = node_data['position']
        plt.plot(x, y, 'o', c="white")  # Plot node

        # Plots edges
        for neighbor_id in node_data['neighbors']:
            neighbor_x, neighbor_y = node_table.get_node_data(neighbor_id)['position']
            plt.plot([x, neighbor_x], [y, neighbor_y], 'w-', lw=0.1, alpha=0.3)  # Plot edge


    if plot_ellipses:
        # Plot uncertainty clouds as ellipses
        for node_id, node_data in node_table.nodes.items():
            node_pos = np.array(node_data['position'])
            stresses = []
            std_devs = []

            for neighbor_id in node_data['neighbors']:
                stress = node_data['neighbor_stress'][neighbor_id]
                stress_std = node_data['neighbor_stress_std'][neighbor_id]

                stresses.append(node_pos + stress)
                std_devs.append(stress_std)

            stresses = np.array(stresses)
            std_devs = np.array(std_devs)

            if len(stresses) >= 2:
                cov = np.cov(stresses.T, aweights=std_devs)
                # Define a range for the standard deviations
                nstd_range = np.linspace(1, 3, num=5)  # Example: 1σ to 3σ in 5 steps
                alpha_values = np.linspace(0.4, 0, num=len(nstd_range))  # Decreasing alpha values for gradient effect


                plot_cov_ellipse(cov=cov, pos=node_pos, nstd_range=nstd_range, ax=plt.gca(),
                                 alpha_values=alpha_values, colormap="viridis")

    else:
        # Plot neighbor stresses as convex hulls
        for node_id, node_data in node_table.nodes.items():
            node_pos = np.array(node_data['position'])
            stresses = np.array([node_pos + stress for stress in node_data['neighbor_stress'].values()])

            if len(stresses) < 3:
                hull_area = 0
                node_table.nodes[node_id]['hull_area'] = hull_area
            else:
                hull = ConvexHull(stresses)
                hull_area = hull.area
                node_table.nodes[node_id]['hull_area'] = hull_area
                for simplex in hull.simplices:
                    plt.plot(stresses[simplex, 0], stresses[simplex, 1], 'r')


    # plt.title('Graph with Node Positions, Edges, and Neighbor Stresses')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')

    plt.savefig(f'{plot_folder}/stats_physics_ellipse_{args.original_title}.pdf')

def plot_node_attributes_correlation(node_table, plot_folder, node_attribute1="hull_area",
                                     node_attribute2="individual_knn"):

    # node attribute: hull_area, total_stress

    # Extract total stress magnitudes and corresponding gta_knn_individual values
    attribute1_magnitudes = []
    attribute2_magnitudes = []

    for node_id, node_data in node_table.nodes.items():
        attribute1 = node_data[node_attribute1]
        attribute2 = node_data[node_attribute2]

        attribute1_magnitudes.append(attribute1)
        attribute2_magnitudes.append(attribute2)


    # Compute correlation

    correlation = np.corrcoef(attribute1_magnitudes, attribute2_magnitudes)[0, 1]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(attribute1_magnitudes, attribute2_magnitudes, alpha=0.7)
    plt.title(f'Correlation between {node_attribute1} and {node_attribute2}: {correlation:.2f}')
    plt.xlabel(f'{node_attribute1}')
    plt.ylabel(f'{node_attribute2} Value')

    plt.savefig(f'{plot_folder}/stats_physics_correlation_{node_attribute2}_{node_attribute1}_{args.args_title}')

    return correlation


def add_list_elements_to_node_table(node_table, list_elements, attribute_name='knn_individual'):
    """
    Apply when the index of the list has direct correspondence with node_id (e.g. node 0 is first element of the list)
    """
    for node_id, element in enumerate(list_elements):
        if node_id in node_table.nodes:
            node_table.nodes[node_id][attribute_name] = element
        else:
            print(f"Warning: Node ID {node_id} not found in node_table")

def standardize_point_cloud(pc):
    # Centering
    centroid = np.mean(pc, axis=0)
    centered_pc = pc - centroid

    # Scaling
    max_distance = np.max(np.linalg.norm(centered_pc, axis=1))
    scaled_pc = centered_pc / max_distance

    return scaled_pc


def compute_correlation(list1, list2):
    """
    Compute the Pearson correlation coefficient between two lists.

    :param list1: First list of numerical values.
    :param list2: Second list of numerical values.
    :return: Pearson correlation coefficient.
    """
    correlation, _ = stats.pearsonr(list1, list2)
    return correlation


def plot_data(args, list1, list2, correlation):
    """
    Create a scatter plot of the two lists.

    :param list1: First list of numerical values.
    :param list2: Second list of numerical values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(list1, list2)
    plt.xlabel('Ground-Truth')
    plt.ylabel('GTA')
    plot_folder = args.directory_map['final_project']
    plt.title(f'Correlation: {correlation}')
    plt.savefig(f'{plot_folder}/correlation_gta_vs_gt_knn')


def compute_and_plot_correlation_2_lists(args, list1, list2):
    correlation = compute_correlation(list1, list2)
    plot_data(args, list1, list2, correlation)


def plot_cov_ellipse(cov, pos, nstd_range,alpha_values, ax=None, colormap='viridis', **kwargs):
    """
    Plots multiple error ellipses for the given covariance matrix (cov) and centroid position (pos)
    for a range of standard deviations (nstd_range), using a specified colormap.
    """
    if ax is None:
        ax = plt.gca()

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    colormap = matplotlib.cm.get_cmap(colormap)
    # Ensure a good spread of colors from the colormap for each standard deviation level

    color_indices = np.linspace(0, 1, len(nstd_range))

    for i, nstd in enumerate(nstd_range):
        width, height = 2 * nstd * np.sqrt(vals)
        facecolor = colormap(color_indices[i])  # Use color index to get color from colormap


        # Remove 'facecolor' from kwargs if it exists
        ellipse_kwargs = {k: v for k, v in kwargs.items() if k != 'facecolor'}

        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor=facecolor, alpha=alpha_values[i], **ellipse_kwargs)
        ax.add_artist(ellip)


##########
# Parameters
args = GraphArgs()
# Specify parameters
args.dim = 2
args.intended_av_degree = 10
args.num_points = 150
args.proximity_mode = "delaunay_corrected"

create_proximity_graph.write_proximity_graph(args)

edge_list_folder = args.directory_map["edge_lists"]
edges_df = pd.read_csv(f"{edge_list_folder}/edge_list_{args.args_title}.csv")



# node_embedding_mode: node2vec, ggvec, landmark_isomap
node_embedding_mode = 'node2vec'
n_reconstructions = 2
#
# # ### New stuff (introduce errors to see if the mappings are the same)  #TODO: remove this
# edge_list_folder = args.directory_map["edge_lists"]
# edges_df = pd.read_csv(f"{edge_list_folder}/edge_list_{args.args_title}.csv")
# edge_list_filtered = edges_df[~edges_df['source'].eq(0)]
# random_nodes = np.random.randint(1, args.num_points, 5)
# print("random error nodes linked to 0", random_nodes)
# new_edges = pd.DataFrame({'source': [0, 0, 1, 3, 6], 'target': random_nodes})
#
# # Add these new edges to the filtered DataFrame
# edge_list_modified = pd.concat([edge_list_filtered, new_edges], ignore_index=True)
# edge_list_modified.to_csv(f'{edge_list_folder}/edge_list_{args.args_title}.csv', index=False)
# edges_df = edge_list_modified
# # # -----------------------

sparse_graph, _ = load_graph(args, load_mode='sparse')


plot_folder = args.directory_map['final_project']
plot_original_or_reconstructed_image(args, image_type="original", edges_df=edges_df)
##########



mean_and_std_distances = run_reconstructions_and_calculate_stats(args=args, n_reconstructions=n_reconstructions,
                                                                 edges_df=edges_df, sparse_graph=sparse_graph,
                                                                 node_embedding_mode=node_embedding_mode)

# Preprocess neighbor information
neighbor_info = preprocess_neighbor_info(mean_and_std_distances)


# Run final reconstruction #TODO: change this? This is mainly for uncertainty, but it would make sense to use MDS with the mean distances (and std)
reconstructed_points, metrics = run_reconstruction(args, sparse_graph=sparse_graph, ground_truth_available=True,
                                          node_embedding_mode=node_embedding_mode)
reconstructed_points = standardize_point_cloud(reconstructed_points) #TODO: here we standardize the points so the distances become comparable across reconstructions



edge_list, node_table = construct_data_structures(reconstructed_points, mean_and_std_distances)




# 2 - Compute stress

# # Gather stress data for all nodes
# all_stress_data = gather_stress_data_for_all_nodes(reconstructed_points, neighbor_info)
#
# print(mean_and_std_distances)
# print(all_stress_data)
calculate_stress(node_table, edge_list)
plot_graph_with_stresses(node_table, plot_folder)

# knn_individual = metrics["ground_truth"].knn_individual
# gta_knn_individual = metrics["gta"].gta_knn_individual
# compute_and_plot_correlation_2_lists(args, knn_individual, gta_knn_individual)
# add_list_elements_to_node_table(node_table, knn_individual, attribute_name="knn_individual")
#
#
# # TODO: order is different or it is not correlated?
# plot_node_attributes_correlation(node_table, plot_folder=plot_folder, node_attribute1="hull_area",
#                                  node_attribute2="knn_individual")
# plot_node_attributes_correlation(node_table, plot_folder=plot_folder, node_attribute1="std_sum",
#                                  node_attribute2="knn_individual")
#
# print(node_table.nodes[0]['hull_area'])
# print(node_table.nodes[0]['knn_individual'])
# print('neighbors', node_table.nodes[0]['neighbors'])
# print(reconstructed_points[0])