import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import math
from collections import defaultdict
import pandas as pd
import os
import random
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

def generate_random_points(num_points, L, dim):
    """
    Generate 'num_points' random 2D/3D points within 'L' range (square or cube)
    Returns:
    - points: list of tuples, each tuple representing the coordinates (x, y)
    """

    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")
    points = np.random.rand(num_points, dim) * L
    points = [tuple(point) for point in points]
    return points


def generate_random_points_anomaly(num_points, L, dim, hotspot=None, anomaly_strength=1.0):
    """
    Generate 'num_points' random 2D/3D points within 'L' range (square or cube) with density anomalies.

    Args:
    - num_points: Number of points to generate.
    - L: Range for each dimension.
    - dim: Dimension of points (2D or 3D).
    - hotspot: The center of the density anomaly (tuple of length 'dim').
               If None, a random hotspot is generated within bounds.
    - anomaly_strength: A multiplier for the density in the hotspot.
                        Higher values create stronger anomalies.

    Returns:
    - points: List of tuples, each tuple representing the coordinates (x, y) or (x, y, z).
    """
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")

    # Generate a random hotspot within bounds if none is provided
    if hotspot is None:
        hotspot = np.random.rand(dim) * L

    points = []
    for _ in range(num_points):
        if np.random.rand() < anomaly_strength * 0.1:
            # Generate points near the hotspot
            point = np.random.normal(loc=hotspot, scale=L*0.1, size=dim)
            # Ensure points are within bounds
            point = np.clip(point, 0, L)
        else:
            # Generate points uniformly
            point = np.random.rand(dim) * L

        points.append(tuple(point))

    return points

def generate_random_points_in_circle_or_sphere(num_points, R, dim):
    """
    Generate 'num_points' random 2D/3D points within a radius 'R' (circle or sphere)
    Returns:
    - points: list of tuples, each tuple representing the coordinates (x, y) or (x, y, z)
    """
    if dim not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3.")
    points = []
    while len(points) < num_points:
        # Generate points in a square/cube of side length 2R, centered at origin
        point = np.random.uniform(-R, R, dim)
        # Check if the point is inside the circle/sphere
        if np.sum(point**2) <= R**2:
            points.append(tuple(point))

    return points

def generate_square_lattice(args):
    """ Generate a square lattice of points. """
    points_per_side = int(np.round(args.num_points ** (1 / args.dim)))
    points = np.linspace(0, args.L, points_per_side)
    return np.array(np.meshgrid(*([points] * args.dim))).T.reshape(-1, args.dim)


def compute_knn_graph(positions, k):
    """
    Computes the k-nearest neighbors graph.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    # Remove distance to self
    return np.delete(distances, 0, 1), np.delete(indices, 0, 1)

def compute_epsilon_ball_graph(positions, radius):
    """
    Computes the epsilon-ball graph.
    """
    nbrs = NearestNeighbors(radius=radius).fit(positions)
    distances, indices = nbrs.radius_neighbors(positions, sort_results=True)
    # Remove self distances and indices
    return [np.delete(dist, 0, 0) for dist in distances], [np.delete(ind, 0, 0) for ind in indices]



def epsilon_bipartite(positions, radius, ratio=2):
    """
    ratio controls which proportion there is between the two types.
    ratio = 2 --> 1:1
    ratio = 3 --> 1:2
    ratio = 4 --> 1:3
    """

    positions = np.array(positions)
    total_len = len(positions)
    indices = np.arange(len(positions))
    half = len(indices) // ratio
    bottom_indices, top_indices = indices[:half], indices[half:]

    # Extract positions for bottom and top sets
    bottom_positions = positions[bottom_indices]
    top_positions = positions[top_indices]

    # epsilon-ball for bottom set using top set
    nbrs_bottom = NearestNeighbors(radius=radius).fit(top_positions)
    distances_bottom, indices_bottom = nbrs_bottom.radius_neighbors(bottom_positions, sort_results=True)
    indices_bottom += half  # Offset the indices

    # epsilon-ball for top set using bottom set
    nbrs_top = NearestNeighbors(radius=radius).fit(bottom_positions)
    distances_top, indices_top = nbrs_top.radius_neighbors(top_positions, sort_results=True)


    ### TODO: Why do these distances are not self included (first one should be 0)?
    distances = np.zeros(total_len, dtype=object)
    indices_combined = np.zeros(total_len, dtype=object)

    distances[:half] = distances_bottom
    distances[half:] = distances_top

    indices_combined[:half] = indices_bottom
    indices_combined[half:] = indices_top

    # # print("distances bottom", distances_bottom)
    # len_dist = []
    # for dist in distances:
    #     len_dist.append(len(dist))
    #     # if len(dist) < 2:
    #         # print(len(dist))
    #         # print("Ha passat")
    #         # print(dist)
    # mean_neig = sum(len_dist)/len(len_dist)
    # len_dist = np.array(len_dist)
    # median = np.median(len_dist)
    # std = np.std(len_dist)
    # # print("MEAN NEIGHBORS", mean_neig)
    # # print("MEDIAN NEIGH", median)
    # # print("STD NEIGH", std)


    return distances, indices_combined

def knn_bipartite(positions, k, ratio=2):
    """
    ratio controls which proportion there is between the two types.
    ratio = 2 --> 1:1
    ratio = 3 --> 1:2
    ratio = 4 --> 1:3
    """

    positions = np.array(positions)
    total_len = len(positions)
    indices = np.arange(len(positions))

    ## This partitions 50-50 (if ratio=2) the bipartite types. It could also be 66/33 if ratio =3 for example
    half = len(indices) // ratio
    bottom_indices, top_indices = indices[:half], indices[half:]

    # Extract positions for bottom and top sets
    bottom_positions = positions[bottom_indices]
    top_positions = positions[top_indices]

    # epsilon-ball for bottom set using top set
    nbrs_bottom = NearestNeighbors(n_neighbors=k).fit(top_positions)
    distances_bottom, indices_bottom = nbrs_bottom.kneighbors(bottom_positions)
    indices_bottom += half  # Offset the indices

    # epsilon-ball for top set using bottom set
    nbrs_top = NearestNeighbors(n_neighbors=k).fit(bottom_positions)
    distances_top, indices_top = nbrs_top.kneighbors(top_positions)

    # Create 2D arrays
    distances = np.zeros((total_len, k))
    indices_combined = np.zeros((total_len, k), dtype=int)

    distances[:half] = distances_bottom
    distances[half:] = distances_top

    indices_combined[:half] = indices_bottom
    indices_combined[half:] = indices_top
    distances = [np.delete(dist, 0, 0) for dist in distances]
    indices_combined = [np.delete(ind, 0, 0) for ind in indices_combined]

    return distances, indices_combined


def get_delaunay_neighbors_set_format(tess):
    neighbors = defaultdict(set)

    for simplex in tess.simplices:
        for idx in simplex:
            other = set(simplex)
            other.remove(idx)
            neighbors[idx] = neighbors[idx].union(other)
    return neighbors

def from_set_to_nparray(set_item):
    nparray_item = [[] for element in set_item]
    # Fill array with set values in an ordered manner (order provided by key)
    for (k,v) in set_item.items():
        value_list = list(v)
        nparray_item[k] = value_list
    # Transform lists into arrays
    nparray_item = [np.array(element) for element in nparray_item]
    nparray_item = np.array(nparray_item, dtype=object)
    return nparray_item
def get_delaunay_neighbors(positions):
    tess = Delaunay(positions)  # positions format np.array([[0,0], [1,2], ...]) . Get tessalation done
    set_neighbors = get_delaunay_neighbors_set_format(tess)
    indices = from_set_to_nparray(set_neighbors)  # list of neighbor indices with np.array() format
    distances = [np.array([math.dist(positions[i], positions[j]) for j in indices[i]]) for i in range(len(indices))]

    return distances, indices


def get_delaunay_neighbors_corrected_simple_threshold(positions):
    tess = Delaunay(positions)  # Delaunay tessellation of the positions
    set_neighbors = get_delaunay_neighbors_set_format(tess)
    indices = from_set_to_nparray(set_neighbors)  # list of neighbor indices with np.array() format

    #### Using threshold distance
    filtered_distances = []
    filtered_indices = []

    if len(positions[0]) == 2:  # 2D
        distance_threshold = 0.1  # TODO: Change this according to the size of the square (and density)!
    else:
        distance_threshold = 0.2
    for i in range(len(indices)):
        neighbor_distances = np.array([math.dist(positions[i], positions[j]) for j in indices[i]])
        within_threshold_mask = neighbor_distances <= distance_threshold
        filtered_distances.append(neighbor_distances[within_threshold_mask])
        filtered_indices.append(np.array(indices[i])[within_threshold_mask])

    return filtered_distances, filtered_indices


def get_delaunay_neighbors_corrected(positions):
    #TODO: check that this works properly, deleting top 5% highest distances
    tess = Delaunay(positions)  # Delaunay tessellation of the positions
    set_neighbors = get_delaunay_neighbors_set_format(tess)
    indices = from_set_to_nparray(set_neighbors)  # list of neighbor indices with np.array() format

    # Compute all distances
    all_distances = []
    for i in range(len(indices)):
        neighbor_distances = np.array([math.dist(positions[i], positions[j]) for j in indices[i]])
        all_distances.extend(neighbor_distances)

    # Sort and find the 95th percentile distance
    all_distances_sorted = np.sort(all_distances)
    top_5_percentile_distance = np.percentile(all_distances_sorted, 98)

    # Filter distances based on the top 5% threshold
    filtered_distances = []
    filtered_indices = []

    for i in range(len(indices)):
        neighbor_distances = np.array([math.dist(positions[i], positions[j]) for j in indices[i]])
        top_5_percent_mask = neighbor_distances <= top_5_percentile_distance
        filtered_distances.append(neighbor_distances[top_5_percent_mask])
        filtered_indices.append(np.array(indices[i])[top_5_percent_mask])

    return filtered_distances, filtered_indices


def compute_proximity_decay_graph(positions, decay_mode='decay_exp', quantile_scale=0.05, power_law_exp=4):
    """
    Computes the proximity decay graph with adjusted decay functions to ensure probabilities are between 0 and 1.
    """
    def decay_powerlaw(distance, distance_scale, exp=2):
        normalized_distance = distance / distance_scale
        probabilities = 1 / (np.power(normalized_distance, exp) + 0.001)
        return probabilities

    def decay_exp(distance, smooth_exponent, scale):
        # Exponential decay naturally falls between 0 and 1
        scaled_distance = distance / scale
        return np.exp(-(scaled_distance)**smooth_exponent)

    print("QUANTILE SCALE: ", quantile_scale)
    # Calculate pairwise distances
    raw_distances = squareform(pdist(positions))
    distances = raw_distances.flatten()

    distance_scale = np.quantile(distances, quantile_scale)

    # Select decay function and compute probabilities
    if decay_mode == 'power_law':
        probabilities = decay_powerlaw(raw_distances, exp=power_law_exp, distance_scale=distance_scale)
    else:  # exponential decay
        probabilities = decay_exp(raw_distances, scale=distance_scale, smooth_exponent=2)

    # # Ensure probabilities do not exceed the bounds [0, 1]
    # probabilities = np.clip(probabilities, 0, 1)
    probabilities /= np.max(probabilities)
    np.fill_diagonal(probabilities, 0)  ## This avoid self edges

    import matplotlib.pyplot as plt


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, probabilities, alpha=0.5)

    plt.xlabel('Distance')
    plt.ylabel('Probability')
    plt.grid(True)
    # plt.show()


    n = len(positions)
    indices = np.transpose(np.triu_indices(n, k=1))
    selected_probabilities = probabilities[indices[:, 0], indices[:, 1]]
    rand_samples = np.random.rand(len(selected_probabilities))
    edge_selection = rand_samples < selected_probabilities
    selected_indices = indices[edge_selection]
    selected_distances = raw_distances[selected_indices[:, 0], selected_indices[:, 1]]


    # Create an adjacency list using a list of lists
    adjacency_list = [[] for _ in range(n)]
    adjacency_distances = [[] for _ in range(n)]
    for edge, distance in zip(selected_indices, selected_distances):
        adjacency_distances[edge[0]].append(distance)
        adjacency_distances[edge[1]].append(distance)
        adjacency_list[edge[0]].append(edge[1])
        adjacency_list[edge[1]].append(edge[0])  # For undirected graph

    # Convert the list of lists into a numpy object array
    adjacency_array = np.array(adjacency_list, dtype=object)
    adjacency_array_distances = np.array(adjacency_distances, dtype=object)


    print("len selected distances", len(selected_distances))


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(selected_distances, bins=50, alpha=0.75)
    plt.xlabel('Edge Distance')
    plt.ylabel('Frequency')
    # plt.show()

    return adjacency_array_distances, adjacency_array


def compute_graph_data(positions, decay_mode='decay_exp', quantile_scale=0.05, power_law_exp=4):
    def decay_powerlaw(distance, scale, exp):
        normalized_distance = distance / scale
        return 1 / (np.power(normalized_distance, exp) + 0.001)

    def decay_exp(distance, scale, smooth_exponent):
        scaled_distance = distance / scale
        return np.exp(-(scaled_distance)**smooth_exponent)

    # Calculate pairwise distances
    distances = squareform(pdist(positions))
    distance_scale = np.quantile(distances.flatten(), quantile_scale)

    if decay_mode == 'power_law':
        probabilities = decay_powerlaw(distances, distance_scale, power_law_exp)
    else:
        probabilities = decay_exp(distances, distance_scale, 2)

    probabilities /= np.max(probabilities)
    np.fill_diagonal(probabilities, 0)  # This avoid self edges

    n = len(positions)
    indices = np.transpose(np.triu_indices(n, k=1))
    selected_probabilities = probabilities[indices[:, 0], indices[:, 1]]
    rand_samples = np.random.rand(len(selected_probabilities))
    edge_selection = rand_samples < selected_probabilities
    selected_indices = indices[edge_selection]
    selected_distances = distances[selected_indices[:, 0], selected_indices[:, 1]]

    return distances.flatten(), probabilities.flatten(), selected_distances


def plot_decay_effects(args, positions, quantile_scales, decay_modes, power_law_exp=4):
    fig1, axes1 = plt.subplots(nrows=len(quantile_scales), ncols=len(decay_modes), figsize=(15, 10), sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(nrows=len(quantile_scales), ncols=len(decay_modes), figsize=(15, 10), sharex=True, sharey=False)


    for i, scale in enumerate(quantile_scales):
        for j, mode in enumerate(decay_modes):
            print("doing this")
            distances, probabilities, edge_distances = compute_graph_data(positions, decay_mode=mode,
                                                                          quantile_scale=scale,
                                                                          power_law_exp=power_law_exp)
            if mode == "power_law":
                mode = "PL"
            elif mode == "decay_exp":
                mode = "exp"

            ax1 = axes1[i, j] if axes1.ndim > 1 else axes1[max(i, j)]
            ax2 = axes2[i, j] if axes2.ndim > 1 else axes2[max(i, j)]

            ax1.scatter(distances, probabilities, alpha=0.5)
            ax1.set_title(f'{mode}, Q={scale}')
            ax1.set_xlabel('Distance')
            ax1.set_ylabel('Probability')

            ax2.hist(edge_distances, bins=50, alpha=0.75)
            ax2.set_title(f'{mode}, Q={scale}')
            ax2.set_xlabel('Edge Distance')
            ax2.set_ylabel('Frequency')

    fig1.tight_layout()
    fig2.tight_layout()
    plot_folder = args.directory_map['distance_decay']
    print("finished 1")
    fig1.savefig(f"{plot_folder}/probability_decay_{args.args_title}.png")
    print("finished 2")
    fig2.savefig(f"{plot_folder}/histogram_edge_decay_{args.args_title}.png")
    print("finished 3")


def compute_epsilon_ball_radius(density, intended_degree, dim, base_proximity_mode):
    if dim == 2:
        radius_coefficient = np.pi  # area circumference
    elif dim == 3:
        radius_coefficient = (4 / 3) * np.pi  # volume sphere
    else:
        raise ValueError("Input dimension should be 2 or 3")

    # Adding the + 1 to not count the origin point itself
    if base_proximity_mode == "epsilon_bipartite":
        intended_degree = 2 * intended_degree + 1
    else:
        intended_degree = intended_degree + 1

    return ((intended_degree) / (radius_coefficient * density)) ** (1 / dim)






def compute_proximity_graph(args, positions):
    """
    Computes the proximity graph based on the positions and the specified proximity mode
    """

    valid_modes = ["knn", "epsilon-ball", "knn_bipartite", "epsilon_bipartite", "delaunay", "delaunay_corrected",
                   "lattice", "random", "distance_decay"]

    # Extract the base proximity mode from the args.proximity_mode
    base_proximity_mode = args.proximity_mode.split("_with_false_edges=")[0]

    # Check if the base mode is valid
    if base_proximity_mode not in valid_modes:
        raise ValueError("Please input a valid proximity graph")



    if base_proximity_mode == "epsilon-ball" or base_proximity_mode == "epsilon_bipartite":
        point_mode = args.point_mode
        if point_mode == "square":
            density = args.num_points
        elif point_mode == "circle":
            if args.dim == 2:
                density = args.num_points / np.pi
            elif args.dim == 3:
                density = args.num_points / (4 / 3 * np.pi)
        else:
            raise ValueError("Please input a valid point mode")
        radius = compute_epsilon_ball_radius(density=density, intended_degree=args.intended_av_degree,
                                             dim=args.dim, base_proximity_mode=base_proximity_mode, )
        print(f"Radius:{radius} for intended degree: {args.intended_av_degree}")

    if base_proximity_mode== "knn":
        k = args.intended_av_degree

        distances, indices = compute_knn_graph(positions, k)
        print("K", k)
    elif base_proximity_mode == "epsilon-ball":
        distances, indices = compute_epsilon_ball_graph(positions, radius)
        average_degree = sum(len(element) for element in indices)/len(indices)
        print("AVERAGE DEGREE EPSILON-BALL:", average_degree)
    elif base_proximity_mode == "delaunay":
        distances, indices = get_delaunay_neighbors(positions)
        average_degree = sum(len(element) for element in indices)/len(indices)
        print("AVERAGE DEGREE DELAUNAY:", average_degree)
    elif base_proximity_mode == "delaunay_corrected":  # delaunay graph
        distances, indices = get_delaunay_neighbors_corrected(positions)
        average_degree = sum(len(element) for element in indices) / len(indices)
        print("AVERAGE DEGREE DELAUNAY CORRECTED:", average_degree)
    elif base_proximity_mode == "epsilon_bipartite":
        distances, indices = epsilon_bipartite(positions, radius=radius)
        args.is_bipartite = True
        average_degree = sum(len(element) for element in indices)/len(indices)
        print("AVERAGE DEGREE EPSILON-BIPARTITE", average_degree)
        print("RADIUS", radius)
    elif base_proximity_mode == "knn_bipartite":
        # k = args.intended_av_degree + 1  # KNN counts itself, so adding +1
        args.is_bipartite = True
        k = args.intended_av_degree
        distances, indices = knn_bipartite(positions, k=k)
        # average_degree = sum(len(element) for element in indices) / len(indices)

    elif base_proximity_mode == "random":
        num_points = args.num_points
        intended_av_degree = args.intended_av_degree

        # Calculate the number of edges needed to achieve the intended average degree
        total_edges = int(num_points * intended_av_degree / 2)

        # Initialize lists to store the indices of the nodes each node is connected to
        indices = [[] for _ in range(num_points)]

        # Create a set to keep track of already connected node pairs to avoid duplicates
        existing_edges = set()

        while len(existing_edges) < total_edges:
            # Randomly select two different nodes
            node1, node2 = random.sample(range(num_points), 2)

            # Check if the pair is already connected or if it's a self-loop
            if (node1, node2) not in existing_edges and (node2, node1) not in existing_edges and node1 != node2:
                # Add the pair to the set of existing edges
                existing_edges.add((node1, node2))

                # Update the indices to reflect the connection
                indices[node1].append(node2)
                indices[node2].append(node1)

        # Assuming distances are not meaningful in this context, we set them to 1 or any arbitrary constant value
        distances = [[1 for _ in neighbor] for neighbor in indices]

        # Calculate the actual average degree to verify
        actual_av_degree = sum(len(neighbor) for neighbor in indices) / num_points
        print(f"Actual Average Degree: {actual_av_degree}")
    elif base_proximity_mode == "distance_decay":

        # TODO: uncomment this if you don't want extra plots with different quantile scales
        # ### example plotting effects
        # start = 0.05
        # stop = 0.30
        # step = 0.05
        start = 0.005
        stop = 0.08
        step = 0.005
        # num = int((stop - start) / step + 1)
        # distance_decay_quantiles_list = np.linspace(start, stop, num)
        # quantile_scales = distance_decay_quantiles_list
        # # quantile_scales = [0.05, 0.15, 0.25, 0.35, 0.75, 0.95]
        # decay_modes = ['power_law', 'decay_exp']
        # # decay_modes = ['decay_exp']
        # plot_decay_effects(args, positions, quantile_scales, decay_modes)


        quantile_scale = args.distance_decay_quantile
        distances, indices = compute_proximity_decay_graph(positions, quantile_scale=quantile_scale)
    else:

        raise ValueError("Please input a valid proximity graph")
    return distances, indices

def compute_lattice(args, positions):
    """ Compute the nearest neighbors in a square or cubic lattice. """
    # Number of neighbors: 4 for a square lattice, 6 for a cubic lattice
    n_neighbors = 4 if args.dim == 2 else 6
    print(positions.shape)
    distances, indices = compute_knn_graph(positions, k=n_neighbors+1)

    return distances, indices


def write_positions(args, np_positions, output_path):
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
    args.positions_path = output_file_path    # defined a little out of the blue? It is for the read_df function
    positions_df.to_csv(output_file_path, index=False)


def compute_decay_rate(positions, quantile=0.5):
    """
    computes a reasonable "scale factor" to weight the graphs
    """
    distance_matrix = pdist(positions)
    distance_matrix = squareform(distance_matrix)
    distances = distance_matrix.flatten()
    nonzero_distances = distances[distances > 0]
    distance_quantile = np.quantile(nonzero_distances, quantile)



    # The decay rate is the inverse of the quantile distance
    decay_rate = 1 / distance_quantile if distance_quantile != 0 else float('inf')
    print(f"Distance quantile: {distance_quantile}")
    print(f"Decay rate: {decay_rate}")
    return decay_rate

def sort_points_by_distance_to_centroid(points):
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
    sorted_indices = np.argsort(distances)
    sorted_points = points[sorted_indices]
    return sorted_points


def sort_points_for_heatmap(points):
    # Calculate the Euclidean distance matrix
    points = np.array(points)
    dist_matrix = ssd.pdist(points, 'euclidean')
    linkage_matrix = sch.linkage(dist_matrix, method='average')
    dendro = sch.dendrogram(linkage_matrix, no_plot=True)
    order = dendro['leaves']
    sorted_points = points[order]
    return sorted_points


def distribute_weights(num_edges, total_weight, max_weight):
    # Generate random weights
    weights = np.random.randint(1, max_weight + 1, size=num_edges)
    while sum(weights) != total_weight:
        # Adjust weights to sum up to total_weight
        diff = sum(weights) - total_weight
        for i in range(num_edges):
            if diff == 0:
                break
            if diff > 0 and weights[i] > 1:
                adjustment = min(weights[i] - 1, diff)
                weights[i] -= adjustment
                diff -= adjustment
            elif diff < 0:
                adjustment = min(max_weight - weights[i], -diff)
                weights[i] += adjustment
                diff += adjustment
    return weights

def add_weights_to_false_edges(edge_df, args, max_weight):
    if args.weighted:
        # Add 'weight' column if it does not exist
        if 'weight' not in edge_df.columns:
            raise ValueError("Weight not in the edge_df columns")
        if args.false_edges_count < 1:
            raise ValueError("False edge counts must be at least 1")

        # Compute and assign weights and distances for false edges
        total_weight = (args.false_edges_count * args.weight_converter.max_weight) / 3  # weight budget
        weights = distribute_weights(args.false_edges_count, total_weight, max_weight)
        for i, edge in enumerate(args.false_edge_ids):
            # Extract the indices for source and target
            source_idx, target_idx = edge[0], edge[1]
            # Assign a random weight between 1 and max_weight
            weight = weights[i]
            edge_df.loc[(edge_df['source'] == source_idx) & (edge_df['target'] == target_idx), 'weight'] = weight
    return edge_df


def write_proximity_graph(args, order_indices=False, point_mode="square"):
    base_proximity_mode = args.proximity_mode.split("_with_false_edges=")[0]
    if base_proximity_mode == "lattice":
        points = generate_square_lattice(args)
        args.num_points = len(points)
        distances, indices = compute_lattice(args, points)

    else:
        # Without density anomalies, square
        if point_mode == "square":
            points = generate_random_points(num_points=args.num_points, L=args.L, dim=args.dim)

        elif point_mode == "circle":
            points = generate_random_points_in_circle_or_sphere(num_points=args.num_points, R=args.L, dim=args.dim)

        # ## With density anomalies
        # points = generate_random_points_anomaly(num_points=args.num_points, L=args.L, dim=args.dim, anomaly_strength=1)

        if order_indices:
            points = sort_points_for_heatmap(points)


        distances, indices = compute_proximity_graph(args, positions=points)

    position_folder = args.directory_map["original_positions"]
    edge_list_folder = args.directory_map["edge_lists"]
    positions = np.array(points)

    # Create the edge list without duplicates
    edges = set()
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            if args.weighted:
                # Include distance with edges for weighted case
                edge = tuple(sorted((i, neighbor))) + (distances[i][j],)
            else:
                # Unweighted case, include only nodes
                edge = tuple(sorted((i, neighbor)))
            edges.add(edge)


    ### Add false edges!
    ### For bipartite graphs:
    if args.is_bipartite:
        half = len(indices) // 2
        # Add false edges for bipartite set
        if args.false_edges_count:
            for _ in range(args.false_edges_count):
                # Select one node from each part of the bipartite graph
                i = random.randint(0, half)
                j = random.randint(half, args.num_points - 1)

                edge = tuple(sorted((i, j)))
                # Check to avoid adding an edge that already exists
                if edge not in edges:
                    args.false_edge_ids.append(edge)
                    distance = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
                    edge = tuple(sorted((i, j))) + (distance,)
                    edges.add(edge)
                else:
                    while edge in edges:
                        j = random.randint(half, args.num_points - 1)
                        edge = tuple(sorted((i, j)))
                        if edge not in edges:
                            args.false_edge_ids.append(edge)
                            distance = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
                            edge = tuple(sorted((i, j))) + (distance,)
                            edges.add(edge)
                            break

    else:
        ### Add false edges
        if args.false_edges_count:
            for _ in range(args.false_edges_count):
                i = random.randint(0, args.num_points - 1)
                j = random.randint(0, args.num_points - 1)
                if i != j:  # Avoid self-loop
                    edge = tuple(sorted((i, j)))
                    args.false_edge_ids.append(edge)
                    if args.weighted:
                        distance = np.linalg.norm(positions[edge[0]] - positions[edge[1]])
                        edge = tuple(sorted((i, j))) + (distance,)
                    edges.add(edge)

    write_positions(args, np_positions=np.array(points), output_path=position_folder)
    columns = ['source', 'target', 'distance'] if args.weighted else ['source', 'target']


    edge_df = pd.DataFrame(list(edges), columns=columns)

    if args.weighted and args.weight_to_distance:
        args.weight_converter.decay_rate = compute_decay_rate(points, quantile=0.01) # TODO: hardcoded, find the proper scaling factor
        if args.weight_to_distance_fun == "exp":
            edge_df['weight'] = edge_df['distance'].apply(args.weight_converter.return_weight_exponential_model)
        else:
            raise ValueError("Please choose valid weight to distance function")
        if args.false_edges_count:
            edge_df = add_weights_to_false_edges(edge_df=edge_df, args=args, max_weight=args.weight_converter.max_weight)

        # TODO: this might be an interesting plot for the paper
        # ### Plots the weight to distance relationship
        # plt.figure(figsize=(10, 6))
        # plt.scatter(edge_df['distance'], edge_df['weight'], color='blue')
        # plt.title('Weight to Distance Relationship')
        # plt.xlabel('Distance')
        # plt.ylabel('Weight')
        # plt.grid(True)
        # plt.show()

        edge_df = edge_df[edge_df['weight'] != 0]  ## remove 0 weights

    # # TODO: revert to this if errors arise
    # edge_df.to_csv(os.path.join(edge_list_folder, f"edge_list_{args.args_title}.csv"), index=False)
    edge_df.to_csv(os.path.join(edge_list_folder, f"{args.edge_list_title}"), index=False)
    return edge_df