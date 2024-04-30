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
                   "lattice", "random"]

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
    args.positions_path = output_file_path
    positions_df.to_csv(output_file_path, index=False)


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

    # Create the edge list without duplicates
    edges = set()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            # Ensure the smaller index always comes first
            edge = tuple(sorted((i, neighbor)))
            edges.add(edge)

    # TODO: False edges only for bipartite set


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
                    edges.add(edge)
                    args.false_edge_ids.append(edge)
                else:
                    while edge in edges:
                        j = random.randint(half, args.num_points - 1)
                        edge = tuple(sorted((i, j)))
                        if edge not in edges:
                            edges.add(edge)
                            args.false_edge_ids.append(edge)
                            break

    else:
        ### Add false edges
        if args.false_edges_count:
            for _ in range(args.false_edges_count):
                i = random.randint(0, args.num_points - 1)
                j = random.randint(0, args.num_points - 1)
                if i != j:  # Avoid self-loop
                    edge = tuple(sorted((i, j)))
                    edges.add(edge)
                    args.false_edge_ids.append(edge)

    write_positions(args, np_positions=np.array(points), output_path=position_folder)
    edge_df = pd.DataFrame(list(edges), columns=['source', 'target'])

    # # TODO: revert to this if errors arise
    # edge_df.to_csv(os.path.join(edge_list_folder, f"edge_list_{args.args_title}.csv"), index=False)
    edge_df.to_csv(os.path.join(edge_list_folder, f"{args.edge_list_title}"), index=False)
    return edge_df