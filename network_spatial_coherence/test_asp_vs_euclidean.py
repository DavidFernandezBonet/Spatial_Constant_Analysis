import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

from structure_and_args import GraphArgs
from create_proximity_graph import compute_proximity_graph
from scipy.sparse import coo_matrix, csr_matrix
import networkx as nx


def create_point_cloud(side_length, num_points, dimension=2):
    """
    Generates a point cloud within a square (2D) or cube (3D).

    Parameters:
    - side_length: the length of the side of the square or cube.
    - num_points: the number of points to generate.
    - dimension: the dimension of the space (2 for square, 3 for cube).

    Returns:
    - points: an array of shape (num_points, dimension) representing point coordinates.
    """

    points = np.random.rand(num_points, dimension) * side_length
    return points


def compute_average_distance(points):
    """
    Computes the average distance between all pairs of points in the point cloud.

    Parameters:
    - points: an array of points of shape (num_points, dimension).

    Returns:
    - avg_distance: the average distance between all pairs of points.
    """
    distances = pdist(points)
    avg_distance = np.mean(distances)
    return avg_distance


def create_csr_matrix_from_edges(edges, num_nodes):
    """
    Creates a CSR matrix from a set of edges without direct use of index pointers.

    Parameters:
    - edges: A set of tuples, where each tuple represents an edge (i, j) in the graph.
    - num_nodes: The total number of nodes in the graph.

    Returns:
    - csr_graph: A CSR matrix representing the graph.
    """
    # Convert the edge set to a numpy array for easy manipulation
    edge_array = np.array(list(edges))

    # Create symmetric edge list by adding both (i, j) and (j, i) for each edge
    symmetric_edges = np.vstack([edge_array, edge_array[:, [1, 0]]])

    # All edges have a weight of 1
    data = np.ones(len(symmetric_edges))

    # Create the COO matrix
    coo_graph = coo_matrix((data, (symmetric_edges[:, 0], symmetric_edges[:, 1])), shape=(num_nodes, num_nodes))

    # Convert COO matrix to CSR format
    csr_graph = coo_graph.tocsr()

    return csr_graph

def get_graph(indices, num_nodes):
    edges = set()
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            # Ensure the smaller index always comes first
            edge = tuple(sorted((i, neighbor)))
            edges.add(edge)
    graph = create_csr_matrix_from_edges(edges, num_nodes)

    return graph


def calculate_average_degree(csr_graph):
    # Count the total number of edges in the graph
    # Note: Each edge is counted twice in an undirected graph represented by a CSR matrix
    num_edges = csr_graph.nnz  # nnz property gives the number of stored (non-zero) values

    # Calculate the average degree
    # Since each edge is counted twice, divide num_edges by 2 to get the actual number of edges
    # Then, divide by the number of nodes to get the average degree
    average_degree = num_edges / ( csr_graph.shape[0])

    return average_degree


def compute_average_shortest_path_distance(csr_graph):
    """
    Computes the average shortest path distance using scipy.sparse.csgraph.
    """
    distance_matrix, _ = shortest_path(csgraph=csr_graph, method='auto', directed=False, return_predecessors=True)
    valid_distances = distance_matrix[np.isfinite(distance_matrix) & (distance_matrix != 0)]
    avg_distance = np.mean(valid_distances)
    return avg_distance

def plot_point_cloud_and_edges(points, csr_graph):
    """
    Plots the point cloud and its edges using matplotlib, given point coordinates and a CSR graph.

    Parameters:
    - points: Numpy array of point coordinates, shape (num_points, dimension).
    - csr_graph: CSR matrix representing the graph connections.
    """
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points')

    # Extract rows, cols from csr_matrix to plot edges
    rows, cols = csr_graph.nonzero()
    for row, col in zip(rows, cols):
        if row < col:  # This condition avoids plotting the same edge twice
            plt.plot([points[row, 0], points[col, 0]], [points[row, 1], points[col, 1]], 'green', linewidth=0.5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Point Cloud with KNN Edges')
    plt.legend()
    plt.show()





def plot_avg_distances_vs_L_with_heatmap(L_values, dim, intended_av_degree):
    avg_distances = []
    avg_shortest_path_distances = []
    L_scaled = []  # To store scaled L values for coloring

    # Adjust these functions according to your implementation
    for L in L_values:
        args = GraphArgs(L=L, dim=dim, intended_av_degree=intended_av_degree, num_points=int(1000 * L ** dim))

        # Generate point cloud and compute metrics
        points = create_point_cloud(args.L, args.num_points, args.dim)
        avg_distance = compute_average_distance(points)
        _, indices = compute_proximity_graph(args, positions=points)  # This needs to be defined
        G = get_graph(indices, num_nodes=args.num_points)  # This needs to be defined
        args.average_degree = calculate_average_degree(G)
        avg_shortest_path_distance = compute_average_shortest_path_distance(G)  # This needs to be defined

        avg_distances.append(avg_distance)
        avg_shortest_path_distances.append(avg_shortest_path_distance)
        L_scaled.append(args.L)  # Use L directly or scale it if needed for coloring

    # Compute correlation
    correlation = np.corrcoef(avg_distances, avg_shortest_path_distances)[0, 1]

    slope, intercept = np.polyfit(avg_distances, avg_shortest_path_distances, 1)
    regression_line = np.array(avg_distances) * slope + intercept
    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(avg_distances, avg_shortest_path_distances, c=L_scaled, cmap='viridis')
    plt.colorbar(scatter, label='Side Length L')
    plt.plot(avg_distances, regression_line, 'r-', linewidth=2, label='Linear Regression Fit')  # Plot the regression line
    plt.xlabel('Average Distance')
    plt.ylabel('Average Shortest Path Distance')
    plt.title(f'Avg. Shortest Path Distance vs Avg. Distance\nCorrelation: {correlation:.2f}')


    # Display the equation of the fit
    equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(min(avg_distances), max(avg_shortest_path_distances), equation_text, fontsize=12, color='red')

    plt.show()


def plot_distances_vs_num_points(L_values, dim, intended_av_degree):
    avg_distances = []
    avg_shortest_path_distances = []
    scaled_quantity_euc_list = []
    scaled_quantity_sp_list = []
    for L in L_values:
        args = GraphArgs(L=L, dim=dim, intended_av_degree=intended_av_degree, num_points=int(1000 * L ** dim))

        num_points = args.num_points
        density = num_points / (L**args.dim)
        scaled_quantity_euc = (num_points/density)**(1/args.dim)
        scaled_quantity_euc_list.append(scaled_quantity_euc)


        # Generate point cloud and compute metrics
        points = create_point_cloud(args.L, args.num_points, args.dim)
        avg_distance = compute_average_distance(points)
        _, indices = compute_proximity_graph(args, positions=points)  # Adjust this call
        G = get_graph(indices, num_nodes=args.num_points)  # Adjust this call
        args.average_degree = calculate_average_degree(G)
        print("average degree", args.average_degree)
        scaled_quantity_sp = (num_points/args.average_degree)**(1/args.dim)
        scaled_quantity_sp_list.append(scaled_quantity_sp)
        avg_shortest_path_distance = compute_average_shortest_path_distance(G)  # Adjust this call

        avg_distances.append(avg_distance)
        avg_shortest_path_distances.append(avg_shortest_path_distance)

    # Linear fits
    slope_dist, intercept_dist = np.polyfit(scaled_quantity_euc_list, avg_distances, 1)
    slope_spd, intercept_spd = np.polyfit(scaled_quantity_sp_list, avg_shortest_path_distances, 1)

    # Equations
    equation_dist = f'y = {slope_dist:.2f}x + {intercept_dist:.2f}'
    equation_spd = f'y = {slope_spd:.2f}x + {intercept_spd:.2f}'

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Average Distance vs Num Points
    axs[0].scatter(scaled_quantity_euc_list, avg_distances, label='Data')
    axs[0].plot(scaled_quantity_euc_list, np.polyval([slope_dist, intercept_dist], scaled_quantity_euc_list), 'r-',
                label='Linear Fit')
    axs[0].text(min(scaled_quantity_euc_list), max(avg_distances), equation_dist, fontsize=12, color='red')
    axs[0].set_xlabel('Number of Points t')
    axs[0].set_ylabel('Average Distance')
    axs[0].set_title('Avg Distance vs Number of Points')


    # Average Shortest Path Distance vs Num Points
    axs[1].scatter(scaled_quantity_sp_list, avg_shortest_path_distances, label='Data')
    axs[1].plot(scaled_quantity_sp_list, np.polyval([slope_spd, intercept_spd], scaled_quantity_sp_list), 'r-', label='Linear Fit')
    axs[1].text(min(scaled_quantity_sp_list), max(avg_shortest_path_distances), equation_spd, fontsize=12, color='red')
    axs[1].set_xlabel('Number of Points')
    axs[1].set_ylabel('Average Shortest Path Distance')
    axs[1].set_title('Avg Shortest Path Distance vs Number of Points')


    plt.tight_layout()
    plt.show()


def plot_distances_vs_average_degree(L, dim, intended_av_degree_values, proximity_mode):
    avg_distances = []
    avg_shortest_path_distances = []
    average_degree_values = []

    for intended_av_degree in intended_av_degree_values:
        args = GraphArgs(L=L, dim=dim, intended_av_degree=int(intended_av_degree), num_points=int(1000 * L ** dim))
        args.proximity_mode = proximity_mode





        # Generate point cloud and compute metrics
        points = create_point_cloud(args.L, args.num_points, args.dim)
        avg_distance = compute_average_distance(points)
        _, indices = compute_proximity_graph(args, positions=points)  # Adjust this call
        G = get_graph(indices, num_nodes=args.num_points)  # Adjust this call
        args.average_degree = calculate_average_degree(G)
        average_degree_values.append((1/(args.average_degree))**(1/args.dim))
        print("average degree", args.average_degree)

        avg_shortest_path_distance = compute_average_shortest_path_distance(G)  # Adjust this call

        avg_distances.append(avg_distance)
        avg_shortest_path_distances.append(avg_shortest_path_distance)

    # Linear fits
    slope_dist, intercept_dist = np.polyfit(average_degree_values, avg_distances, 1)
    slope_spd, intercept_spd = np.polyfit(average_degree_values, avg_shortest_path_distances, 1)

    # Equations

    equation_spd = f'y = {slope_spd:.2f}x + {intercept_spd:.2f}'

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))




    # Average Shortest Path Distance vs Num Points
    axs[1].scatter(average_degree_values, avg_shortest_path_distances, label='Data')
    axs[1].plot(average_degree_values, np.polyval([slope_spd, intercept_spd], average_degree_values), 'r-', label='Linear Fit')
    axs[1].text(min(average_degree_values), max(avg_shortest_path_distances), equation_spd, fontsize=12, color='red')
    axs[1].set_xlabel('Inverse degree to the dimension power')
    axs[1].set_ylabel('Average Shortest Path Distance')



    plt.tight_layout()
    plt.show()

args = GraphArgs()

# Example usage
args.L = 1  # Side length of the square
args.num_points = int(1000 * args.L ** args.dim)  # Number of points in the point cloud
args.dim = 2  # Dimension of the space

args.intended_av_degree = 6  # Number of nearest neighbors
# Part A: Generate point cloud and compute average distance
points = create_point_cloud(args.L, args.num_points, args.dim)

avg_distance = compute_average_distance(points)
print(f"Average distance between points: {avg_distance}")

# Part B: Create KNN graph and compute average shortest path distance
distances, indices = compute_proximity_graph(args, positions=points)
G = get_graph(indices, num_nodes=args.num_points)
avg_shortest_path_distance = compute_average_shortest_path_distance(G)
print(f"Average shortest path distance: {avg_shortest_path_distance}")
# plot_point_cloud_and_edges(points, G)

### Plot several iterations
# Example usage
L_values = np.linspace(1, 2, 10)  # Example range of L values
dim = 2  # Dimension of the space
intended_av_degree = 10  # Intended average degree in the KNN graph

#plot_avg_distances_vs_L_with_heatmap(L_values, dim, intended_av_degree)


### Plot relationship with spatial constant quantity
# plot_distances_vs_num_points(L_values, dim, intended_av_degree)


### Plot relationship with average degree
intended_av_degree_values = np.linspace(4, 15, 10)
plot_distances_vs_average_degree(L=1, dim=dim, intended_av_degree_values=intended_av_degree_values,
                                 proximity_mode="knn")