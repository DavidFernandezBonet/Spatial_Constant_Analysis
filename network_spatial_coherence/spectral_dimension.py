from spatial_constant_analysis import *
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components


def perform_random_walk_csr(csr_graph, start_node, steps):
    """Perform a single random walk from a specified start node for a given number of steps."""
    current_node = start_node
    visited = set([current_node])

    for _ in range(steps):
        neighbors = csr_graph.indices[csr_graph.indptr[current_node]:csr_graph.indptr[current_node + 1]]
        if neighbors.size > 0:
            current_node = np.random.choice(neighbors)
            visited.add(current_node)
    return visited


def simulate_random_walks_varying_lengths(csr_graph, max_steps=100, num_walks_per_step=100, start_walk=10):
    """Simulate random walks of varying lengths on a CSR graph and track the growth of visited nodes."""
    n_nodes = csr_graph.shape[0]
    avg_visited_nodes = np.zeros(max_steps)

    for step in range(start_walk, max_steps + start_walk):
        total_visited = 0
        for _ in range(num_walks_per_step):
            start_node = np.random.randint(0, n_nodes)
            visited = perform_random_walk_csr(csr_graph, start_node, step)
            total_visited += len(visited)
        avg_visited_nodes[step - start_walk] = total_visited / num_walks_per_step

    return avg_visited_nodes


def log_function(x, a, b):
    return a * np.log(x) + b

def lin_function(x, a, b):
    return a*x + b


def compute_eigenvalues_laplacian_csgraph(graph):
    # Convert the adjacency matrix to a sparse graph Laplacian
    L = sp.csgraph.laplacian(graph, normed=False)

    # Compute the second smallest eigenvalue
    # We use which='SM' to request the smallest magnitude eigenvalues and k=2 since we need the second smallest
    eigenvalues, eigenvectors = sp.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=True)

    # The second smallest eigenvalue
    second_smallest_eigenvalue = eigenvalues[1]
    return second_smallest_eigenvalue


# Parameters
args = GraphArgs()
args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
args.proximity_mode = "epsilon-ball"
args.dim = 2

args.intended_av_degree = 15
args.num_points = 1000
### Add random edges? See efect in the dimensionality here
num_edges_to_add = 0

simulation_or_experiment = "simulation"
load_mode = 'sparse'


if simulation_or_experiment == "experiment":
    # # # #Experimental
    # our group:
    # subgraph_2_nodes_44_edges_56_degree_2.55.pickle  # subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle  # subgraph_8_nodes_160_edges_179_degree_2.24.pickle
    # unfiltered pixelgen:
    # pixelgen_cell_2_RCVCMP0000594.csv, pixelgen_cell_1_RCVCMP0000208.csv, pixelgen_cell_3_RCVCMP0000085.csv
    # pixelgen_edgelist_CD3_cell_2_RCVCMP0000009.csv, pixelgen_edgelist_CD3_cell_1_RCVCMP0000610.csv, pixelgen_edgelist_CD3_cell_3_RCVCMP0000096.csv
    # filtered pixelgen:
    # pixelgen_processed_edgelist_Sample01_human_pbmcs_unstimulated_cell_3_RCVCMP0000563.csv
    # pixelgen_processed_edgelist_Sample01_human_pbmcs_unstimulated_cell_2_RCVCMP0000828.csv
    # pixelgen_processed_edgelist_Sample07_pbmc_CD3_capped_cell_3_RCVCMP0000344.csv (stimulated cell)
    # pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv (treated cell)
    # shuai_protein_edgelist_unstimulated_RCVCMP0000133_neigbours_s_proteinlist.csv  (shuai protein list)
    # pixelgen_processed_edgelist_shuai_RCVCMP0000073_cd3_cell_1_RCVCMP0000073.csv (shuai error correction)
    # weinstein:
    # weinstein_data_january_corrected.csv

    args.edge_list_title = "weinstein_data_corrected_february.csv"
    # args.edge_list_title = "mst_N=1024_dim=2_lattice_k=15.csv"  # Seems to have dimension 1.5

    weighted = True
    weight_threshold = 3

    if os.path.splitext(args.edge_list_title)[1] == ".pickle":
        write_nx_graph_to_edge_list_df(args)  # activate if format is .pickle file

    if not weighted:
        sparse_graph, _ = load_graph(args, load_mode='sparse')
    else:
        sparse_graph, _ = load_graph(args, load_mode='sparse', weight_threshold=weight_threshold)
    # plot_graph_properties(args, igraph_graph_original)  # plots clustering coefficient, degree dist, also stores individual spatial constant...

elif simulation_or_experiment == "simulation":
    # # # 1 Simulation
    create_proximity_graph.write_proximity_graph(args)
    sparse_graph, _ = load_graph(args, load_mode='sparse')
    # ## Original data    edge_list = read_edge_list(args)
    # original_positions = read_position_df(args=args)
    # # plot_original_or_reconstructed_image(args, image_type="original", edges_df=edge_list)
    # original_dist_matrix = compute_distance_matrix(original_positions)
else:
    raise ValueError("Please input a valid simulation or experiment mode")



#### Algebraic Connectivity
num_points_range = np.arange(1000, 5000, 500)  # From 100 to 1000, in steps of 100

# Storage for computed and predicted smallest eigenvalues
smallest_eigenvalues = []
predicted_smallest_eigenvalues = []
false_eigenvalues = []

for num_points in num_points_range:
    # Example: Generating a sparse graph representation here.
    # You should replace this with your actual graph generation method based on num_points
    # For demonstration, let's create a random sparse matrix
    args.num_points = num_points
    create_proximity_graph.write_proximity_graph(args)
    sparse_graph, _ = load_graph(args, load_mode='sparse')

    # Compute the smallest eigenvalue for the current graph
    smallest_eigenvalue = compute_eigenvalues_laplacian_csgraph(sparse_graph)
    smallest_eigenvalues.append(smallest_eigenvalue)

    # Predict the smallest eigenvalue
    predicted_small_eigenvalue = ((1 / 6) * (np.pi / (num_points ** (1 / args.dim))) ** 2 *
                                  (args.average_degree + 1) ** ((args.dim + 2) / args.dim))
    predicted_small_eigenvalue = 0
    predicted_smallest_eigenvalues.append(predicted_small_eigenvalue)

    sparse_graph = add_random_edges_to_csrgraph(sparse_graph, num_edges_to_add=10)
    smallest_eigenvalue = compute_eigenvalues_laplacian_csgraph(sparse_graph)
    false_eigenvalues.append(smallest_eigenvalue)



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_points_range, smallest_eigenvalues, label='Computed Smallest Eigenvalue', marker='o')
plt.plot(num_points_range, predicted_smallest_eigenvalues, label='Predicted Smallest Eigenvalue', marker='x')
plt.plot(num_points_range, false_eigenvalues, label='False Eigenvalues', marker='x')
plt.xlabel('Number of Points')
plt.ylabel('Smallest Eigenvalue')
plt.title('Smallest Eigenvalue vs. Number of Points')
plt.legend()

plt.savefig("algebraic_connectivity_by_size")
print("true", smallest_eigenvalue)
print("prediction", predicted_small_eigenvalue)



### Simulate random walks
n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
if n_components > 1:
    print(f"Graph has {n_components} components, consider using a connected graph for meaningful results.")
else:
    # Simulate random walks
    start_walk = 10
    steps = 100
    num_walks = 100
    visited_avg = simulate_random_walks_varying_lengths(sparse_graph, max_steps=steps, num_walks_per_step=num_walks, start_walk=start_walk)
    print("HOLAAA", visited_avg)

    # Time steps
    t = np.arange(start_walk, steps + start_walk)

    print(len(t))
    print(len(visited_avg))

    print(t)
    print(visited_avg)
    # Plotting
    plt.plot(t, visited_avg, 'o', label='Average Visited Nodes')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time Steps (log)')
    plt.ylabel('Average Visited Nodes (log)')
    plt.title('Log-Log Plot of Average Visited Nodes vs. Time Steps')
    plt.legend()
    plt.savefig("spectral_dimension.png")

    # Linear fit to estimate spectral dimension
    log_t = np.log(t)
    log_visited_avg = np.log(visited_avg)
    params, params_covariance = curve_fit(lin_function, log_t, log_visited_avg, p0=[1, np.log(visited_avg[1])])
    print(f"Estimated spectral dimension (d_s/2): {params[0] * 2}")