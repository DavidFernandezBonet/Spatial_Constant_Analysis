import numpy as np

from curve_fitting import CurveFitting
from spatial_constant_analysis import *
from utils import read_position_df
import scipy.stats as stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

font_size = 24
plt.style.use(['no-latex', 'nature'])

sns.set_style("white")  # 'white' is a style option in seaborn

# If you want to use a seaborn style with modifications
sns.set(style="white", rc={
    'axes.labelsize': font_size,
    'axes.titlesize': font_size + 6,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size - 10
})

def compute_node_counts_matrix(distance_matrix):
    """
    Compute a matrix where each row represents a node and each column the count of nodes at a specific distance.

    :param distance_matrix: Matrix of shortest path distances.
    :return: Numpy matrix with node counts at each distance.
    """
    num_nodes = len(distance_matrix)
    max_distance = distance_matrix.max()

    # Initialize the matrix with zeros
    counts_matrix = np.zeros((num_nodes, int(max_distance)), dtype=int)

    # TODO: optimize this given that distance matrix is a numpy array
    # Populate the matrix
    for i, row in enumerate(distance_matrix):
        for distance in row:
            distance = int(distance)
            if distance != 0:  # Ignoring distance 0 (self-loops)
                counts_matrix[i, distance - 1] += 1

    return counts_matrix


def compute_correlation_between_distance_matrices(matrix1, matrix2):
    """
    Compute the Pearson correlation coefficient between two distance matrices.

    :param matrix1: First distance matrix.
    :param matrix2: Second distance matrix.
    :return: Pearson correlation coefficient.
    """
    # Flatten the matrices
    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    # Compute Pearson correlation
    correlation, _ = stats.pearsonr(flat_matrix1, flat_matrix2)
    return correlation

def run_dimension_prediction_continuous(args, distance_matrix, num_bins=10):
    # Determine the range and bin width
    max_distance = np.max(distance_matrix)
    # min_distance = np.min(distance_matrix)
    max_distance = 1

    min_distance = 0
    bin_width = (max_distance - min_distance) / num_bins

    # Create bins for distances
    bins = np.arange(min_distance, max_distance, bin_width)
    # Initialize a matrix to count nodes in each bin
    binned_distance_counts = np.zeros((distance_matrix.shape[0], len(bins)))

    # Group distances into bins and count
    for i in range(distance_matrix.shape[0]):
        for j, bin_edge in enumerate(bins):
            if j == 0:
                continue
            binned_distance_counts[i, j] = np.sum((distance_matrix[i] > bins[j - 1]) & (distance_matrix[i] <= bins[j]))



    # Row with maximum number of nodes
    # Compute the sum of each row
    row_sums = np.sum(binned_distance_counts[:, 0: int(len(bins)/2)], axis=1)

    # Find the index of the row with the maximum sum
    max_sum_index = np.argmax(row_sums)
    print("MAX SUM", np.sum(binned_distance_counts[max_sum_index]))
    print(row_sums)


    # Calculate average counts per bin
    count_by_distance_average = np.mean(binned_distance_counts, axis=0)   #TODO: is the axis right?
    std_distance_average = np.std(binned_distance_counts, axis=0)

    # delete this otherwise
    count_by_distance_average = binned_distance_counts[max_sum_index]  # 1st row  (just to omit the finite size effects)

    # Calculate cumulative counts
    cumulative_count = np.cumsum(count_by_distance_average)

    expected_cumulative_count = args.num_points * (4/3) * np.pi * (bins**3)

    print("EXPECTED CUMULATIVE POINTS", expected_cumulative_count)

    print("sum expected", np.sum(expected_cumulative_count))

    print("BINS", bins)
    print("COUNT BY DISTANCE AVERAGE", count_by_distance_average)
    print("STD AVERAGE DISTANCE", std_distance_average)
    print("CUMULATIVE COUNT", cumulative_count)


    # Fast distance approximation
    predicted_dimensions = (count_by_distance_average / cumulative_count) * np.arange(0, len(cumulative_count))
    print("PREDICTED DIMENSIONS", predicted_dimensions)

    # Try to delete finite size effects
    cumulative_count = cumulative_count[1: int(len(cumulative_count)/2)]
    bins = bins[1: int(len(bins)/2)]

    print("bins", bins)
    print("curated cumulative count", cumulative_count)

    # count_by_distance_average = np.array([15, 68, 222, 440])
    # cumulative_count = np.cumsum(count_by_distance_average)
    # print("CUMULATIVE COUNT GOOD", cumulative_count)  # CUMULATIVE COUNT GOOD [ 15  83 305 745]
    # bins = np.array([0.08, 0.16, 0.24, 0.32])
    # predicted_dimensions = (count_by_distance_average / cumulative_count)
    # print("PREDICTED DIMENSIONS GOOD", predicted_dimensions)

    plot_folder = args.directory_map['plots_predicted_dimension']
    save_path = f'{plot_folder}/dimension_prediction_original_by_node_count_{args.args_title}.png'
    plt.figure()
    curve_fitting_object = CurveFitting(bins, cumulative_count)

    # curve_fitting_object.fixed_a = args.num_points * (4/3) * np.pi
    # func_fit = curve_fitting_object.power_model_fixed



    # # Fixing constant
    # curve_fitting_object.fixed_a = args.average_degree * (1/np.sqrt(args.dim))
    # func_fit = curve_fitting_object.power_model_fixed

    # # Fixing dimension
    # curve_fitting_object.fixed_b = args.dim
    # func_fit = curve_fitting_object.power_model_fixed_exp

    # Unfixed parameters
    func_fit = curve_fitting_object.power_model


    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object.plot_fit_with_uncertainty(func_fit, "Distance", "Node Count",
                                                   "Dimension Prediction", save_path)

    return max_sum_index

def run_dimension_prediction(args, distance_matrix, dist_threshold=6, central_node_index=None):


    distance_count_matrix = compute_node_counts_matrix(distance_matrix)
    # dist_threshold = int(np.max(distance_matrix))  # comment out
    # distance_count_matrix = distance_count_matrix[:, :dist_threshold]

    david_thresh = dist_threshold

    # Row with maximum number of nodes
    # Compute the sum of each row
    row_sums = np.sum(distance_count_matrix[:, 0:david_thresh], axis=1)

    # Find the index of the row with the maximum sum
    max_sum_index = np.argmax(row_sums)
    print("MAX SUM NETWORK", np.sum(distance_count_matrix[:, 0:david_thresh][max_sum_index]))
    print(max_sum_index, central_node_index)


    # Contains the "Surface" --> Number of nodes at a specific distance
    count_by_distance_average = np.mean(distance_count_matrix, axis=0)
    count_by_distance_std = np.std(distance_count_matrix, axis=0)

    if central_node_index:
        count_by_distance_average = distance_count_matrix[central_node_index]
        print("count based on euclidean", count_by_distance_average)
        print(np.cumsum(count_by_distance_average))
    count_by_distance_average = distance_count_matrix[max_sum_index]
    print("count based on network", count_by_distance_average)
    print(np.cumsum(count_by_distance_average))

    # ### Find the diameter of a certain shell level
    # shell_level = 5
    # nodes_at_shell_level_r = np.where(distance_matrix[max_sum_index] == shell_level)[0]
    #
    #
    # alledged_diameter = np.max(distance_matrix[np.ix_(nodes_at_shell_level_r, nodes_at_shell_level_r)])
    # print(f"DIAMETER AT SHELL {shell_level} is: {alledged_diameter}")
    # submatrix = distance_matrix[np.ix_(nodes_at_shell_level_r, nodes_at_shell_level_r)]
    # # Get all pairwise distances between nodes at shell level
    # pairwise_distances = submatrix[np.triu_indices(len(nodes_at_shell_level_r), k=1)]
    # plot_barplot(args, distances=pairwise_distances, title="Distances at a Shell Level")




    # # Maybe taking the mean is biasing it delete otherwise)
    # count_by_distance_average = distance_count_matrix[20]

    # Important step, contains the "Volume" --> Number of nodes at <= distance
    cumulative_count = np.cumsum(count_by_distance_average)
    cumulative_std = np.cumsum(count_by_distance_std)

    print(count_by_distance_average)
    print(cumulative_count)

    ### Adding this here, careful with disruptions
    count_by_distance_average = count_by_distance_average[:david_thresh]
    cumulative_count = cumulative_count[:david_thresh]


    # Fast distance approximation
    predicted_dimensions = (count_by_distance_average / cumulative_count) * np.arange(1, david_thresh + 1)
    print("PREDICTED DIMENSIONS", predicted_dimensions)

    predicted_dimensions_log = np.log(cumulative_count)/np.log(np.arange(1, david_thresh + 1))
    print("OTHER PREDICTED WITH LOG", predicted_dimensions_log)


    plot_folder = args.directory_map['plots_predicted_dimension']
    save_path = f'{plot_folder}/dimension_prediction_by_node_count_{args.args_title}.svg'
    plt.figure()
    x = np.arange(1, david_thresh + 1)
    y = cumulative_count
    y_std = cumulative_std
    x = x[:david_thresh]
    y = y[:david_thresh]
    y_std = y_std[:david_thresh]
    curve_fitting_object = CurveFitting(x, y, y_error_std=None)
    # curve_fitting_object = CurveFitting(x, y, y_error_std=y_std)

    print("YSTD", curve_fitting_object.y_error_std)
    print(len(y_std), len(x))

    # # Fixed power model
    # # curve_fitting_object.fixed_a = args.average_degree
    # curve_fitting_object.fixed_a = cumulative_count[0]
    # func_fit = curve_fitting_object.power_model_fixed

    # # Fixing dimension
    # curve_fitting_object.fixed_b = args.dim
    # func_fit = curve_fitting_object.power_model_fixed_exp

    # Unfixed power model (2 parameters)
    func_fit = curve_fitting_object.power_model

    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object.plot_fit_with_uncertainty(func_fit, "Distance", "Node Count",
                                                   "Dimension Prediction", save_path)


    # Apply the logarithm and just do linear regression
    save_path = f'{plot_folder}/dimension_prediction_by_node_count_LINEAR_{args.args_title}.svg'
    x = np.log(np.arange(1, david_thresh + 1))
    y = np.log(cumulative_count)
    y_std = np.log(cumulative_std)

    # Thresholding the values for finite size effects
    x = x[:david_thresh]
    y = y[:david_thresh]
    y_std = y_std[:david_thresh]
    # curve_fitting_object_linear = CurveFitting(x, y, y_error_std=y_std)
    curve_fitting_object_linear = CurveFitting(x, y, y_error_std=None)
    func_fit = curve_fitting_object_linear.linear_model
    curve_fitting_object_linear.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object_linear.plot_fit_with_uncertainty(func_fit, "Log Distance", "Log Node Count",
                                                   "Dimension Prediction", save_path)

    print(distance_count_matrix)
    print(distance_count_matrix.shape)
    print(count_by_distance_average)

    if curve_fitting_object.fixed_a is not None:
        indx = 0
    else:
        indx = 1
    predicted_dimension = curve_fitting_object.popt[indx]
    r_squared = curve_fitting_object.r_squared
    perr = np.sqrt(np.diag(curve_fitting_object.pcov))
    uncertainty_predicted_dimension = perr[indx]
    results_dimension_prediction = {"predicted_dimension": predicted_dimension, "r2": r_squared,
                                    "std_predicted_dimension": uncertainty_predicted_dimension}


    ### Surface prediction
    save_path = f'{plot_folder}/surface_dimension_prediction_{args.args_title}.svg'
    x = np.arange(1, david_thresh + 1)
    y = count_by_distance_average
    y_std = cumulative_std
    x = x[:david_thresh]
    y = y[:david_thresh]
    y_std = y_std[:david_thresh]
    curve_fitting_object = CurveFitting(x, y, y_error_std=None)
    func_fit = curve_fitting_object.power_model

    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object.plot_fit_with_uncertainty(func_fit, "Distance", "Node Count",
                                                   "Dimension Prediction", save_path)
    return results_dimension_prediction


def reorder_sp_matrix_so_index_matches_nodeid(igraph_graph, sp_matrix):
    for node in igraph_graph.vs:
        print(node['name'], node.index)


def reorder_sp_matrix_so_index_matches_nodeid(igraph_graph, sp_matrix):
    index_to_name = {node.index: node['name'] for node in igraph_graph.vs}
    print("INDEX TO NAME", index_to_name)
    reordered_matrix = np.zeros_like(sp_matrix)

    for current_idx, node in enumerate(igraph_graph.vs):

        new_idx = index_to_name[node.index]   # name as the new index
        print("old index", node.index, "new index", new_idx)
        reordered_matrix[new_idx] = sp_matrix[current_idx]
    return reordered_matrix


def find_nodes_at_distance(sp_matrix, node, distance):
    """
    Find all nodes that are at a specific distance from the given node.
    """
    return np.where(sp_matrix[node] == distance)[0]

def calculate_distances_between_nodes(sp_matrix, nodes1, nodes2):
    """
    Calculate distances between each pair of nodes from two lists of nodes.
    """
    distances = []
    for node1 in nodes1:
        for node2 in nodes2:
            distances.append(sp_matrix[node1, node2])
    return distances

def plot_barplot(args, distances, title):
    """
    Plot a bar plot of the distances with percentages on top of each bar.
    """
    # Count the frequency of each distance
    distances = np.array(distances).astype(int)
    distance_counts = np.bincount(distances)
    max_distance = len(distance_counts)
    total_counts = np.sum(distance_counts)

    # Calculate cumulative percentages
    cumulative_percentages = np.cumsum(distance_counts) / total_counts

    # Initialize a large figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Bar plot on the first subplot
    bars = ax1.bar(range(max_distance), distance_counts, align='center')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval/total_counts:.2%}', va='bottom', ha='center')
    ax1.set_xticks(range(max_distance))
    ax1.set_title("Bar Plot of Distances")
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')

    # Cumulative percentage plot on the second subplot
    ax2.plot(range(max_distance), cumulative_percentages, marker='o', linestyle='-')
    ax2.set_xticks(range(max_distance))
    ax2.set_title("Cumulative Percentage Plot")
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Cumulative Percentage')


    # Set overall title and layout
    plt.suptitle(title)
    plt.tight_layout()

    plot_folder = args.directory_map['plots_predicted_dimension']
    plt.savefig(f"{plot_folder}/barplot_distance_ratio_{title}.png")



def generate_iterative_predictions_data():
    false_edges_list = [0, 20, 40, 60, 80, 100]  # Example list of false edges to add
    original_dims = [2, 3]
    results = []

    for dim in original_dims:
        # Parameters
        args = GraphArgs()
        args.proximity_mode = "knn_bipartite"
        args.dim = dim
        args.intended_av_degree = 10
        args.num_points = 5000
        create_proximity_graph.write_proximity_graph(args)
        sparse_graph, _ = load_graph(args, load_mode='sparse')

        max_false_edges = max(false_edges_list)  # Assume false_edge_list is defined
        all_random_false_edges = select_false_edges_csr(sparse_graph, max_false_edges)

        for num_edges in false_edges_list:
            modified_graph = add_specific_random_edges_to_csrgraph(sparse_graph.copy(), all_random_false_edges,
                                                                   num_edges)
            sp_matrix = np.array(shortest_path(csgraph=modified_graph, directed=False))
            msp = sp_matrix.mean()
            dist_threshold = int(msp) - 2  # finite size effects, careful
            dim_prediction_results = run_dimension_prediction(args, distance_matrix=sp_matrix,
                                                              dist_threshold=dist_threshold, central_node_index=None)
            results.append({
                'original_dim': dim,
                'false_edges': num_edges,
                'predicted_dim': dim_prediction_results['predicted_dimension'],
                'std_predicted_dimension': dim_prediction_results['std_predicted_dimension'],
                'r2': dim_prediction_results['r2']
            })
    return results

def make_dimension_prediction_plot():
    plt.style.use(['no-latex', 'nature'])

    sns.set_style("white")  # 'white' is a style option in seaborn
    font_size = 24
    # If you want to use a seaborn style with modifications
    sns.set(style="white", rc={
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 6,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size - 10
    })

    args = GraphArgs()
    args.proximity_mode = "knn"
    plot_folder = args.directory_map["dimension_prediction_iterations"]
    data = generate_iterative_predictions_data()


    sns.set(style="white")  # Using seaborn for better styling
    fig, ax = plt.subplots()

    for dim in set(d['original_dim'] for d in data):
        dim_data = [d for d in data if d['original_dim'] == dim]
        false_edges = [d['false_edges'] for d in dim_data]
        predicted_dims = [d['predicted_dim'] for d in dim_data]
        # Setting colors based on dim
        if dim == 2:
            color = '#009ADE'
        elif dim == 3:
            color = '#FF1F5B'
        else:
            color = 'gray'  # Default color for other dimensions, if any
        ax.plot(false_edges, predicted_dims, '-o', label=f'Original dim {dim}', color=color)
    ax.legend()
        # ax.errorbar(false_edges, predicted_dims, yerr=std_devs, fmt='-o', label=f'Original dim {dim}', c=)

    ax.set_xlabel('Number of False Edges')
    ax.set_ylabel('Predicted Dimension')
    ax.set_xticks(false_edges)  # Ensuring all false edge counts are shown
    ax.legend(loc='best')


    plt.savefig(f"{plot_folder}/dimension_prediction_iterations.svg", format='svg')


# Parameters
args = GraphArgs()
args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
args.proximity_mode = "knn_bipartite"
args.dim = 3

args.intended_av_degree = 10
args.num_points = 5000


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
    # weinstein_data.csv

    args.edge_list_title = "weinstein_data_january_corrected.csv"
    # args.edge_list_title = "mst_N=1024_dim=2_lattice_k=15.csv"  # Seems to have dimension 1.5

    weighted = True
    weight_threshold = 4

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

    ## Original data
    edge_list = read_edge_list(args)
    original_positions = read_position_df(args=args)
    # plot_original_or_reconstructed_image(args, image_type="original", edges_df=edge_list)
    original_dist_matrix = compute_distance_matrix(original_positions)
else:
    raise ValueError("Please input a valid simulation or experiment mode")





# ### Add random edges? See efect in the dimensionality here
# sparse_graph = add_random_edges_to_csrgraph(sparse_graph, num_edges_to_add=5)

# Compute shortest path matrix
sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))

# node_of_interest = 0
# # Find nodes at distance 3 and 4
# nodes_at_distance_3 = find_nodes_at_distance(sp_matrix, node_of_interest, 3)
# nodes_at_distance_4 = find_nodes_at_distance(sp_matrix, node_of_interest, 4)
# distances = calculate_distances_between_nodes(sp_matrix, nodes_at_distance_3, nodes_at_distance_4)
# plot_barplot(args, distances, "distances_3_4")


msp = sp_matrix.mean()
print("AVERAGE SHORTEST PATH", msp)
print("RANDOM NETWORK AV SP", np.log(args.num_points)/np.log(args.average_degree))
print("ESTIMATED LATTICE SP", (args.num_points/args.average_degree)**(1/args.dim))
print("ESTIMATED LATTICE SP", (args.num_points**(1/args.dim) /args.average_degree))
print("ESTIMATED LATTICE SP CURATED 2D", 1.2*(args.num_points/args.average_degree)**(1/args.dim))
print("ESTIMATED LATTICE SP CURATED 3D", (1.2*0.75)*(args.num_points/args.average_degree)**(1/args.dim))
print("ESTIMATED LATTICE SP CURATED 3D inverse", (1.2*(4/3))*(args.num_points/args.average_degree)**(1/args.dim))
print("ESTIMATED LATTICE SP CURATED 2D BIPARTITE", 1.2*(args.num_points/(args.average_degree*2))**(1/args.dim))
print("ESTIMATED LATTICE SP CURATED 3D BIPARTITE", (1.2*(4/3))*(args.num_points/(args.average_degree*2))**(1/args.dim))
print("ESTIMATED LATTICE SP CURATED 3D 1.1", (1.2*(1.1))*(args.num_points/args.average_degree)**(1/args.dim))
# np.set_printoptions(threshold=np.inf)
# sp_matrix = np.array(sparse_graph.distances())
# reordered_sp_matrix = reorder_sp_matrix_so_index_matches_nodeid(sparse_graph, sp_matrix)

# edge1, edge2 = edge_list.iloc[0][0], edge_list.iloc[0][1]
# print(edge1, edge2)
# correlation = compute_correlation_between_distance_matrices(original_dist_matrix, sp_matrix)
# print("original", original_dist_matrix[edge1][edge2])
# print("shortest path", sp_matrix[edge1][edge2])
# print("Correlation:", correlation)


# # Original dimension prediction
# central_node_index = run_dimension_prediction_continuous(args, distance_matrix=original_dist_matrix, num_bins=50)

## Network dimension prediction
dist_threshold = int(msp) - 1  #finite size effects, careful
run_dimension_prediction(args, distance_matrix=sp_matrix, dist_threshold=dist_threshold, central_node_index=None)





# igraph_graph = load_graph(args, load_mode='igraph')
#
# edge_list = read_edge_list(args)
# original_positions = read_position_df(args=args)
# original_dist_matrix = compute_distance_matrix(original_positions)
#
#
#
# # np.set_printoptions(threshold=np.inf)
# sp_matrix = np.array(igraph_graph.distances())
# reordered_sp_matrix = reorder_sp_matrix_so_index_matches_nodeid(igraph_graph, sp_matrix)
#
# edge1, edge2 = edge_list.iloc[0][0], edge_list.iloc[0][1]
# print(edge1, edge2)
# correlation = compute_correlation_between_distance_matrices(original_dist_matrix, reordered_sp_matrix)
# print("original", original_dist_matrix[edge1][edge2])
# print("shortest path", sp_matrix[edge1][edge2])
# print("Correlation:", correlation)
#
# # dist_threshold = 6  # Get the 1st six columns (finite size effects)
# # run_dimension_prediction(args, distance_matrix=sp_matrix, dist_threshold=dist_threshold)





