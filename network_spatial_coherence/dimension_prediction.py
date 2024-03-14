import matplotlib.pyplot as plt
import numpy as np

from curve_fitting import CurveFitting
from spatial_constant_analysis import *
from utils import read_position_df
import scipy.stats as stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csgraph
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import norm
from scipy.stats import linregress
from data_analysis import calculate_figsize_n_subplots

# font_size = 24
# plt.style.use(['no-latex', 'nature'])
#
# sns.set_style("white")  # 'white' is a style option in seaborn
#
# # If you want to use a seaborn style with modifications
# sns.set(style="white", rc={
#     'axes.labelsize': font_size,
#     'axes.titlesize': font_size + 6,
#     'xtick.labelsize': font_size,
#     'ytick.labelsize': font_size,
#     'legend.fontsize': font_size - 10
# })

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

def run_dimension_prediction_continuous(args, distance_matrix, num_bins=10, plot_heatmap_all_nodes=True):
    # Determine the range and bin width

    max_distance = args.L
    min_distance = 0

    # Create bins for distances
    original_bins = np.linspace(min_distance, max_distance, num_bins + 1)[1:]
    bins = np.linspace(min_distance, max_distance, num_bins+1)[1:]

    # original_bins = np.arange(min_distance, max_distance, bin_width)
    # bins = np.arange(min_distance, max_distance, bin_width)
    # Initialize a matrix to count nodes in each bin

    print("BINS", bins)
    print("DISTANCE MATRIX", distance_matrix)
    binned_distance_counts = np.zeros((distance_matrix.shape[0], len(bins)))

    # Group distances into bins and count
    for i in range(distance_matrix.shape[0]):
        for j, bin_edge in enumerate(bins):
            if j == 0:
                continue
            binned_distance_counts[i, j] = np.sum((distance_matrix[i] > bins[j - 1]) & (distance_matrix[i] <= bins[j]))



    # Row with maximum number of nodes
    # Compute the sum of each row
    row_sums = np.sum(distance_matrix, axis=1)

    # Find the index of the row with the maximum sum (probably central node)
    max_sum_index = np.argmin(row_sums)



    # # Calculate average counts per bin
    # count_by_distance_average = np.mean(binned_distance_counts, axis=0)   #TODO: is the axis right?
    # std_distance_average = np.std(binned_distance_counts, axis=0)

    # delete this otherwise
    count_by_distance_average = binned_distance_counts[max_sum_index]  # 1st row  (just to omit the finite size effects)

    # Calculate cumulative counts
    cumulative_count = np.cumsum(count_by_distance_average)




    # #### Plotting (not developed)
    # # Load original positions
    # original_position_folder = args.directory_map["original_positions"]
    # positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")
    #
    # # Plot
    # plt.close('all')
    # fig = plt.figure(figsize=(10, 6))
    #
    # if plot_in_3d:
    #     ax = fig.add_subplot(111, projection='3d')
    #     scatter = ax.scatter(positions_df['x'], positions_df['y'], positions_df.get('z', 0),
    #                          c=positions_df['predicted_dimension'], cmap='viridis')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z' if 'z' in positions_df.columns else 'Predicted Dimension')
    # else:
    #     scatter = plt.scatter(positions_df['x'], positions_df['y'], c=positions_df['predicted_dimension'],
    #                           cmap='viridis')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')


    print("BINS", bins)
    print("COUNT BY DISTANCE AVERAGE", count_by_distance_average)
    print("CUMULATIVE COUNT", cumulative_count.astype(int))
    old_cum = cumulative_count


    # Fast distance approximation
    predicted_dimensions = (count_by_distance_average / cumulative_count) * np.arange(0, len(cumulative_count))
    print("PREDICTED DIMENSIONS", predicted_dimensions)

    # Try to delete finite size effects
    cumulative_count = cumulative_count[1: int(len(cumulative_count)/2)]
    bins = bins[1: int(len(bins)/2)]

    print("bins", bins)
    print("CUMULATIVE COUNT", old_cum)
    print("curated cumulative count", cumulative_count)
    if args.dim == 3:
        expected_cumulative_count = (args.num_points * (4/3) * np.pi * (bins**3)).astype(int)
    elif args.dim == 2:
        expected_cumulative_count = (args.num_points * np.pi * (bins**2)).astype(int)  # rho * V = N
        expected_cumulative_count_sphere = (args.num_points * (bins ** 2)).astype(int)  # rho * V = N
        print("EXPECTED CUMULATIVE POINTS sphere", expected_cumulative_count_sphere)
    print("EXPECTED CUMULATIVE POINTS", expected_cumulative_count)


    plot_folder = args.directory_map['plots_predicted_dimension']
    save_path = f'{plot_folder}/dimension_prediction_original_by_node_count_{args.args_title}.png'
    plt.figure()
    curve_fitting_object = CurveFitting(bins, cumulative_count)


    # Unfixed parameters
    func_fit = curve_fitting_object.power_model


    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object.plot_fit_with_uncertainty(func_fit, "Distance", "Node Count",
                                                   "Dimension Prediction", save_path)

    dist_threshold = int(len(bins)/2)
    print("BINNED DISTANCE COUNTS", binned_distance_counts)
    print("BINS", bins)
    if plot_heatmap_all_nodes:
        fig_data =compute_and_plot_predicted_dimensions_for_all_nodes(args=args, dist_threshold=dist_threshold,
                                                            distance_count_matrix=binned_distance_counts,
                                                            plot_in_3d=(args.dim == 3), euclidean=True,
                                                            binned_distance=original_bins, central_index=max_sum_index)

        return max_sum_index, fig_data
    else:
        return max_sum_index


def plot_dimension_fit(args, dist_threshold, cumulative_count, surface_count, cumulative_std=None):
    plot_folder = args.directory_map['plots_predicted_dimension']
    save_path = f'{plot_folder}/dimension_prediction_by_node_count_{args.args_title}.svg'
    plt.figure()
    x = np.arange(1, dist_threshold + 1)
    y = cumulative_count
    # y_std = cumulative_std
    x = x[:dist_threshold]
    y = y[:dist_threshold]
    # y_std = y_std[:dist_threshold]
    curve_fitting_object = CurveFitting(x, y, y_error_std=None)
    # curve_fitting_object = CurveFitting(x, y, y_error_std=y_std)

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

    plt.close("all")
    # Apply the logarithm and just do linear regression
    save_path = f'{plot_folder}/dimension_prediction_by_node_count_LINEAR_{args.args_title}.svg'
    x = np.log(np.arange(1, dist_threshold + 1))
    y = np.log(cumulative_count)


    # y_std = np.log(cumulative_std)

    # Thresholding the values for finite size effects
    x = x[:dist_threshold]
    y = y[:dist_threshold]
    # y_std = y_std[:dist_threshold]

    # curve_fitting_object_linear = CurveFitting(x, y, y_error_std=y_std)
    curve_fitting_object_linear = CurveFitting(x, y, y_error_std=None)
    func_fit = curve_fitting_object_linear.linear_model
    curve_fitting_object_linear.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object_linear.plot_fit_with_uncertainty(func_fit, "Log Distance", "Log Node Count",
                                                          "Dimension Prediction", save_path)

    if args.show_plots:
        plt.show()
        plt.close()

    ## Storing it in main plot function
    plot_folder2 = args.directory_map['spatial_coherence']
    save_path2 = f'{plot_folder2}/dimension_prediction_by_node_count_LINEAR_{args.args_title}.svg'
    curve_fitting_object_linear.plot_fit_with_uncertainty(func_fit, "Log Distance", "Log Node Count",
                                                          "Dimension Prediction", save_path2)


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
    if args.verbose:
        print("PREDICTED DIMENSION power fit", predicted_dimension)
        print("UNCERTAINTY PREDICTED DIMENSION", uncertainty_predicted_dimension)
        print("PREDICTED DIMENSION linear fit", curve_fitting_object_linear.popt[indx])
        print("RESULTS DIMENSION PREDICTION", results_dimension_prediction)


    ### Surface prediction
    save_path = f'{plot_folder}/surface_dimension_prediction_{args.args_title}.svg'
    x = np.arange(1, dist_threshold + 1)
    y = surface_count
    # y_std = cumulative_std
    x = x[:dist_threshold]
    y = y[:dist_threshold]
    # y_std = y_std[:dist_threshold]
    curve_fitting_object = CurveFitting(x, y, y_error_std=None)
    func_fit = curve_fitting_object.power_model

    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    curve_fitting_object.plot_fit_with_uncertainty(func_fit, "Distance", "Node Count",
                                                   "Dimension Prediction", save_path)
    plt.close('all')
    return results_dimension_prediction



def compute_local_dimension(args, distance_matrix, central_node_indices, dist_threshold):
    """
    Computes and plots the local dimension for several central nodes, including the mean and std of all central node indices.

    Args:
        args: An object containing configuration parameters and options for the analysis.
        distance_matrix (numpy.ndarray): A 2D numpy array representing the pairwise shortest path distances between nodes.
        central_node_indices (list or numpy.ndarray): Indices of the central nodes.
        dist_threshold (int): The maximum distance threshold to consider.
    """
    # Initialize lists to store predicted dimensions for all central nodes
    all_predicted_dimensions = []

    for central_node_index in central_node_indices:
        distances_from_central = distance_matrix[central_node_index, :]
        count_by_distance = np.bincount(distances_from_central.astype(int), minlength=dist_threshold + 1)[
                            1:dist_threshold + 1]
        cumulative_count = np.cumsum(count_by_distance)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            predicted_dimensions = (count_by_distance / cumulative_count) * np.arange(1, dist_threshold + 1)
            predicted_dimensions[~np.isfinite(predicted_dimensions)] = 0  # Replace inf and NaN with 0

        # Store the predicted dimensions for this central node
        all_predicted_dimensions.append(predicted_dimensions)

    # Convert list of arrays into a 2D numpy array for easier manipulation
    all_predicted_dimensions = np.array(all_predicted_dimensions)

    # Compute mean and std of predicted dimensions across all central nodes
    mean_predicted_dimensions = np.mean(all_predicted_dimensions, axis=0)
    std_predicted_dimensions = np.std(all_predicted_dimensions, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    distances = np.arange(1, dist_threshold + 1)
    plt.plot(distances, mean_predicted_dimensions, label='Mean Predicted Dimension', color='blue', marker='o')

    # Adding the std deviation as a filled color "ribbon"
    plt.fill_between(distances, mean_predicted_dimensions - std_predicted_dimensions,
                     mean_predicted_dimensions + std_predicted_dimensions, color='blue', alpha=0.2,
                     label='Std Deviation')

    plot_folder = args.directory_map['local_dimension']
    plt.xlabel('Distance')
    plt.ylabel('Predicted Dimension')
    plt.title('Local Dimension for Central Nodes')
    plt.legend()


    plt.savefig(f'{plot_folder}/local_dimension_{args.args_title}.png')
    if args.show_plots:
        plt.show()
    plt.close()




def run_dimension_prediction(args, distance_matrix, dist_threshold=6,
                             msp_central_node=False, local_dimension=False, num_central_nodes=10,
                             plot_heatmap_all_nodes=False):
    """
    Performs dimension prediction for a given graph based on its distance matrix. This includes
    computing a node counts matrix, determining the surface and volume growth, and applying curve
    fitting to predict graph dimensions. Optionally, predictions can be based on a specific central
    node or an automatically determined node with the highest connectivity within a given distance threshold.

    Args:
        args: An object containing configuration parameters and options for the analysis, including
              directory mappings (`directory_map`) for saving plots and graph analysis settings.
        distance_matrix: A numpy array representing the pairwise shortest path distances between nodes
                         in the graph.
        dist_threshold (int): The maximum distance threshold to consider for counting nodes and predicting
                              dimensions. Defaults to 6.
        central_node_index (int, optional): The index of a central node to base the predictions on. If not
                                            provided, a node with the maximum sum of connections within the
                                            `dist_threshold` is used.

    Returns:
        dict: A dictionary containing the predicted dimension, the coefficient of determination (R^2) of
              the curve fitting, and the standard deviation of the predicted dimension.

    Raises:
        ValueError: If the provided `distance_matrix` does not meet expected conditions (e.g., not a 2D array).

    Note:
        - The function employs several sub-steps, including computing node counts, averaging counts across
          distances, performing curve fitting for dimension prediction, and generating relevant plots.
        - The dimension prediction is sensitive to the choice of `dist_threshold` and the specified or
          identified `central_node_index`.
        - Plots generated by the function are saved in the directory specified by `args.directory_map['plots_predicted_dimension']`.
    """

    distance_count_matrix = compute_node_counts_matrix(distance_matrix)

    ### Select Central node. If not provide it, we compute it ourselves

    # TODO: modify this to include more nodes
    central_nodes = find_central_nodes(distance_matrix=distance_matrix, num_central_nodes=num_central_nodes)

    ## Grab only the most central node
    max_central_node = central_nodes[0]
    count_by_distance_average = distance_count_matrix[max_central_node]

    ## Central sp distance plot with prediction. #TODO: check if central node is more predictive than just mean count (mean count seems better?)
    compute_centered_average_sp_distance(args, count_by_distance_average=count_by_distance_average, shell_threshold=dist_threshold+5)
    # Important step, contains the "Volume" --> Number of nodes at <= distance
    cumulative_count = np.cumsum(count_by_distance_average)

    if args.verbose:
        print("Surface Counts", count_by_distance_average)
        print("Volume Counts", cumulative_count)

    ### Adding this here, careful with disruptions
    count_by_distance_average = count_by_distance_average[:dist_threshold]
    cumulative_count = cumulative_count[:dist_threshold]

    if msp_central_node:
        # Mean shortest path on the central node only
        msp_approx = np.sum(count_by_distance_average * np.arange(len(count_by_distance_average))) / np.sum(count_by_distance_average)
        if args.verbose:
            print("MEAN SHORTEST PATH APPROXIMATION BY CENTRAL NODE", msp_approx)
            print("MAXIMUM SHELL RANGE USED", len(count_by_distance_average))
            print("PREDICTION OF SP APPROXIMATION", predict_sp(args.dim, n=len(count_by_distance_average)))


    if local_dimension:
        # Local dimension approximation
        predicted_dimensions = (count_by_distance_average / cumulative_count) * np.arange(1, dist_threshold + 1)
        if args.verbose:
            print("PREDICTED DIMENSIONS", predicted_dimensions)
        compute_local_dimension(args=args, distance_matrix=distance_matrix, dist_threshold=dist_threshold,
                                central_node_indices=central_nodes)


    results_dimension_prediction = plot_dimension_fit(args, dist_threshold, cumulative_count, cumulative_std=None,
                                                      surface_count=count_by_distance_average)



    if num_central_nodes > 1:

        # TODO: generate a plot to represent the multiple predicted dimensions
        ### Take into account several central nodes for dimension prediction
        results_dimension_prediction = {}
        predicted_dimensions = []
        std_errors = []

        for idx, central_node in enumerate(central_nodes):
            count_by_distance_average = distance_count_matrix[central_node]
            cumulative_count = np.cumsum(count_by_distance_average)
            x_data = np.arange(1, dist_threshold + 1)
            y_data = cumulative_count[:dist_threshold]
            filtered_indices = y_data > 0  # Create a mask for y_data values that are positive
            filtered_x_data = x_data[filtered_indices]
            filtered_y_data = y_data[filtered_indices]
            log_x_data = np.log(filtered_x_data)
            log_y_data = np.log(filtered_y_data)
            slope, intercept, r_value, p_value, std_err = linregress(log_x_data, log_y_data)
            predicted_dimension = slope
            predicted_dimensions.append(predicted_dimension)
            std_errors.append(std_err)


        variances = np.array(std_errors) ** 2
        weighted_avg_dimension = np.sum(np.array(predicted_dimensions) / variances) / np.sum(1 / variances)
        std_error_weighted_avg = np.sqrt(1 / np.sum(1 / variances))
        results_dimension_prediction['predicted_dimension'] = weighted_avg_dimension
        results_dimension_prediction['std_predicted_dimension'] = std_error_weighted_avg
        results_dimension_prediction['predicted_dimension_list'] = predicted_dimensions
        results_dimension_prediction['std_predicted_dimension_list'] = std_errors
        print("RESULTS DIMENSION PREDICTION FOR SEVERAL CENTRAL NODES", results_dimension_prediction)


    if plot_heatmap_all_nodes:
        fig_data = compute_and_plot_predicted_dimensions_for_all_nodes(args, distance_count_matrix, dist_threshold,
                                                            plot_in_3d=(args.dim == 3), central_index=max_central_node)
        return results_dimension_prediction, fig_data, max_central_node
    else:
        return results_dimension_prediction




def compute_and_plot_predicted_dimensions_for_all_nodes(args, distance_count_matrix, dist_threshold, plot_in_3d=False,
                                                        euclidean=False, binned_distance=None, central_index=None):
    predicted_dimensions = []
    dimension_error = []

    fig_loglog, ax_loglog = plt.subplots()

    # node_count_list = []
    # indices_to_check = []
    # x_data_to_check = []
    # y_data_to_check = []
    for idx, row_count in enumerate(distance_count_matrix):
        skip = False
        count_by_distance_average = row_count
        cumulative_count = np.cumsum(count_by_distance_average)

        # Make sure we have no zeros in cumulative_count for the log operation
        if euclidean and binned_distance is not None:
            # Avoid early and late finite size effects
            x_data = binned_distance[5: int(len(binned_distance)/2) ]
            y_data = cumulative_count[5: int(len(binned_distance)/2)]
            # node_count = y_data[-1]
            # node_count_list.append(node_coun
        else:
            x_data = np.arange(1, dist_threshold + 1)
            y_data = cumulative_count[:dist_threshold]

        log_x_data = np.log(x_data)
        log_y_data = np.log(y_data)

        slope, intercept, r_value, p_value, std_err = linregress(log_x_data, log_y_data)
        predicted_dimension = slope
        if central_index is not None:
            if idx == central_index:

                if args.verbose:
                    print("CENTRAL INDEX", idx)
                    print(f"PREDICTED DIMENSION CENTRAL INDEX, euclidean = {euclidean}", predicted_dimension)
                    print("x data", x_data)
                    print("y data", y_data)
                log_x_data_central = log_x_data
                log_y_data_central = log_y_data
                slope_central = slope
                intercept_central = intercept

                # ## Central Index Plotting  #TODO: right now I plot this with my pipeline
                # ax_loglog.plot(log_x_data, log_y_data, 'o', label='Data Points')
                # ax_loglog.plot(log_x_data, slope * log_x_data + intercept, 'r-',
                #         label=f'Fit: dimension={predicted_dimension:.2f}')
                # ax_loglog.set_xlabel('Log(Distance)')
                # ax_loglog.set_ylabel('Log(Count)')
                # ax_loglog.set_title('Predicted Dimension for Central Node')
                # ax_loglog.legend()
                #
                # if args.show_plots:
                #     plt.show()
                # plt.close()

        predicted_dimensions.append(predicted_dimension)
        dimension_error.append(std_err)



        #     if predicted_dimension > 2:
        #         print("PREDICTED DIMENSION OUTLIER", predicted_dimension)
        #         print("Position")
        #         print("distances", x_data)
        #         print("count", y_data)
        #
        #         expected_cumulative_count_sphere = (args.num_points * (x_data ** 2)).astype(int)  # rho * V = N
        #         print("EXPECTED CUMULATIVE POINTS sphere", expected_cumulative_count_sphere)
        #
        #         print("filtered x data", filtered_x_data)
        #         print("filtered y data", filtered_y_data)
        #         indices_to_check.append(idx)
        #     x_data_to_check.append(x_data)
        #     y_data_to_check.append(y_data)
        # else:
        #     predicted_dimension = 1
        #     predicted_dimensions.append(predicted_dimension)

    # Load original positions
    original_position_folder = args.directory_map["original_positions"]
    positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

    if args.node_ids_map_old_to_new:
        positions_df['node_ID'] = positions_df['node_ID'].map(args.node_ids_map_old_to_new)
        positions_df = positions_df.dropna()
        positions_df['node_ID'] = positions_df['node_ID'].astype(int)

    positions_df['predicted_dimension'] = predicted_dimensions[:len(positions_df)]

    positions_df['dimension_error'] = dimension_error[:len(positions_df)]
    # positions_df['node_count'] = node_count_list[:len(positions_df)]
    #
    # print("HOLAAAAAAAAAAAAAA")
    # print("indices to check", indices_to_check)
    # for idx in indices_to_check:
    #
    #     x_data = x_data_to_check[idx]
    #     y_data = y_data_to_check[idx]
    #     position = (positions_df['x'].iloc[idx], positions_df['y'].iloc[idx])
    #
    #     if position[0] < 0.1:
    #         print("ALERTA")
    #     print("Position", position)
    #     print("distances", x_data)
    #     print("count", y_data)
    #     log_x_data = np.log(x_data)
    #     log_y_data = np.log(y_data)
    #
    #     slope, intercept, r_value, p_value, std_err = linregress(log_x_data, log_y_data)
    #     predicted_dimension = slope
    #     print("slope, intercept", slope, intercept)


    # plt.close('all')
    # fig = plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(positions_df['x'], positions_df['y'], c=positions_df['node_count'],
    #                       cmap='viridis')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.colorbar(scatter, label='Count')
    # plt.show()

    plot_folder = args.directory_map['heatmap_local']
    if euclidean:
        title = "euclidean"
    else:
        title = "network"

    if args.num_points <= 2000:
        form = 'svg'
    else:
        form = 'png'




    # plt.close('all')
    # fig = plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(positions_df['x'], positions_df['y'], c=positions_df['dimension_error'],
    #                       cmap='viridis')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.colorbar(scatter, label='Dimension STD')
    # plt.savefig(f'{plot_folder}/heatmap_predicted_dimension_std_{args.args_title}_{title}', format=form)
    # if args.show_plots:
    #     plt.show()
    # plt.close()

    ### Dimension Prediction
    plt.close('all')
    if 'z' in positions_df.columns:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'],
                             c=positions_df['predicted_dimension'], cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(positions_df['x'], positions_df['y'], c=positions_df['predicted_dimension'], cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    plt.colorbar(scatter, label='Predicted Dimension')
    plt.savefig(f'{plot_folder}/heatmap_predicted_dimension_{args.args_title}_{title}', format=form)
    if args.show_plots:
        plt.show()
    plt.close()

    #### Dimension STD
    plt.close('all')
    if 'z' in positions_df.columns:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'], c=positions_df['dimension_error'],
                             cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(positions_df['x'], positions_df['y'], c=positions_df['dimension_error'], cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    plt.colorbar(scatter, label='Dimension STD')
    plt.savefig(f'{plot_folder}/heatmap_predicted_dimension_std_{args.args_title}_{title}', format=form)
    if args.show_plots:
        plt.show()
    plt.close()

    figure_data = {
        'log_x_data': log_x_data_central,
        'log_y_data': log_y_data_central,
        'slope': slope_central,
        'intercept': intercept_central,
        'predicted_dimension': predicted_dimensions
    }

    return figure_data


def avg_shortest_path_mean_line_segment(args, sparse_graph, distance_matrix, dist_threshold=6, central_node_index=None):
    # TODO: take this seriously and compare predictions. How is it related to the spatial constant?
    distance_count_matrix = compute_node_counts_matrix(distance_matrix)
    dist_threshold = dist_threshold

    ## This is to find a central node
    row_sums = np.sum(distance_count_matrix[:, 0:dist_threshold], axis=1)
    max_sum_index = np.argmax(row_sums)
    print("MAX SUM NETWORK", np.sum(distance_count_matrix[:, 0:dist_threshold][max_sum_index]))
    print(max_sum_index, central_node_index)


    ### Select Central node
    if central_node_index:
        count_by_distance_average = distance_count_matrix[central_node_index]
        print("count based on euclidean", count_by_distance_average)
        print(np.cumsum(count_by_distance_average))
    count_by_distance_average = distance_count_matrix[max_sum_index]
    print("count based on network", count_by_distance_average)
    print(np.cumsum(count_by_distance_average))

    msp = compute_subgraph_mean_shortest_path(sparse_graph=sparse_graph, distance_matrix=distance_matrix,
                                        central_node_index=max_sum_index, dist_threshold=dist_threshold)

    print("shell distance", dist_threshold)
    print("All-pairs mean shortest path", msp)
    print("Predicted (euclidean) MSP 2D", 0.9*dist_threshold)
    print("Predicted (euclidean) MSP 3D", 1.02*dist_threshold)

    max_distance = np.max(distance_matrix[distance_matrix < np.inf])


    ### Compute MSP for different thresholds
    thresholds = range(1, int(max_distance) -5)  # Assuming integer distances
    msp_values = []
    for dist_threshold in thresholds:
        print("current distance threshold", dist_threshold)
        msp = compute_subgraph_mean_shortest_path(sparse_graph, distance_matrix, max_sum_index, dist_threshold)
        msp_values.append(msp)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, msp_values, label='Computed MSP')
    plt.plot(thresholds, [0.9 * d for d in thresholds], '--', label='Predicted MSP 2D (0.9 * distance)')
    plt.plot(thresholds, [1.02 * d for d in thresholds], '--', label='Predicted MSP 3D (1.02 * distance)')

    plt.xlabel('Distance Threshold')
    plt.ylabel('Mean Shortest Path')
    plt.title('Mean Shortest Path vs. Distance Threshold')
    plt.legend()


    plt.savefig(f'{args.directory_map["plots_predicted_dimension"]}/mean_line_segment_{args.args_title}.svg')
    if args.show_plots:
        plt.show()
    plt.close()

    return



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
    if args.show_plots:
        plt.show()
    plt.close()



def generate_iterative_predictions_data(num_central_nodes=1):
    false_edges_list = [0, 20, 40, 60, 80, 100]  # Example list of false edges to add
    original_dims = [2, 3]
    results = []

    for dim in original_dims:
        # Parameters
        args = GraphArgs()

        args.dim = dim
        args.intended_av_degree = 10
        args.num_points = 5000
        args.proximity_mode = "knn_bipartite"
        create_proximity_graph.write_proximity_graph(args)
        sparse_graph = load_graph(args, load_mode='sparse')

        max_false_edges = max(false_edges_list)  # Assume false_edge_list is defined
        all_random_false_edges = select_false_edges_csr(sparse_graph, max_false_edges)

        for num_edges in false_edges_list:
            args.false_edges_count = num_edges

            modified_graph = add_specific_random_edges_to_csrgraph(sparse_graph.copy(), all_random_false_edges,
                                                                   num_edges)
            sp_matrix = np.array(shortest_path(csgraph=modified_graph, directed=False))
            msp = sp_matrix.mean()
            dist_threshold = int(msp) - 2  # finite size effects, careful
            dim_prediction_results = run_dimension_prediction(args=args, distance_matrix=sp_matrix,
                                                              dist_threshold=dist_threshold, local_dimension=False,
                                                              plot_heatmap_all_nodes=True, num_central_nodes=num_central_nodes)
            results.append({
                'original_dim': dim,
                'false_edges': num_edges,
                'predicted_dim': dim_prediction_results['predicted_dimension'],
                'std_predicted_dimension': dim_prediction_results['std_predicted_dimension'],
                # 'r2': dim_prediction_results['r2']
            })
    return results

def make_dimension_prediction_plot(num_central_nodes=1):
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
    data = generate_iterative_predictions_data(num_central_nodes=num_central_nodes)


    sns.set(style="white")  # Using seaborn for better styling
    fig, ax = plt.subplots()

    for dim in set(d['original_dim'] for d in data):
        dim_data = [d for d in data if d['original_dim'] == dim]
        false_edges = [d['false_edges'] for d in dim_data]
        predicted_dims = [d['predicted_dim'] for d in dim_data]
        std_devs = [d['std_predicted_dimension'] for d in dim_data]  # Standard deviations for the ribbon

        # Setting colors based on dim
        if dim == 2:
            color = '#009ADE'
        elif dim == 3:
            color = '#FF1F5B'
        else:
            color = 'gray'  # Default color for other dimensions, if any
        ax.plot(false_edges, predicted_dims, '-o', label=f'Original dim {dim}', color=color)
        # Adding the "ribbon" for standard deviation
        ax.fill_between(false_edges,
                        [d - sd for d, sd in zip(predicted_dims, std_devs)],
                        [d + sd for d, sd in zip(predicted_dims, std_devs)],
                        color=color, alpha=0.2)  # Adjust alpha for ribbon transparency

    ax.legend(loc='best')
        # ax.errorbar(false_edges, predicted_dims, yerr=std_devs, fmt='-o', label=f'Original dim {dim}', c=)

    ax.set_xlabel('Number of False Edges')
    ax.set_ylabel('Predicted Dimension')
    ax.set_xticks(false_edges)  # Ensuring all false edge counts are shown
    ax.legend(loc='best')


    plt.savefig(f"{plot_folder}/dimension_prediction_iterations.svg", format='svg')
    if args.show_plots:
        plt.show()
    plt.close()

def predict_sp(dim=2, n=20):

    # shortest path prediction based on the shortest path with central nodes as origin
    # n is the number of shortest path levels that we take into account

    sum_i = (n*(n+1))/2   # sum(i)
    sum_i2 = (n * (n + 1) * (2 * n + 1)) / 6  # sum(i^2)
    sum_i3 = sum_i**2
    if dim == 2:
        prediction_sp = sum_i2/sum_i
    elif dim == 3:
        prediction_sp = sum_i3/sum_i2
    return prediction_sp

def predict_average_distance_ball(dim=2, distance=1):
    if dim == 2:
        return 2/3 * distance
    elif dim == 3:
        return 3/4 * distance
    else:
        raise ValueError("Wrong dimension")

def compute_centered_average_sp_distance(args, count_by_distance_average, shell_threshold):
    # Calculating msp_approx for each N shell
    n_shell_range = np.arange(1, len(count_by_distance_average) + 1)
    msp_approx_series = [
        np.sum(count_by_distance_average[:n] * np.arange(1, n + 1)) / np.sum(count_by_distance_average[:n]) for n in
        n_shell_range]

    # Assuming a dimension for the predict_sp and your_new_function
    dim = args.dim  # Example dimension, replace with your actual dimension

    # Generating data for y1 and y2 series
    y1_series = [predict_sp(dim, n) for n in n_shell_range]
    y2_series = [predict_average_distance_ball(dim, n) for n in n_shell_range]

    # Calculate the difference between msp_approx_series and y1_series
    difference_series = np.array(msp_approx_series) - np.array(y1_series)

    plt.close('all')
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Creating 2 subplots vertically

    ## thresholding
    y1_series = y1_series[:shell_threshold]
    y2_series = y2_series[:shell_threshold]
    msp_approx_series = msp_approx_series[:shell_threshold]
    difference_series = difference_series[:shell_threshold]
    n_shell_range = n_shell_range[:shell_threshold]

    # First subplot for original data
    axs[0].plot(n_shell_range, y1_series, label='MSP Prediction', marker='o')
    axs[0].plot(n_shell_range, y2_series, label='MSP Euclidean Equivalent', marker='o')
    axs[0].plot(n_shell_range, msp_approx_series, label='MSP Actual', linestyle='--', marker='o')
    axs[0].set_xlabel('N shell')
    axs[0].set_ylabel('MSP')
    axs[0].legend()
    axs[0].set_title('MSP Predictions vs Actual')

    # Second subplot for difference
    axs[1].plot(n_shell_range, difference_series, label='Difference (Actual - Prediction)', marker='o', color='red')
    axs[1].set_xlabel('N shell')
    axs[1].set_ylabel('Difference')
    axs[1].legend()
    axs[1].set_title('Difference between MSP Actual and Prediction')

    plt.tight_layout()  # Adjust layout to not overlap
    plot_folder = args.directory_map['centered_msp']
    plt.savefig(f"{plot_folder}/centered_msp_difference_{args.args_title}.svg")
    if args.show_plots:
        plt.show()
    plt.close()




def compute_subgraph_mean_shortest_path(sparse_graph, distance_matrix, central_node_index, dist_threshold):
    """
    Compute the mean shortest path distance of all pairs within a subgraph defined by a BFS from the central node
    up to a specified distance.

    Parameters:
    - sparse_graph: A sparse matrix representation of the graph.
    - distance_matrix: A dense matrix representing the shortest path distances between all pairs of nodes in the graph.
    - central_node_index: The index of the central node.
    - dist_threshold: The maximum distance from the central node to include nodes in the subgraph.

    Returns:
    - The mean shortest path distance of all pairs within the subgraph.
    """
    n_nodes = sparse_graph.shape[0]
    visited = np.zeros(n_nodes, dtype=bool)
    queue = [(central_node_index, 0)]
    visited[central_node_index] = True
    nodes_in_subgraph = []

    while queue:
        current_node, depth = queue.pop(0)
        if depth <= dist_threshold:
            nodes_in_subgraph.append(current_node)

            neighbors = sparse_graph[current_node].nonzero()[1]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, depth + 1))

    nodes_in_subgraph = np.unique(nodes_in_subgraph)
    subgraph_distances = distance_matrix[nodes_in_subgraph[:, None], nodes_in_subgraph]

    # Compute the mean shortest path distance, excluding infinity and self-loops
    valid_distances = subgraph_distances[np.isfinite(subgraph_distances) & (subgraph_distances > 0)]
    mean_distance = np.mean(valid_distances)

    return mean_distance


def simulate_random_walk(csgraph, num_steps=100, num_walks=100):
    num_nodes = csgraph.shape[0]
    walks = np.zeros((num_walks, num_steps), dtype=int)

    for i in range(num_walks):
        start_node = np.random.randint(0, num_nodes)
        walks[i, 0] = start_node
        for j in range(1, num_steps):
            neighbors = csgraph.indices[csgraph.indptr[walks[i, j - 1]]:csgraph.indptr[walks[i, j - 1] + 1]]
            if len(neighbors) > 0:
                walks[i, j] = np.random.choice(neighbors)
            else:
                walks[i, j] = walks[i, j - 1]
    return walks


def calculate_msd(shortest_paths, walks):
    num_steps = walks.shape[1]
    msd = np.zeros(num_steps)
    for step in range(num_steps):
        displacement = []
        for i in range(walks.shape[0]):
            start_node = walks[i, 0]
            end_node = walks[i, step]
            displacement.append(shortest_paths[start_node, end_node])
        msd[step] = np.mean(np.square(displacement))
    return msd


def estimate_dimension(msd):
    # Use a simple linear regression on the log-log scale to estimate the slope
    log_steps = np.log(np.arange(1, len(msd) + 1))
    log_msd = np.log(msd)
    print(log_steps)
    print(log_msd)
    slope, _ = np.polyfit(log_steps[2:], log_msd[2:], 1)
    print("log slope", slope)
    dimension = slope / 2  # The slope corresponds to 2/D in MSD ~ t^(2/D)
    return dimension


def simulate_random_walk_return_probs(csgraph, num_steps=100, num_walks_per_node=10):
    num_nodes = csgraph.shape[0]
    return_counts = np.zeros(num_steps)

    for start_node in range(num_nodes):
        for _ in range(num_walks_per_node):
            current_node = start_node
            for step in range(1, num_steps):
                neighbors = csgraph.indices[csgraph.indptr[current_node]:csgraph.indptr[current_node + 1]]
                if neighbors.size > 0:
                    current_node = np.random.choice(neighbors)
                if current_node == start_node:
                    return_counts[step] += 1
                    break  # Stop this walk if returned to start

    return_probabilities = return_counts / (num_nodes * num_walks_per_node)
    return return_probabilities


def fit_return_probability(return_probabilities):
    def decay_func(t, dimension):
        return t ** (-dimension / 2.0)

    steps = np.arange(1, len(return_probabilities) + 1)
    params, _ = curve_fit(decay_func, steps[10:], return_probabilities[10:], p0=[2])
    estimated_dimension = params[0]

    # Return both the estimated dimension and the decay function for plotting
    return estimated_dimension, lambda t: decay_func(t, estimated_dimension)


def plot_return_probabilities(return_probabilities, decay_func, estimated_dimension):
    steps = np.arange(1, len(return_probabilities) + 1)
    plt.plot(steps, return_probabilities, marker='o', linestyle='-', color='blue', label='Observed Return Probability')

    # Plotting the fitted line
    fitted_values = decay_func(steps)
    plt.plot(steps, fitted_values, color='red', linestyle='--', label=f'Fitted: $t^{{-{estimated_dimension:.2f}/2}}$')

    plt.xlabel('Step')
    plt.ylabel('Return Probability')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Return Probability vs. Step (Log-Log Scale)')
    plt.legend()
    plt.show()


def euclidean_vs_network_plot(args, figure_data_euclidean, figure_data_network, central_index_euclidean,
                              central_index_network):
    plt.rc('text', usetex=True)  # Enable LaTeX rendering
    plt.rc('font', family='serif')  # Set font to be serif

    original_position_folder = args.directory_map["original_positions"]
    positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Adjusted for square subplots

    # Common setup for highlighting the central node
    central_node_marker = {'marker': 'o', 'color': 'red'}  # Example marker style
    # Determine global min and max for consistent colorbar scales
    vmin = min(min(figure_data_euclidean['predicted_dimension']), min(figure_data_network['predicted_dimension']))
    vmax = max(max(figure_data_euclidean['predicted_dimension']), max(figure_data_network['predicted_dimension']))

    # Adjust each subplot
    for ax in axs.flat:
        ax.set_aspect('equal', 'box')  # Make the subplot square

    # Euclidean Heatmap
    scatter_euc = axs[0, 0].scatter(positions_df['x'], positions_df['y'],
                                    c=figure_data_euclidean['predicted_dimension'], cmap='viridis', vmin=vmin, vmax=vmax)
    central_node_euc = positions_df.iloc[central_index_euclidean]
    axs[0, 0].scatter(central_node_euc['x'], central_node_euc['y'], **central_node_marker)
    fig.colorbar(scatter_euc, ax=axs[0, 0], label='Predicted Dimension')
    axs[0, 0].set_xlabel(r'$X$')
    axs[0, 0].set_ylabel(r'$Y$')

    # Euclidean Log-Log
    axs[0, 1].plot(figure_data_euclidean['log_x_data'], figure_data_euclidean['log_y_data'], 'o', label='Data Points')
    axs[0, 1].plot(figure_data_euclidean['log_x_data'],
                   figure_data_euclidean['slope'] * figure_data_euclidean['log_x_data'] + figure_data_euclidean[
                       'intercept'], 'r-',
                   label=r'Fit: dimension={:.2f}'.format(figure_data_euclidean['slope']))
    axs[0, 1].set_xlabel(r'$\log(r)$')
    axs[0, 1].set_ylabel(r'$\log(N_v)$')
    axs[0, 1].legend()


    # Network Heatmap
    scatter_net = axs[1, 0].scatter(positions_df['x'], positions_df['y'], c=figure_data_network['predicted_dimension'],
                                    cmap='viridis', vmin=vmin, vmax=vmax)
    central_node_net = positions_df.iloc[central_index_network]
    axs[1, 0].scatter(central_node_net['x'], central_node_net['y'], **central_node_marker)
    fig.colorbar(scatter_net, ax=axs[1, 0], label='Predicted Dimension')
    axs[1, 0].set_xlabel(r'$X$')
    axs[1, 0].set_ylabel(r'$Y$')


    # Network Log-Log
    axs[1, 1].plot(figure_data_network['log_x_data'], figure_data_network['log_y_data'], 'o', label='Data Points')
    axs[1, 1].plot(figure_data_network['log_x_data'],
                   figure_data_network['slope'] * figure_data_network['log_x_data'] + figure_data_network['intercept'],
                   'r-', label=r'Fit: dimension={:.2f}'.format(figure_data_network['slope']))
    axs[1, 1].set_xlabel(r'$\log(r)$')
    axs[1, 1].set_ylabel(r'$\log(N_v)$')
    axs[1, 1].legend()


    plt.tight_layout()
    plot_folder = args.directory_map['heatmap_local']
    plt.savefig(f"{plot_folder}/euclidean_vs_network_{args.args_title}.svg")
    if args.show_plots:
        plt.show()
    plt.close()


def make_euclidean_network_dim_pred_comparison_plot():
    # Parameters
    args = GraphArgs()
    args.dim = 2
    args.intended_av_degree = 6
    args.num_points = 3000

    ### Add random edges? See efect in the dimensionality here
    args.false_edges_count = 0
    args.proximity_mode = "knn"
    # # # 1 Simulation
    create_proximity_graph.write_proximity_graph(args, point_mode="circle", order_indices=False)
    sparse_graph = load_graph(args, load_mode='sparse')
    original_positions = read_position_df(args=args)
    original_dist_matrix = compute_distance_matrix(original_positions)
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
    central_node_euc, fig_data_euc  =\
        run_dimension_prediction_continuous(args, distance_matrix=original_dist_matrix, num_bins=50)
    msp = sp_matrix.mean()
    dist_threshold = int(msp) - 1  #finite size effects, careful
    results_dimension_prediction, fig_data_net, central_node_net = (
        run_dimension_prediction(args, distance_matrix=sp_matrix, dist_threshold=dist_threshold,
                                                            plot_heatmap_all_nodes=True))
    euclidean_vs_network_plot(args, figure_data_euclidean=fig_data_euc, figure_data_network=fig_data_net,
                              central_index_network=central_node_net, central_index_euclidean=central_node_euc)



def main():
    # Parameters
    args = GraphArgs()
    args.dim = 2
    args.intended_av_degree = 6
    args.num_points = 3000

    ### Add random edges? See efect in the dimensionality here
    args.false_edges_count = 0
    args.proximity_mode = "knn"



    simulation_or_experiment = "simulation"


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

        args.edge_list_title = "weinstein_data_january_corrected.csv"
        # args.edge_list_title = "mst_N=1024_dim=2_lattice_k=15.csv"  # Seems to have dimension 1.5

        weighted = True
        weight_threshold = 10

        if os.path.splitext(args.edge_list_title)[1] == ".pickle":
            write_nx_graph_to_edge_list_df(args)  # activate if format is .pickle file
        if not weighted:
            sparse_graph = load_graph(args, load_mode='sparse')
        else:
            sparse_graph = load_graph(args, load_mode='sparse', weight_threshold=weight_threshold)
        # plot_graph_properties(args, igraph_graph_original)  # plots clustering coefficient, degree dist, also stores individual spatial constant...

    elif simulation_or_experiment == "simulation":
        # # # 1 Simulation
        create_proximity_graph.write_proximity_graph(args, point_mode="circle", order_indices=False)
        sparse_graph = load_graph(args, load_mode='sparse')

        # Uncomment if you want original data
        ## Original data
        # edge_list = read_edge_list(args)
        original_positions = read_position_df(args=args)
        # plot_original_or_reconstructed_image(args, image_type="original", edges_df=edge_list)
        original_dist_matrix = compute_distance_matrix(original_positions)
    else:
        raise ValueError("Please input a valid simulation or experiment mode")


    # sparse_graph = add_random_edges_to_csrgraph(args=args, csr_graph=sparse_graph, num_edges_to_add=num_edges_to_add)
    # if num_edges_to_add:
    #     args.args_title = args.args_title + f'_false_edges={num_edges_to_add}'

    # Compute shortest path matrix
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))

    # node_of_interest = 0
    # # Find nodes at distance 3 and 4
    # nodes_at_distance_3 = find_nodes_at_distance(sp_matrix, node_of_interest, 3)
    # nodes_at_distance_4 = find_nodes_at_distance(sp_matrix, node_of_interest, 4)
    # distances = calculate_distances_between_nodes(sp_matrix, nodes_at_distance_3, nodes_at_distance_4)
    # plot_barplot(args, distances, "distances_3_4")

    # # Dimension prediction with random walks (MDS)
    # walks = simulate_random_walk(sparse_graph, num_steps=50, num_walks=100)
    # msd = calculate_msd(sp_matrix, walks)
    # dimension = estimate_dimension(msd)
    # print(f"Estimated dimension: {dimension}")
    # plt.plot(msd, label='Mean Squared Displacement')
    # plt.xlabel('Step')
    # plt.ylabel('MSD')
    # plt.legend()
    # plt.show()

    # ## Dimension prediction with return probabilities
    # num_steps = 100
    # num_walks_per_node = 100
    #
    # # Simulate random walks and calculate return probabilities
    # return_probabilities = simulate_random_walk_return_probs(sparse_graph, num_steps, num_walks_per_node)
    #
    # # Estimate dimension and get decay function for plotting
    # estimated_dimension, decay_func = fit_return_probability(return_probabilities)
    # print(f"Estimated Dimension: {estimated_dimension}")
    # plot_return_probabilities(return_probabilities, decay_func, estimated_dimension)

    msp = sp_matrix.mean()
    print("AVERAGE SHORTEST PATH", msp)
    print("RANDOM NETWORK AV SP", np.log(args.num_points)/np.log(args.average_degree))
    print("ESTIMATED LATTICE SP", (args.num_points/args.average_degree)**(1/args.dim))
    print("ESTIMATED LATTICE SP", (args.num_points**(1/args.dim) /args.average_degree))
    print("ESTIMATED LATTICE SP CURATED 2D", 1.2*(args.num_points/args.average_degree)**(1/args.dim))

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


    # Original dimension prediction
    central_node_index = run_dimension_prediction_continuous(args, distance_matrix=original_dist_matrix, num_bins=50)

    ## Network dimension prediction  # TODO: main function!
    # run_dimension_prediction_continuous(args, distance_matrix=original_dist_matrix, num_bins=50)  # Euclidean
    dist_threshold = int(msp) - 1  #finite size effects, careful
    results_dimension_prediction = run_dimension_prediction(args, distance_matrix=sp_matrix, dist_threshold=dist_threshold,
                                                            plot_heatmap_all_nodes=True)
    print(results_dimension_prediction)


    ## All pairs average shortest path from central node prediction  #TODO: investigate further
    dist_threshold = int(msp) - 1  #finite size effects, careful
    avg_shortest_path_mean_line_segment(args, sparse_graph, sp_matrix, dist_threshold=dist_threshold, central_node_index=None)





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



if __name__ == "__main__":
    main()

