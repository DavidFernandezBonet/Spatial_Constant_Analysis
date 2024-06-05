import os.path

import numpy as np
import matplotlib.pyplot as plt
from structure_and_args import GraphArgs
from create_proximity_graph import write_proximity_graph
from utils import load_graph
from data_analysis import run_simulation_subgraph_sampling, run_simulation_subgraph_sampling_by_bfs_depth
import matplotlib.colors as mcolors
import pandas as pd
from algorithms import compute_shortest_path_matrix_sparse_graph, select_false_edges_csr
from gram_matrix_analysis import plot_gram_matrix_eigenvalues
from utils import add_specific_random_edges_to_csrgraph, write_edge_list_sparse_graph
from plots import save_plotting_data
from check_latex_installation import check_latex_installed
from dimension_prediction import run_dimension_prediction
from gram_matrix_analysis import compute_gram_matrix_eigenvalues
import copy
import random
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')  # Use a non-GUI backend, it was throwing errors otherwise when running the experimental setting

# is_latex_in_os = check_latex_installed()
# if is_latex_in_os:
#     plt.style.use(['nature'])
# else:
#     plt.style.use(['no-latex', 'nature'])
# plt.style.use(['no-latex', 'nature'])
# font_size = 24
# plt.rcParams.update({'font.size': font_size})
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 6
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size - 10

is_latex_in_os = check_latex_installed()
if is_latex_in_os:
    plt.style.use(['nature'])
else:
    plt.style.use(['no-latex', 'nature'])
plt.style.use(['no-latex', 'nature'])
# font_size = 24
# plt.rcParams.update({'font.size': font_size})
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 6
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size - 10


base_figsize = (6, 4.5)  # Width, Height in inches
base_fontsize = 18
plt.rcParams.update({
    'figure.figsize': base_figsize,  # Set the default figure size
    'figure.dpi': 300,  # Set the figure DPI for high-resolution images
    'savefig.dpi': 300,  # DPI for saved figures
    'font.size': base_fontsize,  # Base font size
    'axes.labelsize': base_fontsize ,  # Font size for axis labels
    'axes.titlesize': base_fontsize + 2,  # Font size for subplot titles
    'xtick.labelsize': base_fontsize - 4,  # Font size for X-axis tick labels
    'ytick.labelsize': base_fontsize,  # Font size for Y-axis tick labels
    'legend.fontsize': base_fontsize - 6,  # Font size for legends
    'lines.linewidth': 2,  # Line width for plot lines
    'lines.markersize': 6,  # Marker size for plot markers
    'figure.autolayout': True,  # Automatically adjust subplot params to fit the figure
    'text.usetex': False,  # Use LaTeX for text rendering (set to True if LaTeX is installed)
})

np.random.seed(42)
random.seed(42)

def generate_several_graphs(from_one_graph=False, proximity_mode="knn"):
    args_list = []
    # false_edge_list = [0, 20, 40, 60, 80, 100]
    false_edge_list = [0, 2, 5, 10, 20, 1000]



    if not from_one_graph:
        for idx, false_edge_count in enumerate(false_edge_list):
            args = GraphArgs()
            args.verbose = False
            args.proximity_mode = proximity_mode
            args.dim = 2
            args.show_plots = False
            args.intended_av_degree = 10
            args.num_points = 1000
            args.false_edges_count = false_edge_count
            args.network_name = f'FE={args.false_edges_count}'
            edge_list_title = f"edge_list_{args.args_title}_graph_{idx}.csv"  # Assuming 'args.title' exists
            args.edge_list_title = edge_list_title  # update the edge list title
            args.original_edge_list_title = edge_list_title
            write_proximity_graph(args, point_mode="square", order_indices=False)
            # compute_shortest_path_matrix_sparse_graph(graph, args=args)
            args_list.append(args)
    # TODO: just add false edges to one graph, but "create" different ones

    else:
        args = GraphArgs()
        args.verbose = False
        args.proximity_mode = "knn"
        args.dim = 2
        args.show_plots = False
        args.intended_av_degree = 10
        args.num_points = 2000
        write_proximity_graph(args, point_mode="square", order_indices=False)
        load_graph(args, load_mode='sparse')
        all_random_false_edges = select_false_edges_csr(args.sparse_graph, max(false_edge_list))

        for idx, num_edges in enumerate(false_edge_list):
            args_i = copy.copy(args)
            modified_graph = add_specific_random_edges_to_csrgraph(args.sparse_graph, all_random_false_edges,
                                                                   num_edges)
            args_i.sparse_graph = modified_graph
            args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(args=args_i, sparse_graph=args_i.sparse_graph)
            args_i.false_edges_count = num_edges
            edge_list_title = f"edge_list_{args_i.args_title}_graph_{idx}.csv"
            args_i.edge_list_title = edge_list_title  # update the edge list title
            args_i.network_name = f'FE={args_i.false_edges_count}'
            args_i.original_edge_list_title = edge_list_title
            args_i.update_args_title()
            write_edge_list_sparse_graph(args_i, args_i.sparse_graph)
            args_list.append(args_i)

    return args_list

def generate_experimental_graphs(edge_list_titles_dict):
    """
    Load experimental graphs from comparisons. Just input all the edge_list_titles you want in the
    data/edge_lists folder. If you want different thresholds for the same weighted graph, you can input it as the value
    of the dictionary (second element of the tuple).

    Dictionary:  key --> edge list name, value --> 'weight' tuple (see below)
    Weight[0] --> name of the network
    Weight[1] --> it is none, or contains a list of weight thresholds
    """

    args_list = []
    for edge_list, weight_list in edge_list_titles_dict.items():
        if weight_list[1] is None:
            args = GraphArgs()
            args.verbose = False

            args.dim = 2
            args.show_plots = False
            args.edge_list_title = edge_list
            args.proximity_mode = "experimental"
            if args.num_points > 3000:
                args.large_graph_subsampling = True
                args.max_subgraph_size = 3000
            sparse_graph = load_graph(args, load_mode='sparse')
            args.sparse_graph = sparse_graph
            args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(args=args,
                                                                                  sparse_graph=args.sparse_graph)
            args.network_name = weight_list[0]

            args_list.append(args)
        else:
            for weight in weight_list[1]:
                args = GraphArgs()
                args.verbose = False
                args.dim = 2
                args.show_plots = False
                args.weighted = True
                args.weight_threshold = weight
                args.edge_list_title = edge_list
                args.proximity_mode = "experimental"
                if args.num_points > 3000:
                    args.large_graph_subsampling = True
                    args.max_subgraph_size = 3000
                sparse_graph = load_graph(args, load_mode='sparse')
                args.sparse_graph = sparse_graph
                args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(args=args, sparse_graph=args.sparse_graph)
                args.edge_list_title = f"{os.path.splitext(edge_list)[0]}_weight_threshold_{args.weight_threshold}.csv"
                args.network_name = weight_list[0] + f"{args.weight_threshold}"
                write_edge_list_sparse_graph(args, args.sparse_graph)

                args_list.append(args)



    ## add simulated graph for compariosn
    args = GraphArgs()
    args.num_points = 1000
    args.proximity_mode = "delaunay_corrected"
    args.dim = 2
    args.intended_av_degree = 10
    args.verbose = False
    write_proximity_graph(args, point_mode="circle", order_indices=False)
    args.sparse_graph = load_graph(args, load_mode='sparse')
    args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(args=args, sparse_graph=args.sparse_graph)
    args.network_name = "Simulation"
    args_list.append(args)
    return args_list


def get_maximally_separated_colors(num_colors):
    hues = np.linspace(0, 1, num_colors + 1)[:-1]  # Avoid repeating the first color
    colors = [mcolors.hsv_to_rgb([h, 0.7, 0.7]) for h in hues]  # S and L fixed for aesthetic colors
    # Convert to HEX format for broader compatibility
    colors = [mcolors.to_hex(color) for color in colors]
    return colors

def plot_comparative_spatial_constant(results_dfs, args_list, title="", use_depth=False):
    from matplotlib.ticker import MaxNLocator

    ax = plt.figure(figsize=(12, 4.5)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    num_colors = len(args_list)
    colors = get_maximally_separated_colors(num_colors)

    if use_depth:
        size_magnitude = 'depth'
        # s_constant = 'S_depth'
        s_constant = 'S_general'
    else:
        size_magnitude = 'intended_size'
        s_constant = 'S_general'

    data_means = []
    data_depths = []
    data_stds = []
    for i, (results_df_net, args) in enumerate(zip(results_dfs, args_list)):
        unique_sizes = results_df_net[size_magnitude].unique()
        means = []
        std_devs = []
        sizes = []

        for size in unique_sizes:
            subset = results_df_net[results_df_net[size_magnitude] == size]
            mean = subset['S_general'].mean()
            std = subset['S_general'].std()
            means.append(mean)
            std_devs.append(std)
            sizes.append(size)

        sizes_net = np.array(sizes)
        means_net = np.array(means)
        std_devs_net = np.array(std_devs)

        data_means.append(means_net)
        data_depths.append(sizes_net)
        data_stds.append(std_devs_net)

        # Use color from the selected palette
        color = colors[i]

        # Scatter plot and ribbon for mean spatial constants
        plt.plot(sizes, means, label=f'{args.network_name}', marker='o', color=color)
        plt.fill_between(sizes_net, means_net - std_devs_net, means_net + std_devs_net, alpha=0.2,
                          color=color)


    if use_depth:
        plt.xlabel('BFS Depth')
    else:
        plt.xlabel('Size')
    plt.ylabel('Mean Spatial Constant')
    plt.legend()

    # Save the figure
    plot_folder = f"{args_list[0].directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/comparative_spatial_constant_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_spatial_constant_{title}.svg")


    column_names_means = [args.network_name + ' mean spatial constant' for args in args_list]
    column_names_sizes = [args.network_name + ' depths' for args in args_list]
    column_names_stds = [args.network_name + ' stds' for args in args_list]
    column_names = column_names_means + column_names_sizes + column_names_stds
    data = data_means + data_depths + data_stds
    print(len(column_names), len(data))
    save_plotting_data(column_names=column_names, data=data,
                       csv_filename=f"{plot_folder2}/comparative_spatial_constant_{title}.csv")


    if args.show_plots:
        plt.show()
def make_spatial_constant_comparative_plot(args_list, title=""):
    n_samples = 5
    net_results_df_list = []
    for args in args_list:
        size_interval = int(args.num_points / 10)  # collect 10 data points

        ## Network Spatial Constant
        igraph_graph = load_graph(args, load_mode='igraph')  #TODO: make sure igraph is what you need
        # igraph_graph = load_graph(args, load_mode='sparse')

        # ### Run with size
        # net_results_df = run_simulation_subgraph_sampling(args, size_interval=size_interval, n_subgraphs=n_samples,
        #                                                   graph=igraph_graph,
        #                                                   add_false_edges=False, add_mst=False, return_simple_output=False)

        ### Run with depth  #TODO: check that this work
        shortest_path_matrix = args.shortest_path_matrix
        print("NETWORK NAME", args.network_name)

        net_results_df = run_simulation_subgraph_sampling_by_bfs_depth(args, shortest_path_matrix=shortest_path_matrix,
                                                                    n_subgraphs=n_samples, graph=igraph_graph,
                                                                    add_false_edges=False, return_simple_output=False,
                                                                       all_depths=True)

        net_results_df_list.append(net_results_df)


    plot_comparative_spatial_constant(net_results_df_list, args_list, title=title, use_depth=True)


def make_dimension_prediction_comparative_plot(args_list, title=""):
    results_pred_dimension_list = []
    for args in args_list:
        if args.sparse_graph is None:
            sparse_graph = load_graph(args, load_mode='sparse')
            compute_shortest_path_matrix_sparse_graph(sparse_graph=sparse_graph, args=args)
        elif args.shortest_path_matrix is None:
            compute_shortest_path_matrix_sparse_graph(sparse_graph=args.sparse_graph, args=args)
        results_pred_dimension = run_dimension_prediction(args, distance_matrix=args.shortest_path_matrix,
                                                          dist_threshold=int(args.mean_shortest_path),
                                                          num_central_nodes=10,
                                                          local_dimension=False, plot_heatmap_all_nodes=False)
        results_pred_dimension_list.append(results_pred_dimension)
    plot_comparative_predicted_dimension(args_list=args_list, results_predicted_dimension_list=results_pred_dimension_list,
                                         title=title)


def plot_comparative_predicted_dimension(args_list, results_predicted_dimension_list, title=""):
    plt.figure(figsize=(12, 4.5))

    num_colors = len(args_list)
    colors = get_maximally_separated_colors(num_colors)

    # X-axis positions for each violin plot
    x_positions = np.arange(num_colors)

    # Data for plotting, now directly using 'predicted_dimension_list' from each dictionary
    violin_data = [res['predicted_dimension_list'] for res in results_predicted_dimension_list]
    labels = [args.network_name for args in args_list]

    parts = plt.violinplot(violin_data, positions=x_positions, showmeans=False, showmedians=True)
    # Overlay points on the violin plots
    for i, data in enumerate(violin_data):
        # Generating a slight random offset to spread the points horizontally and improve visibility
        x_values = np.random.normal(i, 0.04, size=len(data))
        plt.scatter(x_values, data, alpha=1, color=colors[i], edgecolor='black')  # Adjust alpha as needed for better visibility

    # Coloring each violin plot
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')  # Adding a contrasting edge color for better visibility
        pc.set_alpha(0.7)

    # Aesthetics
    plt.xticks(x_positions, labels, rotation=0, ha="center")
    plt.ylabel('Predicted Dimension')
    plt.title(title if title else 'Predicted Dimension Distribution for Each Graph')
    plt.tight_layout()

    plot_folder = f"{args_list[0].directory_map['dimension_prediction_iterations']}"
    plt.savefig(f"{plot_folder}/comparative_dimension_prediction_violin_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_dimension_prediction_violin_{title}.svg")

    column_names = [args.network_name for args in args_list]
    data = violin_data
    save_plotting_data(column_names=column_names, data=data,
                       csv_filename=f"{plot_folder2}/comparative_dimension_prediction_violin_{title}.csv")

    if args_list[0].show_plots:
        plt.show()


def make_gram_matrix_analysis_comparative_plot(args_list, title=""):
    eigenvalues_list = []
    for i, args in enumerate(args_list):
        if args.sparse_graph is None:
            sparse_graph = load_graph(args, load_mode='sparse')
            compute_shortest_path_matrix_sparse_graph(sparse_graph=sparse_graph, args=args)
        elif args.shortest_path_matrix is None:
            compute_shortest_path_matrix_sparse_graph(sparse_graph=args.sparse_graph, args=args)

        eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=args.shortest_path_matrix)
        eigenvalues_list.append(eigenvalues_sp_matrix)

        first_d_values_contribution,\
        first_d_values_contribution_5_eigen,\
        spectral_gap,\
        last_spectral_gap \
            = plot_gram_matrix_eigenvalues(args=args, shortest_path_matrix=args.shortest_path_matrix)
    plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, title=title)
    plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, consider_first_eigenvalues_only=True, title=title)
    plot_spectral_gap_comparative(args_list, eigenvalues_list, score_method='f', title=title)
    plot_pos_neg_eigenvalue_proportions_comparative(args_list, eigenvalues_list)



def plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, title='', consider_first_eigenvalues_only=False):
    """
    Plot eigenvalue contributions for multiple networks and a comparative bar chart
    of cumulative variance for the first d eigenvalues.

    :param eigenvalues_list: List of arrays, where each array contains the eigenvalues of a network.
    :param args_list: List of args objects, used for labeling.
    """
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1]})

    colors = get_maximally_separated_colors(len(args_list))

    # unicolor (uncomment if you want rainbow color)
    colors = ['#009ADE'] * len(colors)

    cumulative_variances = []

    for i, (S, args) in enumerate(zip(eigenvalues_list, args_list)):
        network_name = args.network_name

        if consider_first_eigenvalues_only:
            S = S[:10]
        total_variance = np.sum(S)
        variance_proportion = S / total_variance
        cumulative_variance = np.cumsum(variance_proportion)
        cumulative_variance_first_d_eigenvalues = cumulative_variance[args.dim - 1]

        selected_cumulative_variances = cumulative_variance[:args.dim]




        color = colors[i]

        # Plot on the first subplot
        axs[0].plot(range(1, len(S) + 1), cumulative_variance, '-o' , color=color)
        axs[0].axvline(x=args.dim, linestyle='--')

        # Append the list of selected cumulative variances instead of a single value
        cumulative_variances.append((selected_cumulative_variances, color))
        # cumulative_variances.append((cumulative_variance_first_d_eigenvalues, color))

    # Setting for the first plot
    axs[0].set_xlabel('Eigenvalue Rank')
    axs[0].set_ylabel('Eigenvalue Contribution')
    axs[0].set_xscale('log')
    # axs[0].legend()

    # # Bar chart for cumulative variance comparison, using the same colors
    # for i, (cumulative_variance, color) in enumerate(cumulative_variances):
    #     axs[1].bar(i, cumulative_variance, color=color)

    for i, (variances, base_color) in enumerate(cumulative_variances):
        # Bottom of the bar stack
        bottom = 0
        color_gradient = [mcolors.to_rgba(base_color, alpha=0.5 + 0.5 * (1 - j / (len(variances) - 1))) if len(
            variances) > 1 else base_color for j in range(len(variances))]
        # Iterate over each cumulative variance up to args.dim
        for j in range(len(variances)):
            # Plot the segment of the stacked bar
            if j == 0:
                height = variances[j]
            else:
                height = variances[j] - variances[j - 1]

            color = color_gradient[j]
            axs[1].bar(i, height,  bottom=bottom, color=color)
            # Update the bottom to the top of the last bar
            bottom = variances[j]

    # Create custom legend
    if args.dim == 3:
        legend_labels = ['1st eigenvalue', '2nd eigenvalue', '3rd eigenvalue']

    elif args.dim == 2:
        legend_labels = ['1st eigenvalue', '2nd eigenvalue']

    color = '#009ADE'  #TODO: change this if there is multicolor, maybe change it to black
    legend_patches = [mpatches.Patch(color=color, alpha=0.5 + 0.5 * (1 - i / (args.dim - 1)), label=label) for i, label in
                      enumerate(legend_labels)]
    axs[1].legend(handles=legend_patches)

    axs[1].set_ylabel('Variance Contribution')
    axs[1].set_xticks(range(len(args_list)))
    axs[1].set_xticklabels([args.network_name for args in args_list], rotation=0, ha="center")
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plot_folder = f"{args_list[0].directory_map['mds_dim']}"
    prefix = 'first_' if consider_first_eigenvalues_only else ''
    plt.savefig(f"{plot_folder}/comparative_{prefix}eigenvalue_contributions_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_{prefix}eigenvalue_contributions_{title}.svg")

    column_names = [args.network_name for args in args_list]
    if consider_first_eigenvalues_only:
        data = cumulative_variance_first_d_eigenvalues
    else:
        data = cumulative_variances
    print("consider first eigenvalues only", consider_first_eigenvalues_only)
    print("cumulative_variances", cumulative_variances)
    print("data", data)
    data = [item[0] for item in data]  # ignore the color tuple (2nd element)
    save_plotting_data(column_names=column_names, data=data,
                       csv_filename=f"{plot_folder2}/comparative_{prefix}eigenvalue_contributions_{title}.csv")
    if args.show_plots:
        plt.show()


def calculate_spectral_score(eigenvalues, args, method):
    from sklearn.metrics.pairwise import cosine_similarity
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    positive_eigenvalues = eigenvalues_sorted[eigenvalues_sorted > 0]
    score = 0

    if method == 'a':
        # Relative Contribution Score
        score = np.sum(positive_eigenvalues[:args.dim]) / np.sum(positive_eigenvalues)
    elif method == 'b':
        # Exponential Decay Score
        penalty = np.exp(-np.sum(positive_eigenvalues[args.dim:])) / np.exp(-np.sum(positive_eigenvalues[:args.dim]))
        score = penalty
    elif method == 'c':
        # Harmonic Mean of Contributions
        contributions = positive_eigenvalues[:args.dim] / np.sum(positive_eigenvalues)
        score = len(contributions) / np.sum(1.0 / contributions)
    elif method == 'd':
        # Squared Difference Score
        ideal_contrib = np.zeros_like(positive_eigenvalues)
        ideal_contrib[:args.dim] = positive_eigenvalues[:args.dim]
        # ideal_contrib[:args.dim] = np.ones(args.dim)
        # print(ideal_contrib)
        score = 1 - (np.sum((ideal_contrib - positive_eigenvalues) ** 2) / np.sum(positive_eigenvalues ** 2))
    elif method == 'e':
        # Cosine Similarity Score
        actual_vector = np.hstack([positive_eigenvalues[:args.dim], np.zeros(len(positive_eigenvalues) - args.dim)])
        ideal_vector = np.zeros_like(actual_vector)
        ideal_vector[:args.dim] = positive_eigenvalues[:args.dim]
        score = cosine_similarity([actual_vector], [ideal_vector])[0][0]
    elif method == 'f':
        # 1st original method, just gap between d and d+1 value
        spectral_gaps = (positive_eigenvalues[:-1] - positive_eigenvalues[1:]) / positive_eigenvalues[:-1]
        score = spectral_gaps[args.dim - 1]
    elif method == 'g':
        # Normalizing using only 1st eigenvalue
        score = (positive_eigenvalues[0] - positive_eigenvalues[args.dim-1]) / positive_eigenvalues[0]

    elif method == 'h':  # squared difference, all eigenvalues d+1 should be 0
        ideal_contrib = np.zeros_like(eigenvalues_sorted[args.dim:])
        score = 1 - (np.sum((ideal_contrib - eigenvalues_sorted[args.dim:]) ** 2) / np.sum(eigenvalues_sorted[:args.dim] ** 2))

    elif method == 'i':
        # Gab between the mean of the first d eigenvalues and the d+1 value --> This seems to work quite well
        # THIS IS THE METHOD I CHOOSE TO USE
        d_values = np.mean(positive_eigenvalues[:args.dim])
        gap = (d_values - positive_eigenvalues[args.dim+1]) / d_values
        score = gap
    elif method == 'negative_mass_fraction':
        # Mass fraction of negative eigenvalues
        negative_eigenvalues = eigenvalues_sorted[eigenvalues_sorted < 0]
        score = np.sum(np.abs(negative_eigenvalues)) / (np.sum(positive_eigenvalues) + np.sum(np.abs(negative_eigenvalues)))
    else:
        raise ValueError("Invalid scoring method specified.")

    return score


def plot_spectral_gap_comparative(args_list, eigenvalues_list, score_method='i', title=''):

    # score method chosen: i or f (i is using mean of all first eigenvalues, f just the last relevant one)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [1, 1]})
    # Retrieve the default color cycle
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = get_maximally_separated_colors(len(args_list))

    # unicolor (uncomment if you want different colors)
    colors = ['#AF58BA'] * len(colors)
    spectral_gap_scores = []

    # Iterate over each network's eigenvalues and args
    for i, (eigenvalues, args) in enumerate(zip(eigenvalues_list, args_list)):
        # Sort the eigenvalues in descending order and select positive ones
        eigenvalues_sorted = np.sort(eigenvalues)[::-1]
        positive_eigenvalues = eigenvalues_sorted[eigenvalues_sorted > 0]  # Limit to first 5 for analysis
        positive_eigenvalues_short = positive_eigenvalues[:5]
        sum_up_to_dim = np.sum(positive_eigenvalues[:args.dim])

        # Calculate spectral gaps for the first few positive eigenvalues
        spectral_gaps = (positive_eigenvalues_short[:-1] - positive_eigenvalues_short[1:]) / positive_eigenvalues_short[:-1]

        # Use the same color for the current network in both plots
        # color = color_cycle[i % len(color_cycle)]
        color = colors[i]

        # Plot the spectral gaps in the first subplot
        axs[0].plot(range(1, len(spectral_gaps) + 1), spectral_gaps, marker='o', linestyle='-', linewidth=2,
                    markersize=8, label=args.network_name, color=color)

        spectral_gap_score = calculate_spectral_score(eigenvalues, args, method=score_method)
        spectral_gap_scores.append(spectral_gap_score)

    # Setting for the first plot (Spectral Gap Analysis)

    axs[0].set_xlabel('Eigenvalue Rank')
    axs[0].set_ylabel('Spectral Gap Ratio')
    axs[0].legend()

    # Create a barplot for spectral gap scores in the second subplot, using the same colors
    for j, spectral_gap_score in enumerate(spectral_gap_scores):
        color = colors[j]
        axs[1].bar(j, spectral_gap_score, color=color)

    axs[1].set_ylabel('Spectral Gap Score')
    axs[1].set_xticks(range(len(args_list)))
    axs[1].set_xticklabels([args.network_name for args in args_list], rotation=0, ha="center")
    axs[1].set_ylim(0, 1)
    plt.tight_layout()




    plot_folder = f"{args_list[0].directory_map['mds_dim']}"
    plt.savefig(f"{plot_folder}/comparative_spectral_gap_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_spectral_gap_{title}.svg")

    column_names = [args.network_name for args in args_list]
    data = spectral_gap_scores
    save_plotting_data(column_names=column_names, data=data,
                       csv_filename=f"{plot_folder2}/comparative_spectral_gap_{title}.csv")

    if args.show_plots:
        plt.show()



def plot_pos_neg_eigenvalue_proportions_comparative(args_list, eigenvalues_list):
    fig, axs = plt.subplots(figsize=(10, 6))

    # Retrieve the default color cycle
    colors = get_maximally_separated_colors(len(args_list))
    proportions = []

    # Iterate over each network's eigenvalues
    for i, (eigenvalues, args) in enumerate(zip(eigenvalues_list, args_list)):

        ## TODO: option 1 ratio neg/pos and use all eigenvalues
        ## todo: option 2 use only eigenvalues > 0 till args.dim and all negative eigenvalues, and ratio pos/neg
        ## TODO: option 3 ratio contribution dim eigenvalues divided by all the rest (using absolute value for the negative)
        # Count positive and negative eigenvalues
        num_positive = np.sum(eigenvalues[eigenvalues > 0])
        num_positive_dim = np.sum(eigenvalues[eigenvalues > 0][:args.dim])
        num_positive_nondim = np.sum(eigenvalues[eigenvalues > 0][args.dim:])
        num_negative = np.abs(np.sum(eigenvalues[eigenvalues < 0]))  # Use absolute sum for negative eigenvalues
        print("positive", num_positive, "negative", num_negative)
        print("positive dim", num_positive_dim)

        # Calculate the proportion of positive to negative eigenvalues
        # If there are no negative eigenvalues, set the proportion to the number of positive eigenvalues
        if num_negative > 0:
            # proportion = num_negative / num_positive
            proportion = num_positive_dim / (num_negative + num_positive_nondim)
            proportion = (num_negative + num_positive_nondim) / num_positive_dim
            proportion =  num_negative / (num_positive + num_negative)  # proportion of badness (negative mass ratio)
            proportion = np.max(num_negative) / np.max(num_positive)   # same but taking into account biggest eigenvalues only
            print("proportion", proportion)
            print("num positive", num_positive, "num negative", num_negative)
        else:
            proportion = num_positive

        proportions.append(proportion)

        # Use the same color for the current network in the bar plot
        color = colors[i]

        # Plot the proportion in the bar plot
        axs.bar(i, proportion, color=color, label=args.network_name)

    axs.set_xlabel('Network')
    axs.set_ylabel('Proportion of Negative/Positive Eigenvalues')
    axs.set_xticks(range(len(args_list)))
    axs.set_xticklabels([args.network_name for args in args_list], rotation=45, ha="right")
    axs.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    what_to_run = "experimental"  # simulation or experimental
    title_experimental = "MPX" # "Experimental Comparison"

    if what_to_run == "simulation":
        ## Simulation with False Edges
        proximity_mode = "knn_bipartite"
        args_list = generate_several_graphs(from_one_graph=True, proximity_mode=proximity_mode)
        title = f"False Edge Comparison v2 {proximity_mode}"

    elif what_to_run == "experimental":
        ### Experimental
        # pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv
        # pixelgen_example_graph.csv

        ### All experimental data
        # edge_list_titles_dict = {"weinstein_data_corrected_february.csv": ('W', [5, 10, 15]),
        #                     "pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv": ('PXL', None),
        #                     "subgraph_8_nodes_160_edges_179_degree_2.24.pickle": ('HL-S', None),
        #                     "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle": ('HL-E', None)}

        # ### Just Weinstein's
        # edge_list_titles_dict = {"weinstein_data_corrected_february.csv": [5, 10, 15],
        #                          }

        # # Weinstein subgraphs with quantile 0.15
        # edge_list_titles_dict = {
        # "weinstein_data_corrected_february_original_image_subgraphs_quantile=0.15.png_12893_subgraph_1.csv": ('S1', None),
        # "weinstein_data_corrected_february_original_image_subgraphs_quantile=0.15.png_3796_subgraph_2.csv": ('S2', None),
        # "weinstein_data_corrected_february_original_image_subgraphs_quantile=0.15.png_2211_subgraph_3.csv": ('S3', None),
        # "weinstein_data_corrected_february_original_image_subgraphs_quantile=0.15.png_1880_subgraph_4.csv": ('S4', None),
        # "weinstein_data_corrected_february_original_image_subgraphs_quantile=0.15.png_1156_subgraph_5.csv": ('S5', None),
        # }

        # # Pixelgen different datasets
        # edge_list_titles_dict = {
        #     "pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv": ('Raji', None),
        #     "pixelgen_processed_edgelist_Sample07_pbmc_CD3_capped_cell_3_RCVCMP0000344.csv": ('CD3', None),
        #     "pixelgen_example_graph.csv": ('Uro', None)
        # }

        # Pixelgen pbmc dataset good, bad, ugly (different gradients of spatial coherence)
        edge_list_titles_dict = {
            "Sample01_human_pbmcs_unstimulated_component_RCVCMP0001392_edgelist.csv": ('PBMC 1', None),
            "Sample01_human_pbmcs_unstimulated_component_RCVCMP0002024_edgelist.csv": ('PBMC 2', None),
            "Sample01_human_pbmcs_unstimulated_component_RCVCMP0000120_edgelist.csv": ('PBMC 3', None)
        }

        title = title_experimental
        args_list = generate_experimental_graphs(edge_list_titles_dict=edge_list_titles_dict)
    else:
        raise ValueError("what_to_run must be 'simulation' or 'experimental'")


    ## Comparative Pipeline
    # make_spatial_constant_comparative_plot(args_list, title=title)
    # make_dimension_prediction_comparative_plot(args_list, title=title)
    make_gram_matrix_analysis_comparative_plot(args_list, title=title)
