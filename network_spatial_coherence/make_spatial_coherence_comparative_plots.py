import os.path

import numpy as np
import matplotlib.pyplot as plt
from structure_and_args import GraphArgs
from create_proximity_graph import write_proximity_graph
from utils import load_graph
from data_analysis import run_simulation_subgraph_sampling
import matplotlib.colors as mcolors
import pandas as pd
from algorithms import compute_shortest_path_matrix_sparse_graph, select_false_edges_csr
from utils import add_specific_random_edges_to_csrgraph, write_edge_list_sparse_graph
from check_latex_installation import check_latex_installed
from dimension_prediction import run_dimension_prediction
from gram_matrix_analysis import compute_gram_matrix_eigenvalues
import copy

is_latex_in_os = check_latex_installed()
if is_latex_in_os:
    plt.style.use(['nature'])
else:
    plt.style.use(['no-latex', 'nature'])
font_size = 24
plt.rcParams.update({'font.size': font_size})
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 6
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size - 10

def generate_several_graphs(from_one_graph=False):
    args_list = []
    # false_edge_list = [0, 20, 40, 60, 80, 100]
    false_edge_list = [0, 2, 5, 10, 20, 1000]
    # false_edge_list = [0, 10, 100, 500, 1000]

    if not from_one_graph:
        for idx, false_edge_count in enumerate(false_edge_list):
            args = GraphArgs()
            args.verbose = False
            args.proximity_mode = "knn"
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
            print(args.edge_list_title)
            args_list.append(args)
    # TODO: just add false edges to one graph, but "create" different ones

    else:
        args = GraphArgs()
        args.verbose = False
        args.proximity_mode = "knn"
        args.dim = 2
        args.show_plots = False
        args.intended_av_degree = 10
        args.num_points = 1000
        write_proximity_graph(args, point_mode="square", order_indices=False)
        load_graph(args, load_mode='sparse')
        all_random_false_edges = select_false_edges_csr(args.sparse_graph, max(false_edge_list))

        for idx, num_edges in enumerate(false_edge_list):
            args_i = copy.copy(args)
            modified_graph = add_specific_random_edges_to_csrgraph(args.sparse_graph, all_random_false_edges,
                                                                   num_edges)
            args_i.sparse_graph = modified_graph
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
            args.proximity_mode = "experimental"
            args.dim = 2
            args.show_plots = False
            args.edge_list_title = edge_list
            load_graph(args, load_mode='sparse')
            args.network_name = weight_list[0]
            args_list.append(args)
        else:
            for weight in weight_list[1]:
                args = GraphArgs()
                args.verbose = False
                args.proximity_mode = "experimental"
                args.dim = 2
                args.show_plots = False
                args.weighted = True
                args.weight_threshold = weight
                args.edge_list_title = edge_list
                load_graph(args, load_mode='sparse')
                args.edge_list_title = f"{os.path.splitext(edge_list)[0]}_weight_threshold_{args.weight_threshold}.csv"
                args.network_name = weight_list[0] + f"{args.weight_threshold}"
                write_edge_list_sparse_graph(args, args.sparse_graph)
                args_list.append(args)

    ## add simulated graph for compariosn
    args = GraphArgs()
    args.num_points = 1000
    args.proximity_mode = "knn"
    args.dim = 2
    args.intended_av_degree = 10
    args.verbose = False
    write_proximity_graph(args, point_mode="square", order_indices=False)
    args.sparse_graph = load_graph(args, load_mode='sparse')
    args.network_name = "KNN"
    args_list.append(args)
    return args_list


def get_maximally_separated_colors(num_colors):
    hues = np.linspace(0, 1, num_colors + 1)[:-1]  # Avoid repeating the first color
    colors = [mcolors.hsv_to_rgb([h, 0.7, 0.7]) for h in hues]  # S and L fixed for aesthetic colors
    # Convert to HEX format for broader compatibility
    colors = [mcolors.to_hex(color) for color in colors]
    return colors

def plot_comparative_spatial_constant(results_dfs, args_list, title=""):
    plt.figure(figsize=(10, 6))
    num_colors = len(args_list)
    colors = get_maximally_separated_colors(num_colors)

    for i, (results_df_net, args) in enumerate(zip(results_dfs, args_list)):
        unique_sizes = results_df_net['intended_size'].unique()
        means = []
        std_devs = []
        sizes = []

        for size in unique_sizes:
            subset = results_df_net[results_df_net['intended_size'] == size]
            mean = subset['S_general'].mean()
            std = subset['S_general'].std()
            means.append(mean)
            std_devs.append(std)
            sizes.append(size)

        sizes_net = np.array(sizes)
        means_net = np.array(means)
        std_devs_net = np.array(std_devs)

        # Use color from the selected palette
        color = colors[i]

        # Scatter plot and ribbon for mean spatial constants
        plt.plot(sizes, means, label=f'{args.network_name}', marker='o', color=color)
        plt.fill_between(sizes_net, means_net - std_devs_net, means_net + std_devs_net, alpha=0.2,
                          color=color)


    plt.xlabel('Subgraph Size')
    plt.ylabel('Mean Spatial Constant')
    plt.legend()

    # Save the figure
    plot_folder = f"{args_list[0].directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/comparative_spatial_constant_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_spatial_constant_{title}.svg")
    if args.show_plots:
        plt.show()
def make_spatial_constant_comparative_plot(args_list, title=""):
    n_samples = 5
    net_results_df_list = []
    for args in args_list:
        print("edge list title", args.edge_list_title)
        print("extra info", args.network_name)

        size_interval = int(args.num_points / 10)  # collect 10 data points
        print(args.edge_list_title)
        ## Network Spatial Constant
        igraph_graph = load_graph(args, load_mode='igraph')
        net_results_df = run_simulation_subgraph_sampling(args, size_interval=size_interval, n_subgraphs=n_samples,
                                                          graph=igraph_graph,
                                                          add_false_edges=False, add_mst=False)
        net_results_df_list.append(net_results_df)


    plot_comparative_spatial_constant(net_results_df_list, args_list, title=title)


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
    plt.figure(figsize=(12, 6))

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
    if args_list[0].show_plots:
        plt.show()


def make_gram_matrix_analysis_comparative_plot(args_list, title=""):
    eigenvalues_list = []
    for args in args_list:
        if args.sparse_graph is None:
            sparse_graph = load_graph(args, load_mode='sparse')
            compute_shortest_path_matrix_sparse_graph(sparse_graph=sparse_graph, args=args)
        elif args.shortest_path_matrix is None:
            compute_shortest_path_matrix_sparse_graph(sparse_graph=args.sparse_graph, args=args)

        eigenvalues_sp_matrix = compute_gram_matrix_eigenvalues(distance_matrix=args.shortest_path_matrix)
        eigenvalues_list.append(eigenvalues_sp_matrix)
    plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, title=title)
    plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, consider_first_eigenvalues_only=True, title=title)
    plot_spectral_gap_comparative(args_list, eigenvalues_list, score_method='i', title=title)
    # plot_pos_neg_eigenvalue_proportions_comparative(args_list, eigenvalues_list)



def plot_eigenvalue_contributions_comparative(eigenvalues_list, args_list, title='', consider_first_eigenvalues_only=False):
    """
    Plot eigenvalue contributions for multiple networks and a comparative bar chart
    of cumulative variance for the first d eigenvalues.

    :param eigenvalues_list: List of arrays, where each array contains the eigenvalues of a network.
    :param args_list: List of args objects, used for labeling.
    """
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})

    colors = get_maximally_separated_colors(len(args_list))

    cumulative_variances = []

    for i, (S, args) in enumerate(zip(eigenvalues_list, args_list)):
        network_name = args.network_name

        if consider_first_eigenvalues_only:
            S = S[:10]
        total_variance = np.sum(S)
        variance_proportion = S / total_variance
        cumulative_variance = np.cumsum(variance_proportion)
        cumulative_variance_first_d_eigenvalues = cumulative_variance[args.dim - 1]

        color = colors[i]

        # Plot on the first subplot
        axs[0].plot(range(1, len(S) + 1), cumulative_variance, '-o' , color=color)
        axs[0].axvline(x=args.dim, linestyle='--')

        cumulative_variances.append((cumulative_variance_first_d_eigenvalues, color))

    # Setting for the first plot
    axs[0].set_xlabel('Eigenvalue Rank')
    axs[0].set_ylabel('Eigenvalue Contribution')
    axs[0].set_xscale('log')
    # axs[0].legend()

    # Bar chart for cumulative variance comparison, using the same colors
    for i, (cumulative_variance, color) in enumerate(cumulative_variances):
        axs[1].bar(i, cumulative_variance, color=color)

    axs[1].set_ylabel('Contribution at Dim=%d' % args_list[0].dim)
    axs[1].set_xticks(range(len(args_list)))
    axs[1].set_xticklabels([args.network_name for args in args_list], rotation=0, ha="center")

    plt.tight_layout()
    plot_folder = f"{args_list[0].directory_map['mds_dim']}"
    prefix = 'first_' if consider_first_eigenvalues_only else ''
    plt.savefig(f"{plot_folder}/comparative_{prefix}eigenvalue_contributions_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_{prefix}eigenvalue_contributions_{title}.svg")
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
        d_values = np.mean(positive_eigenvalues[:args.dim])
        gap = (d_values - positive_eigenvalues[args.dim+1]) / d_values
        score = gap

    else:
        raise ValueError("Invalid scoring method specified.")

    return score


def plot_spectral_gap_comparative(args_list, eigenvalues_list, score_method='i', title=''):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1]})
    # Retrieve the default color cycle
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = get_maximally_separated_colors(len(args_list))
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
    axs[0].set_title('Spectral Gap Analysis', fontweight='bold')
    axs[0].set_xlabel('Eigenvalue Rank')
    axs[0].set_ylabel('Spectral Gap Ratio')
    axs[0].legend()

    # Create a barplot for spectral gap scores in the second subplot, using the same colors
    for j, spectral_gap_score in enumerate(spectral_gap_scores):
        color = colors[j]
        axs[1].bar(j, spectral_gap_score, color=color)
    axs[1].set_xlabel('Network')
    axs[1].set_ylabel('Spectral Gap Score')
    axs[1].set_xticks(range(len(args_list)))
    axs[1].set_xticklabels([args.network_name for args in args_list], rotation=0, ha="center")

    plt.tight_layout()
    plot_folder = f"{args_list[0].directory_map['mds_dim']}"
    plt.savefig(f"{plot_folder}/comparative_spectral_gap_{title}.svg")
    plot_folder2 = f"{args_list[0].directory_map['comparative_plots']}"
    plt.savefig(f"{plot_folder2}/comparative_spectral_gap_{title}.svg")
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

    ## Simulation with False Edges
    args_list = generate_several_graphs(from_one_graph=True)
    title = "False_Eddge_Comparison"

    # ### Experimental
    # edge_list_titles_dict = {"weinstein_data_corrected_february.csv": ('W', [5, 10, 15]),
    #                     "pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv": ('PXL',None),
    #                     "subgraph_8_nodes_160_edges_179_degree_2.24.pickle": ('HL-S', None),
    #                     "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle": ('HL-E', None)}
    # # edge_list_titles_dict = {"weinstein_data_corrected_february.csv": [5,10,15],
    # #                     "subgraph_8_nodes_160_edges_179_degree_2.24.pickle": None,
    # #                          }
    # title = "Experimental Comparison"
    # args_list = generate_experimental_graphs(edge_list_titles_dict=edge_list_titles_dict)


    ## Pipeline
    make_spatial_constant_comparative_plot(args_list, title=title)
    make_dimension_prediction_comparative_plot(args_list, title=title)
    make_gram_matrix_analysis_comparative_plot(args_list, title=title)
