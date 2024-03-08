import numpy as np

from curve_fitting import CurveFitting
from spatial_constant_analysis import *
from utils import read_position_df
import scipy.stats as stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from algorithms import compute_shortest_path_matrix_sparse_graph

# What about seaborn?
import scienceplots
font_size = 24
plt.rcParams.update({'font.size': font_size})
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 6
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size - 10

plt.style.use(['nature'])

sns.set_style("white")  # 'white' is a style option in seaborn

# If you want to use a seaborn style with modifications
sns.set(style="white", rc={
    'axes.labelsize': font_size,
    'axes.titlesize': font_size + 6,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size - 10
})
def compute_correlation_euclidean_sp(euclidean_distances, shortest_path_distances):
    # Flatten the matrices to compute correlation
    euclidean_flat = euclidean_distances.flatten()
    shortest_path_flat = shortest_path_distances.flatten()

    # Compute Pearson correlation coefficient
    correlation, _ = pearsonr(euclidean_flat, shortest_path_flat)

    return correlation



def plot_euclidean_sp_correlation(args, euclidean_distances, shortest_path_distances):
    euclidean_flat = euclidean_distances.flatten()
    shortest_path_flat = shortest_path_distances.flatten()

    correlation = compute_correlation_euclidean_sp(euclidean_distances, shortest_path_distances)
    sns.set_style('white')

    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(shortest_path_flat, euclidean_flat, gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(hb, label='Point Density')

    slope, intercept, _, _, _ = linregress(shortest_path_flat, euclidean_flat)
    x_vals = np.array(plt.xlim())
    y_vals = intercept + slope * x_vals
    trend_color = hb.get_cmap()(0.75)  # Use a color from the hexbin colormap
    plt.plot(x_vals, y_vals, '--', color=trend_color, label=f'$R^2$: {correlation**2:.4f}')

    plt.xlabel('Shortest Path Distance')
    plt.ylabel('Euclidean Distance')
    plt.legend(facecolor='white', framealpha=1)

    plot_folder = args.directory_map["plots_euclidean_sp"]
    plt.savefig(f"{plot_folder}/correlation_euclidean_sp_single_{args.args_title}.svg", format='svg')

def plot_euclidean_sp_correlation_single(ax, euclidean_distances, shortest_path_distances, label, color):
    euclidean_flat = euclidean_distances.flatten()
    shortest_path_flat = shortest_path_distances.flatten()

    correlation = compute_correlation_euclidean_sp(euclidean_distances, shortest_path_distances)
    sns.set_style('white')

    # Create a colormap centered around the specified color
    # cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", color, color])
    cmap = "Blues"



    # Randomly sample a subset of points for KDE density plot
    sample_size = 10000  # Adjust based on your computational capacity
    indices = np.random.choice(range(len(shortest_path_flat)), sample_size, replace=False)
    sampled_shortest_path = shortest_path_flat[indices]
    sampled_euclidean = euclidean_flat[indices]
    print("plotting kde plot...")
    sns.kdeplot(x=sampled_shortest_path, y=sampled_euclidean, cmap=cmap, fill=True, ax=ax)
    print("finished kde...")

    # ## Before I was using hexagons
    # hb = ax.hexbin(shortest_path_flat, euclidean_flat, gridsize=50, cmap=cmap, mincnt=1)
    # plt.colorbar(hb, label='Point Density')

    slope, intercept, _, _, _ = linregress(shortest_path_flat, euclidean_flat)
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals

    # Use the main color for the trend line
    ax.plot(x_vals, y_vals, '--', color=color, label=f'$R^2$ = {correlation**2:.4f}')


    ### Plot representative points
    df = pd.DataFrame({'ShortestPath': sampled_shortest_path, 'Euclidean': sampled_euclidean})
    num_bins = 100  # This could be adjusted based on the range and distribution of shortest path distances
    df['Bin'] = pd.cut(df['ShortestPath'], bins=num_bins, labels=False)
    selected_points = []
    for bin_id in range(num_bins):
        bin_df = df[df['Bin'] == bin_id]
        if len(bin_df) >= 10:
            selected_points.append(bin_df.sample(n=10))
        else:
            selected_points.append(bin_df)
    selected_points_df = pd.concat(selected_points)
    ax.scatter(selected_points_df['ShortestPath'], selected_points_df['Euclidean'], color='red', s=10, edgecolor='k',
               label='Representative Points')


    ax.set_xlabel('Shortest Path Distance')
    ax.set_ylabel('Euclidean Distance')
    ax.legend(facecolor='white', framealpha=1)


def plot_multiple_series(args, distance_matrix_pairs, colors, false_edge_list):
    fig, ax = plt.subplots(figsize=(10, 6))

    for (euclidean_distances, shortest_path_distances), color in zip(distance_matrix_pairs, colors):
        label = f'Series {len(colors) - len(distance_matrix_pairs)}'
        print(color)
        plot_euclidean_sp_correlation_single(ax, euclidean_distances, shortest_path_distances, label, color)


    plot_folder = args.directory_map["plots_euclidean_sp"]
    plt.savefig(f"{plot_folder}/correlation_euclidean_sp_multiple_{args.args_title}.svg", format='svg')

def generate_false_edge_series(sparse_graph, euclidean_distance_matrix, num_edges_to_add_list):
    # Compute the original distance matrix
    original_dist_matrix = euclidean_distance_matrix

    distance_matrix_pairs = []

    for num_edges in num_edges_to_add_list:
        # Create a copy of the original sparse graph
        modified_graph = sparse_graph.copy()  # Assuming sparse_graph is predefined

        # Add random edges
        if num_edges > 0:
            modified_graph = add_random_edges_to_csrgraph(modified_graph, num_edges_to_add=num_edges)

        # Compute shortest path matrix
        sp_matrix = np.array(shortest_path(csgraph=modified_graph, directed=False))

        # Add the pair to the list
        distance_matrix_pairs.append((original_dist_matrix, sp_matrix))

    return distance_matrix_pairs

def plot_r2_vs_false_edges(ax, r2_values, false_edge_list):
    # Ensure no zero values as log scale can't handle them
    false_edge_list = [max(1, num) for num in false_edge_list]

    ax.plot(false_edge_list, r2_values, marker='o', color='#009ADE', linestyle='-')
    # ax.set_xscale('log')
    # ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    #
    # # Set the ticks to be at the false edges positions
    # ax.set_xticks(false_edge_list)
    # ax.set_xticklabels(false_edge_list)

    ax.set_xlabel('Number of False Edges')
    ax.set_ylabel('$R^2$ Coefficient')
    # ax.set_title('R² Coefficient vs. Number of False Edges')

def generate_and_plot_series_with_false_edges(args, euclidean_distance_matrix, sparse_graph, false_edge_list):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Series without false edges
    original_dist_matrix = euclidean_distance_matrix
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))
    plot_euclidean_sp_correlation_single(axes[0], original_dist_matrix, sp_matrix, '', 'skyblue')

    # Compute R2 values for varying false edges
    r2_values = []

    max_false_edges = max(false_edge_list)  # Assume false_edge_list is defined
    all_random_false_edges = select_false_edges_csr(sparse_graph, max_false_edges)


    for num_edges in false_edge_list:
        modified_graph = add_specific_random_edges_to_csrgraph(sparse_graph.copy(), all_random_false_edges, num_edges)
        # modified_graph = add_random_edges_to_csrgraph(sparse_graph.copy(), num_edges)
        sp_matrix = np.array(shortest_path(csgraph=modified_graph, directed=False))
        correlation = compute_correlation_euclidean_sp(original_dist_matrix, sp_matrix)
        r2_values.append(correlation ** 2)

    # Plot R2 vs. False Edges
    plot_r2_vs_false_edges(axes[1], r2_values, false_edge_list)

    plt.tight_layout()
    plot_folder = args.directory_map["plots_euclidean_sp"]
    plt.savefig(f"{plot_folder}/correlation_r2_false_edges_{args.args_title}.svg", format='svg')

def plot_ideal_case(args, original_dist_matrix, shortest_path_distance_matrix):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Plot 1 - Original Distance Matrix
    sns.heatmap(original_dist_matrix, ax=axs[0], cmap="viridis", cbar_kws={'label': 'Euclidean Distance'})
    axs[0].set_title("Euclidean DM")
    axs[0].collections[0].set_rasterized(True)
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")
    axs[0].set_xticklabels([])  # Remove x-axis number labels
    axs[0].set_yticklabels([])  # Remove y-axis number labels
    # Plot 2 - Shortest Path Distance Matrix
    sns.heatmap(shortest_path_distance_matrix, ax=axs[1], cmap="viridis", cbar_kws={'label': 'Shortest Path Distance'})
    axs[1].set_title("Shortest Path DM")
    axs[1].collections[0].set_rasterized(True)
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")
    axs[1].set_xticklabels([])  # Remove x-axis number labels
    axs[1].set_yticklabels([])  # Remove y-axis number labels
    # Plot 3 - Correlation between original distance and shortest path distances
    plot_euclidean_sp_correlation_single(axs[2], original_dist_matrix, shortest_path_distance_matrix, "Correlation", "blue")


    plt.tight_layout()
    plot_folder = args.directory_map["plots_euclidean_sp"]
    plt.savefig(f"{plot_folder}/heatmap_and_correlation_single_case_{args.args_title}", format='svg')
    plt.show()


def plot_single_correlation_euclidean_sp_series(args, original_dist_matrix, sparse_graph):
    ## Single series
    ### Add random edges? See efect in the dimensionality here
    sparse_graph = add_random_edges_to_csrgraph(sparse_graph, num_edges_to_add=0)

    # Compute shortest path matrix
    sp_matrix = np.array(shortest_path(csgraph=sparse_graph, directed=False))

    plot_euclidean_sp_correlation(args, original_dist_matrix, sp_matrix)


def make_euclidean_sp_correlation_plot(single_series=True, multiple_series=False):
    # Parameters
    args = GraphArgs()
    args.proximity_mode = "knn"
    args.dim = 2
    args.intended_av_degree = 15
    args.num_points = 1000  # 2000

    # # # 1 Simulation
    create_proximity_graph.write_proximity_graph(args, order_indices=True)
    sparse_graph = load_graph(args, load_mode='sparse')
    shortest_path_distance_matrix = compute_shortest_path_matrix_sparse_graph(sparse_graph)

    ## Original data
    edge_list = read_edge_list(args)
    original_positions = read_position_df(args=args)
    # plot_original_or_reconstructed_image(args, image_type="original", edges_df=edge_list)
    original_dist_matrix = compute_distance_matrix(original_positions)



    if single_series:
        # # Single series
        # plot_single_correlation_euclidean_sp_series(args, original_dist_matrix, sparse_graph)
        # Plots the heatmap of the matrices also and the correlation
        plot_ideal_case(args, original_dist_matrix, shortest_path_distance_matrix)

    elif multiple_series:
        ## Multiple series
        # false_edge_list = [0, 5, 25, 50, 100]
        false_edge_list = np.linspace(start=0, stop=100, num=10).astype(int)
        generate_and_plot_series_with_false_edges(args, original_dist_matrix, sparse_graph, false_edge_list)


# make_euclidean_sp_correlation_plot()