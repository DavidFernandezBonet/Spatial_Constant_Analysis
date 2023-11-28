import matplotlib.pyplot as plt
import numpy as np
from utils import CurveFitting

def plot_spatial_constant(args, df):
    plt.figure()

    # Plot Spatial Constant
    plt.plot(df['number_of_random_edges'], df['K_sp'], marker='o', label='Spatial Constant', color='tab:blue')

    # Plot Small-World Constant
    plt.plot(df['number_of_random_edges'], df['K_smallworld'], marker='x', label='Small-World Constant', color='tab:red')

    # Set labels and title
    plt.xlabel('Number of Random Edges Added')
    plt.ylabel('Constant')
    plt.title('Spatial and Small-World Constants vs. Number of Random Edges')

    plt.legend()

    # Title and grid
    plt.title('Spatial and Small-World Constants vs. Number of Random Edges')


    # Save the plot
    save_path = args.directory_map['plots_spatial_constant']
    plt.savefig(f"{save_path}/spatial_constant_change_with_false_edges_data.png")

def plot_average_shortest_path(args, df):
    plt.figure()

    # Plot Average Shortest Path Length
    plt.plot(df['number_of_random_edges'], df['mean_shortest_path'], marker='o', color='tab:green')

    # Set labels and title
    plt.xlabel('Number of Random Edges Added')
    plt.ylabel('Average Shortest Path Length')
    plt.title('Average Shortest Path Length vs. Number of Random Edges Added')

    # Save the plot
    save_path = args.directory_map['plots_spatial_constant']
    plt.savefig(f"{save_path}/average_shortest_path_vs_random_edges.png")


def plot_mean_sp_with_graph_growth(args, df):
    plt.figure()
    # Plot Spatial Constant
    plt.plot(df['num_nodes'], df['mean_shortest_path'], marker='o', label='Mean Shortest Path')


    # Set labels and title
    plt.xlabel('Number of Nodes')
    plt.ylabel('Mean Shortest Path')
    plt.title('Mean Shortest Path vs. Number of Nodes')
    plt.legend()

    # Save the plot
    save_path = args.directory_map['plots_spatial_constant']
    plt.savefig(f"{save_path}/mean_sp_vs_graph_growth.png")


def plot_mean_sp_with_graph_growth(args, df, plot_filename, model="spatial_constant_dim=2"):
    # Initialize GraphFitting instance with the data
    curve_fitting = CurveFitting(df['num_nodes'].values, df['mean_shortest_path'].values)
    print(model)
    if model == "spatial_constant_dim=2":
        model_func = curve_fitting.spatial_constant_dim2
    elif model == "spatial_constant_dim=3":
        model_func = curve_fitting.spatial_constant_dim3
    elif model == "small_world":
        model_func = curve_fitting.small_world_model

    # Perform curve fitting using the power model
    curve_fitting.perform_curve_fitting(model_func)

    # Set labels, title, and save path
    xlabel = 'Number of Nodes'
    ylabel = 'Mean Shortest Path'
    title = 'Mean Shortest Path vs. Number of Nodes'
    save_path = f"{args.directory_map['plots_spatial_constant']}/{plot_filename}"

    # Plot the data with the fitted curve
    curve_fitting.plot_fit_with_uncertainty(model_func, xlabel, ylabel, title, save_path)


# def plot_mean_sp_with_num_nodes_large_and_smallworld(args, df, plot_filename):
#     # Initialize the plot
#     fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
#
#     # Initialize GraphFitting instance for original graph
#     df_original = df[df['false_edges_introduced'] == False]
#     curve_fitting_original = CurveFitting(df_original['num_nodes'].values, df_original['mean_shortest_path'].values)
#     curve_fitting_original.perform_curve_fitting(curve_fitting_original.spatial_constant_dim2)
#     # curve_fitting_original.plot_fit_with_uncertainty_for_dataset(df_original['num_nodes'].values,
#     #                                                              df_original['mean_shortest_path'].values,
#     #                                                              curve_fitting_original.spatial_constant_dim2, ax,
#     #                                                              label_prefix='Original', color='tab:blue')
#
#     # Plot Mean Shortest Path for Original Graphs
#     plt.scatter(df_original['num_nodes'], df_original['mean_shortest_path'], marker='o', label='Original Graph',
#                 color='tab:blue')
#     plt.plot(df_original['num_nodes'],
#              curve_fitting_original.spatial_constant_dim2(df_original['num_nodes'], *curve_fitting_original.popt),
#              label='Fit Original', linestyle='--', color='tab:blue')
#
#
#     # Initialize GraphFitting instance for small-world graph
#     df_small_world = df[df['false_edges_introduced'] == True]
#     curve_fitting_small_world = CurveFitting(df_small_world['num_nodes'].values,
#                                              df_small_world['mean_shortest_path'].values)
#     curve_fitting_small_world.perform_curve_fitting(curve_fitting_small_world.small_world_model)
#     # curve_fitting_small_world.plot_fit_with_uncertainty_for_dataset(df_small_world['num_nodes'].values,
#     #                                                                 df_small_world['mean_shortest_path'].values,
#     #                                                                 curve_fitting_small_world.small_world_model, ax,
#     #                                                                 label_prefix='Small-World', color='tab:red')
#
#     # Plot Mean Shortest Path for Small-World Graphs
#     plt.scatter(df_small_world['num_nodes'], df_small_world['mean_shortest_path'], marker='x',
#                 label='Small-World Graph', color='tab:red')
#     plt.plot(df_small_world['num_nodes'],
#              curve_fitting_small_world.small_world_model(df_small_world['num_nodes'], *curve_fitting_small_world.popt),
#              label='Fit Small-World', linestyle='--', color='tab:red')
#
#     # Set labels and title
#     plt.xlabel('Number of Nodes')
#     plt.ylabel('Mean Shortest Path')
#     plt.title('Mean Shortest Path vs. Number of Nodes')
#     plt.legend()
#
#     # Save the plot
#     save_path = f"{args.directory_map['plots_spatial_constant']}/{plot_filename}"
#     plt.savefig(save_path)


def plot_mean_sp_with_num_nodes_large_and_smallworld(args, df, plot_filename):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    series_types = df['series_type'].unique()
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for series, color in zip(series_types, colors):
        if series == "Original" or series == "Smallworld":
            continue
        df_series = df[df['series_type'] == series]

        x_data = df_series['num_nodes'].values
        y_data = df_series['mean_shortest_path'].values
        sorted_indices = np.argsort(x_data)
        sorted_x = x_data[sorted_indices]
        sorted_y = y_data[sorted_indices]


        ax.scatter(sorted_x, sorted_y,  alpha=0.5, edgecolors='w', zorder=3, color=color)
        plt.plot(sorted_x, sorted_y, label=series,
                 linestyle='--', color=color, zorder=2)

    # Plot for Original Graph
    df_original = df[df['series_type'] == "Original"]
    curve_fitting_original = CurveFitting(df_original['num_nodes'].values, df_original['mean_shortest_path'].values)
    curve_fitting_original.plot_fit_with_uncertainty_for_dataset(df_original['num_nodes'].values,
                                                                 df_original['mean_shortest_path'].values,
                                                                 curve_fitting_original.spatial_constant_dim2,
                                                                 ax, label_prefix='Original', color='tab:blue',
                                                                 y_position=0.95)
    print("STD parameters Original", curve_fitting_original.sigma)

    # Plot for Small-World Graph
    df_small_world =  df[df['series_type'] == "Smallworld"]
    curve_fitting_small_world = CurveFitting(df_small_world['num_nodes'].values, df_small_world['mean_shortest_path'].values)
    curve_fitting_small_world.plot_fit_with_uncertainty_for_dataset(df_small_world['num_nodes'].values,
                                                                    df_small_world['mean_shortest_path'].values,
                                                                    curve_fitting_small_world.small_world_model, ax,
                                                                    label_prefix='Small-World', color='tab:red',
                                                                    y_position=0.85)

    print("STD parameters SmallWorld", curve_fitting_small_world.sigma)

    # Set labels and title
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Mean Shortest Path')
    ax.set_title('Mean Shortest Path vs. Number of Nodes')
    ax.legend()

    # Save the plot
    save_path = f"{args.directory_map['plots_spatial_constant']}/{plot_filename}"
    plt.savefig(save_path)
