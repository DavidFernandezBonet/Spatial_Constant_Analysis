import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from curve_fitting import CurveFitting
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


import scienceplots
import os
from scipy.interpolate import griddata

from algorithms import find_geometric_central_node, compute_shortest_path_mapping_from_central_node
from map_image_to_colors import map_points_to_colors
import matplotlib.lines as mlines




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
    if args.show_plots:
        plt.show()
        plt.close()

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
    if args.show_plots:
        plt.show()
        plt.close()





def plot_mean_sp_with_graph_growth(args, df, plot_filename, model="spatial_constant_dim=2", return_s_and_r2=False):
    # Initialize GraphFitting instance with the data
    curve_fitting = CurveFitting(df['num_nodes'].values, df['mean_shortest_path'].values)
    print(model)
    if model == "spatial_constant_dim=2":
        model_func = curve_fitting.spatial_constant_dim2
    elif model == "spatial_constant_dim=3":
        model_func = curve_fitting.spatial_constant_dim3
    elif model == "spatial_constant_dim=2_linearterm":
        model_func = curve_fitting.spatial_constant_dim2_linearterm
    elif model == "spatial_constant_dim=3_linearterm":
        model_func = curve_fitting.spatial_constant_dim3_linearterm


    elif model == "power_model_2d_Sconstant":
        model_func = curve_fitting.power_model_2d_Sconstant
    elif model == "power_model_3d_Sconstant":
        model_func = curve_fitting.power_model_3d_Sconstant
    elif model == "power_model_2d_bi_Sconstant":
        model_func = curve_fitting.power_model_2d_bi_Sconstant
    elif model == "power_model_3d_bi_Sconstant":
        model_func = curve_fitting.power_model_3d_bi_Sconstant


    elif model == "small_world":
        model_func = curve_fitting.small_world_model
    elif model == "power_law":
        model_func = curve_fitting.power_model
    elif model == "power_law_w_constant":
        model_func = curve_fitting.power_model_w_constant

    # Perform curve fitting using the power model
    curve_fitting.perform_curve_fitting(model_func)

    # Set labels, title, and save path
    xlabel = 'Number of Nodes'
    ylabel = 'Mean Shortest Path'
    title = 'Mean Shortest Path vs. Number of Nodes'
    save_path = f"{args.directory_map['plots_spatial_constant_gg']}/{plot_filename}"

    # Plot the data with the fitted curve
    curve_fitting.plot_fit_with_uncertainty(model_func, xlabel, ylabel, title, save_path)
    if return_s_and_r2:
        return curve_fitting.popt[0], curve_fitting.r_squared  # return spatial constant and r2
    else:
        return




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
    if args.dim == 2:
        func_fit = curve_fitting_original.spatial_constant_dim2
    elif args.dim == 3:
        func_fit = curve_fitting_original.spatial_constant_dim3
    else:
        raise ValueError("Dimension must be 2 or 3")
    curve_fitting_original.plot_fit_with_uncertainty_for_dataset(df_original['num_nodes'].values,
                                                                 df_original['mean_shortest_path'].values,
                                                                 func_fit,
                                                                 ax, label_prefix='Original', color='tab:blue',
                                                                 y_position=0.95)
    print("STD parameters Original", curve_fitting_original.sigma)

    # Plot for Small-World Graph
    df_small_world =  df[df['series_type'] == "Smallworld"]
    curve_fitting_small_world = CurveFitting(df_small_world['num_nodes'].values, df_small_world['mean_shortest_path'].values)
    curve_fitting_small_world.plot_fit_with_uncertainty_for_dataset(df_small_world['num_nodes'].values,
                                                                    df_small_world['mean_shortest_path'].values,
                                                                    curve_fitting_small_world.power_model, ax,
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



def plot_clustering_coefficient_distribution(args, clustering_coefficients, title="Clustering Coefficient Distribution"):
    plt.figure(figsize=(12, 6))  # Adjusted figure size for potential subplots

    if args.is_bipartite:
        # Plotting two subplots for each set in a bipartite graph
        mean_clustering_coefficient_set1 = args.mean_clustering_coefficient_set1
        mean_clustering_coefficient_set2 = args.mean_clustering_coefficient_set2

        # Clustering Coefficients for Set 1
        plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
        plt.hist(clustering_coefficients[0], bins=20, color='blue', alpha=0.7, rwidth=0.85)
        plt.title(title + " - Set 1")
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.legend([f"Mean Clustering Coefficient: {mean_clustering_coefficient_set1:.2f}"])

        # Clustering Coefficients for Set 2
        plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
        plt.hist(clustering_coefficients[1], bins=20, color='red', alpha=0.7, rwidth=0.85)
        plt.title(title + " - Set 2")
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.legend([f"Mean Clustering Coefficient: {mean_clustering_coefficient_set2:.2f}"])

    else:
        # Plotting a single histogram for non-bipartite graphs
        mean_clustering_coefficient = args.mean_clustering_coefficient
        plt.hist(clustering_coefficients, bins=20, color='blue', alpha=0.7, rwidth=0.85)
        plt.title(title)
        plt.xlabel('Clustering Coefficient')
        plt.ylabel('Frequency')
        plt.legend([f"Mean Clustering Coefficient: {mean_clustering_coefficient:.2f}"])

    plot_folder = args.directory_map['plots_clustering_coefficient']
    plt.savefig(f"{plot_folder}/clust_coef_{args.args_title}", format="png")
    if args.show_plots:
        plt.show()
        plt.close()



def plot_degree_distribution(args, degree_distribution, title="Degree Distribution"):
    plt.figure(figsize=(12, 6))  # Adjusted figure size for potential subplots

    if args.is_bipartite:
        # Plotting two subplots for each set in a bipartite graph
        average_degree_set1 = args.average_degree_set1
        average_degree_set2 = args.average_degree_set2

        # Degree Distribution for Set 1
        plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
        plt.bar(range(len(degree_distribution[0])), degree_distribution[0], color='blue', alpha=0.7)
        plt.title(title + " - Set 1")
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend([f"Average Degree: {average_degree_set1:.2f}"])

        # Degree Distribution for Set 2
        plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
        plt.bar(range(len(degree_distribution[1])), degree_distribution[1], color='red', alpha=0.7)
        plt.title(title + " - Set 2")
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend([f"Average Degree: {average_degree_set2:.2f}"])

    else:
        # Plotting a single bar chart for non-bipartite graphs
        average_degree = args.average_degree
        plt.bar(range(len(degree_distribution)), degree_distribution, color='green', alpha=0.7)
        plt.title(title)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend([f"Average Degree: {average_degree:.2f}"])

    plot_folder = args.directory_map['plots_degree_distribution']
    print(args.args_title)
    plt.savefig(f"{plot_folder}/degree_dist_{args.args_title}", format="png")
    if args.show_plots:
        plt.show()
    plt.close()

def plot_shortest_path_distribution(args, shortest_path_dist, mean_shortest_path, title="Shortest Path Distribution"):
    # Plotting a single bar chart for non-bipartite graphs
    plt.figure(figsize=(12, 6))  # Adjusted figure size for potential subplots
    unique_paths, counts = np.unique(shortest_path_dist, return_counts=True)
    plt.bar(unique_paths, counts, color='blue', alpha=0.7)

    # Add a vertical line for the mean shortest path
    plt.axvline(x=mean_shortest_path, color='red', linestyle='--', linewidth=2, label=f"Mean: {mean_shortest_path:.2f}")

    plt.title(title)
    plt.xlabel('Shortest Path')
    plt.ylabel('Frequency')
    plt.legend()

    plot_folder = args.directory_map['plots_shortest_path_distribution']
    plt.savefig(f"{plot_folder}/plots_shortest_path_distribution_{args.args_title}", format="png")
    if args.show_plots:
        plt.show()
    plt.close()



### Functions for spatial constant variation analysis



def classify_row(row):
    if row['dim'] == 2 and 'bipartite' not in row['proximity_mode']:
        return '2D Non-Bipartite'
    elif row['dim'] == 2 and 'bipartite' in row['proximity_mode']:
        return '2D Bipartite'
    elif row['dim'] == 3 and 'bipartite' not in row['proximity_mode']:
        return '3D Non-Bipartite'
    else:  # row['dim'] == 3 and 'bipartite' in row['proximity_mode']
        return '3D Bipartite'

def plot_spatial_constant_variation(args, spatial_constant_variation_results_df):
    # quantity_of_interest = "S_general" # spatial constant
    quantity_of_interest = "relative_msp_prediction_error"
    ### Predictions
    constant_scaler = np.sqrt(4.5)
    bipartite_correction = 1 / 1.2  # 1/np.sqrt(2)   #TODO: i don't know exactly how the correction should look like!
    super_spatial_constant_3d = 0.66 * constant_scaler
    super_spatial_constant_2d = 0.517 * constant_scaler
    super_spatial_constant_2d_bipartite = super_spatial_constant_2d * bipartite_correction
    super_spatial_constant_3d_bipartite = super_spatial_constant_3d * bipartite_correction
    predicted_medians = [super_spatial_constant_2d, super_spatial_constant_2d_bipartite, super_spatial_constant_3d,
                         super_spatial_constant_3d_bipartite]

    groups = ['2D Non-Bipartite', '2D Bipartite', '3D Non-Bipartite', '3D Bipartite']
    n_class_groups = 4
    # Apply the classification function to each row
    spatial_constant_variation_results_df['classification'] = spatial_constant_variation_results_df.apply(classify_row, axis=1)

    # Step 2: Create the Violin Plot
    plt.figure(figsize=(10, 6))

    # sns.violinplot(x='classification', y='S_general', data=spatial_constant_variation_results_df)
    # Changed it to mean shortest path
    sns.violinplot(x='classification', y=quantity_of_interest, data=spatial_constant_variation_results_df)
    plt.title('Violin Plot Spatial Constant')
    plt.ylabel('Spatial Constant')  # Replace with the name of your variable

    # Plot predictions
    for i in range(n_class_groups):
        plt.scatter(x=i, y=predicted_medians[i], color='red', zorder=3)

    plot_folder = args.directory_map["plots_spatial_constant_variation"]
    plt.savefig(f'{plot_folder}/spatial_constant_variation_violin.png', bbox_inches='tight')

    # Interactive plot
    import plotly.express as px

    # Create a new column to identify if 'proximity_mode' contains 'false_edges'
    spatial_constant_variation_results_df['color_group'] = spatial_constant_variation_results_df[
        'proximity_mode'].apply(
        lambda x: 'With False Edges' if 'false_edges' in x else 'Without False Edges'
    )

    # Plot the main violin plot
    fig = px.violin(spatial_constant_variation_results_df, x='classification', y=quantity_of_interest, color='color_group',
                    box=False, points='all', hover_data=spatial_constant_variation_results_df.columns)

    # Add predicted medians as scatter points
    for i, group in enumerate(groups):
        fig.add_scatter(x=[group], y=[predicted_medians[i]], mode='markers', marker=dict(color='green'))

    # Update the layout
    fig.update_layout(
        title='Interactive Violin Plot of S_general for Different Groups',
        xaxis_title='Group',
        yaxis_title=quantity_of_interest
    )

    # Show the plot
    fig.write_html("violin_plot_S_variation" + '.html')
    fig.show()


def plot_original_or_reconstructed_image(args, image_type="original", edges_df=None, position_filename=None,
                                         plot_weights_against_distance=False, positions_df=None):
    """Plots the original or reconstructed image based on the provided arguments.

    This function handles the plotting of either the original or reconstructed graph images,
    with options to include weights against distance. It sets up the plot based on the dimensionality
    specified in `args`, retrieves edge data, and plots node positions with optional coloring.

    Args:
        args: An object containing configuration and graph arguments, including directory paths,
              dimensionality (`dim`), color mapping, and more.
        image_type (str): The type of image to plot, options are "original", "mst", or "reconstructed".
                          Defaults to "original".
        edges_df (pandas.DataFrame, optional): DataFrame containing edge data. If None, the edge data
                                               is loaded from a file specified in `args`. Defaults to None.
        position_filename (str, optional): Filename of the position data CSV file. If None, the position
                                           data is loaded based on `image_type` and configurations in `args`.
                                           Defaults to None.
        plot_weights_against_distance (bool): Flag to enable plotting of weights against distance for edges.
                                              Defaults to False.

    Raises:
        ValueError: If `image_type` is not one of the expected options ("original", "reconstructed", "mst").
        ValueError: If the edge list numbering is not valid, indicating a mismatch between edge data and node positions.
    """
    def setup_plot(args, figsize=(10, 8)):
        fig = plt.figure(figsize=figsize)
        if args.dim == 3:
            ax = fig.add_subplot(111, projection='3d')
        elif args.dim == 2:
            ax = fig.add_subplot(111)
        return ax

    def plot_positions(ax, positions_df, args, color_df=None, simulated_colors=False):
        # Merge positions_df with color_df on 'node_ID' if color_df is provided

        if color_df is not None:
            # merged_df = positions_df.merge(color_df, on='node_ID')
            # TODO: this is because shuai colors are incomplete (beacons are missing)

            merged_df = positions_df.merge(color_df, on='node_ID', how='left')
            # merged_df['color'] = merged_df['color'].fillna("gray")

            colors = merged_df['color'].tolist()

            color_priorities = {
                'red': 2,  # Highest priority
                'green': 1,
                'gray': 0,
            }
            priority_list = merged_df['color'].map(color_priorities).tolist()  # Use .fillna(0) to handle any undefined colors with the lowest priority


            if args.dim == 3:
                ax.scatter(merged_df['x'], merged_df['y'], merged_df['z'], color=colors, zorder=priority_list)
            elif args.dim == 2:
                ax.scatter(merged_df['x'], merged_df['y'], color=colors)







        else:
            if simulated_colors:

                if image_type == "reconstructed" and args.node_ids_map_old_to_new is not None:
                    id_to_color = {}
                    for old_index, new_index in args.node_ids_map_old_to_new.items():
                        id_to_color[new_index] = args.id_to_color_simulation[old_index]

                else:
                    id_to_color = args.id_to_color_simulation


                colors = [id_to_color.get(node_id, 'b')  # Default to 'b' if node_id is not in the dictionary
                          for node_id in positions_df['node_ID']]

                if args.dim == 3:
                    ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'],
                               facecolors=colors)
                elif args.dim == 2:
                    ax.scatter(positions_df['x'], positions_df['y'],
                               facecolors=colors)

            else:

                if args.dim == 3:
                    ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'], facecolors='none',
                               edgecolors='b')
                elif args.dim == 2:
                    ax.scatter(positions_df['x'], positions_df['y'], facecolors='none', edgecolors='b')


    # image_type: original, mst, reconstructed


    # Simulated colors should only be used for simulation proximity modes
    if args.colorfile is not None:
        if args.colorfile[-4:] != '.csv' and args.proximity_mode == "experimental":
            args.colorfile = None


    # TODO: this should not be working when we reconstruct with sparse matrices right? The order of the edge list is different (so positions and graph are different)
    if edges_df is None:
        edge_list_folder = args.directory_map["edge_lists"]
        edges_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")
        if args.verbose:
            print(f"retrieving edges from {args.edge_list_title}")

    # Get the positions
    if image_type == "original" or image_type == "mst":

        if positions_df is None:
            original_position_folder = args.directory_map["original_positions"]
            if position_filename is None:
                positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")
                # positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.args_title}.csv")
            else:
                positions_df = pd.read_csv(f"{original_position_folder}/{position_filename}")

        if args.colorfile is not None:
            # Check if the colorfile is an image
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']
            colorfile_extension = os.path.splitext(args.colorfile)[1].lower()
            if colorfile_extension in image_extensions:
                args.id_to_color_simulation = map_points_to_colors(positions_df, args.colorfile, args=args)

    elif image_type == "reconstructed":
        edge_list_folder = args.directory_map["edge_lists"]
        edges_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")
        reconstructed_position_folder = args.directory_map["reconstructed_positions"]
        if position_filename is None:
            positions_df = pd.read_csv(f"{reconstructed_position_folder}/positions_{args.args_title}.csv")
        else:
            positions_df = pd.read_csv(f"{reconstructed_position_folder}/{position_filename}")

        is_valid, reason = validate_edge_list_numbers(edges_df, np.array(positions_df))
        if not is_valid:
            print("Reason edge list is not valid:", reason)
            raise ValueError("Edge list numbering is not valid! It should go from 0 to N-1 points")
    else:
        raise ValueError("Please specify a correct value for image_type, either original or reconstructed.")

    ax = setup_plot(args)


    # Plot data
    if (args.colorfile is not None) and args.proximity_mode == "experimental":
        # Change the position ID for weinstein case also
        if args.colorfile[:4] == "wein" and image_type == 'original':  # mapping should only be applied when we have original image
            if args.node_ids_map_old_to_new is not None:
                positions_df['node_ID'] = positions_df['node_ID'].map(args.node_ids_map_old_to_new)
                positions_df = positions_df.dropna()
                positions_df['node_ID'] = positions_df['node_ID'].astype(int)

        ### Plot from color ID file
        color_folder = args.directory_map["colorfolder"]
        print(f"{color_folder}/{args.colorfile}")
        color_df = pd.read_csv(f"{color_folder}/{args.colorfile}")


        color_df['color'] = color_df['color'].map(args.colorcode)
        # If we selected a largest component then we need to map the node ids to the proper color
        if args.node_ids_map_old_to_new is not None:
            color_df['node_ID'] = color_df['node_ID'].map(args.node_ids_map_old_to_new)
            color_df = color_df.dropna()


        # Additional processing for color_df if required
        plot_positions(ax, positions_df, args, color_df)
    else:
        if args.id_to_color_simulation is not None:
            plot_positions(ax, positions_df, args, simulated_colors=True)
        else:
            plot_positions(ax, positions_df, args)

        # TODO: check that this makes sense for every case where we have old and new indices
        ## This applies the new mapping so the (new) edges get the right positions
        if args.node_ids_map_old_to_new is not None and image_type == 'original':
            positions_df['node_ID'] = positions_df['node_ID'].map(args.node_ids_map_old_to_new)
            positions_df = positions_df.dropna()
            positions_df['node_ID'] = positions_df['node_ID'].astype(int)

    if plot_weights_against_distance:
        distances = []
        weights = []


    # Draw edges
    for _, row in edges_df.iterrows():
        filtered_df = positions_df[positions_df['node_ID'] == row['source']]
        if not filtered_df.empty:
            source = filtered_df.iloc[0]
        else:
            continue

        source = positions_df[positions_df['node_ID'] == row['source']].iloc[0]     # TODO: is this the most efficient?
        target = positions_df[positions_df['node_ID'] == row['target']].iloc[0]

        edge_color = 'red' if (row['source'], row['target']) in args.false_edge_ids or (
        row['target'], row['source']) in args.false_edge_ids else 'k'

        edge_alpha = 1 if (row['source'], row['target']) in args.false_edge_ids or (
        row['target'], row['source']) in args.false_edge_ids else 0.2

        edge_linewidth = 1 if (row['source'], row['target']) in args.false_edge_ids or (  #TODO: adjust linewidth
        row['target'], row['source']) in args.false_edge_ids else 0.5

        if args.dim == 3:
            ax.plot([source['x'], target['x']], [source['y'], target['y']], [source['z'], target['z']],
                    edge_color, linewidth=edge_linewidth, alpha=edge_alpha)
        else:
            ax.plot([source['x'], target['x']], [source['y'], target['y']],
                    edge_color, linewidth=edge_linewidth, alpha=edge_alpha)

        if plot_weights_against_distance:
            distance = np.sqrt((source['x'] - target['x']) ** 2 + (source['y'] - target['y']) ** 2)
            distances.append(distance)
            weights.append(row['weight'])



    format_extension = '.svg' if args.num_points < 1000 else '.png'
    if image_type == "original":
        plot_folder = args.directory_map["plots_original_image"]
        plt.savefig(f"{plot_folder}/original_image_{args.args_title}{format_extension}")

        plot_folder2 = args.directory_map['spatial_coherence']
        plt.savefig(f"{plot_folder2}/original_image_{args.args_title}{format_extension}")
        # plt.savefig(f"{plot_folder}/original_image_{args.args_title}", format='svg')
    elif image_type == "mst":
        plot_folder = args.directory_map["plots_mst_image"]
        plt.savefig(f"{plot_folder}/mst_image_{args.args_title}{format_extension}")
        # plt.savefig(f"{plot_folder}/mst_image_{args.args_title}", format='svg')
        plot_folder2 = args.directory_map['spatial_coherence']
        plt.savefig(f"{plot_folder2}/mst_image_{args.args_title}{format_extension}")
    else:
        plot_folder = args.directory_map["plots_reconstructed_image"]
        plot_folder2 = args.directory_map['spatial_coherence']
        plot_folder3 = args.directory_map['rec_images_subgraphs']
        plt.savefig(f"{plot_folder}/reconstructed_image_{args.args_title}{format_extension}")
        plt.savefig(f"{plot_folder2}/reconstructed_image_{args.args_title}{format_extension}")
        plt.savefig(f"{plot_folder3}/reconstructed_image_{args.args_title}{format_extension}")
        # plt.savefig(f"{plot_folder}/reconstructed_image_{args.args_title}", format='svg')


    if plot_weights_against_distance:
        #### Create new dataframe
        # Merge edges with positions for source nodes
        merged_df = edges_df.merge(positions_df, how='left', left_on='source', right_on='node_ID',
                                   suffixes=('', '_source'))
        merged_df.rename(columns={'x': 'x_source', 'y': 'y_source', 'z': 'z_source'}, inplace=True)

        # Merge edges with positions for target nodes
        merged_df = merged_df.merge(positions_df, how='left', left_on='target', right_on='node_ID',
                                    suffixes=('', '_target'))
        merged_df.rename(columns={'x': 'x_target', 'y': 'y_target', 'z': 'z_target'}, inplace=True)

        # Calculate Euclidean distance
        if 'z_source' in merged_df.columns:  # Check if 3D data is available
            merged_df['distance'] = np.sqrt(
                (merged_df['x_source'] - merged_df['x_target']) ** 2 +
                (merged_df['y_source'] - merged_df['y_target']) ** 2 +
                (merged_df['z_source'] - merged_df['z_target']) ** 2
            )
        else:  # For 2D data
            merged_df['distance'] = np.sqrt(
                (merged_df['x_source'] - merged_df['x_target']) ** 2 +
                (merged_df['y_source'] - merged_df['y_target']) ** 2
            )

        # Create the improved edge list
        improved_edge_list = merged_df[['source', 'target', 'weight', 'distance']]
        plot_folder = args.directory_map["plots_original_image"]
        improved_edge_list.to_csv(f'{plot_folder}/improved_edge_list_{args.args_title}.csv', index=False)

        print("esta passant...")
        plt.figure(figsize=(10, 6))
        plt.scatter(distances, weights, color='blue', edgecolor='k')
        plt.title('Euclidean Distance vs. Weight')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Weight')
        plt.savefig(f"{plot_folder}/weight_distance_{args.args_title}_linear")
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig(f"{plot_folder}/weight_distance_{args.args_title}_log")

        from scipy.stats import gaussian_kde
        # Logarithmic transformation with a small shift to handle zero weights
        log_weights = np.log10(np.array(weights) + 1)
        log_distances = np.log10(np.array(distances) + 1)

        # Calculate the point density
        xy = np.vstack([log_distances, log_weights])
        z = gaussian_kde(xy)(xy)
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(log_distances, log_weights, c=z, s=50, edgecolor='none', cmap='viridis')
        fig.colorbar(sc, ax=ax, label='Density')

        ax.set_title('Log-Transformed Weights vs. Distances with Density Coloring')
        ax.set_xlabel('Log of Euclidean Distance')
        ax.set_ylabel('Log of Weight')
        plt.savefig(f"{plot_folder}/weight_distance_{args.args_title}_log_with_density")


        # ### Boxplot
        # bins = np.linspace(min(distances), max(distances), num=10)  # Adjust num for more or fewer bins
        # binned_distances = np.digitize(distances, bins)
        #
        # fig, ax = plt.subplots(figsize=(10, 6))
        # for i in range(1, len(bins)):
        #     data = [weights[binned_distances == j] for j in range(1, len(bins))]
        #     ax.boxplot(data, positions=[(bins[j - 1] + bins[j]) / 2 for j in range(1, len(bins))],
        #                widths=np.diff(bins) * 0.6)
        #
        # ax.set_title('Binned Euclidean Distance vs. Weight')
        # ax.set_xlabel('Euclidean Distance')
        # ax.set_ylabel('Weight')
        # plt.xticks(ticks=bins, labels=[f"{bins[i]:.2f}" for i in range(len(bins))], rotation=45)
        # plt.savefig(f"{plot_folder}/binned_weight_distance_boxplot.png")


    if args.show_plots:
        plt.show()

    plt.close('all')



def plot_shortest_path_heatmap(args, shortest_path_matrix, ax, vmin, vmax, positions_df=None,
                               edges_df=None, n_false_edges=0):
    original_position_folder = args.directory_map["original_positions"]
    if positions_df is None:
        positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

    if edges_df is None:
        edge_list_folder = args.directory_map["edge_lists"]
        edges_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")
        if args.verbose:
            print(f"retrieving edges from {args.edge_list_title}")

    central_node_ID = find_geometric_central_node(positions_df=positions_df)
    node_ID_to_shortest_path_mapping = (
        compute_shortest_path_mapping_from_central_node(central_node_ID=central_node_ID, positions_df=positions_df,
                                                    shortest_path_matrix=shortest_path_matrix))

    distances_df = pd.DataFrame(list(node_ID_to_shortest_path_mapping.items()), columns=['node_ID', 'distance'])
    merged_df = pd.merge(positions_df, distances_df, on='node_ID')
    partition_int = np.ceil(np.sqrt(args.num_points))
    grid_x, grid_y = np.mgrid[min(merged_df.x):max(merged_df.x):complex(0, partition_int),
                     min(merged_df.y):max(merged_df.y):complex(0, partition_int)]
    grid_z = griddata((merged_df.x, merged_df.y), merged_df.distance, (grid_x, grid_y), method='cubic')


    # Use ax (the subplot axes) for plotting instead of plt directly
    image = ax.imshow(grid_z.T, extent=(min(merged_df.x), max(merged_df.x), min(merged_df.y), max(merged_df.y)),
                      origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

    positions_indexed = positions_df.set_index('node_ID')
    if args.false_edge_ids and edges_df is not None:
        for edge in args.false_edge_ids[:n_false_edges]:
            # Extract source and target from the tuple, handling both directions
            source, target = edge
            # Ensure the edge exists in the DataFrame
            if (source in positions_indexed.index) and (target in positions_indexed.index):
                source_pos = positions_indexed.loc[source]
                target_pos = positions_indexed.loc[target]
                ax.plot([source_pos.x, target_pos.x], [source_pos.y, target_pos.y], color='red', linewidth=0.5, alpha=0.3)

    # Optionally set titles, labels, etc., using the ax object
    ax.set_title(f'False Edges {n_false_edges}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Return the image object for potential use with colorbars, etc.
    return image

    # # Plotting
    # plt.figure(figsize=(10, 8))
    # plt.imshow(grid_z.T, extent=(min(merged_df.x), max(merged_df.x), min(merged_df.y), max(merged_df.y)),
    #            origin='lower')
    # plt.colorbar(label='Shortest path distance to central node')
    #
    # plt.xlabel('X ')
    # plt.ylabel('Y ')
    # plt.grid(False)
    #
    # plot_folder = args.directory_map['plots_shortest_path_heatmap']
    # plt.savefig(f'{plot_folder}_heatmap_sp_{args.args_title}')
    # plt.show()

def compute_grid_z(args, shortest_path_matrix, positions_df=None):
    if positions_df is None:
        original_position_folder = args.directory_map["original_positions"]
        positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

    central_node_ID = find_geometric_central_node(positions_df=positions_df)
    node_ID_to_shortest_path_mapping = compute_shortest_path_mapping_from_central_node(
        central_node_ID=central_node_ID, positions_df=positions_df, shortest_path_matrix=shortest_path_matrix
    )

    distances_df = pd.DataFrame(list(node_ID_to_shortest_path_mapping.items()), columns=['node_ID', 'distance'])
    merged_df = pd.merge(positions_df, distances_df, on='node_ID')
    partition_int = np.ceil(np.sqrt(args.num_points))  # Adjust partition based on number of points
    grid_x, grid_y = np.mgrid[min(merged_df.x):max(merged_df.x):complex(0, partition_int),
                              min(merged_df.y):max(merged_df.y):complex(0, partition_int)]
    grid_z = griddata((merged_df.x, merged_df.y), merged_df.distance, (grid_x, grid_y), method='cubic')

    return grid_z

def plot_multiple_shortest_path_heatmaps(args, sp_matrices, false_edge_list):
    # Create a figure for the subplots
    num_plots = len(sp_matrices)
    fig, axs = plt.subplots(1, num_plots, figsize=(20, 5), constrained_layout=True)

    # vmin = np.min(sp_matrices[0])
    # vmax = np.max(sp_matrices[0])
    # for matrix in sp_matrices:
    #     vmin = min(vmin, np.min(matrix))
    #     vmax = max(vmax, np.max(matrix))

    all_grid_z = [compute_grid_z(args, sp_matrix) for sp_matrix in sp_matrices]

    vmin = np.min([np.nanmin(grid) for grid in all_grid_z])
    vmax = np.max([np.nanmax(grid) for grid in all_grid_z])


    # Now plot each with consistent color scale
    for i, sp_matrix in enumerate(sp_matrices):
        # Assuming plot_shortest_path_heatmap can accept subplot axes and color limits
        n_false_edges = false_edge_list[i]
        plot_shortest_path_heatmap(args, shortest_path_matrix=sp_matrix, ax=axs[i], vmin=vmin, vmax=vmax, n_false_edges=n_false_edges)

    # Create a single colorbar for the whole figure
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap="viridis"), ax=axs,
                 orientation='horizontal', fraction=0.02, pad=0.04, label="Shortest Path Distance")
    plot_folder = args.directory_map['plots_shortest_path_heatmap']
    plt.savefig(f'{plot_folder}/heatmap_several_sp_matrix{args.args_title}.png')
    plt.savefig(f'{plot_folder}/heatmap_several_sp_matrix{args.args_title}.svg')
    if args.show_plots:
        plt.show()
    plt.close()



def calculate_predicted_s(args):
    ### Predictions
    constant_scaler = np.sqrt(4.5)
    bipartite_correction = 1 / 1.2  # 1/np.sqrt(2)   #TODO: i don't know exactly how the correction should look like!
    super_spatial_constant_3d = 0.66 * constant_scaler
    super_spatial_constant_2d = 0.517 * constant_scaler
    super_spatial_constant_2d_bipartite = super_spatial_constant_2d * bipartite_correction
    super_spatial_constant_3d_bipartite = super_spatial_constant_3d * bipartite_correction

    if args.dim == 2:
        if args.is_bipartite:
            return super_spatial_constant_2d_bipartite  # Replace with actual calculation
        else:
            return super_spatial_constant_2d  # Replace with actual calculation
    elif args.dim == 3:
        if args.is_bipartite:
            return super_spatial_constant_3d_bipartite  # Replace with actual calculation
        else:
            return super_spatial_constant_3d  # Replace with actual calculation

def plot_sample_spatial_constant(args, dataframe):

    unique_sizes = dataframe['intended_size'].unique()
    plt.figure(figsize=(10, 6))

    # Determine the global range for all histograms
    min_value = dataframe['S_general'].min()
    max_value = dataframe['S_general'].max()
    bin_edges = np.linspace(min_value, max_value, 21)  # 20 bins across the full range

    for size in unique_sizes:
        subset = dataframe[dataframe['intended_size'] == size]
        plt.hist(subset['S_general'], alpha=0.5, label=f'Size {size}', bins=bin_edges)

    predicted_s = calculate_predicted_s(args=args)
    plt.axvline(predicted_s, color='r', linestyle='dashed', linewidth=2, label='Predicted S')

    plt.xlabel('S_general')
    plt.ylabel('Frequency')
    plt.title('Histogram of S_general for each Intended Size')
    plt.legend()

    plot_folder = f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/subgraph_sampling_{args.args_title}")
    if args.show_plots:
        plt.show()
    plt.close()


def plot_spatial_constant_against_subgraph_size(args, dataframe):
    unique_sizes = dataframe['intended_size'].unique()
    means = []
    std_devs = []
    sizes = []

    # Calculate mean and standard deviation for each size
    for size in unique_sizes:
        subset = dataframe[dataframe['intended_size'] == size]
        mean = subset['S_general'].mean()
        std = subset['S_general'].std()
        means.append(mean)
        std_devs.append(std)
        sizes.append(size)

    plt.figure(figsize=(10, 6))

    sizes = np.array(sizes)
    means = np.array(means)
    std_devs = np.array(std_devs)

    # Scatter plot
    plt.scatter(sizes, means, label='Mean Spatial Constant')

    # Elegant ribbon style
    ribbon_color = '#6FC276'
    contour_ribon_color = '#006400'
    plt.fill_between(sizes, means - std_devs, means + std_devs, color=ribbon_color, alpha=0.3, edgecolor=contour_ribon_color, linewidth=1, linestyle='--')

    # Optionally, plot the predicted S line if needed
    predicted_s = calculate_predicted_s(args=args)
    plt.axhline(predicted_s, color='r', linestyle='dashed', linewidth=2, label='Predicted S')

    plt.xlabel('Subgraph Size')
    plt.ylabel('Mean Spatial Constant')

    plt.legend()

    plot_folder = f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}.png")
    # if args.show_plots:
    #     plt.show()
    plt.close()



def plot_spatial_constant_against_subgraph_size_with_false_edges(args, dataframes, false_edge_list, mst_case_df=None):
    plt.close('all')
    # Main plot spatial constant
    plt.figure(figsize=(6, 4.5))



    for dataframe, false_edge_count in zip(dataframes, false_edge_list):
        unique_sizes = dataframe['intended_size'].unique()
        means = []
        std_devs = []
        sizes = []

        # Calculate mean and standard deviation for each size
        for size in unique_sizes:
            subset = dataframe[dataframe['intended_size'] == size]
            mean = subset['S_general'].mean()
            std = subset['S_general'].std()
            means.append(mean)
            std_devs.append(std)
            sizes.append(size)

        sizes = np.array(sizes)
        means = np.array(means)
        std_devs = np.array(std_devs)


        # Plot each series
        label = f'False Edges: {false_edge_count}'
        plt.plot(sizes, means, label=label, marker='o')
        plt.fill_between(sizes, means - std_devs, means + std_devs, alpha=0.3)



    if mst_case_df is not None:
        unique_sizes = mst_case_df['intended_size'].unique()
        means = [mst_case_df[mst_case_df['intended_size'] == size]['S_general'].mean() for size in unique_sizes]
        std_devs = [mst_case_df[mst_case_df['intended_size'] == size]['S_general'].std() for size in unique_sizes]

        plt.plot(unique_sizes, means, label='MST', marker='o')
        plt.fill_between(unique_sizes, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs), alpha=0.3)

    plt.xlabel('Subgraph Size')
    plt.ylabel('Mean Spatial Constant')
    # plt.title('Mean Spatial Constant vs. Subgraph Size')
    plt.legend()

    plot_folder = args.directory_map['plots_spatial_constant_subgraph_sampling']
    plot_folder2 = args.directory_map['spatial_coherence']

    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_version.svg")
    plt.savefig(f"{plot_folder2}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_version.svg")

    if args.show_plots:
        plt.show()
    plt.close()

    plt.yscale('log')
    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_loglin_version.png")
    plt.close('all')

    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_loglog_version.png")
    plt.close('all')


def plot_spatial_constant_against_subgraph_size_with_multiple_series(args, dataframe, false_edge_counts, ax=None, title=None, mst_case_df=None):
    # Use provided axis or get/create current active axis
    if ax is None:
        ax = plt.gca()


    # Unique values in the 'proximity_mode' column
    proximity_modes = dataframe['proximity_mode'].unique()

    # Loop through each proximity mode
    for mode in proximity_modes:
        # Extract the number of false edges from the proximity mode string
        match = re.search(r'with_false_edges=(\d+)', mode)
        false_edge_count = int(match.group(1)) if match else 0

        # Filter the DataFrame for the current proximity mode
        filtered_df = dataframe[dataframe['proximity_mode'] == mode]
        unique_sizes = filtered_df['intended_size'].unique()
        means = []
        std_devs = []
        sizes = []

        # Calculate mean and standard deviation for each size
        for size in unique_sizes:
            subset = filtered_df[filtered_df['intended_size'] == size]
            mean = subset['S_general'].mean()
            std = subset['S_general'].std()
            means.append(mean)
            std_devs.append(std)
            sizes.append(size)

        sizes = np.array(sizes)
        means = np.array(means)
        std_devs = np.array(std_devs)

        # Plot each series on the given axis
        label = f'False Edges: {false_edge_count}'
        ax.plot(sizes, means, label=label, marker='o')
        ax.fill_between(sizes, means - std_devs, means + std_devs, alpha=0.3)


    if mst_case_df is not None:
        pass

    ax.set_xlabel('Subgraph Size')
    ax.set_ylabel('Mean Spatial Constant')
    if title:
        ax.set_title(title)
    ax.legend()

    # Save plot if ax is not provided (assumed to be a standalone plot)
    if ax is None:
        plot_folder = args.directory_map['plots_spatial_constant_subgraph_sampling']
        plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_version.png")


def plot_spatial_constants_subplots(args, all_dataframes, all_false_edge_lists, weight_thresholds, mst_case_df=None):
    num_plots = len(weight_thresholds)
    fig, axs = plt.subplots(1, num_plots, figsize=(10 * num_plots, 6), sharey=True)

    # Ensure axs is iterable when there's only one subplot
    if num_plots == 1:
        axs = [axs]

    for idx, wt in enumerate(weight_thresholds):
        # Extract the DataFrames and false_edge_counts for the current weight threshold
        dataframes_for_wt = all_dataframes[idx]
        false_edge_counts_for_wt = all_false_edge_lists[idx]

        # Call the modified plot function for each subplot
        plot_spatial_constant_against_subgraph_size_with_multiple_series(
            args, dataframes_for_wt, false_edge_counts_for_wt, ax=axs[idx], title=f"Weight Threshold: {wt}", mst_case_df=mst_case_df
        )

    plt.tight_layout()
    plot_folder = args.directory_map['plots_spatial_constant_subgraph_sampling']
    plt.savefig(f"{plot_folder}/spatial_constant_subgraphs_different_weight_threshold_{args.args_title}_false_edge_version.png")
    plt.savefig(
        f"{plot_folder}/spatial_constant_subgraphs_different_weight_threshold_{args.args_title}_false_edge_version.pdf")

def plot_false_edges_against_spatial_constant(args, processed_false_edge_series):
    plot_folder = args.directory_map['plots_spatial_constant_false_edge_difference']

    # Determine the unique intended_subgraph_sizes across all dataframes
    all_sizes = set(list(processed_false_edge_series.keys()))


    # Create a single plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    cmap_name = 'Oranges_d'
    colors = sns.color_palette(cmap_name, len(all_sizes))
    # colors = cm.get_cmap('magma_r', len(all_sizes))
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    # Normalizing the size values for the color mapping
    norm = mcolors.Normalize(vmin=min(all_sizes), vmax=max(all_sizes))

    # Exctract each series and plot
    for i, size in enumerate(sorted(all_sizes)):
        plot_folder_fit = args.directory_map['plots_spatial_constant_false_edge_difference_fits']
        false_edges = processed_false_edge_series[size]["false_edges"]
        means = processed_false_edge_series[size]["means"]  # Spatial Constant means
        std_devs = processed_false_edge_series[size]["std_devs"]

        # Plotting the series for the current subgraph size
        ax1.errorbar(false_edges, means, yerr=std_devs, color=colors[i], label=f'Size: {size}', marker='o')
        # plt.errorbar(false_edges, means, yerr=std_devs, color=colors(i), label=f'Size: {size}', marker='o')  # for matplotlib

        log_false_edges = np.log(np.array(false_edges) + 1)  # +1 to avoid zero error
        log_means = np.log(means)
        curve_fitting_object = CurveFitting(log_false_edges, log_means)
        func_fit = curve_fitting_object.linear_model
        curve_fitting_object.perform_curve_fitting(model_func=func_fit)
        save_path = f"{plot_folder_fit}/false_edge_powerlaw_fit_size={size}_{args.args_title}_fedgelist={false_edges}.pdf"
        curve_fitting_object.plot_fit_with_uncertainty(model_func=func_fit, xlabel="Log False Edge Count",
                                                       ylabel="Log Spatial Constant", title=f"Size {size}", save_path=save_path)

    ax1.set_title('Spatial Constant vs False Edge Count')
    ax1.set_xlabel('False Edge Count')
    ax1.set_ylabel('Spatial Constant')

    # Creating a colorbar for the sizes
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Subgraph Size')

    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


    fig1.tight_layout()
    fig1.savefig(f"{plot_folder}/normal_false_edge_difference_{args.args_title}_fedgelist={false_edges}.pdf")

    ax1.set_xscale('log')
    fig1.savefig(f"{plot_folder}/semilogX_false_edge_difference_{args.args_title}_fedgelist={false_edges}.pdf")

    ax1.set_yscale('log')
    ax1.set_xscale('linear')
    fig1.savefig(f"{plot_folder}/semilogY_false_edge_difference_{args.args_title}_fedgelist={false_edges}.pdf")

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    fig1.savefig(f"{plot_folder}/loglog_false_edge_difference_{args.args_title}_fedgelist={false_edges}.pdf")

def plot_weight_distribution(args, edge_list_with_weight_df):
    # # Histogram Plot
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    # plt.hist(edge_list_with_weight_df['weight'], bins=20, color='skyblue', edgecolor='black', log=True)
    # plt.title('Histogram of Weights')
    # plt.xscale('log')
    # plt.xlabel('Weight')
    # plt.ylabel('Frequency')

    unique_weights = edge_list_with_weight_df['weight'].unique()
    num_unique_weights = len(unique_weights)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.hist(edge_list_with_weight_df['weight'], bins=num_unique_weights, color='skyblue', edgecolor='black', log=True)
    plt.title('Histogram of Weights')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')


    # Prepare DataFrame for Scatter Plot - Ordered by weight
    sorted_df = edge_list_with_weight_df.sort_values(by='weight')
    sorted_df['rank'] = range(1, len(sorted_df) + 1)

    # Scatter Plot - Ordered by weight
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.scatter(sorted_df['rank'], sorted_df['weight'], color='tomato')
    plt.title('Scatter Plot of Weights (Ordered by Rank)')
    plt.xlabel('Rank')
    plt.ylabel('Weight')


    # Show plots
    plt.tight_layout()
    plot_folder = args.directory_map['plots_weight_distribution']
    plt.savefig(f"{plot_folder}/weight_distribution_{args.args_title}.png")
    if args.show_plots:
        plt.show()
    plt.close()


def plot_s_general_vs_weight_threshold(args, results):

    weight_thresholds, s_generals = zip(*results)

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.plot(weight_thresholds, s_generals, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.title('Mean S_general vs. Weight Threshold')
    plt.xlabel('Weight Threshold')
    plt.ylabel('Mean S_general')
    plot_folder = args.directory_map['plots_spatial_constant_weighted_threshold']
    plt.savefig(f"{plot_folder}/spatial_constant_vs_weight_threshold{args.args_title}.png")
    plt.savefig(f"{plot_folder}/spatial_constant_vs_weight_threshold{args.args_title}.pdf")

def validate_edge_list_numbers(edge_list, reconstructed_positions):
    n = len(reconstructed_positions) - 1
    expected_set = set(range(n + 1))

    # Create a set of all values in 'source' and 'target'
    edge_values = set(edge_list['source']).union(set(edge_list['target']))

    if edge_values == expected_set:
        return True, "Edge list is valid."

    missing = expected_set - edge_values
    extra = edge_values - expected_set

    mismatch_info = []
    if missing:
        mismatch_info.append(f"Missing nodes: {missing}")
    if extra:
        mismatch_info.append(f"Extra nodes: {extra}")

    return False, "; ".join(mismatch_info)

    return edge_values == expected_set


def visualize_communities_positions(args, communities, modularity, edges_within_communities, edges_between_communities,
                                    edge_score_dict=None, betweenness_dict=None, rank_score_dict=None):
    import networkx as nx
    """
    Visualize the network with community colors, false edges, and modularity score.

    :param df: DataFrame with columns ['node_id', 'x', 'y'] for node positions.
    :param communities: Array or list of community labels corresponding to df['node_id'].
    :param false_edges: List of tuples (node_id_start, node_id_end) representing false edges.
    :param modularity: Modularity score of the community partition.
    """

    plt.close()
    plt.rcdefaults()

    def read_position_df(args, return_df=False):
        if hasattr(args, 'reconstruction_mode') and args.reconstruction_mode in args.args_title:
            old_args_title = args.args_title.replace(f"_{args.reconstruction_mode}",
                                                     "")
        else:
            old_args_title = args.args_title

        if args.proximity_mode == "experimental" and args.original_positions_available:
            filename = args.original_edge_list_title
            print("FILENAME", filename)
            match = re.search(r"edge_list_(.*?)\.csv", filename)
            if match:
                extracted_part = match.group(1)
                old_args_title = extracted_part
            else:
                old_args_title = filename[:-4]


        original_points_path = f"{args.directory_map['original_positions']}/positions_{old_args_title}.csv"
        original_points_df = pd.read_csv(original_points_path)
        # Choose columns based on the dimension specified in args.dim
        if args.dim == 2:
            columns_to_read = ['x', 'y']
        elif args.dim == 3:
            columns_to_read = ['x', 'y', 'z']
        else:
            raise ValueError("Invalid dimension specified. Choose '2D' or '3D'.")

        # Read the specified columns from the DataFrame
        original_points_array = np.array(original_points_df[columns_to_read])

        if return_df:
            return original_points_df
        else:
            return original_points_array

    if args.node_ids_map_old_to_new is not None:
        original_points = read_position_df(args, return_df=True)
        original_points['node_ID'] = original_points['node_ID'].map(args.node_ids_map_old_to_new)
        original_points = original_points.dropna()
        original_points['node_ID'] = original_points['node_ID'].astype(int)
        original_points_df = original_points.sort_values(by='node_ID')
        original_points = original_points_df[['x', 'y']].to_numpy()
    else:
        original_position_folder = args.directory_map["original_positions"]
        original_points_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

        # original_points_df = read_position_df(args, return_df=True)


    # ### Plotting
    # Load positions DataFrame
    positions_df = original_points_df
    # node_ids_map_new_to_old = {v: k for k, v in args.node_ids_map_old_to_new.items()}
    # Load edges DataFrame
    edge_list_folder = args.directory_map["edge_lists"]
    edges_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")

    # TODO: this plots everything to do with community detection and false edges
    plot_edge_communities_and_distances(edges_within_communities, edges_between_communities, original_positions_df=positions_df)
    if edge_score_dict is not None:
        plot_edge_scores_vs_distances(args, edge_score_dict, original_positions_df=positions_df, score_interpretation='negative')

    if betweenness_dict is not None:
        plot_edge_scores_vs_distances(args, betweenness_dict, original_positions_df=positions_df, score_interpretation='positive')

    if rank_score_dict is not None:
        plot_edge_scores_vs_distances(args, rank_score_dict, original_positions_df=positions_df, score_interpretation='negative')

    plot_false_true_edges_percentages(args, edges_within_communities, edges_between_communities)
    plot_edge_categories_and_distances(edges_within_communities, edges_between_communities, args.false_edge_ids,
                                       original_positions_df=positions_df)
    # Convert positions_df to a dictionary for efficient access
    positions_dict = positions_df.set_index('node_ID')[['x', 'y']].T.to_dict('list')

    # Map communities to colors
    if isinstance(communities[0], (np.ndarray, np.generic)):
        communities = communities[0]

    print(communities)
    communities_dict = {i: community for i, community in enumerate(communities)}
    unique_communities = set(communities_dict.values())
    # # community_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))  # Adjust color map as needed
    # # community_to_color = {community: color for community, color in zip(unique_communities, community_colors)}
    community_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_to_color = {community: color for community, color in zip(unique_communities, community_colors)}




    plt.figure(figsize=(20, 30))

    # Plot original points with community colors
    for node_ID, (x, y) in positions_dict.items():
        community_id = communities_dict[node_ID]
        plt.plot(x, y, 'o', color=community_to_color[community_id])

    # Plot edges
    for _, row in edges_df.iterrows():
        source_new_id = row['source']
        target_new_id = row['target']

        if source_new_id in positions_dict and target_new_id in positions_dict:
            source_pos = positions_dict[source_new_id]
            target_pos = positions_dict[target_new_id]

            # Check if the edge is a false edge
            is_false_edge = (row['source'], row['target']) in args.false_edge_ids or (
            row['target'], row['source']) in args.false_edge_ids

            edge_linewidth = 0.5
            edge = (row['source'], row['target'])
            if edge in edges_within_communities:
                edge_color = "blue"
            elif edge in edges_between_communities:
                edge_color = "green"
                edge_linewidth = 3
            if is_false_edge:
                edge_color = "red"
                edge_linewidth = 1
            edge_alpha = 0.1 if is_false_edge else 1

            # distance = np.sqrt((source_pos[0] - target_pos[0]) ** 2 + (source_pos[1] - target_pos[1]) ** 2)
            # if distance > 0.5:
            #     if (row['source'], row['target']) in edges_within_communities:
            #         edge_linewidth += 2  # Increase by 0.5 or any other value you see fit



            plt.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], color=edge_color, alpha=edge_alpha,
                     linewidth=edge_linewidth)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Modularity: {modularity}')




    # Assuming the rest of your plotting code is in place and you've already created 'community_to_color'

    # Create a list of proxy artists for the legend
    legend_handles = [mlines.Line2D([], [], color=community_to_color[community], marker='o', linestyle='None',
                                    markersize=10, label=f'Community {community}') for community in unique_communities]

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Make sure the plot is displayed properly with the legend outside
    plt.tight_layout()

    print("Modularity: ", modularity)
    plt.show()


def plot_edge_communities_and_distances(edges_within_communities, edges_between_communities, original_positions_df):
    # Function to compute Euclidean distance between two points
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Initialize lists to store distances
    distances_within = []
    distances_between = []

    # Create a dictionary for quick node ID to position lookup
    positions_dict = original_positions_df.set_index('node_ID').to_dict('index')

    # Compute distances for edges within communities
    for edge in edges_within_communities:
        node1, node2 = edge
        pos1 = positions_dict[node1]
        pos2 = positions_dict[node2]
        distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
        distances_within.append(distance)

    # Compute distances for edges between communities
    for edge in edges_between_communities:
        node1, node2 = edge
        pos1 = positions_dict[node1]
        pos2 = positions_dict[node2]
        distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
        distances_between.append(distance)

    # Plotting the histograms
    plt.figure(figsize=(10, 6))
    plt.hist(distances_within, bins=20, alpha=0.5, label='Within Communities')
    plt.hist(distances_between, bins=20, alpha=0.5, label='Between Communities')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Distances')
    plt.legend(loc='upper right')
    plt.show()


def plot_edge_scores_vs_distances(args, edge_score_dict, original_positions_df, score_interpretation='positive'):
    # Function to compute Euclidean distance between two points
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Create a dictionary for quick node ID to position lookup
    positions_dict = original_positions_df.set_index('node_ID').to_dict('index')

    # Initialize lists to store scores and distances
    scores = []
    distances = []
    colors = []
    labels = []  # This will hold the binary labels needed for ROC computation


    # Compute distances and retrieve scores for edges
    for edge, score in edge_score_dict.items():
        node1, node2 = edge
        pos1 = positions_dict[node1]
        pos2 = positions_dict[node2]
        distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])

        if edge in args.false_edge_ids:
            colors.append('red')
            labels.append(1)  # False edge
        else:
            colors.append('blue')
            labels.append(0)  # True edge

        distances.append(distance)
        if score_interpretation == 'negative':
            # Transform score if higher scores indicate a false edge
            score = max(edge_score_dict.values()) + 1 - score
        scores.append(score)

    # Set up the figure and the subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Now three subplots

    # Scatter plot of scores vs distances
    axs[0].scatter(distances, scores, alpha=0.5, color=colors)
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Score')
    axs[0].set_title('Plot of Score vs. Distance for Edges')

    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    axs[1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    axs[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('Receiver Operating Characteristic')
    axs[1].legend(loc="lower right")

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    axs[2].plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    axs[2].set_xlabel('Recall')
    axs[2].set_ylabel('Precision')
    axs[2].set_title('Precision-Recall Curve')
    axs[2].legend(loc="best")

    # Display the plot
    plt.tight_layout()
    plt.show()

    f_scores = (2 * precision * recall) / (np.maximum(precision + recall, 1e-8))
    optimal_idx = np.argmax(f_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]


    #### Additional plotting for a specific threshold of the PR curve
    # Plotting the Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {auc(recall, precision):.2f})')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, edgecolor='k', zorder=5)
    plt.text(recall[optimal_idx], precision[optimal_idx], f'  Threshold={optimal_threshold:.2f}',
             verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold Highlighted')
    plt.legend()
    plt.show()

    predictions = [1 if x >= optimal_threshold else 0 for x in scores]
    conf_matrix = confusion_matrix(labels, predictions)

    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix at Optimal Threshold: {:.2f}'.format(optimal_threshold))
    plt.xticks([0.5, 1.5], ['True Edge', 'False Edge'], rotation=0)
    plt.yticks([0.5, 1.5], ['True Edge', 'False Edge'], rotation=0)
    plt.show()



def plot_false_true_edges_percentages(args, edges_within_communities, edges_between_communities):
    false_edges = set(args.false_edge_ids)  # Assuming this is already in the form of tuples (node1, node2)

    false_within = sum(edge in false_edges for edge in edges_within_communities)
    true_within = len(edges_within_communities) - false_within
    false_between = sum(edge in false_edges for edge in edges_between_communities)
    true_between = len(edges_between_communities) - false_between

    total_false_edges = len(false_edges)

    # Calculate the percentages for the main plot
    total_within = len(edges_within_communities)
    total_between = len(edges_between_communities)
    percentages_within = [false_within / total_within * 100, true_within / total_within * 100]
    percentages_between = [false_between / total_between * 100, true_between / total_between * 100]

    # Calculate the percentages of false edges that are 'within' and 'between'
    false_within_percentage = (false_within / total_false_edges) * 100 if total_false_edges > 0 else 0
    false_between_percentage = (false_between / total_false_edges) * 100 if total_false_edges > 0 else 0

    # Setup for subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Main plot with stacked bars
    categories = ['Within Communities', 'Between Communities']
    axs[0].bar(categories, [percentages_within[0], percentages_between[0]], label='False Edges', color='r')
    axs[0].bar(categories, [percentages_within[1], percentages_between[1]],
               bottom=[percentages_within[0], percentages_between[0]], label='True Edges', color='g')
    axs[0].set_ylabel('Percentage')
    axs[0].set_title('Percentage of False and True Edges')
    axs[0].legend()

    # Subplot for percentage of false edges that are within/between
    axs[1].bar(['False Edges Distribution'], [false_within_percentage], label='Within', color='blue')
    axs[1].bar(['False Edges Distribution'], [false_between_percentage], bottom=[false_within_percentage],
               label='Between', color='orange')
    axs[1].set_ylabel('Percentage')
    axs[1].set_title('Distribution of False Edges')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_edge_categories_and_distances(edges_within_communities, edges_between_communities,
                                                         false_edges, original_positions_df):
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    positions_dict = original_positions_df.set_index('node_ID').to_dict('index')
    categories = ['False Within', 'True Within', 'False Between', 'True Between']
    distances = {category: [] for category in categories}

    for edge in edges_within_communities.union(edges_between_communities):
        node1, node2 = edge
        pos1, pos2 = positions_dict[node1], positions_dict[node2]
        distance = euclidean_distance(pos1['x'], pos1['y'], pos2['x'], pos2['y'])
        if edge in edges_within_communities:
            if edge in false_edges:
                distances['False Within'].append(distance)
            else:
                distances['True Within'].append(distance)
        else:
            if edge in false_edges:
                distances['False Between'].append(distance)
            else:
                distances['True Between'].append(distance)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    for i, category in enumerate(categories):
        axs[i].hist(distances[category], bins=20, alpha=0.7, label=category)
        axs[i].set_title(category)
        axs[i].set_xlabel('Distance')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()

    plt.tight_layout()
    plt.show()