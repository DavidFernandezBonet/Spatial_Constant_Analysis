import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import CurveFitting
import seaborn as sns
import scienceplots

plt.style.use(['science', 'nature'])


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

    print("hello", args.args_title)
    plot_folder = args.directory_map['plots_clustering_coefficient']
    plt.savefig(f"{plot_folder}/clust_coef_{args.args_title}.png")



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
    plt.savefig(f"{plot_folder}/degree_dist_{args.args_title}")
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
    plt.savefig(f"{plot_folder}/plots_shortest_path_distribution_{args.args_title}")



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
    sns.violinplot(x='classification', y='S_general', data=spatial_constant_variation_results_df)
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
    fig = px.violin(spatial_constant_variation_results_df, x='classification', y='S_general', color='color_group',
                    box=False, points='all', hover_data=spatial_constant_variation_results_df.columns)

    # Add predicted medians as scatter points
    for i, group in enumerate(groups):
        fig.add_scatter(x=[group], y=[predicted_medians[i]], mode='markers', marker=dict(color='green'))

    # Update the layout
    fig.update_layout(
        title='Interactive Violin Plot of S_general for Different Groups',
        xaxis_title='Group',
        yaxis_title='S_general'
    )

    # Show the plot
    fig.write_html("violin_plot_S_variation" + '.html')
    fig.show()


def plot_original_image(args):

    edge_list_folder = args.directory_map["edge_lists"]
    original_position_folder = args.directory_map["original_positions"]
    edges_df = pd.read_csv(f"{edge_list_folder}/edge_list_{args.original_title}.csv")
    positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")

    # Create a plot
    fig = plt.figure(figsize=(10, 8))
    if args.dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions_df['x'], positions_df['y'], positions_df['z'], facecolors='none', edgecolors='b')
    elif args.dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(positions_df['x'], positions_df['y'], facecolors='none', edgecolors='b')

    # Draw edges
    for _, row in edges_df.iterrows():
        source = positions_df[positions_df['node_ID'] == row['source']].iloc[0]
        target = positions_df[positions_df['node_ID'] == row['target']].iloc[0]
        edge_color = 'red' if (row['source'], row['target']) in args.false_edge_ids or (
        row['target'], row['source']) in args.false_edge_ids else 'k'

        edge_alpha = 1 if (row['source'], row['target']) in args.false_edge_ids or (
        row['target'], row['source']) in args.false_edge_ids else 0.5

        if args.dim == 3:
            ax.plot([source['x'], target['x']], [source['y'], target['y']], [source['z'], target['z']],
                    edge_color, linewidth=0.5, alpha=edge_alpha)
        else:
            ax.plot([source['x'], target['x']], [source['y'], target['y']],
                    edge_color, linewidth=0.5, alpha=edge_alpha)

    plot_folder = args.directory_map["plots_original_image"]
    plt.savefig(f"{plot_folder}/original_image_{args.args_title}")

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


import matplotlib.pyplot as plt
import numpy as np

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
    plt.title('Mean Spatial Constant vs. Subgraph Size')
    plt.legend()

    plot_folder = f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}.png")



def plot_spatial_constant_against_subgraph_size_with_false_edges(args, dataframes, false_edge_list, mst_case_df=None):
    plt.figure(figsize=(10, 6))

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
    plt.title('Mean Spatial Constant vs. Subgraph Size')
    plt.legend()

    plot_folder = args.directory_map['plots_spatial_constant_subgraph_sampling']
    plt.savefig(f"{plot_folder}/mean_s_general_vs_intended_size_{args.args_title}_false_edge_version.png")