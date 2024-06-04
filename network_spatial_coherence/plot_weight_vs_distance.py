import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from structure_and_args import GraphArgs
import seaborn as sns
import os
import networkx as nx
import matplotlib.colors as mcolors

import scienceplots


plt.style.use(['science', 'no-latex', 'nature'])
base_figsize = (6, 4.5)  # Width, Height in inches
base_fontsize = 18
plt.rcParams.update({
    'figure.figsize': base_figsize,  # Set the default figure size
    'figure.dpi': 300,  # Set the figure DPI for high-resolution images
    'savefig.dpi': 300,  # DPI for saved figures
    'font.size': base_fontsize,  # Base font size
    'axes.labelsize': base_fontsize ,  # Font size for axis labels
    'axes.titlesize': base_fontsize + 2,  # Font size for subplot titles
    'xtick.labelsize': base_fontsize,  # Font size for X-axis tick labels
    'ytick.labelsize': base_fontsize,  # Font size for Y-axis tick labels
    'legend.fontsize': base_fontsize - 6,  # Font size for legends
    'lines.linewidth': 2,  # Line width for plot lines
    'lines.markersize': 6,  # Marker size for plot markers
    'figure.autolayout': True,  # Automatically adjust subplot params to fit the figure
    'text.usetex': False,  # Use LaTeX for text rendering (set to True if LaTeX is installed)
})

def plot_weight_distance_with_violin(dataframe, plot_folder):
    """
    Plot the log-transformed weights vs. distances with density coloring and
    a violin plot for the distribution of distances for the first 10 unique weights.

    Parameters:
        dataframe (pd.DataFrame): Input data containing 'weight' and 'distance' columns.
        plot_folder (str): Path to the folder where the plot will be saved.
        args_title (str): Title suffix for the plot file.

    """
    dataframe = dataframe.sample(frac=0.1, random_state=42)  # random_state for reproducibility
    # Clean data: Remove or handle any inf or NaN values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()
    # Ensure the plot_folder directory exists
    import os
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Logarithmic transformation with a small shift to handle zero weights and distances
    # log_weights = np.log10(dataframe['weight'] + 1)
    # log_distances = np.log10(dataframe['distance'] + 1)

    log_weights = np.log10(dataframe['weight'].clip(lower=0) + 1)
    log_distances = np.log10(dataframe['distance'].clip(lower=0) + 1)

    # Calculate the point density
    xy = np.vstack([log_distances, log_weights])
    z = gaussian_kde(xy)(xy)

    # Create the subplot figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # Scatter plot with density coloring
    sc = ax1.scatter(log_distances, log_weights, c=z, s=50, edgecolor='none', cmap='viridis')
    fig.colorbar(sc, ax=ax1, label='Density')
    ax1.set_title('Log-Transformed Weights vs. Distances with Density Coloring')
    ax1.set_xlabel('Log of Euclidean Distance')
    ax1.set_ylabel('Log of Weight')

    # Violin plot for the first 10 unique weights
    # Identify the first 10 unique weights
    first_10_weights = dataframe['weight'].drop_duplicates().sort_values()[:10]
    filtered_df = dataframe[dataframe['weight'].isin(first_10_weights)]

    # Create violin plot
    sns.violinplot(x='weight', y='distance', data=filtered_df, ax=ax2, scale='width')
    ax2.set_title('Distance Distribution for First 10 Weights')
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Distance')
    ax2.set_ylim(-0.5, 2)

    # Calculate the median of unique weights
    median_weight = dataframe['weight'].drop_duplicates().median()
    sorted_weights = dataframe['weight'].drop_duplicates().sort_values()
    median_index = sorted_weights.searchsorted(median_weight)
    start_index = max(median_index - 5, 0)
    end_index = min(median_index + 5,
                    len(sorted_weights))
    if end_index - start_index < 10:
        start_index = max(end_index - 10, 0)  # Adjust start_index to gather 10 weights if possible
    middle_weights = sorted_weights[start_index:end_index]
    filtered_df_middle = dataframe[dataframe['weight'].isin(middle_weights)]


    sns.violinplot(x='weight', y='distance', data=filtered_df_middle, ax=ax3, scale='width')
    ax3.set_title('Distance Distribution for Middle 10 Weights')
    ax3.set_xlabel('Weight')
    ax3.set_ylabel('Distance')
    ax3.set_ylim(-0.5, 2)  # Adjust as needed


    # Last 10 unique weights violin plot
    last_10_weights = dataframe['weight'].drop_duplicates().sort_values().tail(10)
    filtered_df_last = dataframe[dataframe['weight'].isin(last_10_weights)]
    sns.violinplot(x='weight', y='distance', data=filtered_df_last, ax=ax4, scale='width')
    ax4.set_title('Distance Distribution for Last 10 Weights')
    ax4.set_xlabel('Weight')
    ax4.set_ylabel('Distance')
    ax4.set_ylim(-0.5, 2)  # Adjust as needed

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"weight_vs_distance_weinstein.png"))
    plt.show()  # Optionally display the plot as well



def categorize_weights(weight):
    if weight > 4:
        return '>4'
    else:
        return str(weight)

def plot_weight_distance_with_violin_v2(dataframe, plot_folder):
    """
    Plot the violin plot for the distribution of distances for categorized weights.

    Parameters:
        dataframe (pd.DataFrame): Input data containing 'weight' and 'distance' columns.
        plot_folder (str): Path to the folder where the plot will be saved.
    """
    # Clean data: Remove or handle any inf or NaN values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure the plot_folder directory exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Categorize weights
    dataframe['categorized_weight'] = dataframe['weight'].apply(categorize_weights)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='categorized_weight', y='distance', data=dataframe, scale='width', order=['1', '2', '3', '4', '>4'])
    plt.title('Distance Distribution for Categorized Weights')
    plt.xlabel('Weight')
    plt.ylabel('Distance')
    plt.ylim(-0.5, 2)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "categorized_weight_vs_distance_violin.png"))
    plt.show()  # Optionally display the plot as well


def plot_distance_histogram_with_weight_stacks(dataframe, plot_folder, num_bins=20, num_colors=5):
    """
    Plots a stacked barplot of 'distance' with each bar stacked by 'weight' categories.

    Parameters:
        dataframe (pd.DataFrame): Input data containing 'weight' and 'distance' columns.
        plot_folder (str): Path to the folder where the plot will be saved.
        num_bins (int): Number of bins for the distance histogram.
        num_colors (int): Number of colors to use for distinct weights.
    """
    # Ensure the plot_folder directory exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Clean data: Remove or handle any inf or NaN values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # Define bin edges
    bin_edges = np.linspace(np.log10(dataframe['distance'].min()), np.log10(dataframe['distance'].max()), num_bins+1)
    bin_centers = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)
    bin_widths = np.diff(10 ** bin_edges)  # Calculate the width of each bin

    # Determine unique weights and corresponding colors
    unique_weights = sorted(dataframe['weight'].unique())
    num_unique_weights = min(len(unique_weights), num_colors)
    colors = plt.cm.get_cmap('viridis', num_unique_weights)(np.linspace(0, 1, num_unique_weights))
    # Assign the last color to weights beyond num_colors
    colors_extended = list(colors) + [colors[-1]] * max(0, len(unique_weights) - num_colors)

    # Create histogram data
    hist_data = []
    for weight in unique_weights:
        hist, _ = np.histogram(dataframe[dataframe['weight'] == weight]['distance'], bins=10 ** bin_edges)
        hist_data.append(hist)

    # Create stacked barplot
    plt.figure(figsize=(12, 6))
    for i, weight in enumerate(unique_weights):
        color_index = min(i, num_colors - 1)  # Assign colors up to num_colors, then the last color for the rest
        plt.bar(bin_centers, hist_data[i], width=bin_widths, bottom=np.sum(hist_data[:i], axis=0), label=f'Weight: {weight}', color=colors_extended[color_index])

    # Plot customization
    plt.title('Stacked Barplot of Distance with Weight Categories (Log-Log Scale)')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Weight Categories')

    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "distance_stacked_barplot_by_weight.png"))
    plt.show()


def plot_distance_histogram_with_linear_bins(dataframe, plot_folder, num_bins=200, num_colors=5):
    """
    Plots a stacked barplot of 'distance' with each bar stacked by 'weight' categories using linear bin widths.

    Parameters:
        dataframe (pd.DataFrame): Input data containing 'weight' and 'distance' columns.
        plot_folder (str): Path to the folder where the plot will be saved.
        num_bins (int): Number of bins for the distance histogram.
        num_colors (int): Number of colors to use for distinct weights.
    """
    # Ensure the plot_folder directory exists
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Clean data: Remove or handle any inf or NaN values
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    # Define bin edges
    min_distance = dataframe['distance'].min()
    max_distance = dataframe['distance'].max()
    bin_edges = np.linspace(min_distance, max_distance, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)  # Calculate the width of each bin

    # Determine unique weights and corresponding colors
    unique_weights = sorted(dataframe['weight'].unique())
    num_unique_weights = min(len(unique_weights), num_colors)
    colors = plt.cm.get_cmap('viridis', num_unique_weights)(np.linspace(0, 1, num_unique_weights))
    # Assign the last color to weights beyond num_colors
    colors_extended = list(colors) + [colors[-1]] * max(0, len(unique_weights) - num_colors)

    # Create histogram data
    hist_data = []
    for weight in unique_weights:
        hist, _ = np.histogram(dataframe[dataframe['weight'] == weight]['distance'], bins=bin_edges)
        hist_data.append(hist)

    # Create stacked barplot
    plt.figure(figsize=(12, 6))
    labels_added = set()  # Track which labels have been added to avoid duplicates in the legend
    group_label_added = False  # Flag to check if group label has been added
    for i, weight in enumerate(unique_weights):
        color_index = min(i, num_colors - 1)
        if weight >= num_colors:
            if not group_label_added:
                label_text = f'Weight: > {num_colors - 1}'
                group_label_added = True  # Set the flag to True after adding group label
            else:
                label_text = None  # Avoid adding label to the bar
        else:
            label_text = f'Weight: {weight}'
        plt.bar(bin_centers, hist_data[i], width=bin_widths, bottom=np.sum(hist_data[:i], axis=0), label=label_text,
                color=colors_extended[color_index])

    plt.xlabel('Reconstructed Distance')
    plt.ylabel('Frequency')
    plt.legend()

    # Save and show plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, "distance_stacked_barplot_by_weight_linear.png"))
    plt.show()

def filter_dataframe_by_quantile_distance(dataframe, lower_quantile=0, upper_quantile=0.5):
    """
    Filters a DataFrame by 'distance' between specified quantiles.

    Parameters:
        dataframe (pd.DataFrame): Input data containing a 'distance' column.
        lower_quantile (float): Lower quantile to use as the minimum distance filter.
        upper_quantile (float): Upper quantile to use as the maximum distance filter.

    Returns:
        pd.DataFrame: A DataFrame filtered by distance within the specified quantiles.
    """
    # Validate quantile values
    if not 0 <= lower_quantile <= 1 or not 0 <= upper_quantile <= 1:
        raise ValueError("Quantiles must be between 0 and 1")
    if lower_quantile > upper_quantile:
        raise ValueError("Lower quantile must be less than or equal to the upper quantile")

    # Calculate quantile values for distance
    quantile_values = dataframe['distance'].quantile([lower_quantile, upper_quantile])
    min_distance = quantile_values.loc[lower_quantile]
    max_distance = quantile_values.loc[upper_quantile]

    # Filter DataFrame by the quantile range
    filtered_df = dataframe[(dataframe['distance'] >= min_distance) & (dataframe['distance'] <= max_distance)]

    return filtered_df

def plot_network(position_dataframe, edge_dataframe, colorcode_dataframe=None, save_path=None, draw_edges=True):
    """
    Plots a network graph using node positions and edge connections using only matplotlib, including only nodes that appear in the edge list.

    Parameters:
        position_dataframe (pd.DataFrame): DataFrame containing 'x', 'y', and 'node_ID' for node positions.
        edge_dataframe (pd.DataFrame): DataFrame containing 'source', 'target', 'weight', and 'distance' for edges.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Filter nodes that are either source or target in the edge list
    involved_nodes = pd.concat([edge_dataframe['source'], edge_dataframe['target']]).unique()
    filtered_position_df = position_dataframe[position_dataframe['node_ID'].isin(involved_nodes)]

    color_map = {-1: 'grey', 0: 'grey', 1: 'green', 2: 'red'}
    default_color = 'blue'  # Default color if no colorcode provided or a node's color code is not in color_map



    if colorcode_dataframe is not None:
        filtered_position_df = filtered_position_df.merge(colorcode_dataframe, on='node_ID', how='left')
        filtered_position_df['color'] = filtered_position_df['color'].map(color_map).fillna(default_color)
    else:
        filtered_position_df['color'] = default_color
    positions = {row['node_ID']: (row['x'], row['y'], row['color']) for index, row in filtered_position_df.iterrows()}

    print(positions)
    if draw_edges:
        # Draw the edges
        for index, row in edge_dataframe.iterrows():
            print(index)
            source_position = positions.get(int(row['source']))[:-1]
            target_position = positions.get(int(row['target']))[:-1]
            if source_position and target_position:  # Check if both source and target positions exist
                # Optionally, use weight or distance to modify the line properties
                line_width = 0.5
                alpha = 0.2
                ax.plot([source_position[0], target_position[0]], [source_position[1], target_position[1]],
                        'k-', lw=line_width, alpha=alpha)  # 'k-' is for black line, modify as needed


    x_coords = [pos[0] for pos in positions.values()]
    y_coords = [pos[1] for pos in positions.values()]
    colors = [pos[2] for pos in positions.values()]  # pos[2] is the color

    # Draw the nodes using scatter
    ax.scatter(x_coords, y_coords, s=0.1, color=colors)  # Plot all nodes at once

    # Set labels and grid

    # Set labels and grid
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')


    # Optionally set limits if your points are too dispersed
    ax.set_xlim([filtered_position_df['x'].min() - 1, filtered_position_df['x'].max() + 1])
    ax.set_ylim([filtered_position_df['y'].min() - 1, filtered_position_df['y'].max() + 1])

    # Show plot

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def get_maximally_separated_colors(num_colors):
    hues = np.linspace(0, 1, num_colors + 1)[:-1]  # Avoid repeating the first color
    colors = [mcolors.hsv_to_rgb([h, 0.7, 0.7]) for h in hues]  # S and L fixed for aesthetic colors
    # Convert to HEX format for broader compatibility
    colors = [mcolors.to_hex(color) for color in colors]
    return colors

def plot_network_subgraphs(position_dataframe, edge_dataframe, save_path=None):
    """
    Plots a network graph using node positions and edge connections, detects subgraphs and colors them differently.

    Parameters:
        position_dataframe (pd.DataFrame): DataFrame containing 'x', 'y', and 'node_ID' for node positions.
        edge_dataframe (pd.DataFrame): DataFrame containing 'source', 'target' for edges.
        save_path (str): Optional path to save the plot image.
    """
    # Create graph
    G = nx.Graph()

    # Add nodes and positions
    positions = position_dataframe.set_index('node_ID')[['x', 'y']].to_dict('index')
    for node, pos in positions.items():
        G.add_node(node, pos=(pos['x'], pos['y']))

    # Add edges
    edges = edge_dataframe[['source', 'target']].values
    G.add_edges_from(edges)

    subgraphs = list(nx.connected_components(G))
    subgraphs = sorted(subgraphs, key=len, reverse=True)  # Sort by size, largest first

    big_subgraphs = 5
    # Select a colormap and get colors for the five largest subgraphs

    colors = get_maximally_separated_colors(big_subgraphs+1)

    # Initialize node colors, default color for those not in top 5
    node_colors = {node: 'grey' for component in subgraphs[big_subgraphs:] for node in component}  # Assign grey to smaller subgraphs

    # Assign colors to only the top 5 largest subgraphs
    for i, component in enumerate(subgraphs[:big_subgraphs]):
        color = colors[i]  # Get color from colormap
        size = len(component)
        print("size subgraph", size)

        ### Store the subgraphs in CSV
        subgraph_nodes = list(component)
        subgraph_edges = edge_dataframe[edge_dataframe['source'].isin(subgraph_nodes) & edge_dataframe['target'].isin(subgraph_nodes)]

        # Save dataframe as CSV
        subgraph_edges.to_csv(f'{save_path}_{size}_subgraph_{i+1}.csv', index=False)

        for node in component:
            node_colors[node] = color  # Assign color

        # Create a subgraph with surrounding nodes
        subgraph = G.subgraph(component)
        surrounding_nodes = set(component)
        for node in component:
            surrounding_nodes.update(G.neighbors(node))

        # Draw the subgraph with surrounding nodes
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.get_node_attributes(subgraph, 'pos')
        nx.draw_networkx_nodes(subgraph, pos, node_color=[node_colors[node] for node in subgraph.nodes()], ax=ax, node_size=0.5)
        nx.draw_networkx_edges(subgraph, pos, ax=ax, alpha=0.3)
        # ax.set_xlim(min(pos[node][0] for node in surrounding_nodes) - 10, max(pos[node][0] for node in surrounding_nodes) + 10)
        # ax.set_ylim(min(pos[node][1] for node in surrounding_nodes) - 10, max(pos[node][1] for node in surrounding_nodes) + 10)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.axis('off')  # Turn off the axis

        if save_path:
            plt.savefig(f'{save_path}_subgraph_{i+1}.png')
        plt.close(fig)

    # Draw the entire network
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors[node] for node in G.nodes()], ax=ax, node_size=0.1)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)

    # Set labels and grid
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    plt.axis('off')  # Turn off the axis

    # Optionally save to file
    if save_path:
        plt.savefig(save_path)
    plt.show()
def select_portion_of_reconstructed_image(edge_dataframe, position_dataframe, x_range, y_range):
    """
    Filters the position and edge dataframes to include only nodes and edges within the specified x and y ranges.

    Parameters:
    - edge_dataframe (pd.DataFrame): DataFrame containing 'source' and 'target' columns for edges.
    - position_dataframe (pd.DataFrame): DataFrame containing 'node_ID', 'x', and 'y' columns.
    - x_range (tuple): A tuple (x_min, x_max) defining the range of x coordinates.
    - y_range (tuple): A tuple (y_min, y_max) defining the range of y coordinates.

    Returns:
    - filtered_position_dataframe (pd.DataFrame): The filtered position DataFrame.
    - filtered_edge_dataframe (pd.DataFrame): The filtered edge DataFrame.
    """

    # Filter position_dataframe for the given ranges
    x_min, x_max = x_range
    y_min, y_max = y_range
    filtered_position_dataframe = position_dataframe[
        (position_dataframe['x'] >= x_min) & (position_dataframe['x'] <= x_max) &
        (position_dataframe['y'] >= y_min) & (position_dataframe['y'] <= y_max)
    ]

    # Filter edge_dataframe to include only edges with nodes in the filtered position_dataframe
    filtered_node_ids = set(filtered_position_dataframe['node_ID'])
    filtered_edge_dataframe = edge_dataframe[
        (edge_dataframe['source'].isin(filtered_node_ids)) &
        (edge_dataframe['target'].isin(filtered_node_ids))
    ]

    return filtered_position_dataframe, filtered_edge_dataframe

args = GraphArgs()
input_folder = args.directory_map['weinstein']
edge_dataframe = pd.read_csv(f"{input_folder}/weinstein_data_corrected_february_edge_list_and_original_distance_data.csv")
plot_folder = args.directory_map['plots_weight_distribution']
color_dataframe = pd.read_csv(f'{input_folder}/weinstein_colorcode_february_corrected.csv')

# Plot the original image
original_pos_folder = args.directory_map['original_positions']
position_dataframe_name = f'{original_pos_folder}/positions_weinstein_data_corrected_february.csv'
position_dataframe = pd.read_csv(position_dataframe_name)

## Plots the best version for violin plots, to see the relation between weights and reconstructed distance
# plot_weight_distance_with_violin_v2(dataframe, plot_folder)
# plot_distance_histogram_with_weight_stacks(dataframe, plot_folder)
# plot_distance_histogram_with_linear_bins(dataframe, plot_folder)

## Selects only the central part of the reconstructed image
# x_range = (-4, 1.5)
# y_range = (-2, 3.5)
x_range = (-6, 6)
y_range = (-6, 6)
position_dataframe, edge_dataframe = select_portion_of_reconstructed_image(edge_dataframe=edge_dataframe,
                                                                           position_dataframe=position_dataframe,
                                                                           x_range=x_range, y_range=y_range)

# ### Filter by distance quantile
quantile = 0.2
# edge_dataframe = filter_dataframe_by_quantile_distance(edge_dataframe, lower_quantile=0, upper_quantile=quantile)
# # edge_dataframe.to_csv(f"{plot_folder}/weinstein_data_corrected_february_edge_list_and_original_distance_selected_square_region_quantile_{quantile}.csv", index=False)
# # plot_distance_histogram_with_linear_bins(dataframe, plot_folder)
#
# # ## Store different edge lists by quantile (the quantile is the top % closest distance in edges)
# # quantile_list = [0.8,0.6, 0.4, 0.2, 0.1, 0.05]
# # for quantile in quantile_list:
# #     dataframe = filter_dataframe_by_quantile_distance(edge_dataframe, lower_quantile=0, upper_quantile=quantile)
# #     dataframe.to_csv(f"{plot_folder}/weinstein_data_corrected_february_edge_list_and_original_distance_selected_square_region_quantile_{quantile}.csv", index=False)
#
# ## Plots the network using the "edge dataframe" --> dataframe, and in particular for weinstein's case
#
# # plot_network(position_dataframe=position_dataframe, edge_dataframe=edge_dataframe, colorcode_dataframe=color_dataframe,
# #              save_path=f'{plot_folder}/weinstein_data_corrected_february_original_image_quantile={quantile}.png', draw_edges=False)
# plot_network_subgraphs(position_dataframe=position_dataframe, edge_dataframe=edge_dataframe,
#              save_path=f'{plot_folder}/weinstein_data_corrected_february_original_image_subgraphs_quantile={quantile}.png')


### Store multiple quantiles with their subgraphs
quantile_list = [0.15]
for quantile in quantile_list:
    edge_dataframe_copy = edge_dataframe.copy()
    edge_dataframe_copy = filter_dataframe_by_quantile_distance(edge_dataframe_copy, lower_quantile=0, upper_quantile=quantile)
    plot_network_subgraphs(position_dataframe=position_dataframe, edge_dataframe=edge_dataframe_copy,
                 save_path=f'{plot_folder}/weinstein_data_corrected_february_original_image_subgraphs_quantile={quantile}.png')