import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from structure_and_args import GraphArgs
import seaborn as sns


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

args = GraphArgs()
input_folder = args.directory_map['weinstein']
dataframe = pd.read_csv(f"{input_folder}/weinstein_data_corrected_february_edge_list_and_original_distance_data.csv")
plot_folder = args.directory_map['plots_weight_distribution']
plot_weight_distance_with_violin(dataframe, plot_folder)