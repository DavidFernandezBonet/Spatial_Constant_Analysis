from pixelator import read
from structure_and_args import GraphArgs
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data(file_path):
    return read(file_path)

def create_edge_rank_df(data):
    edge_rank_df = data.adata.obs[['edges']].copy()
    edge_rank_df["rank"] = edge_rank_df['edges'].rank(ascending=False, method="first")
    return edge_rank_df

def plot_edge_rank(df, plot_path, threshold):
    plot = sns.relplot(data=df, x="rank", y="edges", aspect=1.6)
    plot.set(xscale="log", yscale="log")
    plot.set_xlabels("Component rank (by number of edges)")
    plot.set_ylabels("Number of edges")
    plot.fig.axes[0].axhline(threshold, linestyle="--")
    plot.savefig(f'{plot_path}_{title_sample}.png')

def filter_edge_data(data, threshold):
    return data.adata[data.adata.obs["edges"] >= threshold]

def plot_metrics(data, plot_path):
    variables = ["edges", "umi_per_upia", "mean_reads"]
    data = data.obs[["edges", "umi_per_upia", "mean_reads"]]
    melted_df = data[variables].melt()

    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))

    for i, variable in enumerate(variables):
        sns.violinplot(ax=axes[i], y=data[variable])
        axes[i].set_title(f"Violin Plot of {variable}")

    plt.tight_layout()
    plt.savefig(f'{plot_path}/metrics_{title_sample}.png')


def tau_outlier_removal(data, plot_path, tau_column="tau", umi_column="umi_per_upia", tau_type_column="tau_type"):
    """
    Perform tau outlier removal and generate a plot.

    :param data: DataFrame containing the tau, umi_per_upia, and tau_type data.
    :param tau_column: Name of the column containing tau values.
    :param umi_column: Name of the column containing umi_per_upia values.
    :param tau_type_column: Name of the column containing tau_type.
    :return: Filtered DataFrame with only 'normal' tau_type.
    """
    # Plotting

    tau_plot = sns.relplot(
        data=data.obs,
        x=tau_column,
        y=umi_column,
        hue=tau_type_column
    )
    tau_plot.set_xlabels("Marker specificity (Tau)")
    tau_plot.set_ylabels("Pixel content (UMI/UPIA)")
    plt.savefig(f'{plot_path}/tau_outlier_removal_{title_sample}.png')

    # Filtering
    filtered_data = data[data.obs[tau_type_column] == "normal"]
    return filtered_data




def plot_control_markers_ann_data_and_filter(data, plot_folder, threshold_control_marker_count=10):
    """
    Create violin plots for control markers in an AnnData object.

    :param data: AnnData object containing the data.
    :param control_markers: List of control markers to plot.
    """
    control_markers = ["ACTB", "mIgG1", "mIgG2a", "mIgG2b"]
    # Find indices of control markers
    marker_indices = [data.var.index.get_loc(marker) for marker in control_markers if marker in data.var.index]

    # Extract expression data for control markers
    control_data = data.X[:, marker_indices]
    # Convert to DataFrame
    control_df = pd.DataFrame(control_data, columns=control_markers)

    # Melting the DataFrame for plotting
    melted_control_df = control_df.melt()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="variable", y="value", data=melted_control_df)
    plt.xticks(rotation=45)
    plt.title("Violin Plots of Control Markers")
    plt.savefig(f'{plot_folder}/control_markers_{title_sample}.png')

    ### Filter any components where any control marker exceeds the threshold
    exceeds_threshold = np.any(control_data > threshold_control_marker_count, axis=1)
    filtered_data = data[~exceeds_threshold, :]
    return filtered_data


def filter_pxl_data(pxl_data, threshold_edges=5000, threshold_control_markers=10):
    control_markers = ["ACTB", "mIgG1", "mIgG2a", "mIgG2b"]

    # Find indices of control markers
    marker_indices = [pxl_data.adata.var.index.get_loc(marker) for marker in control_markers]
    # Extract expression data for control markers and check if below threshold
    below_threshold_mask = (pxl_data.adata.X[:, marker_indices] < threshold_control_markers).all(axis=1)
    # Combine with your existing conditions
    combined_mask = below_threshold_mask & \
                    (pxl_data.adata.obs["edges"] >= threshold_edges) & \
                    (pxl_data.adata.obs["tau_type"] == "normal")
    # Filter components
    components_to_keep = pxl_data.adata.obs[combined_mask].index
    print(f"Kept {len(components_to_keep)} out of {len(pxl_data.adata)} components (cells)")
    pxl_data_filtered = pxl_data.filter(components=components_to_keep)
    return pxl_data_filtered


def plots_analysis_pxl_data(pxl_data, plot_folder):
    threshold = 5000  # Edge count threshold (cells should have at least this number of edges)
    control_markers = ["ACTB", "mIgG1", "mIgG2a", "mIgG2b"]  # negative control markers (should be there only in a low amount)

    edge_rank_df = create_edge_rank_df(pxl_data)
    plot_edge_rank(edge_rank_df, f'{plot_folder}/edgerank_plot.png', threshold)
    filtered_data = filter_edge_data(pxl_data, threshold)
    # Plot some metrics
    plot_metrics(filtered_data, plot_folder)
    # Filter by tau value
    filtered_data = tau_outlier_removal(data=filtered_data, plot_path=plot_folder)
    # Filter by antibody count (negative control markers) that should not be there
    filtered_data = plot_control_markers_ann_data_and_filter(data=filtered_data, plot_folder=plot_folder)
    # Filter out the control markers
    filtered_data = filtered_data[:, ~filtered_data.var.index.isin(control_markers)]

# Load
args = GraphArgs()
title_sample = 'Sample04_Raji_Rituximab_treated'  # no extensions here
data_folder = args.directory_map['pixelgen_data']
data_path = f"{data_folder}/{title_sample}.dataset.pxl"
plot_folder = args.directory_map['plots_pixelgen']
pxl_data = read_data(data_path)
pxl_data.edgelist.to_csv(f'{data_folder}/unfiltered_edge_list_{title_sample}.csv')
# Generate analysis plots
plots_analysis_pxl_data(pxl_data=pxl_data, plot_folder=plot_folder)
# Filter by: number edges, normal tau value, low quantity of control markers
filtered_pxl_data = filter_pxl_data(pxl_data, threshold_control_markers=10, threshold_edges=5000)
# Write the filtered edge list

filtered_pxl_data.edgelist.to_csv(f'{data_folder}/filtered_edge_list_{title_sample}.csv')






