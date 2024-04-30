import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from structure_and_args import GraphArgs
from datetime import datetime
import numpy as np
import random
import scienceplots
from sklearn.cluster import KMeans

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


np.random.seed(42)
random.seed(42)
def add_properties_from_filenames(folder_path):
    # TODO: parts not identifying amplitude and width
    # List all CSV files in the specified folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'quantitative_metrics' in f]

    # Loop through each file and process
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        parts = file_name.split('_')
        filtered_index = [i for i, part in enumerate(parts) if 'filtered' in part]

        if filtered_index:
            filtered = True
            index_offset = 1  # Adjusts the index for amplitude and width
        else:
            filtered = False
            index_offset = 0  # No adjustment needed

        print("parts", parts, "filtered_index", filtered_index, "index_offset", index_offset)
        # Extract parameters based on whether 'filtered' is present
        start = 10
        amplitude = parts[start + index_offset]
        width = parts[start+1 + index_offset]
        if filtered:
            false_edge_ratio = float("0."+parts[start+2 + index_offset][1:])  # 'f0' -> '0'
            if 'q' in parts[start + 3 + index_offset]:  # Check if quantile is part of the last known parameter
                quantile = parts[start + 3 + index_offset][1:]  # 'q01' -> '01'
                quantile = quantile[:-4]
                quantile = float("0." + quantile)
            else:
                quantile = 'None'  # Default in case 'quantile' is not part of the filename
        else:
            false_edge_ratio = parts[start+2 + index_offset][1:]
            false_edge_ratio = float("0."+false_edge_ratio[:-4])
            # quantile = 'None'
            quantile = 0

        print("false edge ratio", false_edge_ratio, "parts[start+1 + index_offset]", parts[start+1 + index_offset])




        # Append new rows to the dataframe
        new_rows = [
            {"Property": "Amplitude", "Value": amplitude, "Category": "Parameter"},
            {"Property": "Width", "Value": width, "Category": "Parameter"},
            {"Property": "False Edge Ratio", "Value": false_edge_ratio, "Category": "Parameter"},
            {"Property": "Quantile", "Value": quantile, "Category": "Parameter"},
            {"Property": "Filtered", "Value": filtered, "Category": "Parameter"}
        ]
        new_df = pd.DataFrame(new_rows)

        # Combine with existing dataframe
        updated_df = pd.concat([df, new_df], ignore_index=True)

        # Optionally, save the updated dataframe back to the same file or a new file
        updated_df.to_csv(file_path, index=False)

        print(f"Updated file saved: {file_path}")


def create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters={}):
    """
    Creates a heatmap of the data in the specified folder. The folder must contain several dataframes and 3
    interesting quantities (x, y, z) where z is the color
    """

    c_bar_limits = {
    "network_dim": (0.5, 3.5),
    "gram_total_contribution": (0.3, 1),
    "gram_spectral_gap": (0, 1),
    "gram_last_spectral_gap": (0, 1),
    "largeworldness": (0, 1),
     }

    label_mappings = {
        "network_dim": "Network Dimension",
        "gram_total_contribution": "Eigenvalue Contribution",
        "gram_spectral_gap": "Spectral Gap (mean)",
        "gram_last_spectral_gap": "Spectral Gap",
        "largeworldness": "Large Worldness",
        "false_edges_ratio": "False Edges Ratio",
        "false_edges_count": "False Edges",
        "Quantile": "Filtering Power",
        "False Edge Ratio": "False Edges Ratio",
        "true_edges_deletion_ratio": "Missing Edges Ratio",

    }

    colormap_styles = {
        "network_dim": "Spectral_r",
        "gram_total_contribution": "magma",
        "gram_spectral_gap": "magma",
        "gram_last_spectral_gap": "magma",
    }



    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    combined_data = pd.DataFrame()


    for file in all_files:
        df = pd.read_csv(file)

        for key, value in omit_df_with_parameters.items():
            if key in df['Property'].values:
                if df.loc[df['Property'] == key, 'Value'].iloc[0] == value:
                    continue  # Skip this file if the condition matches

        # Filter rows where Property matches parameter_x or parameter_y, or quantity_to_evaluate
        if 'false_edges_count' in df['Property'].unique():
            filtered_df = df[df['Property'].isin([parameter_x, parameter_y, quantity_to_evaluate, 'false_edges_count'])]
        else:
            filtered_df = df[df['Property'].isin([parameter_x, parameter_y, quantity_to_evaluate])]

        pivoted = filtered_df.pivot_table(index='Property', values='Value', aggfunc='first').T
        combined_data = pd.concat([combined_data, pivoted], ignore_index=True)

        # false_edge_coun
    if 'false_edges_count' in combined_data.columns:
        unique_false_edge_counts = combined_data['false_edges_count'].dropna().unique()
        num_clusters = len(unique_false_edge_counts)
        print(num_clusters)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(combined_data['false_edges_ratio'].values.reshape(-1, 1))
        print(combined_data['false_edges_ratio'])
        combined_data['false_edges_ratio'] = kmeans.cluster_centers_[kmeans.labels_].round(3)
        print(combined_data['false_edges_ratio'])

    # Ensure the DataFrame is complete with all required columns
    if not all(col in combined_data.columns for col in [parameter_x, parameter_y, quantity_to_evaluate]):
        print(combined_data.columns)
        raise ValueError("Some parameters are missing in the data.")

    combined_data = combined_data.astype('float')
    heatmap_data = combined_data.pivot_table(index=parameter_y, columns=parameter_x, values=quantity_to_evaluate,
                                             aggfunc='mean')
    heatmap_data = heatmap_data.astype(float)
    heatmap_data = heatmap_data.sort_index(ascending=False)



    # Check if the quantity to evaluate has predefined limits
    colorbar_min, colorbar_max = c_bar_limits.get(quantity_to_evaluate,
                                            (heatmap_data.min().min(), heatmap_data.max().max()))
    # Plot the heatmap
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(heatmap_data, annot=True, cmap=colormap_styles.get(quantity_to_evaluate), fmt=".2f", cbar_kws={'label': label_mappings.get(quantity_to_evaluate)},
                vmin=colorbar_min, vmax=colorbar_max, linewidths=2, linecolor='white')

    plt.xlabel(label_mappings.get(parameter_x))
    plt.ylabel(label_mappings.get(parameter_y))
    args = GraphArgs()
    plot_folder = args.directory_map['dataframes']

    print(label_mappings.get(parameter_y))
    print(parameter_y)


    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{plot_folder}/heatmap_{quantity_to_evaluate}_by_{parameter_x}_and_{parameter_y}_{current_time}"

    plt.savefig(filename + '.svg')
    plt.show()



# ### Process Johana files adding new properties to the dataframe based on title of the files
# add_properties_from_filenames("/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Fused_nodes_DF/Fused_nodes_Dataframe/")

# Run heatmaps
# #   bipartite  "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240424_160524_false_edges_count_true_edges_deletion_ratio/"
## delaunay "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240424_135604_false_edges_count_true_edges_deletion_ratio/"
folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Fused_nodes_DF/Fused_nodes_Dataframe/"
# "/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Spurious_crosslinks_DF/Spurious_crosslinks_Dataframe/"

# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240424_160524_false_edges_count_true_edges_deletion_ratio/"

parameter_x = 'False Edge Ratio'  # false_edges_count, false_edges_ratio, False Edge Ratio (alex data)
parameter_y = 'Quantile'  # true_edges_deletion_ratio, Quantile (alex data)
quantity_to_evaluate = 'network_dim'  # gram_total_contribution, gram_spectral_gap, gram_last_spectral_gap, network_dim, largeworldness
# omit_df_with_parameters = {'Filtered': False}
omit_df_with_parameters = {}
create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters=omit_df_with_parameters)