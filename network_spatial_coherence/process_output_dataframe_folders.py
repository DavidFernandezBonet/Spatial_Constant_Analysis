import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from structure_and_args import GraphArgs
from datetime import datetime
import numpy as np
import random
import re
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

# Constants
C_BAR_LIMITS = {
    "network_dim": (0.5, 3.5),
    "gram_total_contribution": (0.3, 1),
    "gram_spectral_gap": (0, 1),
    "gram_last_spectral_gap": (0, 1),
    "largeworldness": (0, 1),
}

LABEL_MAPPINGS = {
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
    "intended_av_degree": r"$\langle k \rangle$",
}

COLORMAP_STYLES = {
    "network_dim": "Spectral_r",
    "gram_total_contribution": "magma",
    "gram_spectral_gap": "magma",
    "gram_last_spectral_gap": "magma",
}


def load_data(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters):
    combined_data = pd.DataFrame()
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in all_files:
        df = pd.read_csv(file)
        if should_omit_file(df, omit_df_with_parameters):
            continue

        if parameter_x == "Total_Mode":
            df = append_total_mode(df)

        filtered_df = filter_data(df, parameter_x, parameter_y, quantity_to_evaluate)
        pivoted = pivot_data(filtered_df)
        combined_data = pd.concat([combined_data, pivoted], ignore_index=True)

    return combined_data


def load_data_v2(folder_path, parameter_x, quantity_to_evaluate, parameter_y=None, omit_df_with_parameters={}):
    combined_data = pd.DataFrame()
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in all_files:
        df = pd.read_csv(file)
        if should_omit_file(df, omit_df_with_parameters):
            continue

        if parameter_x == "Total_Mode":
            df = append_total_mode(df)

        # Filter data based on the provided parameters and quantity to evaluate
        if parameter_y:
            filtered_df = filter_data(df, parameter_x, parameter_y, quantity_to_evaluate)
            pivoted = pivot_data(filtered_df, index=parameter_y, columns=parameter_x, values=quantity_to_evaluate)
        else:
            # If parameter_y is not provided, pivot differently or skip pivoting
            print(file)
            print(df)
            filtered_df = filter_data(df, parameter_x, None, quantity_to_evaluate)
            filtered_df = filtered_df.drop_duplicates(subset=['Property'], keep='first')
            print(filtered_df)
            pivoted = pivot_data(filtered_df)

        combined_data = pd.concat([combined_data, pivoted], ignore_index=True)

    return combined_data

def load_data_1Dplot(folder_path, quantity_to_evaluate):
    combined_data = pd.DataFrame()
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in all_files:
        df = pd.read_csv(file)
        filtered_df = filter_data(df, "edge_list_title", None, quantity_to_evaluate)
        filtered_df = filtered_df.drop_duplicates(subset=['Property'], keep='first')
        pivoted_data = pivot_data(filtered_df)
        # property_data = filtered_df[filtered_df['Property'] == quantity_to_evaluate]
        # pivoted_data = property_data[['Value']].rename(columns={'Value': quantity_to_evaluate}).reset_index(drop=True)
        combined_data = pd.concat([combined_data, pivoted_data], ignore_index=True)

    return combined_data

def should_omit_file(df, omit_df_with_parameters):
    for key, value in omit_df_with_parameters.items():
        if key in df['Property'].values and df.loc[df['Property'] == key, 'Value'].iloc[0] == value:
            return True
    return False


def append_total_mode(df):
    modes = df[df['Property'].isin(['dim', 'proximity_mode', 'bipartiteness'])]
    modes = modes.set_index('Property')['Value'].to_dict()
    total_mode = f"{modes.get('proximity_mode', '')}_{modes.get('dim', '')}_{modes.get('bipartiteness', '')}"
    total_mode_row = pd.DataFrame({'Property': ['Total_Mode'], 'Value': [total_mode], 'Category': ['Parameter']})
    return pd.concat([df, total_mode_row], ignore_index=True)


def filter_data(df, parameter_x, parameter_y, quantity_to_evaluate):
    properties = [parameter_y, quantity_to_evaluate, parameter_x]
    filtered_df = df[df['Property'].isin(properties)]
    if 'Category' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['Category'])
    return filtered_df


def pivot_data(df):
    pivot = df.set_index('Property').T
    return pivot

def reformat_total_mode(total_mode):
    parts = total_mode.split('_')
    proximity_mode = parts[0]
    dimension = parts[1]
    bipartitedness = "bipartite" if parts[2] == "True" else "unipartite"

    # Special handling for modes where bipartiteness is implied in the name
    if "bipartite" in proximity_mode:
        return f"{proximity_mode} {dimension}"
    else:
        return f"{proximity_mode} {dimension} {bipartitedness}"
def normalize_mode_label(total_mode):
    bipartiteness = "Bipartite" if "bipartite" in total_mode else "Unipartite"
    dimension_label = "2D" if "2" in total_mode else "3D"
    print("total_mode", total_mode)
    return f"{bipartiteness} {dimension_label}"

def plot_2_variables_figure(data, parameter_x, quantity_to_evaluate):
    # Reformat data if parameter_x is a special format like 'Total_Mode'
    if parameter_x == 'Total_Mode':
        data[parameter_x] = data[parameter_x].apply(reformat_total_mode)
        data = data.astype({parameter_x: 'float', quantity_to_evaluate: 'float'})
    else:
        data = data.astype({parameter_x: 'float', quantity_to_evaluate: 'float'})

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=parameter_x, y=quantity_to_evaluate, s=100, color='blue', edgecolor='w', alpha=0.6)

    # Styling
    plt.title(f'Scatter Plot of {quantity_to_evaluate} by {parameter_x}')
    plt.xlabel(parameter_x)  # Assuming LABEL_MAPPINGS has a label mapping for parameter_x
    plt.ylabel(quantity_to_evaluate)  # Assuming LABEL_MAPPINGS has a label mapping for quantity_to_evaluate
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save and show the plot
    args = GraphArgs()
    plot_folder = args.directory_map['dataframes']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{plot_folder}/scatter_{quantity_to_evaluate}_by_{parameter_x}_{current_time}.svg"

    plt.savefig(filename)
    plt.show()
def plot_heatmap(data, parameter_x, parameter_y, quantity_to_evaluate):
    if parameter_x == 'Total_Mode':
        numeric_columns = data.columns.drop([parameter_x])
        data[numeric_columns] = data[numeric_columns].astype('float')
        data[parameter_x] = data[parameter_x].apply(reformat_total_mode)
    else:
        data = data.astype('float')

    print(data[parameter_x].unique())  # Check unique values of parameter_x before plotting
    heatmap_data = data.pivot_table(index=parameter_y, columns=parameter_x, values=quantity_to_evaluate, aggfunc='mean')
    heatmap_data = heatmap_data.sort_index(ascending=False)

    colorbar_min, colorbar_max = C_BAR_LIMITS.get(quantity_to_evaluate,
                                                  (heatmap_data.min().min(), heatmap_data.max().max()))

    plt.figure(figsize=(6, 4.5))
    sns.heatmap(heatmap_data, annot=True, cmap=COLORMAP_STYLES.get(quantity_to_evaluate, 'viridis'), fmt=".2f",
                cbar_kws={'label': LABEL_MAPPINGS.get(quantity_to_evaluate)},
                vmin=colorbar_min, vmax=colorbar_max, linewidths=2, linecolor='white')
    plt.xlabel(LABEL_MAPPINGS.get(parameter_x))
    plt.ylabel(LABEL_MAPPINGS.get(parameter_y))

    args = GraphArgs()
    plot_folder = args.directory_map['dataframes']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{plot_folder}/heatmap_{quantity_to_evaluate}_by_{parameter_x}_and_{parameter_y}_{current_time}"

    plt.savefig(filename + '.svg')
    plt.show()



def plot_heatmap_proximity_modes(data, proximity_modes, parameter_x, parameter_y, quantity_to_evaluate):
    if parameter_x == 'Total_Mode':
        numeric_columns = data.columns.drop([parameter_x])
        data[numeric_columns] = data[numeric_columns].astype('float')
    else:
        data = data.astype('float')
    for mode in proximity_modes:
        mode_data = data[data['Total_Mode'].str.contains(mode)]
        mode_data['Total_Mode'] = mode_data['Total_Mode'].apply(normalize_mode_label)
        print(mode_data)
        heatmap_data = mode_data.pivot_table(index=parameter_y, columns='Total_Mode', values=quantity_to_evaluate, aggfunc='mean')
        column_order = ['Unipartite 2D', 'Bipartite 2D', 'Unipartite 3D', 'Bipartite 3D']
        heatmap_data = heatmap_data.reindex(column_order, axis=1)  # Reorder the columns
        heatmap_data = heatmap_data.sort_index(ascending=False)
        colorbar_min, colorbar_max = C_BAR_LIMITS.get(quantity_to_evaluate,
                                                      (heatmap_data.min().min(), heatmap_data.max().max()))
        plt.figure(figsize=(6, 4.5))
        print(heatmap_data)
        ax = sns.heatmap(heatmap_data, annot=False, cmap=COLORMAP_STYLES.get(quantity_to_evaluate, 'viridis'), fmt=".2f",
                    cbar_kws={'label': LABEL_MAPPINGS.get(quantity_to_evaluate)},
                    vmin=colorbar_min, vmax=colorbar_max, linewidths=2, linecolor='white')

        y_labels = [int(float(label.get_text())) if float(label.get_text()).is_integer() else label.get_text() for label
                    in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels)
        plt.title(f"Proximity Mode: {mode}")
        plt.ylabel(LABEL_MAPPINGS.get(parameter_y))
        plt.xlabel(LABEL_MAPPINGS.get(parameter_x))
        print(parameter_x)
        print(LABEL_MAPPINGS.get(parameter_x))
        args = GraphArgs()
        plot_folder = args.directory_map['dataframes']
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{plot_folder}/heatmap_{quantity_to_evaluate}_by_{parameter_x}_and_{parameter_y}_{current_time}_{mode}"
        plt.savefig(filename + '.svg')


def create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters={}):

    combined_data = load_data(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters)
    if parameter_x == 'Total_Mode':
        proximity_modes = ['knn', 'epsilon']
        plot_heatmap_proximity_modes(combined_data, proximity_modes, parameter_x, parameter_y, quantity_to_evaluate)
    else:
        plot_heatmap(combined_data, parameter_x, parameter_y, quantity_to_evaluate)



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

def create_2d_plot(folder_path, parameter_x, quantity_to_evaluate, omit_df_with_parameters={}):
    combined_data = load_data_v2(folder_path, parameter_x, quantity_to_evaluate, omit_df_with_parameters,)
    plot_2_variables_figure(combined_data, parameter_x, quantity_to_evaluate)

def create_violin_plot(folder_path, quantity_to_evaluate):
    combined_data = load_data_1Dplot(folder_path, quantity_to_evaluate)
    combined_data[quantity_to_evaluate] = pd.to_numeric(combined_data[quantity_to_evaluate], errors='coerce')
    print(combined_data)
    # Plotting the violin plot
    sns.violinplot(data=combined_data, y=quantity_to_evaluate)
    # sns.swarmplot(data=combined_data, y=quantity_to_evaluate, color='black', alpha=0.5)  # 'alpha' is for transparency
    # sns.stripplot(data=combined_data, y=quantity_to_evaluate, color='black', jitter=0.1, size=5,
    #               alpha=0.5)  # 'jitter' adds a small horizontal variation

    # Calculate maximum, median, and minimum values
    max_value = combined_data[quantity_to_evaluate].max()
    second_max_value = combined_data[quantity_to_evaluate].nlargest(2).values[1]
    min_value = combined_data[quantity_to_evaluate].min()
    second_min_value = combined_data[quantity_to_evaluate].nsmallest(2).values[1]
    ## Calculate median
    median_value = np.percentile(combined_data[quantity_to_evaluate], 50)
    top_90_percent_value = np.percentile(combined_data[quantity_to_evaluate], 97)

    # Find rows corresponding to these values (or the closest match in the data)
    closest_median = \
    combined_data.iloc[(combined_data[quantity_to_evaluate] - median_value).abs().argsort()[:1]][
        quantity_to_evaluate].values[0]
    closest_top_90_percent = \
    combined_data.iloc[(combined_data[quantity_to_evaluate] - top_90_percent_value).abs().argsort()[:1]][
        quantity_to_evaluate].values[0]

    specific_title_quantity = combined_data[combined_data['edge_list_title'] ==
                                            'Sample01_human_pbmcs_unstimulated_component_RCVCMP0000120_edgelist.csv'][quantity_to_evaluate].iloc[0]
    # Identify the rows corresponding to these values
    special_values = [max_value, closest_top_90_percent, specific_title_quantity]



    # # Identifying the top two outliers
    # special_values = combined_data[quantity_to_evaluate].nlargest(2)
    outlier_data = combined_data[combined_data[quantity_to_evaluate].isin(special_values)]

    print(special_values)
    print(outlier_data)
    # Highlight the outliers with a different color and larger point
    sns.swarmplot(data=outlier_data, y=quantity_to_evaluate, color='red', size=8)

    # Annotate the outliers
    for index, row in outlier_data.iterrows():
        title = row['edge_list_title']
        # Parsing the string to extract desired parts
        match = re.search(r"RCVCMP(\d+)_", title)
        if match:
            # Extracting the part immediately after "RCVCMP" and before the next underscore
            display_text = match.group(1)
        else:
            display_text = "Info unavailable"

        plt.text(x=0.05, y=row[quantity_to_evaluate],
                 s=display_text,
                 horizontalalignment='left', verticalalignment='center',
                 color='green')

    # Finalize the plot
    plt.ylabel(LABEL_MAPPINGS[quantity_to_evaluate])

    # Set y-axis limits and ticks
    # plt.ylim(0, 1)  # Set the limits of y-axis from 0 to 1
    # plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis ticks at intervals of 0.1
    args = GraphArgs()
    plot_folder = args.directory_map['dataframes']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{plot_folder}/violin_plot_{quantity_to_evaluate}_{current_time}"
    plt.savefig(filename + '.svg')
    print(f"Saved plot to {filename}.svg")

    plt.show()



# def create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters={}):
#     """
#     Creates a heatmap of the data in the specified folder. The folder must contain several dataframes and 3
#     interesting quantities (x, y, z) where z is the color
#     """
#
#     c_bar_limits = {
#     "network_dim": (0.5, 3.5),
#     "gram_total_contribution": (0.3, 1),
#     "gram_spectral_gap": (0, 1),
#     "gram_last_spectral_gap": (0, 1),
#     "largeworldness": (0, 1),
#      }
#
#     label_mappings = {
#         "network_dim": "Network Dimension",
#         "gram_total_contribution": "Eigenvalue Contribution",
#         "gram_spectral_gap": "Spectral Gap (mean)",
#         "gram_last_spectral_gap": "Spectral Gap",
#         "largeworldness": "Large Worldness",
#         "false_edges_ratio": "False Edges Ratio",
#         "false_edges_count": "False Edges",
#         "Quantile": "Filtering Power",
#         "False Edge Ratio": "False Edges Ratio",
#         "true_edges_deletion_ratio": "Missing Edges Ratio",
#
#     }
#
#     colormap_styles = {
#         "network_dim": "Spectral_r",
#         "gram_total_contribution": "magma",
#         "gram_spectral_gap": "magma",
#         "gram_last_spectral_gap": "magma",
#     }
#
#
#
#     all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
#     combined_data = pd.DataFrame()
#
#
#     for file in all_files:
#         df = pd.read_csv(file)
#
#         for key, value in omit_df_with_parameters.items():
#             if key in df['Property'].values:
#                 if df.loc[df['Property'] == key, 'Value'].iloc[0] == value:
#                     continue  # Skip this file if the condition matches
#
#         # Create Total_Mode if parameter_x is set to it
#         if parameter_x == "Total_Mode":
#             # Extract values from the DataFrame
#             modes = df[df['Property'].isin(['dim', 'proximity_mode', 'bipartiteness'])]
#             modes = modes.set_index('Property')['Value'].to_dict()
#             total_mode = f"{modes.get('proximity_mode', '')}_{modes.get('dim', '')}_{modes.get('bipartiteness', '')}"
#
#             # Append the new Total_Mode row to the DataFrame
#             total_mode_row = pd.DataFrame({'Property': ['Total_Mode'], 'Value': [total_mode]})
#             df = pd.concat([df, total_mode_row], ignore_index=True)
#
#
#         # Filter rows where Property matches parameter_x or parameter_y, or quantity_to_evaluate
#         if 'false_edges_count' in df['Property'].unique():
#             filtered_df = df[df['Property'].isin([parameter_x, parameter_y, quantity_to_evaluate, 'false_edges_count'])]
#         else:
#             filtered_df = df[df['Property'].isin([parameter_x, parameter_y, quantity_to_evaluate])]
#
#
#         pivoted = filtered_df.pivot_table(index='Property', values='Value', aggfunc='first').T
#         combined_data = pd.concat([combined_data, pivoted], ignore_index=True)
#
#         # false_edge_coun
#     if 'false_edges_count' in combined_data.columns:
#         unique_false_edge_counts = combined_data['false_edges_count'].dropna().unique()
#         num_clusters = len(unique_false_edge_counts)
#         print(num_clusters)
#         kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(combined_data['false_edges_ratio'].values.reshape(-1, 1))
#         print(combined_data['false_edges_ratio'])
#         combined_data['false_edges_ratio'] = kmeans.cluster_centers_[kmeans.labels_].round(3)
#         print(combined_data['false_edges_ratio'])
#
#     # Ensure the DataFrame is complete with all required columns
#     if not all(col in combined_data.columns for col in [parameter_x, parameter_y, quantity_to_evaluate]):
#         print(combined_data.columns)
#         raise ValueError("Some parameters are missing in the data.")
#
#
#     print("combined data", combined_data)
#     if parameter_x == 'Total_Mode':
#         numeric_columns = combined_data.columns.drop(
#             [parameter_x])  # Assuming 'parameter_x' is 'Total_Mode' and not numeric
#         combined_data[numeric_columns] = combined_data[numeric_columns].astype('float')
#     else:
#         combined_data = combined_data.astype('float')
#     heatmap_data = combined_data.pivot_table(index=parameter_y, columns=parameter_x, values=quantity_to_evaluate,
#                                              aggfunc='mean')
#
#     # heatmap_data = heatmap_data.astype(float)
#     heatmap_data = heatmap_data.sort_index(ascending=False)
#
#
#     colorbar_min, colorbar_max = c_bar_limits.get(quantity_to_evaluate,
#                                             (heatmap_data.min().min(), heatmap_data.max().max()))
#     # Plot the heatmap
#     plt.figure(figsize=(6, 4.5))
#     sns.heatmap(heatmap_data, annot=True, cmap=colormap_styles.get(quantity_to_evaluate), fmt=".2f", cbar_kws={'label': label_mappings.get(quantity_to_evaluate)},
#                 vmin=colorbar_min, vmax=colorbar_max, linewidths=2, linecolor='white')
#     plt.xlabel(label_mappings.get(parameter_x))
#     plt.ylabel(label_mappings.get(parameter_y))
#
#
#
#     args = GraphArgs()
#     plot_folder = args.directory_map['dataframes']
#     current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     filename = f"{plot_folder}/heatmap_{quantity_to_evaluate}_by_{parameter_x}_and_{parameter_y}_{current_time}"
#
#     plt.savefig(filename + '.svg')
#     plt.show()



# ### Process Johana files adding new properties to the dataframe based on title of the files
# add_properties_from_filenames("/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Fused_nodes_DF/Fused_nodes_Dataframe/")

# Run heatmaps
# #   bipartite  "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240424_160524_false_edges_count_true_edges_deletion_ratio/"
## delaunay "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240424_135604_false_edges_count_true_edges_deletion_ratio/"

# I think the Fused Nodes is what we use in the paper
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Fused_nodes_DF/Fused_nodes_Dataframe/"
# "/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Spurious_crosslinks_DF/Spurious_crosslinks_Dataframe/"


# ### Different proximity graphs
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240517_105854_proximity_mode_intended_av_degree_dim/"
# ##High average degree
# # folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240517_120259_proximity_mode_intended_av_degree_dim/"
#
# # folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/data/weinstein/Alexanders_Simulation_DF_results/Spurious_crosslinks_DF/Spurious_crosslinks_Dataframe/"
#
# parameter_x = 'Total_Mode'  # false_edges_count, false_edges_ratio, False Edge Ratio (alex data), Total_Mode (simulated prox graphs)
# parameter_y = 'intended_av_degree'  # true_edges_deletion_ratio, Quantile (alex data), intended_av_degree
# quantity_to_evaluate = 'gram_last_spectral_gap'  # gram_total_contribution, gram_spectral_gap, gram_last_spectral_gap, network_dim, largeworldness
# # omit_df_with_parameters = {'Filtered': False}
# omit_df_with_parameters = {}

# ## Single heatmap  (3D plot)
# create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters=omit_df_with_parameters)

# ## Multiple heatmaps  (3D plot)
# quantities_list = ['gram_last_spectral_gap', 'gram_spectral_gap', 'gram_total_contribution', 'network_dim', 'largeworldness']
# for quantity_to_evaluate in quantities_list:
#     create_heatmap(folder_path, parameter_x, parameter_y, quantity_to_evaluate, omit_df_with_parameters=omit_df_with_parameters)


# TODO: heatmaps are for 3-dimensions, make a simple plot for 2 dimensions

# ## Scatter - 2D plot
# # folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240528_151805_distance_decay_quantile_proximity_mode/"  # small quantiles
# ## middle ranges
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240528_152730_distance_decay_quantile_proximity_mode/"
# # folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240528_134903_distance_decay_quantile_proximity_mode/" # big quantiles, just gram matrix
# parameter_x = 'distance_decay_quantile' # false_edges_count, false_edges_ratio, False Edge Ratio (alex data), Total_Mode (simulated prox graphs)
# parameter_y = None
# quantity_to_evaluate = 'gram_last_spectral_gap'  # gram_total_contribution, gram_total_contribution_all_eigens, gram_spectral_gap, gram_last_spectral_gap, network_dim, largeworldness
# omit_df_with_parameters = {}
# create_2d_plot(folder_path, parameter_x, quantity_to_evaluate, omit_df_with_parameters=omit_df_with_parameters)


### Violin plot (1D plot)

### raji cells
## edge threshold 8000 and above
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240531_170544_proximity_mode_edge_list_title_raji/"
## edge threshold 2000-8000
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240603_143439_proximity_mode_edge_list_title_raji_2000-8000/"

# # pbmc human cells
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240531_170946_proximity_mode_edge_list_title_pbmc/"
## edge threshold 2000-8000
folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240603_153707_proximity_mode_edge_list_title_pbmc_2000-8000/"

# # uropod
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240603_095632_proximity_mode_edge_list_title_uropod/"
## edge threshold 2000-8000
# folder_path = "/home/david/PycharmProjects/Spatial_Constant_Analysis/results/output_dataframe/20240603_151811_proximity_mode_edge_list_title_uropod_2000-8000/"
quantity_to_evaluate = 'gram_total_contribution'

create_violin_plot(folder_path, quantity_to_evaluate)

