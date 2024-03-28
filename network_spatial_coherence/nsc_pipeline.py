import sys
from pathlib import Path

import pandas as pd

# This is so the script works as a stand alone and as a package
package_root = Path(__file__).parent
if str(package_root) not in sys.path:
    sys.path.append(str(package_root))
import matplotlib.pyplot as plt
import time
from create_proximity_graph import write_proximity_graph
from structure_and_args import GraphArgs
from data_analysis import plot_graph_properties, run_simulation_subgraph_sampling
import warnings
from plots import plot_original_or_reconstructed_image
from utils import *
from spatial_constant_analysis import run_reconstruction
from dimension_prediction import run_dimension_prediction
from gram_matrix_analysis import plot_gram_matrix_eigenvalues
from gram_matrix_analysis import plot_gram_matrix_first_eigenvalues_contribution
from structure_and_args import create_project_structure
from functools import wraps
from memory_profiler import memory_usage
from check_latex_installation import check_latex_installed
import scienceplots


is_latex_in_os = check_latex_installed()
if is_latex_in_os:
    plt.style.use(['nature'])
else:
    plt.style.use(['no-latex', 'nature'])
plt.style.use(['science', 'no-latex', 'nature'])
# font_size = 24
# plt.rcParams.update({'font.size': font_size})
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 6
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size - 10


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



# Global storage for profiling data
profiling_data = {
    'functions': [],
    'time': [],
    'memory': []
}

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        mem_usage_before = memory_usage(max_usage=True)
        result = func(*args, **kwargs)
        mem_usage_after = memory_usage(max_usage=True)
        end_time = time.time()

        # Store profiling data
        profiling_data['functions'].append(func.__name__)
        profiling_data['time'].append(end_time - start_time)
        profiling_data['memory'].append(mem_usage_after - mem_usage_before)

        return result
    return wrapper

def plot_profiling_results(args):
    functions = profiling_data['functions']
    time_taken = profiling_data['time']
    memory_used = profiling_data['memory']
    indices = range(len(functions))
    # Creating the figure and subplots
    fig, (ax1, ax_legend) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plotting the performance metrics
    color = 'tab:red'
    ax1.set_ylabel('Time (seconds)', color=color)
    line1, = ax1.plot(functions, time_taken, color=color, marker='o', linestyle='--', label='Time (s)')
    ax1.set_xticks(indices)  # Set x-ticks to numerical indices
    ax1.set_xticklabels([str(i+1) for i in indices])  # Label x-ticks with numerical indices
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Memory (MiB)', color=color)
    line2, = ax2.plot(functions, memory_used, color=color, marker='o', linestyle='--', label='Memory (MiB)')
    ax2.tick_params(axis='y', labelcolor=color)

    # Configure the legend subplot
    ax_legend.axis('off')  # Turn off the axis
    legend_text = "\n".join(f'{i+1}: {name}' for i, name in enumerate(functions))
    ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', fontsize=9)

    # Additional plot adjustments
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the space between the plots

    # Save the plot
    plot_folder = args.directory_map['profiler']
    plt.savefig(f'{plot_folder}/function_performance_{args.args_title}.svg', bbox_inches='tight')


@profile
def load_and_initialize_graph(args=None):
    """
    Step 1: Load the graph with provided arguments and perform initial checks.
    """
    if args is None:
        args = GraphArgs()
    print("proximity_mode", args.proximity_mode)
    if args.proximity_mode != "experimental":
        write_proximity_graph(args)
        print("Number Nodes", args.num_points)
        print("Average Degree", args.average_degree)
    print("Title Edge List", args.edge_list_title)
    return load_graph(args, load_mode='sparse'), args
@profile
def subsample_graph_if_necessary(graph, args):
    """
    Subsamples the graph if it is too large for efficient processing.
    """
    min_points_subsampling = args.max_subgraph_size
    if args.num_points > min_points_subsampling and args.large_graph_subsampling:
        warnings.warn(f"Large graph. Subsampling using BFS for efficiency purposes.\nSize of the sample; {min_points_subsampling}")
        return sample_csgraph_subgraph(args, graph, min_nodes=min_points_subsampling)
    return graph
@profile
def plot_and_analyze_graph(graph, args):
    """
    Plots the original graph and analyzes its properties.
    """
    if args.original_positions_available and args.plot_original_image:
        if args.proximity_mode == "experimental":
            warnings.warn("Make sure the original image is available for experimental mode. If not, "
                          "set original_positions_available to False")
            positions_file = f"positions_{args.network_name}.csv"
            if "weinstein" in args.network_name:
                args.colorfile = 'weinstein_colorcode_february_corrected.csv'

        else:
            positions_file = None

        plot_original_or_reconstructed_image(args, image_type='original', position_filename=positions_file)

    if args.plot_graph_properties:
        plot_graph_properties(args, igraph_graph=graph)
@profile
def compute_shortest_paths(graph, args):
    """
    Step 1.5: Compute shortest paths and store it in args. This is done only once.
    """
    compute_shortest_path_matrix_sparse_graph(args=args, sparse_graph=graph)
    return args


@profile
def spatial_constant_analysis(graph, args, false_edge_list=None):
    """
    Step 2: Analyze spatial constant
    """
    if false_edge_list is None:
        false_edge_list = np.arange(0, 101, step=20)
    size_interval = int(args.num_points / 10)  # collect 10 data points
    combined_df = run_simulation_subgraph_sampling(args, size_interval=size_interval, n_subgraphs=10, graph=graph,
                                     add_false_edges=True, add_mst=False, false_edge_list=false_edge_list)
    combined_df['Category'] = 'Spatial Coherence'
    filtered_df = combined_df

    spatial_slope = filtered_df['Slope'].iloc[0]
    spatial_r_squared = filtered_df['R_squared'].iloc[0] if 'R_squared' in filtered_df.columns else None

    spatial_slope_false_edge_100 = filtered_df['Slope'].iloc[-1]
    spatial_r_squared_false_edge_100 = filtered_df['R_squared'].iloc[-1] if 'R_squared' in filtered_df.columns else None

    ## Update in main results dictionary
    args.spatial_coherence_quantiative_dict['slope_spatial_constant'] = spatial_slope
    args.spatial_coherence_quantiative_dict['r2_slope_spatial_constant'] = spatial_r_squared
    args.spatial_coherence_quantiative_dict['slope_spatial_constant_false_edge_100'] = spatial_slope_false_edge_100
    args.spatial_coherence_quantiative_dict['r2_slope_spatial_constant_false_edge_100'] = spatial_r_squared_false_edge_100
    args.spatial_coherence_quantiative_dict['ratio_slope_0_to_100_false_edges'] = spatial_slope_false_edge_100 / spatial_slope
    return args, combined_df

@profile
def network_dimension(args):
    """
    Steps 3: Predict the dimension of the graph
    """
    if args.proximity_mode != 'experimental':
        plot_all_heatmap_nodes = True
    else:
        plot_all_heatmap_nodes = False
    print("plot_all_heatmap_nodes", plot_all_heatmap_nodes)
    print("mean shortest path", args.mean_shortest_path)
    results_dimension_prediction = run_dimension_prediction(args, distance_matrix=args.shortest_path_matrix,
                                                      dist_threshold=int(args.mean_shortest_path),
                                                      num_central_nodes=12,
                                                      local_dimension=False, plot_heatmap_all_nodes=plot_all_heatmap_nodes,
                                                      msp_central_node=False, plot_centered_average_sp_distance=False)
    if args.verbose:
        print("Results predicted dimension", results_dimension_prediction)

    if results_dimension_prediction is not None:
        predicted_dimension = results_dimension_prediction['predicted_dimension']
        std_predicted_dimension = results_dimension_prediction['std_predicted_dimension']
        args.spatial_coherence_quantiative_dict.update({
            'network_dim': predicted_dimension,
            'network_dim_std': std_predicted_dimension
        })
        return args, results_dimension_prediction
    else:
        return
@profile
def rank_matrix_analysis(args):
    """
    Step 4. Analyze the rank matrix
    """
    first_d_values_contribution,\
    first_d_values_contribution_5_eigen,\
    spectral_gap, \
        = plot_gram_matrix_eigenvalues(args=args, shortest_path_matrix=args.shortest_path_matrix)

    # results_dict = {"first_d_values_contribution": first_d_values_contribution, "first_d_values_contribution_5_eigen":
    #     first_d_values_contribution_5_eigen, "spectral_gap": spectral_gap}
    #
    # results_dict = pd.DataFrame(results_dict, index=[0])
    # results_dict['Category'] = 'Spatial_Coherence'
    # return results_dict
    args.spatial_coherence_quantiative_dict.update( {
        'gram_total_contribution': first_d_values_contribution_5_eigen,
        'gram_spectral_gap': spectral_gap
    })

    return args, first_d_values_contribution




@profile
def reconstruct_graph(graph, args):
    """
    Reconstructs the graph if required based on the specifications in the `args` object.
    This involves running a graph reconstruction process, which may include converting the graph
    to a specific format, and potentially considering ground truth availability based on the
    reconstruction mode specified in `args`.

    The reconstruction process is conditionally executed based on the `reconstruct` flag within
    the `args` object. If reconstruction is performed, the function also handles the determination
    of ground truth availability and executes the reconstruction process accordingly.

    Args:
        graph: The graph to potentially reconstruct. This graph should be compatible with the
               reconstruction process and might be converted to a different format as part of
               the reconstruction.
        args: An object containing various configuration options and flags for the graph analysis
              and reconstruction process. This includes:
              - `reconstruct` (bool): Whether the graph should be reconstructed.
              - `reconstruction_mode` (str): The mode of reconstruction to be applied.
              - `proximity_mode` (str): The mode of proximity used for the graph, affecting ground
                truth availability.
              - `large_graph_subsampling` (bool): A flag indicating whether subsampling for large
                graphs is enabled, also affecting ground truth availability.

    Note:
        The function directly prints updates regarding the reconstruction process, including the
        mode of reconstruction and whether ground truth is considered available.
    """

    print("running reconstruction...")
    print("reconstruction mode:", args.reconstruction_mode)
    # ground_truth_available = not (args.proximity_mode == "experimental" or args.large_graph_subsampling)
    # TODO: is large_graph_-subsampling messinge up the indices or something? Why did exclude it

    ground_truth_available = args.proximity_mode == "experimental" and args.original_positions_available


    print("ground truth available:", ground_truth_available)
    reconstructed_points, metrics =(
        run_reconstruction(args, sparse_graph=graph, ground_truth_available=ground_truth_available,
                       node_embedding_mode=args.reconstruction_mode))

    if ground_truth_available:
        args.spatial_coherence_quantiative_dict.update(metrics['ground_truth'])
    args.spatial_coherence_quantiative_dict.update(metrics['gta'])
    return args, metrics



def collect_graph_properties(args):
    # Create a dictionary with the graph properties
    args.num_edges = args.sparse_graph.nnz // 2
    properties_dict = {
        'Property': ['Number of Points', 'Number of Edges', 'Average Degree', 'Clustering Coefficient',
                     'Mean Shortest Path'],
        'Value': [
            args.num_points,
            args.num_edges ,
            args.average_degree ,
            args.mean_clustering_coefficient,
            args.mean_shortest_path
        ]
    }

    # Create DataFrame
    graph_properties_df = pd.DataFrame(properties_dict)
    graph_properties_df['Category'] = 'Graph Properties'  # Adding a category column for consistency
    if args.num_points:
        args.spatial_coherence_quantiative_dict['num_points'] = args.num_points
    if args.num_edges:
        args.spatial_coherence_quantiative_dict['num_edges'] = args.num_edges
    if args.average_degree:
        args.spatial_coherence_quantiative_dict['average_degree'] = args.average_degree
    if args.mean_clustering_coefficient:
        args.spatial_coherence_quantiative_dict['clustering_coefficient'] = args.mean_clustering_coefficient
    if args.mean_shortest_path:
        args.spatial_coherence_quantiative_dict['mean_shortest_path'] = args.mean_shortest_path
    return args, graph_properties_df

def output_df_category_mapping():
    category_mapping = {
        'num_points': 'Graph Property',
        'num_edges': 'Graph Property',
        'average_degree': 'Graph Property',
        'clustering_coefficient': 'Graph Property',
        'mean_shortest_path': 'Graph Property',
        'slope_spatial_constant': 'Spatial Constant',
        'r2_slope_spatial_constant': 'Spatial Constant',
        'slope_spatial_constant_false_edge_100': 'Spatial Constant',
        'r2_slope_spatial_constant_false_edge_100': 'Spatial Constant',
        'ratio_slope_0_to_100_false_edges': 'Spatial Constant',
        'network_dim': 'Network Dimension',
        'network_dim_std': 'Network Dimension',
        'gram_total_contribution': 'Gram Matrix',
        'gram_spectral_gap': 'Gram Matrix',
        'KNN': 'Reconstruction',
        'CPD': 'Reconstruction',
        'GTA_KNN': 'Reconstruction',
        'GTA_CPD': 'Reconstruction'
    }
    return category_mapping

def write_output_data(args):
    output_df = pd.DataFrame(list(args.spatial_coherence_quantiative_dict.items()), columns=['Property', 'Value'])
    category_mapping = output_df_category_mapping()
    expected_properties = set(category_mapping.keys())
    missing_properties = expected_properties - set(output_df['Property'])
    for prop in missing_properties:
        output_df = output_df._append({'Property': prop, 'Value': "not computed"}, ignore_index=True)
    output_df['Category'] = output_df['Property'].map(category_mapping)

    df_folder = args.directory_map['output_dataframe']
    output_df.to_csv(f"{df_folder}/quantitative_metrics_{args.args_title}.csv", index=False)
def run_pipeline(graph, args):
    """
    Main function: graph loading, processing, and analysis.
    """

    # Assuming subsample_graph_if_necessary, plot_and_analyze_graph, compute_shortest_paths
    # don't return DataFrames and are just part of the processing
    # graph = subsample_graph_if_necessary(graph, args)  # this is done with the load function now
    plot_and_analyze_graph(graph, args)
    args = compute_shortest_paths(graph, args)

    # Collect graph properties into DataFrame
    args, graph_properties_df = collect_graph_properties(args)
    # Conditional analysis based on args
    if args.spatial_coherence_validation['spatial_constant']:
        args, spatial_constant_df = spatial_constant_analysis(graph, args)
    if args.spatial_coherence_validation['network_dimension']:
        args, results_pred_dimension_df = network_dimension(args)
    if args.spatial_coherence_validation['gram_matrix']:
        args, results_gram_matrix_df = rank_matrix_analysis(args)

    # Reconstruction metrics
    if args.reconstruct:
        args, reconstruction_metrics_df = reconstruct_graph(graph, args)

    write_output_data(args)
    return args

if __name__ == "__main__":
    create_project_structure()  # Create structure if not done before
    # Load and process the graph


    graph, args = load_and_initialize_graph()

    if args.handle_all_subgraphs and type(graph) is list:
        graph_args_list = graph
        for i, graph_args in enumerate(graph_args_list):
            print("iteration:", i, "graph size:", graph_args.num_points)
            if graph_args.num_points > 30:  # only reconstruct big enough graphs
                single_graph_args = run_pipeline(graph=graph_args.sparse_graph, args=graph_args)
                # optionally profile every time
                # plot_profiling_results(single_graph_args)  # Plot the results at the end

    else:
        single_graph_args = run_pipeline(graph, args)
        plot_profiling_results(single_graph_args)  # Plot the results at the end

    # # Modify individual parameters
    # args = GraphArgs()
    # args.show_plots = True
    # args.dim = 2
    # args.plot_graph_properties = False
    # args.colorfile = 'dna_cool2.png'
    # args.proximity_mode = 'delaunay_corrected'
    # args.num_points = 1000
    # args.large_graph_subsampling = True
    #
    #
    # # TODO: solve how it is updated, solve random plots popping up, solve graph with false edges after running spatial constant
    #
    # # Load and process the graph
    # graph, args = load_and_initialize_graph(args=args)
    #
    # # Run the pipeline and plot the results
    # single_graph_args = run_pipeline(graph, args)
    # plot_profiling_results(single_graph_args)

