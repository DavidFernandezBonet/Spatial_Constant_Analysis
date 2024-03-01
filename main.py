
from create_proximity_graph import write_proximity_graph
from structure_and_args import GraphArgs
from data_analysis import plot_graph_properties, run_simulation_subgraph_sampling
import warnings
from plots import plot_original_or_reconstructed_image
from utils import *
from spatial_constant_analysis import run_reconstruction
from dimension_prediction import run_dimension_prediction
from gram_matrix_analysis import plot_gram_matrix_eigenvalues
from structure_and_args import create_project_structure




def load_and_initialize_graph():
    """
    Step 1: Load the graph with provided arguments and perform initial checks.
    """
    args = GraphArgs()
    print(args.proximity_mode)

    if args.proximity_mode != "experimental":
        write_proximity_graph(args)

    print(args.num_points)
    print(args.edge_list_title)
    return load_graph(args, load_mode='sparse'), args

def subsample_graph_if_necessary(graph, args):
    """
    Subsamples the graph if it is too large for efficient processing.
    """
    min_points_subsampling = args.max_subgraph_size
    if args.num_points > min_points_subsampling and args.large_graph_subsampling:
        warnings.warn(f"Large graph. Subsampling using BFS for efficiency purposes.\nSize of the sample; {min_points_subsampling}")
        return sample_csgraph_subgraph(args, graph, min_nodes=min_points_subsampling)
    return graph

def plot_and_analyze_graph(graph, args):
    """
    Plots the original graph and analyzes its properties.
    """
    if args.proximity_mode != "experimental" and args.plot_original_image:
        plot_original_or_reconstructed_image(args, image_type='original')

    if args.plot_graph_properties:
        plot_graph_properties(args, igraph_graph=graph)

def compute_shortest_paths(graph, args):
    """
    Step 1.5: Compute shortest paths and store it in args. This is done only once.
    """
    compute_shortest_path_matrix_sparse_graph(args=args, sparse_graph=graph)



def spatial_constant_analysis(graph, args, false_edge_list=None):
    """
    Step 2: Analyze spatial constant
    """
    if false_edge_list is None:
        false_edge_list = np.arange(0, 101, step=20)
    size_interval = int(args.num_points / 10)  # collect 10 data points
    print(size_interval)
    run_simulation_subgraph_sampling(args, size_interval=size_interval, n_subgraphs=10, graph=graph,
                                     add_false_edges=True, add_mst=False, false_edge_list=false_edge_list)

def network_correlation_dimension(args):
    """
    Steps 3: Predict the dimension of the graph
    """
    results_pred_dimension = run_dimension_prediction(args, distance_matrix=args.shortest_path_matrix,
                                                      dist_threshold=int(args.mean_shortest_path),
                                                      central_node_index=find_central_node(args.shortest_path_matrix))
    print("Results predicted dimension", results_pred_dimension)

def rank_matrix_analysis(args):
    """
    Step 4. Analyze the rank matrix
    """
    first_d_values_contribution = plot_gram_matrix_eigenvalues(args=args, shortest_path_matrix=args.shortest_path_matrix)
    print("First d values contribution", first_d_values_contribution)

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
    if args.reconstruct:
        print("running reconstruction...")
        print("reconstruction mode:", args.reconstruction_mode)
        ground_truth_available = not (args.proximity_mode == "experimental" or args.large_graph_subsampling)
        run_reconstruction(args, sparse_graph=graph, ground_truth_available=ground_truth_available,
                           node_embedding_mode=args.reconstruction_mode)




def main():
    """
    Main function: graph loading, processing, and analysis.
    """
    create_project_structure()  # Create structure if not done before
    # Load and process the graph
    graph, args = load_and_initialize_graph()
    graph = subsample_graph_if_necessary(graph, args)
    plot_and_analyze_graph(graph, args)
    compute_shortest_paths(graph, args)

    # # Spatial Coherence Validation
    spatial_constant_analysis(graph, args)
    network_correlation_dimension(args)
    rank_matrix_analysis(args)

    # # Reconstruction
    reconstruct_graph(graph, args)

if __name__ == "__main__":
    main()