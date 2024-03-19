import sys
import os
import matplotlib.pyplot as plt


# script_dir = "/home/david/PycharmProjects/Spatial_Graph_Denoising"
# # Add this directory to the Python path
# if script_dir not in sys.path:
#     sys.path.append(script_dir)


from create_proximity_graph import write_proximity_graph
from algorithms import *
from utils import *
from utils import load_graph
from data_analysis import *
from plots import *
from metrics import *
from bfs_animation import *

import scienceplots
plt.style.use(['science', 'nature'])
# plt.rcParams.update({'font.size': 16, 'font.family': 'serif'})
font_size = 24
plt.rcParams.update({'font.size': font_size})
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 6
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size - 10

# plt.style.use(['science','no-latex', 'nature'])

def run_reconstruction(args, sparse_graph, node_embedding_mode='ggvec', manifild_learning_mode='UMAP',
                       ground_truth_available=False):
    """
    Performs reconstruction of a graph from a sparse matrix representation, employing node embedding
    and manifold learning techniques. The process includes inferring node positions, plotting the
    reconstructed graph, and computing quality metrics related to the reconstruction accuracy.

    Args:
        args: An object containing configuration parameters and options for the analysis, including
              the dimensionality (`dim`), directory mappings (`directory_map`), and graph analysis
              settings.
        sparse_graph: A sparse matrix representation of the graph to be reconstructed.
        node_embedding_mode (str): The mode of node embedding to use for initial node position inference.
                                   Defaults to 'ggvec'. Other options include 'landmark_isomap'.
        manifild_learning_mode (str): The manifold learning technique to apply for dimensionality
                                      reduction. Defaults to 'UMAP'.
        ground_truth_available (bool): Indicates whether ground truth data is available for evaluating
                                       the reconstruction quality. Defaults to False.

    Returns:
        tuple: A tuple containing:
               - reconstructed_points (numpy array): The inferred positions of nodes in the target
                 dimensionality.
               - metrics (dict): A dictionary of quality metrics assessing the reconstruction, with
                 keys for ground truth-based metrics ("ground_truth") and metrics in the absence of
                 ground truth ("gta").

    Note:
        The function modifies the `args` object by appending the `node_embedding_mode` to `args_title`
        for identification.
    """
    sparse_graph = convert_graph_type(graph=sparse_graph, args=args, desired_type='sparse')
    metrics = {}
    args.args_title = args.args_title + '_' + node_embedding_mode
    reconstruction = ImageReconstruction(graph=sparse_graph, dim=args.dim, node_embedding_mode=node_embedding_mode,
                                         manifold_learning_mode=manifild_learning_mode)
    reconstructed_points = reconstruction.reconstruct(do_write_positions=True, args=args)

    # # Assessing Weinstein reconstruction algo # TODO: it is not working well I think
    # position_filename = 'reconstructed_positions_weinstein_febraury_corrected.csv'
    # original_position_folder = args.directory_map["original_positions"]
    # positions_df = pd.read_csv(f"{original_position_folder}/{position_filename}")
    # positions_df['node_ID'] = positions_df['node_ID'].map(args.node_ids_map_old_to_new)
    # positions_df = positions_df.dropna()
    # positions_df['node_ID'] = positions_df['node_ID'].astype(int)
    # reconstructed_points = positions_df[['x', 'y']].to_numpy()


    plot_original_or_reconstructed_image(args, image_type='reconstructed')
    edge_list = read_edge_list(args)

    # Ground Truth-based quality metrics
    if ground_truth_available:
        original_points = read_position_df(args)
        qm = QualityMetrics(original_points, reconstructed_points)
        qm.evaluate_metrics()
        metrics["ground_truth"] = qm
    # GTA metrics
    gta_qm = GTA_Quality_Metrics(edge_list=edge_list, reconstructed_points=reconstructed_points)
    gta_qm.evaluate_metrics()
    metrics["gta"] = gta_qm
    return reconstructed_points, metrics



def make_spatial_constant_euc_vs_network():
    args = GraphArgs()
    args.proximity_mode = "knn"
    args.dim = 3
    args.intended_av_degree = 10
    args.num_points = 1000
    create_proximity_graph.write_proximity_graph(args, point_mode="square", order_indices=False)
    # Parameters for the particular plot
    size_interval = int(args.num_points / 10)  # collect 10 data points
    n_samples = 5

    ## Euclidean Spatial Constant
    euclidean_positions = read_position_df(args=args)
    euc_results_df = get_spatial_constant_euclidean_df(args, positions_array=euclidean_positions, size_interval=size_interval,
                                                       num_samples=n_samples)
    # plot_euc_spatial_constant_against_size_threshold(args, results_df=euc_results_df)
    ## Network Spatial Constant
    igraph_graph = load_graph(args, load_mode='igraph')

    net_results_df = run_simulation_subgraph_sampling(args, size_interval=size_interval, n_subgraphs=n_samples,
                                                      graph=igraph_graph,
                                     add_false_edges=False, add_mst=False)

    ### Plotting function that encompasses both of them
    plot_spatial_constant_euc_vs_network(args, results_df_euc=euc_results_df, results_df_net=net_results_df)



def main():
    # TODO: args_title is not instantiated if you don't call the parameters (maybe just make a config file with the parameters and call them all)
    args = GraphArgs()

    args.proximity_mode = "lattice"
    args.dim = 2

    # print("Proximity_mode after setting to 'knn':", args.proximity_mode)
    # print("Setting false_edges_count to 5...")
    args.false_edges_count = 0   #TODO: this only adds false edges to simulated graphs!
    # print("Proximity_mode after setting false_edges_count to 5:", args.proximity_mode)
    print(args.proximity_mode)
    args.intended_av_degree = 10
    args.num_points = 1000
    args.colorfile = "weinstein_colorcode_february_corrected.csv"  # colorful_spiral.jpeg, weinstein_colorcode_february_corrected.csv, None

    simulation_or_experiment = "simulation"
    reconstruct = False


    if simulation_or_experiment == "experiment":
        # # # #Experimental
        # our group:
        # subgraph_2_nodes_44_edges_56_degree_2.55.pickle  # subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle  # subgraph_8_nodes_160_edges_179_degree_2.24.pickle

        # slidetag:
        # edge_list_nbead_7_filtering_simon.csv, edge_list_distance_150_filtering_simon_connected.csv (this one is good, uses a little ground truth) # new simon data from slidetag
        # edge_list_distance_300_filtering_simon.csv
        # now with good indexing:
        # nbead_7_goodindex_simon.csv, edge_list_distance_150_filtering_goodindex_simon.csv

        # unfiltered pixelgen:
        # pixelgen_cell_2_RCVCMP0000594.csv, pixelgen_cell_1_RCVCMP0000208.csv, pixelgen_cell_3_RCVCMP0000085.csv
        # pixelgen_edgelist_CD3_cell_2_RCVCMP0000009.csv, pixelgen_edgelist_CD3_cell_1_RCVCMP0000610.csv, pixelgen_edgelist_CD3_cell_3_RCVCMP0000096.csv
        # filtered pixelgen:
        # pixelgen_processed_edgelist_Sample01_human_pbmcs_unstimulated_cell_3_RCVCMP0000563.csv
        # pixelgen_processed_edgelist_Sample01_human_pbmcs_unstimulated_cell_2_RCVCMP0000828.csv
        # pixelgen_processed_edgelist_Sample07_pbmc_CD3_capped_cell_3_RCVCMP0000344.csv (stimulated cell)
        # pixelgen_processed_edgelist_Sample04_Raji_Rituximab_treated_cell_3_RCVCMP0001806.csv (treated cell)
        # shuai_protein_edgelist_unstimulated_RCVCMP0000133_neigbours_s_proteinlist.csv  (shuai protein list)
        # pixelgen_processed_edgelist_shuai_RCVCMP0000073_cd3_cell_1_RCVCMP0000073.csv (shuai error correction)

        # weinstein:
        # weinstein_data.csv
        # weinstein_data_corrected_january.csv
        # weinstein_data_corrected_february.csv (new indices shuai)
        # slidetag_processed_edgelist_10X_bc_to_cell_bc_SRR07.csv, edge_list_abundant_beads_cut_beadsum_thresholds_8_256_SRR11.csv SRR11
        # slidetag_processed_edgelist_edge_list_filtered_by_bed_n_connections_thresholds_2-16.csv
        args.proximity_mode = "experimental"  # define proximity mode before name!
        args.edge_list_title = "edge_list_distance_150_filtering_goodindex_simon.csv"
        weighted = False
        weight_threshold = 5

        if os.path.splitext(args.edge_list_title)[1] == ".pickle":
            write_nx_graph_to_edge_list_df(args)    # activate if format is .pickle file

        # # TODO: uncomment this
        # if not weighted:
        #     igraph_graph_original = load_graph(args, load_mode='igraph')
        # else:
        #     igraph_graph_original = load_graph(args, load_mode='igraph', weight_threshold=weight_threshold)

        # # Plot original weinstein. Activate if you want to plot
        # plot_original_or_reconstructed_image(args, image_type='original',
        #                                      position_filename='reconstructed_positions_weinstein_febraury_corrected.csv',
        #                                      plot_weights_against_distance=True)


        if reconstruct:
            args.edge_list_title = "weinstein_data_corrected_february.csv"
            # # Reconstruct with weights
            # igraph_graph_original, _ = load_graph(args, load_mode='sparse_weighted', weight_threshold=7)
            # # Reconstruct unweighted
            igraph_graph_original, _ = load_graph(args, load_mode='sparse', weight_threshold=weight_threshold)
            run_reconstruction(args, sparse_graph=igraph_graph_original, ground_truth_available=False,
                               node_embedding_mode="ggvec")

        # plot_graph_properties(args, igraph_graph_original)  # plots clustering coefficient, degree dist, also stores individual spatial constant...


    ### Create simulated graph if it is a simulation
    elif simulation_or_experiment == "simulation":
        # # # 1 Simulation
        create_proximity_graph.write_proximity_graph(args)

        ## Plot the original graph
        # igraph_graph_original = load_graph(args, load_mode='igraph')
        # # igraph_graph_original = get_minimum_spanning_tree_igraph(igraph_graph_original)  # careful with activating this
        #
        # plot_original_or_reconstructed_image(args, image_type='original')
    else:
        raise ValueError("Input simulation or experiment")

    ### Animation
    # main_animation(args, igraph_graph_original, n_graphs=50)
    # igraph_graph_clean = remove_false_edges_igraph(igraph_graph_original, args.false_edge_ids)
    # main_animation(args, igraph_graph_clean, n_graphs=50, title="clean")
    # raise ValueError("finito")

    ### Efficiency calculation
    # print("ORIGINAL global efficiency", global_efficiency(igraph_graph_original))
    # print("ORIGINAL local efficiency", local_efficiency(igraph_graph_original))
    #
    # igraph_graph_original = add_random_edges_igraph(igraph_graph_original, num_edges_to_add=10000)
    #
    # print("global efficiency", global_efficiency(igraph_graph_original))  # TODO: efficiency is too dependent on the size (normalization constant heaivly affected)
    # print("local efficiency", local_efficiency(igraph_graph_original))


    ## ------------------------------------------------------------------
    ##### Run subgraph sampling (main spatial constant function)
    # Linear spaced
    false_edge_list = np.arange(0, 101, step=20)


    # # Log spaced
    # false_edge_list = np.logspace(start=0, stop=3, num=20, base=10, dtype=int)
    # false_edge_list = np.insert(false_edge_list, 0, 0)

    # ## Artificially add random edges
    # igraph_graph_original = add_random_edges_igraph(igraph_graph_original, num_edges_to_add=10)


    #### Run subgraph sampling simulation  #TODO: main function for the spatial constant plots
    run_simulation_subgraph_sampling(args, size_interval=100, n_subgraphs=10, graph=igraph_graph_original,
                                     add_false_edges=True, add_mst=False, false_edge_list=false_edge_list)
    ## ------------------------------------------------------------------


    ### Run dimension prediction ##TODO: I think this is still in development phase, not working too well?
    # get_dimension_estimation(args, graph=igraph_graph_original, n_samples=20, size_interval=100, start_size=100)  # TODO: start_size matters a lot if not uncertainty


    # #### For Weinstein data get a sample only  (I think this is not operative any longer?)
    # sample_size = 3000
    # igraph_graph_original = get_one_bfs_sample(igraph_graph_original, sample_size=sample_size)### Get only a sample
    # args.num_points = sample_size

    ## Different weighted thresholds
    # # weight_thresholds = [4, 5, 7, 9, 12]
    # weight_thresholds = [10]
    # subgraph_sampling_analysis_for_different_weight_thresholds(args, weight_thresholds, edge_list_title=args.edge_list_title)

    # ## Weight threshold analysis
    # weight_thresholds = np.arange(1, 50)
    # spatial_constant_and_weight_threshold_analysis(args, weight_thresholds, edge_list_title=args.edge_list_title)


    # ## Reconstruction (load graph as sparse!)  #TODO: main recons function
    # sparse_graph_original, _ = load_graph(args, load_mode='sparse')
    # run_reconstruction(args, sparse_graph=sparse_graph_original, ground_truth_available=False,
    #                    node_embedding_mode="node2vec")

    # ### Plot cool SP images with heatmaps
    sparse_graph_original = load_graph(args, load_mode='sparse')
    ## Individual

    # shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(sparse_graph_original)
    # plot_shortest_path_heatmap(args, shortest_path_matrix=shortest_path_matrix)

    ## Multiple   #TODO: add pipeline that deals with making several graphs and plotting them in comparison
    false_edge_list = [0, 10, 50, 100]
    sp_matrix_list = compute_several_sp_matrices(args, sparse_graph_original, false_edges_list=false_edge_list)
    plot_multiple_shortest_path_heatmaps(args, sp_matrix_list, false_edge_list=false_edge_list)


    ### Old stuff?
    # # num_random_edges = 0
    # # model_func = "spatial_constant_dim=2"  # small_world
    # # # model_func = "small_world"
    #
    #
    # # run_simulation_false_edges(args, max_edges_to_add=100)
    # # run_simulation_graph_growth(args, n_graphs=50, num_random_edges=num_random_edges, model_func=model_func)
    # run_simulation_comparison_large_and_small_world(args, start_n_nodes=500, end_n_nodes=5000, n_graphs=10, num_random_edges_ratio=0.015)
    #
    # run_simulation_graph_growth(args, n_graphs=50, num_random_edges=0, model_func="spatial_constant_dim=2_linearterm")



    # ### Many graphs simulation --> Obtains violin plots for spatial constant for several proximity graphs
    #
    # num_points_list = [500, 1000, 2000]
    # # proximity_mode_list = ["knn", "epsilon-ball", "epsilon_bipartite", "knn_bipartite", "delaunay_corrected"]
    # proximity_mode_list = ["knn",   "knn_bipartite", "delaunay_corrected", "epsilon-ball", "epsilon_bipartite", "lattice"]
    # intended_av_degree_list = [6, 9, 15, 30]
    # false_edges_list = [0, 5, 20, 100]
    # dim_list = [2, 3]
    #
    #
    # # # Simple simulation to test stuff
    # # num_points_list = [500, 1000]
    # # proximity_mode_list = ["knn",   "knn_bipartite"]
    # # intended_av_degree_list = [6]
    # # false_edges_list = [0]
    # # dim_list = [2, 3]
    #
    #
    # spatial_constant_variation_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list)

if __name__ == "__main__":
    main()