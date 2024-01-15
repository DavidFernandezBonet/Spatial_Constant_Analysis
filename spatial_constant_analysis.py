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


def run_reconstruction(args, sparse_graph, node_embedding_mode='ggvec', manifild_learning_mode='UMAP',
                       ground_truth_available=False):
    """
    Given a sparse matrix representing a graph:
    1 - Infer the reconstructed positions
    2 - Plot the resulting image with the original graph edges (stored in results/plots/reconstructed_image)
    3 - Return GTA quality metrics (measures the goodness of the reconstruction in Ground Truth Absence)
    Note: If the graph is not simulated, set "ground_truth_available" to False

    node_embedding_mode : 'landmark_isomap' (fast), 'ggvec' (more robust)

    """
    reconstruction = ImageReconstruction(graph=sparse_graph, dim=args.dim, node_embedding_mode=node_embedding_mode,
                                         manifold_learning_mode=manifild_learning_mode)
    reconstructed_points = reconstruction.reconstruct(do_write_positions=True, args=args)
    plot_original_or_reconstructed_image(args, image_type='reconstructed')
    edge_list = read_edge_list(args)
    # Ground Truth-based quality metrics
    if ground_truth_available:
        original_points = read_position_df(args)
        qm = QualityMetrics(original_points, reconstructed_points)
        qm.evaluate_metrics()
    # GTA metrics
    gta_qm = GTA_Quality_Metrics(edge_list=edge_list, reconstructed_points=reconstructed_points)
    gta_qm.evaluate_metrics()


def main():
    # TODO: args_title is not instantiated if you don't call the parameters (maybe just make a config file with the parameters and call them all)
    args = GraphArgs()
    args.proximity_mode = "knn"
    args.dim = 2
    # print("Proximity_mode after setting to 'knn':", args.proximity_mode)
    # print("Setting false_edges_count to 5...")
    args.false_edges_count = 0   #TODO: this only adds false edges to simulated graphs!
    # print("Proximity_mode after setting false_edges_count to 5:", args.proximity_mode)
    print(args.proximity_mode)
    args.intended_av_degree = 10
    args.num_points = 1000

    ### Creates all the necessary folders!
    args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args


    # # # # #Experimental
    # # subgraph_2_nodes_44_edges_56_degree_2.55.pickle  # subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle  # subgraph_8_nodes_160_edges_179_degree_2.24.pickle
    # # pixelgen_cell_2_RCVCMP0000594.csv, pixelgen_cell_1_RCVCMP0000208.csv, pixelgen_cell_3_RCVCMP0000085.csv
    # # pixelgen_edgelist_CD3_cell_2_RCVCMP0000009.csv, pixelgen_edgelist_CD3_cell_1_RCVCMP0000610.csv, pixelgen_edgelist_CD3_cell_3_RCVCMP0000096.csv
    # # weinstein_data.csv
    # args.proximity_mode = "experimental"  # define proximity mode before name!
    # args.edge_list_title = "weinstein_data.csv"
    #
    # if os.path.splitext(args.edge_list_title)[1] == ".pickle":
    #     write_nx_graph_to_edge_list_df(args)    # activate if format is .pickle file
    #
    # # # Unweighted graph
    # # igraph_graph_original = load_graph(args, load_mode='igraph')
    #
    # # ## Weighted graph
    # igraph_graph_original = load_graph(args, load_mode='igraph', weight_threshold=15)


    # plot_graph_properties(args, igraph_graph_original)  # plots clustering coefficient, degree dist, also stores individual spatial constant...


    # # # 1 Simulation
    create_proximity_graph.write_proximity_graph(args)
    igraph_graph_original = load_graph(args, load_mode='igraph')
    # igraph_graph_original = get_minimum_spanning_tree_igraph(igraph_graph_original)  # careful with activating this
    # plot_graph_properties(args, igraph_graph_original)
    plot_original_or_reconstructed_image(args, image_type='original')


    # # # # Watts-Storgatz
    # # Parameters for the Watts-Strogatz graph
    # p = 0   # Rewiring probability
    # # Create the Watts-Strogatz small-world graph
    # args.dim=1
    # igraph_graph_original = ig.Graph.Watts_Strogatz(args.dim, args.num_points, 1, p)
    # print("graph_created!")


    # #### Run subgraph sampling simulation
    # run_simulation_subgraph_sampling(args, size_interval=100, n_subgraphs=20, graph=igraph_graph_original,
    #                                  add_false_edges=True, add_mst=False)

    ### Run dimension prediction ##TODO: I think this is still in development phase, not working too well?
    # get_dimension_estimation(args, graph=igraph_graph_original, n_samples=20, size_interval=100, start_size=100)  # TODO: start_size matters a lot if not uncertainty


    # #### For Weinstein data get a sample only
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





    ### Reconstruction pipeline
    igraph_graph_original, _ = load_graph(args, load_mode='sparse')
    run_reconstruction(args, sparse_graph=igraph_graph_original, ground_truth_available=True)


    # /home/david/PycharmProjects/Node_Embedding_Clean/Input_Documents/Edge_Lists/subgraph_8_nodes_160_edges_179_degree_2.24.pickle  # simon good viz


    # num_random_edges = 0
    # model_func = "spatial_constant_dim=2"  # small_world
    # # model_func = "small_world"
    #
    #
    # # run_simulation_false_edges(args, max_edges_to_add=100)
    # # run_simulation_graph_growth(args, n_graphs=50, num_random_edges=num_random_edges, model_func=model_func)
    # run_simulation_comparison_large_and_small_world(args, start_n_nodes=500, end_n_nodes=5000, n_graphs=10, num_random_edges_ratio=0.015)

    # run_simulation_graph_growth(args, n_graphs=50, num_random_edges=0, model_func="spatial_constant_dim=2_linearterm")



    ### Many graphs simulation

    # num_points_list = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # # proximity_mode_list = ["knn", "epsilon-ball", "epsilon_bipartite", "knn_bipartite", "delaunay_corrected"]
    # proximity_mode_list = ["knn",   "knn_bipartite", "delaunay_corrected", "epsilon-ball", "epsilon_bipartite"]
    # intended_av_degree_list = [6, 9, 15, 30]
    # false_edges_list = [0, 1, 10, 100]
    # dim_list = [2, 3]


    # # Simple simulation to test stuff
    # num_points_list = [500, 1000]
    # proximity_mode_list = ["knn",   "knn_bipartite"]
    # intended_av_degree_list = [6]
    # false_edges_list = [0]
    # dim_list = [2, 3]


    # spatial_constant_variation_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list)

if __name__ == "__main__":
    main()