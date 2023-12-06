import sys
import os
import matplotlib.pyplot as plt
script_dir = "/home/david/PycharmProjects/Spatial_Graph_Denoising"
# Add this directory to the Python path
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now you can import your module (assuming the file is named your_script.py)
import create_proximity_graph

from algorithms import *
from utils import *
from data_analysis import *
from plots import *
from metrics import *

import scienceplots
plt.style.use(['science', 'nature'])






# TODO: args_title is not instantiated if you don't call the parameters (maybe just make a config file with the parameters and call them all)
args = GraphArgs()
print("Initial proximity_mode:", args.proximity_mode)
print("Setting proximity_mode to 'knn'...")
args.proximity_mode = "knn"
print("Proximity_mode after setting to 'knn':", args.proximity_mode)
print("Setting false_edges_count to 5...")
args.false_edges_count = 5
print("Proximity_mode after setting false_edges_count to 5:", args.proximity_mode)
print(args.proximity_mode)
args.intended_av_degree = 6
args.num_points = 1000
print(args.args_title)
args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args


# # Experimental
# # subgraph_2_nodes_44_edges_56_degree_2.55.pickle  # subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle  # subgraph_8_nodes_160_edges_179_degree_2.24.pickle
# args.edge_list_title = "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle"
# write_nx_graph_to_edge_list_df(args)

# # # 1 Simulation
# create_proximity_graph.write_proximity_graph(args)


# igraph_graph_original = load_graph(args, load_mode='igraph')
# plot_graph_properties(args, igraph_graph_original)

#### Reconstruction pipeline
# igraph_graph_original, _ = load_graph(args, load_mode='sparse')
# print(type(igraph_graph_original))
#
# reconstruction = ImageReconstruction(graph=igraph_graph_original, dim=2)
# reconstructed_points = reconstruction.reconstruct()
#
#
# edge_list = read_edge_list(args)
# original_points = read_position_df(args)
#
# qm = QualityMetrics(original_points, reconstructed_points)
# qm.evaluate_metrics()
#
# gta_qm = GTA_Quality_Metrics(edge_list=edge_list, reconstructed_points=reconstructed_points)
# gta_qm.evaluate_metrics()


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



#### Many graphs simulation
# num_points_list = [1000, 2000, 3000]
num_points_list = [1000, 2000, 5000, 10000]
# proximity_mode_list = ["knn", "epsilon-ball", "epsilon_bipartite", "knn_bipartite", "delaunay_corrected"]
proximity_mode_list = ["knn",   "knn_bipartite", "delaunay_corrected", "epsilon-ball", "epsilon_bipartite"]
intended_av_degree_list = [6, 9, 15, 30]
false_edges_list = [0, 1, 10, 100]
dim_list = [2, 3]
spatial_constant_variation_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list)

