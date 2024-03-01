# Base settings common to all scenarios
base = {
    "proximity_mode": "delaunay_corrected",  # Options 1) Simulation: knn, epsilon_ball, lattice, delaunay_corrected... 2) experimental
    "dim": 2,
    "false_edges_count": 0,
    "colorfile": 'dna_cool2.png',  # For coloring reconstruction. Alternatives: colorful_spiral.jpeg, colored_squares.png, dna.jpg None
    "plot_graph_properties": False,

    "large_graph_subsampling": True,   # If the graph is large, subsample it to save time and memory. Cap at 3000 nodes  #TODO: implement this
    "max_subgraph_size": 3000,
    "reconstruct": False,
    "reconstruction_mode": "node2vec"
}

if base['proximity_mode'] != 'experimental':
    # Settings specific to simulation scenarios
    simulation = {
        "num_points": 10000,
        "intended_av_degree": 15,
        'plot_original_image': False
    }

else:
    # Settings specific to experimental scenarios
    experiment = {
        "edge_list_title": "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle",  # edge_list_distance_150_filtering_goodindex_simon.csv
        "weighted": False,
        "weight_threshold": 0,
    }