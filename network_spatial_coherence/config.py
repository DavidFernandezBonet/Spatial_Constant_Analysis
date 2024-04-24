# Base settings common to all scenarios
base = {
    "proximity_mode": "knn_bipartite",  # Options 1) Simulation: knn, epsilon-ball, lattice, delaunay_corrected... 2) experimental
    "dim": 2,
    "false_edges_count": 0,
    "true_edges_deletion_ratio": 0,
    "colorfile": None,  # For coloring reconstruction. Alternatives: colorful_spiral.jpeg, colored_squares.png, dna.jpg, dna_cool2.png, weinstein_colorcode_february_corrected.csv, None
    "plot_graph_properties": False,
    "show_plots": False,

    "large_graph_subsampling": False,   # If the graph is large, subsample it to save time and memory. Cap at 3000 nodes  #TODO: implement this
    "max_subgraph_size": 500,
    "reconstruct": False,
    "reconstruction_mode": "STRND",  # STRND, ggvec, landmark_isomap, PyMDE, MDS

    "spatial_coherence_validation": {"spatial_constant": False, "network_dimension": True, "gram_matrix": True},
    "community_detection": False,
    "handle_all_subgraphs": False,
    'plot_original_image': True,
    "verbose": False
}


# Settings specific to simulation scenarios
simulation = {
    "num_points": 1000,
    "intended_av_degree": 10,
}


# Settings specific to experimental scenarios
experiment = {
    # pixelgen_example_graph.csv  #edge_list_nbead_0_filtering_march_8.csv, # edge_list_us_counties.csv # weinstein_data_corrected_february.csv,
    "edge_list_title": "edge_list_us_counties.csv",  # example_edge_list.pickle, edge_list_distance_150_filtering_goodindex_simon.csv, nbead_7_goodindex_simon.csv, edge_list_nbead_4_filtering.csv, subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle (erik's data)
    "weighted": False,
    "weight_threshold": 15,
    "original_positions_available": True,
}
