# Base settings common to all scenarios
base = {
    "proximity_mode": "lattice",  # Options 1) Simulation: knn, epsilon_ball, lattice, delaunay_corrected... 2) experimental
    "dim": 2,
    "false_edges_count": 0,
    "colorfile": None,  # For coloring reconstruction. Alternatives: colorful_spiral.jpeg, colored_squares.png, dna.jpg, dna_cool2.png, None
    "plot_graph_properties": False,

    "large_graph_subsampling": False,   # If the graph is large, subsample it to save time and memory. Cap at 3000 nodes  #TODO: implement this
    "max_subgraph_size": 3000,
    "reconstruct": True,
    "reconstruction_mode": "node2vec",

    "spatial_coherence_validation": False,
    "handle_all_subgraphs": False,
}

if base['proximity_mode'] != 'experimental':
    # Settings specific to simulation scenarios
    simulation = {
        "num_points": 1000,
        "intended_av_degree": 10,
        'plot_original_image': True
    }

else:
    # Settings specific to experimental scenarios
    experiment = {
        # pixelgen_example_graph.csv  #edge_list_nbead_0_filtering_march_8.csv
        "edge_list_title": "edge_list_nbead_0_filtering_march_8.csv",  # edge_list_distance_150_filtering_goodindex_simon.csv, nbead_7_goodindex_simon.csv, edge_list_nbead_4_filtering.csv
        "weighted": False,
        "weight_threshold": 0,
    }