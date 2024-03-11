# Default configuration template

# Base settings common to all scenarios
base = {   'colorfile': None,
    'dim': 2,
    'false_edges_count': 0,
    'handle_all_subgraphs': False,
    'large_graph_subsampling': False,
    'max_subgraph_size': 3000,
    'plot_graph_properties': False,
    'proximity_mode': 'lattice',
    'reconstruct': True,
    'reconstruction_mode': 'node2vec',
    'show_plots': False,
    'spatial_coherence_validation': False}

# Settings specific to simulation scenarios
simulation = {'intended_av_degree': 10, 'num_points': 1000, 'plot_original_image': True}

# Settings specific to experimental scenarios
experiment = {   'edge_list_title': 'edge_list_nbead_0_filtering_march_8.csv',
    'weight_threshold': 0,
    'weighted': False}
