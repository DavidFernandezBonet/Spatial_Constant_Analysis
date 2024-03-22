# Default configuration template

# Base settings common to all scenarios
base = {   'colorfile': 'dna_cool2.png',
    'dim': 2,
    'false_edges_count': 0,
    'handle_all_subgraphs': False,
    'large_graph_subsampling': True,
    'max_subgraph_size': 500,
    'plot_graph_properties': False,
    'proximity_mode': 'lattice',
    'reconstruct': False,
    'reconstruction_mode': 'STRND',
    'show_plots': True,
    'spatial_coherence_validation': {   'gram_matrix': True,
                                        'network_dimension': True,
                                        'spatial_constant': True},
    'verbose': True}

# Settings specific to simulation scenarios
simulation = {'intended_av_degree': 10, 'num_points': 1000, 'plot_original_image': False}

# Settings specific to experimental scenarios
experiment = {   'edge_list_title': 'example_edge_list.pickle',
    'weight_threshold': 0,
    'weighted': False}
