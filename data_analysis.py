import numpy as np

def get_spatial_constant_results(args, mean_shortest_path):

    proximity_mode = args.proximity_mode
    dim = args.dim
    num_nodes = args.num_points

    K_sp = mean_shortest_path / ((num_nodes)**(1/dim))  # spatial constant
    K_log = mean_shortest_path / np.log(num_nodes)      # small-world constant

    # Save metrics to CSV
    spatial_constant_results = {
        'proximity_mode': proximity_mode,
        'average_degree': args.average_degree,  # TODO: compute average degree
        'mean_shortest_path': mean_shortest_path,
        'num_nodes': num_nodes,
        'dim': dim,
        'K_sp': K_sp,
        'K_smallworld': K_log
    }
    return spatial_constant_results