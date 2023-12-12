import numpy as np
import pandas as pd

from plots import *
from algorithms import *
from structure_and_args import *
from utils import *
import sys

script_dir = "/home/david/PycharmProjects/Spatial_Graph_Denoising"
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now you can import your module (assuming the file is named your_script.py)
import create_proximity_graph
import itertools
import multiprocessing

def spatial_constant_variation_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list):
    spatial_constant_variation_results = []

    args = GraphArgs()  # caution, this is just to get plot folders but specific graph values are default
    args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
    # Iterate over all combinations of parameters
    for num_points, proximity_mode, dim, false_edges in itertools.product(num_points_list, proximity_mode_list, dim_list, false_edges_list):
        if proximity_mode == "delaunay_corrected":
            # For delaunay_corrected, use only the first value in intended_av_degree_list
            intended_av_degree = intended_av_degree_list[0]
            result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=True)
            spatial_constant_variation_results.append(result)
        else:
            for intended_av_degree in intended_av_degree_list:
                result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=True)
                spatial_constant_variation_results.append(result)

    spatial_constant_variation_results_df = pd.DataFrame(spatial_constant_variation_results)

    # plot_variation_with_num_points(args, spatial_constant_variation_results_df, fixed_proximity_mode='knn', fixed_av_degree=6)
    # plot_variation_with_proximity_mode(args, spatial_constant_variation_results_df, fixed_num_points=1000, fixed_av_degree=6)
    # plot_variation_with_av_degree(args, spatial_constant_variation_results_df, fixed_num_points=1000, fixed_proximity_mode='knn')
    # plot_variation_with_av_degree(args, spatial_constant_variation_results_df)
    plot_spatial_constant_variation(args, spatial_constant_variation_results_df)

    csv_folder = args.directory_map['plots_spatial_constant_variation']
    spatial_constant_variation_results_df.to_csv(f'{csv_folder}/spatial_constant_variation_df.csv')
    print("DF")
    print(spatial_constant_variation_results_df)
    return spatial_constant_variation_results_df


def perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=False):
    args = GraphArgs()

    args.proximity_mode = proximity_mode
    args.num_points = num_points
    args.dim = dim  # assuming dimension is an important parameter
    args.directory_map = create_project_structure()
    args.intended_av_degree = intended_av_degree
    args.false_edges_count = false_edges
    create_proximity_graph.write_proximity_graph(args)



    # Load the graph
    igraph_graph = load_graph(args, load_mode='igraph')
    mean_shortest_path = get_mean_shortest_path(igraph_graph)
    spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path,
                                                            average_degree=args.average_degree, num_nodes=args.num_points)

    if graph_growth:
        if args.dim == 2:
            _, S_fit, r_squared, dim_fit, r_squared_dim = run_simulation_graph_growth(args, n_graphs=20, num_random_edges=0, graph=igraph_graph,
                                                              model_func="spatial_constant_dim=2_linearterm")
        elif args.dim == 3:
            _, S_fit, r_squared, dim_fit, r_squared_dim = run_simulation_graph_growth(args, n_graphs=20, num_random_edges=0, graph=igraph_graph,
                                                              model_func="spatial_constant_dim=3_linearterm")
        else:
            raise ValueError("Input valid dimension")

        spatial_constant_results['S_fit'] = S_fit
        spatial_constant_results['r_squared_S'] = r_squared
        spatial_constant_results['dim_fit'] = 1/dim_fit  # taking the inverse to get the dimension
        spatial_constant_results['r_squared_dim'] = r_squared_dim

    return spatial_constant_results


def get_spatial_constant_results(args, mean_shortest_path, average_degree, num_nodes):

    proximity_mode = args.proximity_mode
    dim = args.dim


    S = mean_shortest_path / ((num_nodes)**(1/dim))  # spatial constant
    K_log = mean_shortest_path / (np.log(num_nodes) / np.log(average_degree))     # small-world constant
    S_general = mean_shortest_path / ((num_nodes ** (1 / dim)) * (average_degree ** (-1 / dim)))

    # Save metrics to CSV
    spatial_constant_results = {
        'proximity_mode': proximity_mode,
        'average_degree': average_degree,
        'intended_av_degree': args.intended_av_degree,
        'mean_shortest_path': mean_shortest_path,
        'num_nodes': num_nodes,
        'dim': dim,
        'S': S,
        'S_general': S_general,
        'S_smallworld_general': K_log
    }
    return spatial_constant_results


# def plot_graph_properties(args, igraph_graph):
#     # Get properties
#     clustering_coefficients = get_local_clustering_coefficients(igraph_graph)
#     degree_distribution = get_degree_distribution(igraph_graph)
#     args.mean_clustering_coefficient = np.mean(clustering_coefficients)
#
#     # Plot them
#     plot_clustering_coefficient_distribution(args, clustering_coefficients)
#     plot_degree_distribution(args, degree_distribution)


def plot_graph_properties(args, igraph_graph):
    """
    Plot clustering coefficient and degree distribution
    Both for unipartite and bipartite grpahs
    """


    if args.is_bipartite:
        # If the graph is bipartite, compute properties separately for each set
        clustering_coefficients_set1, mean_clustering_coefficient_set1, \
            clustering_coefficients_set2, mean_clustering_coefficient_set2 = bipartite_clustering_coefficient_optimized(args,
            igraph_graph)

        degree_distribution_set1, degree_distribution_set2 = get_bipartite_degree_distribution(args, igraph_graph)

        args.mean_clustering_coefficient_set1 = mean_clustering_coefficient_set1
        args.mean_clustering_coefficient_set2 = mean_clustering_coefficient_set2
        args.average_degree_set1 = np.mean(degree_distribution_set1)
        args.average_degree_set1 = (np.sum([element * i for i, element in enumerate(degree_distribution_set1)]) /
                                    np.sum(degree_distribution_set1))

        args.average_degree_set2 = (np.sum([element * i for i, element in enumerate(degree_distribution_set2)]) /
                                    np.sum(degree_distribution_set2))

        # Plot them for both sets
        plot_clustering_coefficient_distribution(args, [clustering_coefficients_set1, clustering_coefficients_set2])
        plot_degree_distribution(args, [degree_distribution_set1, degree_distribution_set2])




    else:
        # For non-bipartite graphs, compute and plot as before
        clustering_coefficients = get_local_clustering_coefficients(igraph_graph)
        degree_distribution = get_degree_distribution(igraph_graph)
        args.mean_clustering_coefficient = np.mean(clustering_coefficients)
        # args.average_degree = np.mean(degree_distribution)

        plot_clustering_coefficient_distribution(args, clustering_coefficients)
        plot_degree_distribution(args, degree_distribution)

    # Shortest paths
    mean_shortest_path, shortest_path_dist = get_mean_shortest_path(igraph_graph, return_all_paths=True)
    plot_shortest_path_distribution(args, shortest_path_dist, mean_shortest_path)

    # Store spatial constant results
    spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path, average_degree=args.average_degree,
                                                            num_nodes=args.num_points)
    spatial_constant_results_df = df = pd.DataFrame([spatial_constant_results])
    df_path = args.directory_map["s_constant_results"]
    spatial_constant_results_df.to_csv(f"{df_path}/s_constant_results_{args.args_title}.csv")


def run_simulation_false_edges(args, max_edges_to_add=10):
    results = []

    # Load the initial graph
    igraph_graph = load_graph(args, load_mode='igraph')

    for i in range(1, max_edges_to_add + 1):
        # Add i random edges
        igraph_graph = add_random_edges_igraph(igraph_graph, i)

        num_nodes = igraph_graph.vcount()
        degrees = igraph_graph.degree()  # This gets the degree of each vertex
        avg_degree = sum(degrees) / num_nodes


        # Compute mean shortest path and other results
        mean_shortest_path = get_mean_shortest_path(igraph_graph)
        spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path, average_degree=avg_degree, num_nodes=num_nodes)
        spatial_constant_results['number_of_random_edges'] = i

        # Append the results to the DataFrame
        results.append(spatial_constant_results)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/spatial_constant_change_with_false_edges_data.csv")
    plot_spatial_constant(args, results_df)
    plot_average_shortest_path(args, results_df)
    return results_df

def run_simulation_graph_growth(args, start_n_nodes=100, n_graphs=10, num_random_edges=0, model_func="spatial_constant_dim=2",
                                graph=None):
    results = []
    if graph == None:
        # Load the initial graph
        igraph_graph = load_graph(args, load_mode='igraph')
    else:
        igraph_graph = graph

    # Add random edges if specified
    if num_random_edges > 0:
        igraph_graph = add_random_edges_igraph(igraph_graph, num_random_edges)

    # Generate subgraphs with BFS
    subgraphs = grow_graph_bfs(igraph_graph, nodes_start=start_n_nodes, nodes_finish=args.num_points, n_graphs=n_graphs)

    for subgraph in subgraphs:
        # Compute mean shortest path and other results
        num_nodes = subgraph.vcount()
        degrees = subgraph.degree()  # This gets the degree of each vertex
        avg_degree = sum(degrees) / num_nodes

        # Compute mean shortest path and other results
        mean_shortest_path = get_mean_shortest_path(subgraph)
        spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path, average_degree=avg_degree, num_nodes=num_nodes)

        # Append the results to the DataFrame
        results.append(spatial_constant_results)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Filename based on whether random edges were added
    filename_suffix = f"_random_edges_{num_random_edges}" if num_random_edges > 0 else ""
    csv_filename = f"spatial_constant_change_with_graph_growth_data_{args.args_title}_{filename_suffix}.csv"
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant_gg']}/{csv_filename}")


    plot_filename_S = f"mean_sp_vs_graph_growth_Sfit_{args.args_title}_{filename_suffix}.png"
    plot_filename_dim = f"mean_sp_vs_graph_growth_dimfit_{args.args_title}_{filename_suffix}.png"


    if "bipartite" in args.proximity_mode:
        if args.dim == 2:
            model_func_dim = "power_model_2d_Sconstant"
        elif args.dim == 3:
            model_func_dim = "power_model_3d_Sconstant"
    else:
        if args.dim == 2:
            model_func_dim = "power_model_2d_bi_Sconstant"
        elif args.dim == 3:
            model_func_dim = "power_model_3d_bi_Sconstant"

    # Plot graph growth and inferred spatial constant with fixed dimension
    fit_S, r_squared_S = plot_mean_sp_with_graph_growth(args, results_df, plot_filename_S, model=model_func,
                                                        return_s_and_r2=True)
    # Plot graph growth and inferred dimension with fixed predicted spatial constant
    fit_dim, r_squared_dim = plot_mean_sp_with_graph_growth(args, results_df, plot_filename_dim, model=model_func_dim,
                                                        return_s_and_r2=True)
    return results_df, fit_S, r_squared_S, fit_dim, r_squared_dim


def run_simulation_subgraph_sampling(args, size_interval=100, n_subgraphs=10, graph=None, add_false_edges=False, add_mst=False):
    if graph is None:
        # Load the initial graph
        igraph_graph = load_graph(args, load_mode='igraph')
    else:
        igraph_graph = graph.copy()  # Create a copy if graph is provided

    size_subgraph_list = np.arange(50, args.num_points, size_interval)
    size_subgraph_list = np.append(size_subgraph_list, args.num_points)
    size_subgraph_list = np.unique(size_subgraph_list)



    if add_mst:
        # Work on a copy of the graph for MST
        igraph_graph_mst = get_minimum_spanning_tree_igraph(igraph_graph.copy())
        args.false_edges_count = -1
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        tasks = [(size_subgraphs, args, igraph_graph_mst, n_subgraphs) for size_subgraphs in size_subgraph_list]
        results = pool.starmap(process_subgraph__bfs_parallel, tasks)
        pool.close()
        pool.join()
        flat_results = [item for sublist in results for item in sublist]
        mst_df = pd.DataFrame(flat_results)



    if add_false_edges:
        all_results = []
        false_edge_list = [0, 5, 20, 100]
        for false_edge_number in false_edge_list:
            args.false_edges_count = false_edge_number
            # Work on a copy of the graph for false edges
            igraph_graph_false = add_random_edges_igraph(igraph_graph.copy(), num_edges_to_add=false_edge_number)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            tasks = [(size_subgraphs, args, igraph_graph_false, n_subgraphs) for size_subgraphs in size_subgraph_list]
            results = pool.starmap(process_subgraph__bfs_parallel, tasks)
            pool.close()
            pool.join()
            flat_results = [item for sublist in results for item in sublist]
            results_df = pd.DataFrame(flat_results)
            all_results.append(results_df)

        if add_mst:
            plot_spatial_constant_against_subgraph_size_with_false_edges(args, all_results, false_edge_list, mst_case_df=mst_df)  # also adding the mst case
            all_results.append(mst_df)

        else:
            plot_spatial_constant_against_subgraph_size_with_false_edges(args, all_results, false_edge_list)
        csv_filename = f"spatial_constant_subgraph_sampling_{args.args_title}_with_false_edges.csv"
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}/{csv_filename}")

    else:
        #     # Generate subgraphs with BFS
        igraph_graph_copy = igraph_graph.copy()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        tasks = [(size_subgraphs, args, igraph_graph_copy, n_subgraphs) for size_subgraphs in size_subgraph_list]
        results = pool.starmap(process_subgraph__bfs_parallel, tasks)
        pool.close()
        pool.join()
        # Flatten the list of results
        flat_results = [item for sublist in results for item in sublist]

        # Create DataFrame from results
        results_df = pd.DataFrame(flat_results)

        csv_filename = f"spatial_constant_subgraph_sampling_{args.args_title}.csv"
        results_df.to_csv(f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}/{csv_filename}")
        plot_sample_spatial_constant(args, results_df)
        plot_spatial_constant_against_subgraph_size(args, results_df)
    return results_df


def process_subgraph__bfs_parallel(size_subgraphs, args, igraph_graph, n_subgraphs):
    results = []
    if size_subgraphs > args.num_points:
        print("Careful, assigned sampling of nodes is greater than total number of nodes!")
        size_subgraphs = args.num_points
    print("size:", size_subgraphs)
    n_subgraphs = 1 if size_subgraphs == args.num_points else n_subgraphs

    subgraphs = get_bfs_samples(igraph_graph, n_graphs=n_subgraphs, min_nodes=size_subgraphs)

    for subgraph in subgraphs:
        num_nodes = subgraph.vcount()
        degrees = subgraph.degree()
        avg_degree = sum(degrees) / num_nodes

        mean_shortest_path = get_mean_shortest_path(subgraph)
        spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path, average_degree=avg_degree,
                                                                num_nodes=num_nodes)
        spatial_constant_results["intended_size"] = size_subgraphs
        results.append(spatial_constant_results)

    return results

def run_simulation_comparison_large_and_small_world(args, start_n_nodes=50, end_n_nodes=1000, n_graphs=10, num_random_edges_ratio=0.015):
    results = []
    node_counts = np.linspace(start_n_nodes, end_n_nodes, n_graphs, dtype=int)
    # args.dim = 3

    for i, node_count in enumerate(node_counts):
        print(f"graph {i}")
        args.num_points = node_count
        create_proximity_graph.write_proximity_graph(args)
        igraph_graph_original = load_graph(args, load_mode='igraph')

        # Series: Original, Almost Regular, Almost Smallworld, Smallworld
        series_ratios = {'Original': 0, 'Quasi Largeworld': num_random_edges_ratio / 10, 'Quasi Smallworld': num_random_edges_ratio / 2, 'Smallworld': num_random_edges_ratio}

        for series_name, ratio in series_ratios.items():
            # Add random edges based on the specified ratio for each series
            num_random_edges = int(ratio * igraph_graph_original.ecount())
            modified_graph = add_random_edges_igraph(igraph_graph_original.copy(), num_random_edges)

            # Compute mean shortest path and other results
            mean_shortest_path = get_mean_shortest_path(modified_graph)
            spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path)
            spatial_constant_results['num_random_edges'] = num_random_edges
            spatial_constant_results['series_type'] = series_name

            # Append the results to the DataFrame
            results.append(spatial_constant_results)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Save the DataFrame
    filename_suffix = f"_smallworld_e={num_random_edges_ratio}"
    csv_filename = f"mean_sp_vs_num_nodes_data{filename_suffix}.csv"
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/{csv_filename}")

    # Plot
    plot_filename = f"mean_sp_vs_num_nodes{filename_suffix}.png"
    plot_mean_sp_with_num_nodes_large_and_smallworld(args, results_df, plot_filename)

    return results_df