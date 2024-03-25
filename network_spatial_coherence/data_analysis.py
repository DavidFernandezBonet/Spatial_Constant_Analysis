import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

from plots import *
from algorithms import *
from structure_and_args import *
from utils import *


import itertools
import multiprocessing
import copy
from functools import partial


import create_proximity_graph
from scipy.stats import linregress

def spatial_constant_variation_analysis(num_points_list, proximity_mode_list, intended_av_degree_list, dim_list, false_edges_list):
    spatial_constant_variation_results = []
    graph_growth = False  # Turn to true if you want fits for the dimension
    args = GraphArgs()  # caution, this is just to get plot folders but specific graph values are default
    args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
    # Iterate over all combinations of parameters
    for num_points, proximity_mode, dim, false_edges in itertools.product(num_points_list, proximity_mode_list, dim_list, false_edges_list):
        if proximity_mode == "delaunay_corrected" or proximity_mode == "lattice":
            # For delaunay_corrected, use only the first value in intended_av_degree_list
            intended_av_degree = intended_av_degree_list[0]
            result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=graph_growth)
            spatial_constant_variation_results.append(result)
        else:
            for intended_av_degree in intended_av_degree_list:
                result = perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=graph_growth)
                spatial_constant_variation_results.append(result)

    spatial_constant_variation_results_df = pd.DataFrame(spatial_constant_variation_results)

    # plot_variation_with_num_points(args, spatial_constant_variation_results_df, fixed_proximity_mode='knn', fixed_av_degree=6)
    # plot_variation_with_proximity_mode(args, spatial_constant_variation_results_df, fixed_num_points=1000, fixed_av_degree=6)
    # plot_variation_with_av_degree(args, spatial_constant_variation_results_df, fixed_num_points=1000, fixed_proximity_mode='knn')
    # plot_variation_with_av_degree(args, spatial_constant_variation_results_df)
    plot_spatial_constant_variation(args, spatial_constant_variation_results_df)

    csv_folder = args.directory_map['plots_spatial_constant_variation']
    spatial_constant_variation_results_df.to_csv(f'{csv_folder}/spatial_constant_variation_df.csv', index=False)
    print("DF")
    print(spatial_constant_variation_results_df)
    return spatial_constant_variation_results_df


def perform_simulation(num_points, proximity_mode, intended_av_degree, dim, false_edges, graph_growth=False):
    args = GraphArgs()

    args.proximity_mode = proximity_mode
    args.num_points = num_points
    args.dim = dim  # assuming dimension is an important parameter
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

    if args.dim == 2:
        if args.is_bipartite:
            mean_shortest_path_prediction = 1.2 * (args.num_points/(args.average_degree*2))**(1/args.dim)
        else:
            mean_shortest_path_prediction = 1.2 * (args.num_points / args.average_degree) ** (1 / args.dim)
    elif args.dim == 3:
        if args.is_bipartite:
            mean_shortest_path_prediction = (1.2*(4/3)) * (args.num_points/(args.average_degree*2))**(1/args.dim)
        else:
            mean_shortest_path_prediction = (1.2*(4/3)) * (args.num_points / args.average_degree) ** (1 / args.dim)

    msp_prediction_error = mean_shortest_path_prediction - mean_shortest_path
    relative_msp_prediction_error = msp_prediction_error / mean_shortest_path

    # Save metrics to CSV
    spatial_constant_results = {
        'proximity_mode': proximity_mode,
        'average_degree': average_degree,
        # 'intended_av_degree': args.intended_av_degree,  # TODO: commenting out this, intended degree should not be relevant
        'mean_shortest_path': mean_shortest_path,
        'num_nodes': num_nodes,
        'dim': dim,
        'S': S,
        'S_general': S_general,
        'S_smallworld_general': K_log,
        'mean_shortest_path_prediction': mean_shortest_path_prediction,
        'msp_prediction_error': msp_prediction_error,
        'relative_msp_prediction_error': relative_msp_prediction_error
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
    Plots various graph properties including clustering coefficients, degree distributions,
    and shortest path distributions. It supports both unipartite and bipartite graphs.
    For bipartite graphs, properties are computed and plotted separately for each set.

    This function also computes and stores spatial constant results based on the mean shortest path
    and average degree of the graph.

    Args:
        args: An object containing configuration parameters and options for the graph analysis. This object
              should include fields for bipartite graph checks (`is_bipartite`), directory mappings
              (`directory_map`), graph titles (`args_title`), and placeholders for results
              (`mean_clustering_coefficient`, `average_degree`, etc.).
        igraph_graph: An igraph graph object. If the graph is not of the desired igraph type, it will be
                      converted within the function.

    Outputs:
        - Plots of clustering coefficient distributions and degree distributions saved to specified directories.
        - A CSV file containing spatial constant results saved in the specified directory.

    Side Effects:
        - Modifies the `args` object by setting various properties such as `mean_clustering_coefficient`,
          `average_degree`, and others depending on whether the graph is bipartite or not.
        - Generates and saves plots to the filesystem.
        - Saves spatial constant results as a CSV file to the filesystem.

    Note:
        - The function relies on several helper functions (`convert_graph_type`,
          `bipartite_clustering_coefficient_optimized`, `get_bipartite_degree_distribution`,
          `plot_clustering_coefficient_distribution`, `plot_degree_distribution`, `get_local_clustering_coefficients`,
          `get_degree_distribution`, `get_mean_shortest_path`, `plot_shortest_path_distribution`,
          `get_spatial_constant_results`) to perform its tasks.
        - Ensure that all necessary fields are present in the `args` object before calling this function.
    """
    igraph_graph = convert_graph_type(args, graph=igraph_graph, desired_type='igraph')

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
    spatial_constant_results_df = pd.DataFrame([spatial_constant_results])
    df_path = args.directory_map["s_constant_results"]
    spatial_constant_results_df.to_csv(f"{df_path}/s_constant_results_{args.args_title}.csv", index=False)


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
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/spatial_constant_change_with_false_edges_data.csv", index=False)
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
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant_gg']}/{csv_filename}", index=False)


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


def run_simulation_subgraph_sampling(args, graph, size_interval=100, n_subgraphs=10, add_false_edges=False,
                                     add_mst=False, parallel=True, false_edge_list=[0,1,2,3,4],
                                     plot_spatial_constant_against_false_edges=False):
    """
     Runs a simulation that samples subgraphs from a given graph (using BFS) to analyze various properties,
     optionally adding minimum spanning trees (MST) and/or false edges to the graph before sampling.
     The function supports parallel processing to speed up computations.

     Args:
         args: An object containing configuration parameters and options for the graph analysis,
               including directory mappings and proximity mode settings.
         graph: An igraph graph object to be analyzed. The graph is converted to the igraph format
                if not already in that format.
         size_interval (int): The interval size for subgraph sampling, determining the range of subgraph
                              sizes to analyze. Defaults to 100.
         n_subgraphs (int): The number of subgraphs to sample at each size interval. Defaults to 10.
         add_false_edges (bool): Whether to add false edges to the graph before sampling. Defaults to False.
         add_mst (bool): Whether to compute and analyze the minimum spanning tree of the graph. Defaults to False.
         false_edge_list (list of int): A list specifying the numbers of false edges to add for each simulation run.
                                        Only relevant if `add_false_edges` is True. Defaults to [0,1,2,3,4].
         plot_spatial_constant_against_false_edges (bool): Whether to plot the spatial constant against the number
                                                           of false edges added. Only relevant if `add_false_edges`
                                                           is True. Defaults to False.

     Returns:
         pandas.DataFrame: A DataFrame containing the aggregated results of the subgraph sampling simulation,
                           including spatial constant calculations for various subgraph sizes and configurations.

     Raises:
         ValueError: If `args` does not contain the necessary configuration for the simulation.

     Note:
         This function modifies the `args` object by updating it with results from the simulation, such as
         mean shortest path and clustering coefficients. Ensure that `args` is properly configured before
         calling this function. It plots the spatial constant plot.

     """

    # TODO: introduce option to not parallelize (can run into memory problems)

    # Needs to be an igraph
    igraph_graph = convert_graph_type(args, graph, desired_type='igraph')

    # # Set parallel = False if you run into memory issues
    # if graph is None:
    #     # Load the initial graph
    #     igraph_graph = load_graph(args, load_mode='igraph')
    # else:
    #     igraph_graph = graph.copy()  # Create a copy if graph is provided

    if args.num_points < 10000:   # for too large of a graph it is difficult to get the full shortest path, #TODO: implement sampling
        size_subgraph_list = np.arange(50, args.num_points, size_interval)
        size_subgraph_list = np.append(size_subgraph_list, args.num_points)
        size_subgraph_list = np.unique(size_subgraph_list)
    else:
        size_subgraph_list = np.arange(50, 3000, size_interval)  # For large graphs we just do up to 3000 nodes






    if add_mst:
        # Work on a copy of the graph for MST
        igraph_graph_mst = get_minimum_spanning_tree_igraph(igraph_graph.copy())
        print("SHORTEST PATH MST", get_mean_shortest_path(igraph_graph_mst))

        # plot the mst (only when ground truth is available)
        if args.proximity_mode != 'experimental':
            node_ids = igraph_graph_mst.vs['name']
            edge_list = igraph_graph_mst.get_edgelist()
            mapped_edge_list = [(node_ids[source], node_ids[target]) for source, target in edge_list]
            print(edge_list)
            edge_df = pd.DataFrame(mapped_edge_list, columns=['source', 'target'])
            edge_list_folder = args.directory_map["edge_lists"]
            edge_df.to_csv(f"{edge_list_folder}/mst_{args.args_title}.csv",index=False)
            plot_original_or_reconstructed_image(args, image_type='mst', edges_df=edge_df)


        args.false_edges_count = -1
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        tasks = [(size_subgraphs, args, igraph_graph_mst, n_subgraphs) for size_subgraphs in size_subgraph_list]
        results = pool.starmap(process_subgraph__bfs_parallel, tasks)
        pool.close()
        pool.join()
        flat_results = [item for sublist in results for item in sublist]
        mst_df = pd.DataFrame(flat_results)



    if add_false_edges:
        max_false_edges = max(false_edge_list)  # Assume false_edge_list is defined
        all_random_false_edges = select_false_edges(igraph_graph, max_false_edges)

        all_results = []
        # false_edge_list = [0, 5, 20, 100]   # now it is a default argument
        for false_edge_number in false_edge_list:
            args_copy = copy.deepcopy(args)
            args_copy.false_edges_count = false_edge_number
            # Work on a copy of the graph for false edges
            # # This adds previously computed false edges, so when you add 1 false edge (a,b) that edge will also be present when you add 2 false edges (a,b) (c,d)
            igraph_graph_false = add_specific_random_edges_igraph(igraph_graph.copy(), all_random_false_edges,
                                                                  false_edge_number)
            # # This adds random edges by the number
            # igraph_graph_false = add_random_edges_igraph(igraph_graph.copy(), num_edges_to_add=false_edge_number)
            pool = multiprocessing.Pool(multiprocessing.cpu_count())


            tasks = [(size_subgraphs, args_copy, igraph_graph_false, n_subgraphs) for size_subgraphs in size_subgraph_list]
            results = pool.starmap(process_subgraph__bfs_parallel, tasks)
            pool.close()
            pool.join()
            flat_results = [item for sublist in results for item in sublist]
            results_df = pd.DataFrame(flat_results)
            all_results.append(results_df)

        if add_mst:
            plot_spatial_constant_against_subgraph_size_with_false_edges(args, all_results, false_edge_list,
                                                                         mst_case_df=mst_df)  # also adding the mst case
            all_results.append(mst_df)

        else:



            # Spatial constant against subgraph size
            plot_spatial_constant_against_subgraph_size_with_false_edges(args, all_results, false_edge_list)

            if plot_spatial_constant_against_false_edges:
                # Spatial constant against false edge count
                processed_false_edge_series = aggregate_spatial_constant_by_size(all_results, false_edge_list)
                plot_false_edges_against_spatial_constant(args, processed_false_edge_series)

        csv_filename = f"spatial_constant_subgraph_sampling_{args.args_title}_with_false_edges.csv"
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}/{csv_filename}", index=False)

    else:   # If we don't add false edges
        #     # Generate subgraphs with BFS
        print("running normal bfs")
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
        results_df.to_csv(f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}/{csv_filename}", index=False)
        # plot_sample_spatial_constant(args, results_df)  # This is an old barplot
        processed_spatial_constant = process_spatial_constant_false_edge_df(combined_df=results_df,
                                                                            false_edge_list=false_edge_list)
        combined_df = processed_spatial_constant

        csv_filename = f"spatial_constant_subgraph_sampling_processed_{args.args_title}.csv"
        results_df.to_csv(f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}/{csv_filename}",
                          index=False)

        plot_spatial_constant_against_subgraph_size(args, results_df)
        combined_df = results_df

    if 'experimental' in args.proximity_mode:
        args.false_edges_count = 0
        args.proximity_mode = 'experimental'


    return combined_df


def process_spatial_constant_false_edge_df(combined_df, false_edge_list):
    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()

    for dataframe, false_edge_count in zip(combined_df, false_edge_list):
        unique_sizes = dataframe['intended_size'].unique()
        means = []
        std_devs = []
        sizes = []

        # Calculate mean and standard deviation for each size
        for size in unique_sizes:
            subset = dataframe[dataframe['intended_size'] == size]
            mean = subset['S_general'].mean()
            std = subset['S_general'].std()
            means.append(mean)
            std_devs.append(std)
            sizes.append(size)

        sizes = np.array(sizes)
        means = np.array(means)
        std_devs = np.array(std_devs)

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(sizes, means)
        r_squared = r_value ** 2  # Coefficient of determination

        # Create a DataFrame for the current false_edge_count results
        temp_df = pd.DataFrame({
            'False Edge Count': false_edge_count,
            'Sizes': sizes,
            'Means': means,
            'Standard Deviation': std_devs,
            'Slope': np.repeat(slope, len(sizes)),
            'R_squared': np.repeat(r_squared, len(sizes))
        })
        # Append the temporary DataFrame to the results DataFrame
        results_df = pd.concat([results_df, temp_df], ignore_index=True)

    spatial_constant_df = results_df
    return spatial_constant_df



def process_subgraph__bfs_parallel(size_subgraphs, args, igraph_graph, n_subgraphs):
    results = []
    if size_subgraphs > args.num_points:
        print("Careful, assigned sampling of nodes is greater than total number of nodes!")
        size_subgraphs = args.num_points

    if args.verbose:
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
        spatial_constant_results["false_edge_number"] = args.false_edges_count
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
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/{csv_filename}", index=False)

    # Plot
    plot_filename = f"mean_sp_vs_num_nodes{filename_suffix}.png"
    plot_mean_sp_with_num_nodes_large_and_smallworld(args, results_df, plot_filename)

    return results_df


def subgraph_sampling_analysis_for_different_weight_thresholds(args, weight_thresholds, edge_list_title):
    combined_dfs = []
    false_edge_counts = []

    # Step 1: Load graph for different weight_threshold values and run simulation
    for wt in weight_thresholds:
        args.edge_list_title = edge_list_title
        igraph_graph_original = load_graph(args, load_mode='igraph', weight_threshold=wt)
        false_edge_count = [0, 5, 20, 100]
        combined_df = run_simulation_subgraph_sampling(args, size_interval=100, n_subgraphs=20, graph=igraph_graph_original,
                                         add_false_edges=True, add_mst=False, false_edge_list=false_edge_count)
        combined_dfs.append(combined_df)
        false_edge_counts.append(false_edge_count)


    # Generate combined plot with subplots
    plot_spatial_constants_subplots(args, combined_dfs, false_edge_counts, weight_thresholds)


def spatial_constant_and_weight_threshold_analysis(args, weight_thresholds, edge_list_title):
    results = []
    variable_of_interest = 'S_general'
    # TODO: ensure that the graph is weighted
    for wt in weight_thresholds:
        print("Weight threshold", wt)
        args.edge_list_title = edge_list_title
        igraph_graph_original = load_graph(args, load_mode='igraph', weight_threshold=wt)
        mean_shortest_path = get_mean_shortest_path(igraph_graph_original)
        sp_results = get_spatial_constant_results(args, mean_shortest_path=mean_shortest_path,
                                                  average_degree=args.average_degree, num_nodes=args.num_points)

        spatial_constant = sp_results[variable_of_interest]
        results.append((wt, spatial_constant))

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=['Weight Threshold', variable_of_interest])
    df_folder = args.directory_map['plots_spatial_constant_weighted_threshold']
    output_file = f'{df_folder}/spatial_constant_vs_weight_threshold{args.args_title}.csv'  # Define your output file path here
    df.to_csv(output_file, index=False)

    # Now results is a list of tuples (weight_threshold, mean_s_general)
    plot_s_general_vs_weight_threshold(args, results)


def get_dimension_estimation(args, graph, n_samples=20, size_interval=100, start_size=200):
    """
    n_samples: how many subgraphs to get for the same subgraph size (gain statistical power)
    size_interval: interval distance between the sizes
    """

    # Compute all the sizes list
    if args.num_points < 10000:   # for too large of a graph it is difficult to get the full shortest path, #TODO: implement sampling
        size_subgraph_list = np.arange(start_size, args.num_points, size_interval)
        size_subgraph_list = np.append(size_subgraph_list, args.num_points)
        size_subgraph_list = np.unique(size_subgraph_list)
    else:
        size_subgraph_list = np.arange(start_size, 3000, size_interval)


    mean_shortest_path = get_mean_shortest_path(graph)
    sp_results_big = get_spatial_constant_results(args, mean_shortest_path=mean_shortest_path,
                                                  average_degree=args.average_degree, num_nodes=args.num_points)

    # Generate local statistics by performing several BFS sampling
    igraph_graph_copy = graph.copy()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tasks = [(size_subgraphs, args, igraph_graph_copy, n_samples) for size_subgraphs in size_subgraph_list]
    sp_results_df_list = pool.starmap(process_subgraph__bfs_parallel, tasks)
    flat_sp_results_df_list = [item for sublist in sp_results_df_list for item in sublist]
    sp_results_dataframe = pd.DataFrame(flat_sp_results_df_list)

    csv_plot_folder = args.directory_map['plots_predicted_dimension']
    sp_results_dataframe.to_csv(f"{csv_plot_folder}/dimension_prediction_{args.args_title}.csv", index=False)



    # Group resulting data
    sizes, means_L, std_devs_L = compute_mean_std_per_group(sp_results_dataframe, 'intended_size', 'mean_shortest_path')
    # sizes, means_k, std_devs_k = compute_mean_std_per_group(sp_results_df, 'intended_size', 'average_degree')
    std_devs_L = np.nan_to_num(std_devs_L)
    std_devs_L[-1] = 0.1

    # Perform the fit
    # TODO: introduce y_error (right now not working with the fit. probably std 0)
    ### This gets priority to the final points, as 1st points have more uncertainty... Doesn't work well with Weinstein data, but well with simulated?
    curve_fit = CurveFitting(x_data=sizes, y_data=means_L, y_error_std=std_devs_L)
    print("STD", std_devs_L)
    ### Fit does not consider stds here
    # curve_fit = CurveFitting(x_data=sizes, y_data=means_L)


    fixed_a = sp_results_big['S']  # Fixed parameter
    # fixed_a_model_func = partial(curve_fit.power_model, a=fixed_a)
    print("final S", sp_results_big['S'])

    # fixed_a_model_func = curve_fit.power_model_2d_Sconstant


    fixed_a_model_func = curve_fit.power_model  # 2 parameters - constant and exponent

    curve_fit.perform_curve_fitting(model_func=fixed_a_model_func)
    plot_folder = args.directory_map['plots_predicted_dimension']
    curve_fit.plot_fit_with_uncertainty(model_func=fixed_a_model_func, xlabel='N', ylabel='Mean Shortest Path',
                                        title='Mean Shortest Path vs. N - Dim fit',
                                        save_path=f'{plot_folder}/dimension_prediction_{args.args_title}.pdf')


def aggregate_spatial_constant_by_size(dataframes, false_edge_list):
    """
    WHen adding false edges, we have a "series" for every size of subgraph.
    The series has "spatial constant" as y-axis and "false edge count" as x-axis
    This function takes the dataframe and neatly returns each individual series by size ( a dictionary of dictionaries)
    1st dictionary: size: series
    2nd dictionary: "false_edges": false_edge_list, "means": spatial_constant_mean_list, "std": ...
    """
    # Determine the unique intended_subgraph_sizes across all dataframes
    all_sizes = set()
    for df in dataframes:
        all_sizes.update(df['intended_size'].unique())

    # Initialize a dictionary to hold the aggregated data
    aggregated_data = {}

    for size in sorted(all_sizes):
        means = []
        std_devs = []
        false_edges = []

        # Iterating over each false edge count
        for false_edge_count in false_edge_list:
            combined_subset = None

            # Combine data from all dataframes for the current false edge count
            for dataframe in dataframes:
                subset = dataframe[(dataframe['intended_size'] == size) &
                                   (dataframe['false_edge_number'] == false_edge_count)]
                if combined_subset is None:
                    combined_subset = subset
                else:
                    combined_subset = pd.concat([combined_subset, subset])

            if combined_subset is not None and not combined_subset.empty:
                # Calculate mean and standard deviation of S_general
                mean = combined_subset['S_general'].mean()
                std = combined_subset['S_general'].std()

                means.append(mean)
                std_devs.append(std)
                false_edges.append(false_edge_count)

        # Store the aggregated data
        aggregated_data[size] = {
            'false_edges': false_edges,
            'means': means,
            'std_devs': std_devs
        }

    return aggregated_data


def compute_several_sp_matrices(args, sparse_graph, false_edges_list):
    sp_matrices = []
    max_false_edges = max(false_edges_list)  # Assume false_edge_list is defined
    all_random_false_edges = select_false_edges_csr(sparse_graph, max_false_edges, args=args)

    for num_edges in false_edges_list:
        modified_graph = add_specific_random_edges_to_csrgraph(sparse_graph.copy(), all_random_false_edges, num_edges)
        shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(modified_graph)
        sp_matrices.append(shortest_path_matrix)
    return sp_matrices


def compute_average_distance(points):
    """
    Compute the average distance between points in the given array.
    """
    if len(points) < 2:
        return 0
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1))
    # Exclude the diagonal (distance to self) and divide by 2 (since matrix is symmetric)
    avg_distance = np.sum(distances) / (len(points) * (len(points) - 1))
    return avg_distance


def get_spatial_constant_euclidean_df(args, positions_array, size_interval, num_samples=10):
    final_results = []
    size_threshold_list = np.arange(50, args.num_points, size_interval)

    for size_threshold in size_threshold_list:
        spatial_constants = []  # Collect spatial constants for current size_threshold

        for _ in range(num_samples):
            # Pick a random node and create a subset
            random_index = random.randint(0, positions_array.shape[0] - 1)
            random_node = positions_array[random_index]

            # Calculate distances and sort indices
            distances = np.linalg.norm(positions_array - random_node, axis=1)
            sorted_indices = np.argsort(distances)
            subset_indices = sorted_indices[:size_threshold]
            subset = positions_array[subset_indices]

            # Compute average distance
            avg_distance = compute_average_distance(subset)

            print(f"Average distance: {avg_distance} at size_threshold: {size_threshold}")

            # Compute spatial constant
            number_points = len(subset)
            spatial_constant = avg_distance * ((args.num_points ) / number_points) ** (1 / args.dim)
            # spatial_constant = avg_distance * (((args.num_points)/(np.pi)) / number_points) ** (1 / args.dim)

            print(f"Spatial constant: {spatial_constant} at size_threshold: {size_threshold}")

            spatial_constants.append(spatial_constant)

        # Calculate mean and standard deviation of spatial constants for the current size_threshold
        mean_spatial_constant = np.mean(spatial_constants)
        std_spatial_constant = np.std(spatial_constants)

        # Store the mean and std for each size_threshold
        final_results.append({
            'size_threshold': size_threshold,
            'mean_spatial_constant': mean_spatial_constant,
            'std_spatial_constant': std_spatial_constant
        })

    # Convert final results to DataFrame
    results_df = pd.DataFrame(final_results)
    return results_df


def plot_euc_spatial_constant_against_size_threshold(args, results_df):
    plt.figure(figsize=(10, 6))

    # Data from results_df
    sizes = results_df['size_threshold'].values
    means = results_df['mean_spatial_constant'].values
    std_devs = results_df['std_spatial_constant'].values

    # Scatter plot for mean spatial constants
    plt.scatter(sizes, means, label='Mean Spatial Constant', color='blue')

    # Ribbon style for standard deviation
    ribbon_color = '#ADD8E6'
    contour_ribbon_color = '#00008B'
    plt.fill_between(sizes, means - std_devs, means + std_devs, color=ribbon_color, alpha=0.3,
                     edgecolor=contour_ribbon_color, linewidth=1, linestyle='--')

    plt.xlabel('Subgraph Size')
    plt.ylabel('Mean Spatial Constant')
    plt.title('Mean Spatial Constant vs. Subgraph Size')
    plt.legend()

    # If there's a specific plot folder defined in args, save the plot there
    # plot_folder = args.get('directory_map', {}).get('plots_spatial_constant', 'current_directory')
    # plt.savefig(f"{plot_folder}/mean_spatial_constant_vs_size_threshold_{args.get('args_title', 'default')}.png")
    plt.show()


def plot_spatial_constant_euc_vs_network(args, results_df_euc, results_df_net, useful_plot_folder):

    plt.figure(figsize=(6, 4.5))

    ### Euclidean
    # Data from results_df_euc
    sizes_euc = results_df_euc['size_threshold'].values
    means_euc = results_df_euc['mean_spatial_constant'].values
    std_devs_euc = results_df_euc['std_spatial_constant'].values

    # Scatter plot and ribbon for mean spatial constants (Euclidean)



    plt.plot(sizes_euc, means_euc, label='Euclidean', color='#00CD6C', marker='o')
    plt.fill_between(sizes_euc, means_euc - std_devs_euc, means_euc + std_devs_euc, color='#00CD6C', alpha=0.3,)

    ### Network
    # Data from results_df_net

    unique_sizes = results_df_net['intended_size'].unique()
    means = []
    std_devs = []
    sizes = []

    # Calculate mean and standard deviation for each size
    for size in unique_sizes:
        subset = results_df_net[results_df_net['intended_size'] == size]
        mean = subset['S_general'].mean()
        std = subset['S_general'].std()
        means.append(mean)
        std_devs.append(std)
        sizes.append(size)


    sizes_net = np.array(sizes)
    means_net = np.array(means)
    std_devs_net = np.array(std_devs)


    plt.plot(sizes_net, means_net, label='Network', color='#009ADE', marker='o')
    plt.fill_between(sizes_net, means_net - std_devs_net, means_net + std_devs_net, color='#009ADE', alpha=0.3)

    plt.xlabel('Subgraph Size')
    plt.ylabel('Mean Spatial Constant')
    plt.legend()


    # Save the figure
    plot_folder = f"{args.directory_map['plots_spatial_constant_subgraph_sampling']}"
    plt.savefig(f"{plot_folder}/mean_spatial_constant_euc_vs_network_{args.args_title}.svg")
    plot_folder2 = f"{args.directory_map['spatial_coherence']}"
    plt.savefig(f"{plot_folder2}/mean_spatial_constant_euc_vs_network_{args.args_title}.svg")
    plt.savefig(f"{useful_plot_folder}/mean_spatial_constant_euc_vs_network_{args.args_title}.svg")
    plt.show()


def calculate_figsize_n_subplots(n_subplots, base_subplot_size=(4, 3), additional_height_per_subplot=0.5, top_margin=2):
    """
    Calculates the figsize for a figure based on the number of central nodes.

    Parameters:
    - n_central_nodes: The number of central nodes (int).
    - base_subplot_size: A tuple representing the width and height (in inches) of each subplot.
    - additional_height_per_subplot: Extra height (in inches) to add per subplot for labels, titles, etc.
    - top_margin: Extra space (in inches) to add at the top for the overall figure.

    Returns:
    - A tuple representing the figsize (width, height) in inches.
    """
    base_width, base_height = base_subplot_size
    total_height = n_subplots * (base_height + additional_height_per_subplot) + top_margin
    figsize = (base_width, total_height)
    return figsize