import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.sparse import find
from scipy.sparse import csr_matrix, isspmatrix

import scipy.stats
import igraph as ig
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
import pickle
import os

from algorithms import *
from plots import plot_weight_distribution
from plots import plot_original_or_reconstructed_image
from structure_and_args import GraphArgs
import warnings
import re
import json
from scipy.spatial.distance import pdist

def get_largest_component_sparse(args, sparse_graph, original_node_ids, weighted=False):
    n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
    if n_components > 1:  # If not connected
        if not args.handle_all_subgraphs:   # Just get the largest component if we don't handle all subgraphs
            print("Disconnected (or disordered) graph! Finding largest component...")
            num_nodes = sparse_graph.shape[0]  # or sparse_graph.shape[1], as it should be a square matrix

            # Find the largest component
            largest_component_label = np.bincount(labels).argmax()
            component_node_indices = np.where(labels == largest_component_label)[0]
            component_node_ids = original_node_ids[component_node_indices]
            largest_component = sparse_graph[component_node_indices][:, component_node_indices]

            args.num_points = largest_component.shape[0]
            if args.verbose:
                print("Size of the total graph", num_nodes)
                print("Size of largest connected component:", args.num_points)
            # Largeset component to an edge list
            rows, cols, weight = find(largest_component)

            if not weighted:
                edges = list(zip(rows, cols))
            else:
                edges = list(zip(rows, cols, weight))

            if weighted:
                columns = ['source', 'target', 'weight']
            else:
                columns = ['source', 'target']
            edge_df = pd.DataFrame(edges, columns=columns)
            edge_list_folder = args.directory_map["edge_lists"]
            args.edge_list_title = f"edge_list_{args.args_title}.csv"
            edge_df.to_csv(f"{edge_list_folder}/{args.edge_list_title}", index=False)

            if args.colorcode or args.reconstruct:  # Interested in indices if we want to reconstruct the graph
                # Component ids to dictionary
                node_id_mapping = {old_id: new_index for new_index, old_id in enumerate(component_node_ids)}
                args.node_ids_map_old_to_new = node_id_mapping
                # store also the old edge list
                node_ids_map_new_to_old = {new_index: old_id for new_index, old_id in enumerate(component_node_ids)}
                old_index_edge_df = pd.DataFrame()
                old_index_edge_df['source'] = edge_df['source'].map(node_ids_map_new_to_old)
                old_index_edge_df['target'] = edge_df['target'].map(node_ids_map_new_to_old)
                if weighted:
                    old_index_edge_df['weight'] = edge_df['weight']
                old_index_edge_df.to_csv(f"{edge_list_folder}/old_index_{args.edge_list_title}", index=False)

            # Update largest component properties
            degrees = largest_component.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
            average_degree = np.mean(degrees)
            args.average_degree = average_degree
            if args.verbose:
                print(f"Average Degree sparse: {average_degree}")
            args.num_points = largest_component.shape[0]
            args.component_node_ids = component_node_ids

            if args.false_edges_count and not args.false_edge_ids:  # TODO: adapt for bipartite case
                largest_component = add_random_edges_to_csrgraph(args, largest_component, args.false_edges_count)
                print(args.false_edge_ids)

            # Save the graph in "args"
            args.sparse_graph = largest_component

            return largest_component

        else:   # handling multiple subgraphs
            args_subgraph_list = []
            for component_label in range(n_components):

                ### Have a graph args for every subgraph
                args_subgraph = GraphArgs(override_config_path=args.override_config_path, data_dir=args.data_dir)
                component_node_indices = np.where(labels == component_label)[0]
                component_node_ids = original_node_ids[component_node_indices]
                component_sparse = sparse_graph[component_node_indices][:, component_node_indices]
                args_subgraph.num_points = component_sparse.shape[0]

                if args_subgraph.num_points < 30:  # only interested in subgraphs with more than 30 points
                    continue

                if weighted:
                    columns = ['source', 'target', 'weight']
                else:
                    columns = ['source', 'target']
                args_subgraph.num_points = component_sparse.shape[0]
                print("Size of the connected component:", args_subgraph.num_points)
                # Largeset component to an edge list
                rows, cols, weight = find(component_sparse)
                if not weighted:
                    edges = list(zip(rows, cols))
                else:
                    edges = list(zip(rows, cols, weight))
                edge_df = pd.DataFrame(edges, columns=columns)
                edge_list_folder = args_subgraph.directory_map["edge_lists"]
                args_subgraph.edge_list_title = f"edge_list_{args_subgraph.args_title}_component_{component_label}.csv"
                edge_df.to_csv(f"{edge_list_folder}/{args_subgraph.edge_list_title}", index=False)


                # Component ids to dictionary
                node_id_mapping = {old_id: new_index for new_index, old_id in enumerate(component_node_ids)}
                args_subgraph.node_ids_map_old_to_new = node_id_mapping

                # store also the old edge list
                node_ids_map_new_to_old = {new_index: old_id for new_index, old_id in enumerate(component_node_ids)}
                old_index_edge_df = pd.DataFrame()
                old_index_edge_df['source'] = edge_df['source'].map(node_ids_map_new_to_old)
                old_index_edge_df['target'] = edge_df['target'].map(node_ids_map_new_to_old)
                if weighted:
                    # Directly copy the 'weight' column from the original DataFrame
                    old_index_edge_df['weight'] = edge_df['weight']
                old_index_edge_df.to_csv(f"{edge_list_folder}/old_index_{args_subgraph.edge_list_title}", index=False)


                ### Add subgraph properties
                degrees = component_sparse.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
                average_degree = np.mean(degrees)
                args_subgraph.average_degree = average_degree
                if args.verbose:
                    print(f"Average Degree sparse: {average_degree}")
                args_subgraph.component_node_ids = component_node_ids

                if args_subgraph.false_edges_count and not args.false_edge_ids:  # TODO: adapt for bipartite case
                    component_sparse = add_random_edges_to_csrgraph(args_subgraph, component_sparse,
                                                                     args_subgraph.false_edges_count)
                    if args.verbose:
                        print(args_subgraph.false_edge_ids)

                args_subgraph.sparse_graph = component_sparse
                args_subgraph_list.append(args_subgraph)

            # Returns a list of graph args, containing the graph in "sparse_graph",
            # the node ids in "node_ids_map_old_to_new"
            return args_subgraph_list

    else:   # If the graph is completely connected
        degrees = sparse_graph.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        if args.verbose:
            print(f"Average Degree sparse: {average_degree}")
        args.num_points = sparse_graph.shape[0]

        if args.false_edges_count and not args.false_edge_ids:  # TODO: adapt for bipartite case
            sparse_graph = add_random_edges_to_csrgraph(args, sparse_graph, args.false_edges_count)
            if args.verbose:
                print(args.false_edge_ids)

        # Save the graph in "args"
        args.sparse_graph = sparse_graph
        return sparse_graph

def get_largest_component_igraph(args, igraph_graph, weighted=False):
    components = igraph_graph.clusters()
    if len(components) > 1:
        print("Disconnected Graph!")
        largest = components.giant()

        args.num_points = largest.vcount()
        # Write the new edge list with largest component
        edges = largest.get_edgelist()


        edge_df = pd.DataFrame(edges, columns=['source', 'target'])

        if weighted:
            weights = largest.es['weight']  # Access the weights of the edges
            edge_df['weight'] = weights

        edge_list_folder = args.directory_map["edge_lists"]
        args.edge_list_title = f"edge_list_{args.args_title}.csv"
        edge_df.to_csv(f"{edge_list_folder}/{args.edge_list_title}", index=False)

        # TODO: make sure that this works
        component_node_ids = [node['name'] for node in largest.vs]
        if args.colorcode:  # We are only interested in keeping the indices if we want to plot colors in principle
            # Component ids to dictionary
            node_id_mapping = {old_id: new_index for new_index, old_id in enumerate(component_node_ids)}
            args.node_ids_map_old_to_new = node_id_mapping

        return largest
    return igraph_graph

def get_largest_component_networkx(networkx_graph):
    if not nx.is_connected(networkx_graph):
        largest_component = max(nx.connected_components(networkx_graph), key=len)
        subgraph = networkx_graph.subgraph(largest_component).copy()
        return subgraph
    return networkx_graph

def read_edge_list(args):
    file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    edge_list_df = pd.read_csv(file_path)

    return edge_list_df


def read_position_df(args, return_df=False):
    if args.proximity_mode == "experimental" and args.original_positions_available:
        filename = args.original_edge_list_title
        print(f"Original edge list: {filename}")
        if "weinstein" in filename:
            old_args_title = filename[:-4]
            original_points_path = f"{args.directory_map['original_positions']}/positions_weinstein_data_corrected_february.csv"
        else:
            match = re.search(r"edge_list_(.*?)\.csv", filename)
            if match:
                extracted_part = match.group(1)

            old_args_title = extracted_part
            original_points_path = f"{args.directory_map['original_positions']}/positions_{old_args_title}.csv"
    else:
        original_points_path = args.positions_path
    original_points_df = pd.read_csv(original_points_path)
    # Choose columns based on the dimension specified in args.dim
    if args.dim == 2:
        columns_to_read = ['x', 'y']
    elif args.dim == 3:
        columns_to_read = ['x', 'y', 'z']
    else:
        raise ValueError("Invalid dimension specified. Choose '2D' or '3D'.")

    # Read the specified columns from the DataFrame
    original_points_array = np.array(original_points_df[columns_to_read])

    if return_df:
        return original_points_df
    else:
        return original_points_array


def write_nx_graph_to_edge_list_df(args):
    # Load the graph from the pickle file
    print(args.edge_list_title)
    pickle_file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    with open(pickle_file_path, 'rb') as f:
        G = pickle.load(f)

    edge_list = G.edges()

    edge_df = pd.DataFrame(edge_list, columns=["source", "target"])

    # Splitting the filename and extension
    new_edge_list_name, _ = os.path.splitext(args.edge_list_title)

    args.edge_list_title = new_edge_list_name + ".csv"

    edge_df.to_csv(f"{args.directory_map['edge_lists']}/{args.edge_list_title}", index=False)
    return


def write_nx_graph_to_edge_list_df(args):
    # Load the graph from the pickle file
    print(args.edge_list_title)
    pickle_file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    with open(pickle_file_path, 'rb') as f:
        G = pickle.load(f)

    edge_list = G.edges()

    # Initial DataFrame with original source and target
    edge_df = pd.DataFrame(edge_list, columns=["source", "target"])

    # Creating a mapping for unique sequences
    unique_nodes = pd.unique(edge_df[['source', 'target']].values.ravel('K'))
    node_to_int = {node: idx for idx, node in enumerate(unique_nodes)}

    # Apply the mapping to create new columns and rename original columns
    edge_df['source (seq)'] = edge_df['source']
    edge_df['target (seq)'] = edge_df['target']
    edge_df['source'] = edge_df['source (seq)'].map(node_to_int)
    edge_df['target'] = edge_df['target (seq)'].map(node_to_int)



    # Splitting the filename and extension
    new_edge_list_name, _ = os.path.splitext(args.edge_list_title)

    # Saving the modified DataFrame
    args.edge_list_title = new_edge_list_name + ".csv"
    edge_df.to_csv(f"{args.directory_map['edge_lists']}/{args.edge_list_title}", index=False)
    return args

def check_edge_list_columns(args, edge_list_df):
    # Define allowed columns
    allowed_columns = {'source', 'target', 'weight', 'source (seq)', 'target (seq)'}
    mandatory_columns = {'source', 'target'}

    # Check for extra columns
    extra_columns = set(edge_list_df.columns) - allowed_columns
    if extra_columns:
        warnings.warn(f"Extra columns found: {extra_columns}")

    # Check for mandatory columns
    if not mandatory_columns.issubset(edge_list_df.columns):
        missing_columns = mandatory_columns - set(edge_list_df.columns)
        raise ValueError(f"Mandatory columns missing: {missing_columns}")

    if args.verbose:
        if 'weight' in edge_list_df.columns:
            print("Column 'weight' exists. Threshold filtering will be performed with minimum weight...")
        else:
            print("Unweighted graph")

    if 'source (seq)' in edge_list_df.columns:
        edge_list_df = edge_list_df.drop('source (seq)', axis=1)
    if 'target (seq)' in edge_list_df.columns:
        edge_list_df = edge_list_df.drop('target (seq)', axis=1)

    edge_list_df[['source', 'target']] = edge_list_df.apply(lambda x: sorted([x['source'], x['target']]), axis=1, result_type='expand')
    duplicates = edge_list_df.duplicated(keep=False)  # 'keep=False' marks all duplicates as True
    if duplicates.any():
        warnings.warn("Duplicate edges found in the network. Please check the edge list.")
        print(edge_list_df[duplicates])
    else:
        if args.verbose:
            print("No duplicates found.")

    if args.verbose:
        print("Edge list columns are valid.")
    return edge_list_df
def load_graph(args, load_mode='sparse'):
    """
        Load a graph from an edge list CSV file, compute its average degree, and
        update the provided args object with the average degree and the number of
        nodes in the largest connected component of the graph.

        If the graph is not fully connected, only the largest connected component
        is considered for the computation of the average degree and the number of nodes.

        Parameters:
        - args: An object that must have a 'directory_map' attribute, which is a
                dictionary with keys including 'edge_lists', and an 'args_title'
                attribute that is used to construct the file path for the CSV.
                This object will be updated with 'average_degree' and 'num_points'
                attributes reflecting the loaded graph's properties.
        - load_mode (str): The mode for loading the graph. Supported values are
                           'sparse', 'igraph', and 'networkx'.

        Returns:
        - For 'sparse': A tuple of the largest connected component as a sparse matrix
                        and an array of node IDs corresponding to the original graph.
        - For 'igraph': The largest connected component as an igraph graph.
        - For 'networkx': The largest connected component as a NetworkX graph.

        Raises:
        - ValueError: If an invalid load_mode is provided.

        Side Effects:
        - The 'args' object is updated with 'average_degree' and 'num_points' attributes.
        """

    # TODO: implement different input files, e.g. edge list, pickle networkx... (csv and pickle compatible now)
    # TODO: update edge list if graph is disconnected! Done for igraph and sparse
    # TODO: false edge implementation for other types apart from igraph? Is it necessary¿
    # TODO: implementation for weighed graph

    if args.edge_list_title == None:
        raise ValueError('Please make sure that a) the edgelist in the data/edge_lists folder and b)'
                         'the name of the edgelist is correct.')


    if os.path.splitext(args.edge_list_title)[1] == ".pickle":
        write_nx_graph_to_edge_list_df(args)  # write a .csv edge list if it is in pickle format

    file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    df = pd.read_csv(file_path)  # edge list
    df = check_edge_list_columns(args=args, edge_list_df=df)

    if args.original_title is None:
        if hasattr(args, 'args_title'):
            args.original_title = args.args_title
        else:
            args.original_title = None  # Example: Assigning a default value of None

    # # TODO: check that source is not contained in target and viceversa
    # # Convert columns to sets
    # source_set = set(df['source'])
    # target_set = set(df['target'])
    #
    # # Maximum value for each set
    # max_source = max(source_set)
    # max_target = max(target_set)
    #
    # # Check if sets go from 0 to N
    # source_sequence_check = source_set == set(range(max_source + 1))
    # target_sequence_check = target_set == set(range(max_target + 1))
    #
    # # Intersection
    # intersection = source_set.intersection(target_set)
    #
    # # Percentage of intersection
    # percentage_source = (len(intersection) / len(source_set)) * 100
    # percentage_target = (len(intersection) / len(target_set)) * 100

    # print((source_sequence_check, target_sequence_check, len(intersection), percentage_source, percentage_target))


    # Handling of weighted graphs, for now just a simple threshold
    if "weight" in df.columns:
        if args.weight_threshold == None:
            raise ValueError("Please select a weight threshold when calling the load_graph function. It can be 0"
                             "(same effect as no threshold)")
        else:
            print(f"Weighted graphs will be treated as unweighted with a minimum weight threshold filtering = {args.weight_threshold}")
            # Plot weight distribution here

            if args.plot_graph_properties:
                plot_weight_distribution(args, edge_list_with_weight_df=df)
                if args.original_positions_available:
                    positions_file = f"positions_{args.network_name}.csv"
                    if "weinstein" in args.network_name:
                        args.colorfile = 'weinstein_colorcode_february_corrected.csv'
                    plot_original_or_reconstructed_image(args, image_type='original', edges_df=df,  position_filename=positions_file,
                                                         plot_weights_against_distance=True)
            df = df[df["weight"] > args.weight_threshold]

            if load_mode == 'sparse' and args.weight_to_distance:
                load_mode = 'sparse_weighted'


    if load_mode == 'sparse':
        # TODO: bipartite stuff
        # TODO: this returns also the node_ids as sparse matrices do not keep track of them. If it is used be aware you  need the IDs
        n_nodes = int(df[['source', 'target']].max().max()) + 1  # Assuming the nodes start from 0
        # Create symmetric edge list: add both (source, target) and (target, source)
        edges = np.vstack([df[['source', 'target']].values, df[['target', 'source']].values])
        data = np.ones(len(edges))  # Edge weights (1 for each edge)
        # Create sparse matrix
        sparse_graph_coo = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
        # Convert COO matrix to CSR format
        sparse_graph = sparse_graph_coo.tocsr()
        original_node_ids = np.arange(n_nodes)
        # TODO: careful, if we have "handle_all_subgraphs" activated, this will return a list of GraphArgs objects
        # TODO: if not, it will return a single *sparse_graph* object
        largest_component = get_largest_component_sparse(args, sparse_graph, original_node_ids)

        if args.large_graph_subsampling and args.num_points > args.max_subgraph_size:
            warnings.warn(
                f"Large graph. Subsampling using BFS for efficiency purposes.\nSize of the sample; {args.max_subgraph_size}")
            largest_component = sample_csgraph_subgraph(args, largest_component, min_nodes=args.max_subgraph_size)
        return largest_component

    elif load_mode == "sparse_weighted":
        # TODO: copy other functions for sparse, such as adding false edges
        largest_component = load_graph_sparse_weighted(args, df=df)

        if not args.weight_to_distance:
            args.sparse_graph = largest_component
        else:    # If we transform weights to distances
            if args.proximity_mode == "experimental":
                # TODO: set properly the decay_rate in experimental case
                args.weight_converter.decay_rate = 1/(7/100000)   # seems to not matter a lot, but would be nice with right units

                # TODO: maybe the maximum weight should be even higher than the maximum observed weight!
                # args.weight_converter.max_weight = np.quantile(largest_component.data, 0.95)  # choose 95% percentile for weight

                if "weinstein" in args.edge_list_title:
                    args.weight_converter.max_weight = 1326  # maximum weight for all the dataset
                else:
                    args.weight_converter.max_weight = np.max(largest_component.data)
                print("Max weight", args.weight_converter.max_weight)



            # args.weight_converter.decay_rate = 0.01
            # args.weight_converter.max_weight = 100
            # args.weight_converter.max_weight = 10000000
            # args.weight_converter.decay_rate = 0.00001

            transformed_data = np.vectorize(args.weight_converter.return_distance_exponential_model)(
                largest_component.data)
            largest_component.data = transformed_data
            args.sparse_graph = largest_component

            if args.proximity_mode == "experimental":
                # TODO: if the edge list is not symmetric (duplicate edges) then you will run into index length problems
                ### Adding the "distance" column additional to the "weight" column here
                edge_list_folder = args.directory_map["edge_lists"]
                edge_df = pd.read_csv(f"{edge_list_folder}/{args.edge_list_title}")
                edge_df['distance'] = transformed_data
                edge_df.to_csv(f"{edge_list_folder}/{args.edge_list_title}", index=False)

                old_edge_df = pd.read_csv(f"{edge_list_folder}/old_index_{args.edge_list_title}")
                old_edge_df['distance'] = transformed_data
                old_edge_df.to_csv(f"{edge_list_folder}/old_index_{args.edge_list_title}", index=False)



        return largest_component

    elif load_mode == 'igraph':
        if "weight" in df.columns:
            weighted = True
            tuples = [tuple(x) for x in df[['source', 'target']].values]
            igraph_graph = ig.Graph.TupleList(tuples, directed=False, edge_attrs=None)
            igraph_graph.es['weight'] = df['weight'].tolist()
        else:
            weighted = False
            tuples = [tuple(x) for x in df.values]
            igraph_graph = ig.Graph.TupleList(tuples, directed=False)


        largest_component = get_largest_component_igraph(args, igraph_graph, weighted=weighted)

        degrees = largest_component.degree()
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        args.num_points = largest_component.vcount()    # TODO: calling args.num_points changes the edge list title
        if args.verbose:
            print("average degree igraph", average_degree)
            print("num points", args.num_points)


        # Check bipartitedness
        is_bipartite, types = largest_component.is_bipartite(return_types=True)
        args.is_bipartite = is_bipartite
        if is_bipartite:
            args.bipartite_sets = types

        # Add false edges if necessary
        if args.false_edges_count and not args.false_edge_ids:  #TODO: adapt for bipartite
            largest_component = add_random_edges_igraph(args, largest_component, args.false_edges_count)

        # Save the graph in "args"
        args.igraph_graph = largest_component

        return largest_component

    elif load_mode == 'networkx':
        networkx_graph = nx.from_pandas_edgelist(df, 'source', 'target')
        if not nx.is_connected(networkx_graph):
            largest_cc = max(nx.connected_components(networkx_graph), key=len)
            networkx_graph = networkx_graph.subgraph(largest_cc).copy()
        args.average_degree = sum(dict(networkx_graph.degree()).values()) / float(networkx_graph.number_of_nodes())
        args.num_points = networkx_graph.number_of_nodes()
        return networkx_graph

    else:
        raise ValueError("Invalid load_mode. Choose 'sparse', 'igraph', or 'networkx'.")





def add_random_edges_to_csrgraph(args, csr_graph, num_edges_to_add):
    """
    Add a specified number of random edges to a graph in CSR format.

    :param csr_graph: Graph in CSR format.
    :param num_edges_to_add: Number of random edges to add.
    :param max_weight: Maximum weight of the edges to be added.
    :return: Graph in CSR format with added edges.
    """
    # TODO: adapt for weighted case (see below, done for spatial constant)
    lil_graph = csr_graph.tolil()  # Convert to LIL format for easier modifications
    num_nodes = lil_graph.shape[0]

    for _ in range(num_edges_to_add):
        # Randomly select two different nodes
        node_a, node_b = np.random.choice(num_nodes, 2, replace=False)
        args.false_edge_ids.append((node_a, node_b))
        lil_graph[node_a, node_b] = 1
        lil_graph[node_b, node_a] = 1

    # Convert back to CSR format
    return lil_graph.tocsr()

def add_specific_random_edges_to_csrgraph(csr_graph, false_edges_ids, num_edges_to_add, weighted=False, args=None):
    """
    Add a specified number of random edges to a graph in CSR format.

    :param csr_graph: Graph in CSR format.
    :param num_edges_to_add: Number of random edges to add.
    :param max_weight: Maximum weight of the edges to be added.
    :return: Graph in CSR format with added edges.
    """
    lil_graph = csr_graph.tolil()  # Convert to LIL format for easier modifications
    edges_to_add = false_edges_ids[:num_edges_to_add]


    if weighted and args is None:
        raise ValueError("args cannot be None if weighted is True")
    if weighted:
        total_weight = int((num_edges_to_add * args.weight_converter.max_weight) / 3)  # weight budget
        weights = distribute_weights(num_edges_to_add, total_weight, args.weight_converter.max_weight)
    for i, edge in enumerate(edges_to_add):
        # Randomly select two different nodes
        node_a, node_b = edge[0], edge[1]
        if weighted:
            weight = weights[i]
            distance = args.weight_converter.return_distance_exponential_model(weight)
            # TODO: make sure this is what you want, it is adding a "false distance" based on "weight"
            lil_graph[node_a, node_b] = distance
            lil_graph[node_b, node_a] = distance
        else:
            lil_graph[node_a, node_b] = 1
            lil_graph[node_b, node_a] = 1
    # Convert back to CSR format
    return lil_graph.tocsr()




def distribute_weights(num_edges, total_weight, max_weight, min_weight=1):
    if num_edges * max_weight < total_weight or num_edges * min_weight > total_weight:
        raise ValueError("Total weight cannot be achieved with given constraints")

    # Even initial distribution within the allowed range
    weights = np.full(num_edges, min_weight)
    remaining_weight = total_weight - np.sum(weights)

    # Distribute remaining weight with random adjustments within the bounds
    while remaining_weight > 0:
        for i in np.random.permutation(num_edges):
            if remaining_weight <= 0:
                break
            increment = min(max_weight - weights[i], remaining_weight)
            if increment > 0:
                added_weight = np.random.randint(1, increment + 1)
                weights[i] += added_weight
                remaining_weight -= added_weight

    # Randomly adjust to exact total if over or under due to min/max restrictions
    adjustment = sum(weights) - total_weight
    while adjustment != 0:
        for i in np.random.permutation(num_edges):
            if adjustment == 0:
                break
            if adjustment > 0 and weights[i] > min_weight:  # Need to decrease some weights
                decrement = min(weights[i] - min_weight, adjustment)
                weights[i] -= decrement
                adjustment -= decrement
            elif adjustment < 0 and weights[i] < max_weight:  # Need to increase some weights
                increment = min(max_weight - weights[i], -adjustment)
                weights[i] += increment
                adjustment += increment
    return weights

def remove_false_edges_igraph(graph, false_edges):
    for edge in false_edges:
        # Find the edge based on 'name' attribute of the nodes
        source = graph.vs.find(name=edge[0]).index
        target = graph.vs.find(name=edge[1]).index

        # Check if the edge exists and then delete it
        if graph.are_connected(source, target):
            edge_id = graph.get_eid(source, target)
            graph.delete_edges(edge_id)

    return graph

def validate_edge_list_numbers(edge_list, reconstructed_positions):
    """
    Validate the edge list.

    Parameters:
    edge_list (pd.DataFrame): DataFrame containing the edge list with 'source' and 'target' columns.
    reconstructed_positions (list or array): List or array of reconstructed positions.

    Returns:
    bool: True if the edge list is valid, False otherwise.
    """
    n = len(reconstructed_positions) - 1
    expected_set = set(range(n + 1))

    # Create a set of all values in 'source' and 'target'
    edge_values = set(edge_list['source']).union(set(edge_list['target']))

    if edge_values == expected_set:
        return True, "Edge list is valid."

    missing = expected_set - edge_values
    extra = edge_values - expected_set

    mismatch_info = []
    if missing:
        mismatch_info.append(f"Missing nodes: {missing}")
    if extra:
        mismatch_info.append(f"Extra nodes: {extra}")

    return False, "; ".join(mismatch_info)


def load_graph_sparse_weighted(args, df):
    """
    Load a weighted graph from an edge list dataframe, compute its average degree, and
    update the provided args object with the average degree and the number of nodes in the
    largest connected component of the graph.

    Parameters:
    - df: DataFrame containing the edge list with columns ['source', 'target', 'weight']
    - args: An object to be updated with graph properties

    Returns:
    - A tuple of the largest connected component as a sparse matrix and an array of node IDs
    """
    # Ensure nodes are labeled from 0 to n-1
    unique_nodes = np.union1d(df['source'].unique(), df['target'].unique())
    n_nodes = unique_nodes.size
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}

    # Map original node IDs to new, continuous range
    mapped_edges = np.vectorize(node_mapping.get)(df[['source', 'target']].values)
    weights = df['weight'].values

    # Create symmetric edge list: add both (source, target) and (target, source) with their weights
    edges_symmetric = np.vstack([mapped_edges, mapped_edges[:, [1, 0]]])
    weights_symmetric = np.hstack([weights, weights])  # Duplicate weights for symmetry

    # Create weighted sparse graph
    sparse_graph_coo = coo_matrix((weights_symmetric, (edges_symmetric[:, 0], edges_symmetric[:, 1])),
                                  shape=(n_nodes, n_nodes))

    # Convert COO matrix to CSR format for efficiency
    sparse_graph = sparse_graph_coo.tocsr()

    # Extract largest connected component, update args, and compute graph properties
    largest_component = get_largest_component_sparse(args, sparse_graph, unique_nodes, weighted=True)

    # Compute average degree: sum of weights divided by number of nodes
    degrees = largest_component.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
    average_degree = np.mean(degrees) / 2  # Adjust for symmetric duplication
    args.average_degree = average_degree
    args.num_points = largest_component.shape[0]


    return largest_component


def convert_graph_type(args, graph, desired_type="igraph"):
    def is_verbose():
        # Check if args is not None and verbose attribute is set to True
        return args is not None and getattr(args, 'verbose', False)

    if desired_type == "igraph":
        # Check if the graph is not an igraph instance and needs conversion
        if not isinstance(graph, ig.Graph):
            if is_verbose():
                print("Graph is not an igraph instance. Converting to igraph...")
            # Check for preloaded igraph graph in args
            igraph_graph = getattr(args, 'igraph_graph', None) if args else None
            if igraph_graph is None:
                igraph_graph = load_graph(args, load_mode="igraph")
            return igraph_graph
        else:
            return graph

    elif desired_type == "sparse":
        # Check if the graph is already a sparse matrix
        if not isspmatrix(graph):
            if is_verbose():
                print(f"Original graph format/type: {type(graph).__name__}")
                print("Graph is not a csrgraph instance. Converting to csrgraph...")
            # Check for preloaded sparse graph in args
            sparse_graph = getattr(args, 'sparse_graph', None) if args else None
            if sparse_graph is None:
                sparse_graph = load_graph(args, load_mode='sparse')
            return sparse_graph
        else:
            return graph
    else:
        raise ValueError("Unsupported desired graph type. Currently, only 'sparse' and 'igraph' types are supported.")


def write_edge_list_sparse_graph(args, sparse_graph):
    rows, cols = sparse_graph.nonzero()
    edges = [(i, j) for i, j in zip(rows, cols) if i < j]
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_list_folder = args.directory_map["edge_lists"]
    edge_list_title = f"{args.edge_list_title}"
    edge_df.to_csv(f"{edge_list_folder}/{edge_list_title}", index=False)

def reload_graph_from_edge_list(args, edge_list, new_edge_list_title):

    # TODO: adapt for weighted grpahs
    # 1 - save edge list as a csv with columns source and target
    edge_list_folder = args.directory_map["edge_lists"]
    edge_df = pd.DataFrame(edge_list, columns=['source', 'target'])
    args.edge_list_title = new_edge_list_title  # update the edge list title
    edge_df.to_csv(f"{edge_list_folder}/{new_edge_list_title}", index=False)

    # # 2 - Update the node IDs mapping
    # if args.node_ids_map_old_to_new is not None:  # TODO: case where we have originally a disconnected subgraph. Not sure if this works, and it is relevant for experimental data that is disconnected
    #     # Reverse the mapping to get from new (csgraph) indices to old (original) indices
    #     new_to_old_index_mapping = {new: old for old, new in args.node_ids_map_old_to_new.items()}
    #     args.node_ids_map_old_to_new = {old: new for old, new in args.node_ids_map_old_to_new.items() if
    #                                     new in unique_nodes}
    #     original_indices = [new_to_old_index_mapping.get(idx, idx) for idx in sorted_visited_nodes]
    args.sparse_graph = load_graph(args, load_mode='sparse')

    args.num_edges = args.sparse_graph.nnz // 2
    args.num_points = args.sparse_graph.shape[0]

    ## should I comment it out? Does it do anything
    args.shortest_path_matrix = compute_shortest_path_matrix_sparse_graph(args.sparse_graph, None)
    args.mean_shortest_path = args.shortest_path_matrix.mean()
    return args.sparse_graph, args


def write_network_in_json_format(args, positions_df, edges_df, network_type):
    # Determine if the positions are in 2D or 3D
    if 'z' in positions_df.columns:
        node_fields = ['node_ID', 'x', 'y', 'z']
    else:
        positions_df['z'] = 0  # Add a 'z' column with 0 values if it is a 2D DataFrame
        node_fields = ['node_ID', 'x', 'y', 'z']


    # Calculate distances
    coordinates = positions_df[['x', 'y', 'z']].to_numpy()
    distances = pdist(coordinates)  # Compute the pairwise distances between rows
    median_distance = np.median(distances)

    # Determine scale factor and val based on the median distance
    scale_factor = 100 / median_distance
    val = 0.01

    # Apply scaling
    positions_df['x'] *= scale_factor
    positions_df['y'] *= scale_factor
    positions_df['z'] *= scale_factor
    positions_df['val'] = val
    positions_df['name'] = positions_df['node_ID']

    # Validate the positions DataFrame
    for field in node_fields:
        if field not in positions_df.columns:
            raise ValueError(f"Missing required field in positions DataFrame: {field}")

    # Validate the edges DataFrame
    if 'source' not in edges_df.columns or 'target' not in edges_df.columns:
        raise ValueError("Edges DataFrame must contain 'source' and 'target' columns")

    # Convert positions DataFrame to the nodes list
    nodes = positions_df.rename(columns={'node_ID': 'id'}).to_dict(orient='records')

    # Convert edges DataFrame to the links list
    links = edges_df.to_dict(orient='records')

    # Create the graph dictionary
    graph_data = {
        "nodes": nodes,
        "links": links
    }

    # Construct the file path
    data_folder = args.directory_map["json"]
    file_path = f'{data_folder}/json_format_{network_type}_{args.args_title}.json'

    # Write the graph dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(graph_data, f, indent=4)

    print(f"Graph JSON file saved to {file_path}")

