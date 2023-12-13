import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
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

def get_largest_component_sparse(sparse_graph, original_node_ids):
    n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
    if n_components > 1:  # If not connected
        # Find the largest component
        largest_component_label = np.bincount(labels).argmax()
        component_node_indices = np.where(labels == largest_component_label)[0]
        component_node_ids = original_node_ids[component_node_indices]
        largest_component = sparse_graph[component_node_indices][:, component_node_indices]
        return largest_component, component_node_ids
    return sparse_graph, original_node_ids

def get_largest_component_igraph(args, igraph_graph):
    components = igraph_graph.clusters()
    if len(components) > 1:
        print("Disconnected Graph!")
        largest = components.giant()


        # Write the new edge list with largest component
        edges = largest.get_edgelist()
        edge_df = pd.DataFrame(edges, columns=['source', 'target'])
        edge_list_folder = args.directory_map["edge_lists"]
        edge_df.to_csv(f"{edge_list_folder}/edge_list_{args.args_title}.csv")

        return largest
    return igraph_graph

def get_largest_component_networkx(networkx_graph):
    if not nx.is_connected(networkx_graph):
        largest_component = max(nx.connected_components(networkx_graph), key=len)
        subgraph = networkx_graph.subgraph(largest_component).copy()
        return subgraph
    return networkx_graph

def read_edge_list(args):
    file_path = f"{args.directory_map['edge_lists']}/edge_list_{args.args_title}.csv"
    edge_list_df = pd.read_csv(file_path)
    return edge_list_df


def read_position_df(args):
    original_points_path = f"{args.directory_map['original_positions']}/positions_{args.args_title}.csv"
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

def check_edge_list_columns(edge_list_df):
    # Define allowed columns
    allowed_columns = {'source', 'target', 'weight'}
    mandatory_columns = {'source', 'target'}

    # Check for extra columns
    extra_columns = set(edge_list_df.columns) - allowed_columns
    if extra_columns:
        raise ValueError(f"Extra columns found: {extra_columns}")

    # Check for mandatory columns
    if not mandatory_columns.issubset(edge_list_df.columns):
        missing_columns = mandatory_columns - set(edge_list_df.columns)
        raise ValueError(f"Mandatory columns missing: {missing_columns}")

    # Check if 'weight' column exists
    if 'weight' in edge_list_df.columns:
        print("Column 'weight' exists. Threshold filtering will be performed with minimum weight...")
    else:
        print("Column 'weight' does not exist. Continuing with unweighted graph procedure")

    print("Edge list columns are valid.")
def load_graph(args, load_mode='igraph', input_file_type='edge_list', weight_threshold=0):
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


    # TODO: implement different input files, e.g. edge list, pickle networkx...
    # TODO: update edge list if graph is disconnected!
    # TODO: false edge implementation for other types apart from igraph? Is it necessary¿
    # TODO: implementation for weighed graph
    file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    df = pd.read_csv(file_path)
    check_edge_list_columns(edge_list_df=df)
    args.original_title = args.args_title

    # Handling of weighted graphs, for now just a simple threshold
    if "weight" in df.columns:
        if weight_threshold == None:
            raise ValueError("Please select a weight threshold when calling the load_graph function. It can be 0"
                             "(same effect as no threshold)")
        else:
            print(f"Weighted graphs will be treated as unweighted with a minimum weight threshold filtering = {weight_threshold}")
            # Plot weight distribution here
            plot_weight_distribution(args, edge_list_with_weight_df=df)
            df = df[df["weight"] > weight_threshold]


    if load_mode == 'sparse':
        # TODO: this returns also the node_ids as sparse matrices do not keep track of them. If it is used be aware you  need the IDs
        n_nodes = df.max().max() + 1  # Assuming the nodes start from 0
        # Create symmetric edge list: add both (source, target) and (target, source)
        edges = np.vstack([df[['source', 'target']].values, df[['target', 'source']].values])
        data = np.ones(len(edges))  # Edge weights (1 for each edge)

        # Create sparse matrix
        sparse_graph_coo = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
        # Convert COO matrix to CSR format
        sparse_graph = sparse_graph_coo.tocsr()

        original_node_ids = np.arange(n_nodes)
        largest_component, component_node_ids = get_largest_component_sparse(sparse_graph, original_node_ids)
        degrees = largest_component.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        args.num_points = largest_component.shape[0]
        return largest_component, component_node_ids

    elif load_mode == 'igraph':
        tuples = [tuple(x) for x in df.values]
        igraph_graph = ig.Graph.TupleList(tuples, directed=False)
        largest_component = get_largest_component_igraph(args, igraph_graph)
        degrees = largest_component.degree()
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        args.num_points = largest_component.vcount()
        print("average degree", average_degree)
        print("num points", args.num_points)

        # Check bipartitedness
        is_bipartite, types = largest_component.is_bipartite(return_types=True)
        args.is_bipartite = is_bipartite
        if is_bipartite:
            args.bipartite_sets = types

        # # Add false edges if necessary  #TODO: how to guarantee that the false edges are in the largest component? Might be unlucky
        # if args.false_edges_count:  #TODO: adapt for bipartite case
        #     print("ha passat")
        #     largest_component = add_random_edges_igraph(largest_component, args.false_edges_count)

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






