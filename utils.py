import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.sparse import find
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

def get_largest_component_sparse(args, sparse_graph, original_node_ids):
    n_components, labels = connected_components(csgraph=sparse_graph, directed=False, return_labels=True)
    if n_components > 1:  # If not connected
        print("Disconnected graph! Finding largest component...")
        # Find the largest component
        largest_component_label = np.bincount(labels).argmax()
        component_node_indices = np.where(labels == largest_component_label)[0]
        component_node_ids = original_node_ids[component_node_indices]
        largest_component = sparse_graph[component_node_indices][:, component_node_indices]

        args.num_points = largest_component.shape[0]
        print("Size of largest connected component:", args.num_points)
        # Largeset component to an edge list
        rows, cols, _ = find(largest_component)
        edges = list(zip(rows, cols))
        edge_df = pd.DataFrame(edges, columns=['source', 'target'])
        edge_list_folder = args.directory_map["edge_lists"]
        args.edge_list_title = f"edge_list_{args.args_title}.csv"
        edge_df.to_csv(f"{edge_list_folder}/{args.edge_list_title}", index=False)

        if args.colorcode:  # We are only interested in keeping the indices if we want to plot colors in principle
            # Component ids to dictionary
            node_id_mapping = {old_id: new_index for new_index, old_id in enumerate(component_node_ids)}
            args.node_ids_map_old_to_new = node_id_mapping

        return largest_component, component_node_ids
    return sparse_graph, original_node_ids

def get_largest_component_igraph(args, igraph_graph, weighted=False):
    components = igraph_graph.clusters()
    if len(components) > 1:
        print("Disconnected Graph!")
        largest = components.giant()

        args.num_points = largest.vcount()
        # Write the new edge list with largest component
        edges = largest.get_edgelist()
        weights = largest.es['weight']  # Access the weights of the edges

        edge_df = pd.DataFrame(edges, columns=['source', 'target'])
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

def check_edge_list_columns(edge_list_df):
    # Define allowed columns
    allowed_columns = {'source', 'target', 'weight', 'source (seq)', 'target (seq)'}
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

    if 'source (seq)' in edge_list_df.columns:
        edge_list_df = edge_list_df.drop('source (seq)', axis=1)
    if 'target (seq)' in edge_list_df.columns:
        edge_list_df = edge_list_df.drop('target (seq)', axis=1)

    print("Edge list columns are valid.")
    return edge_list_df
def load_graph(args, load_mode='igraph', weight_threshold=0):
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

    file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    df = pd.read_csv(file_path)  # edge list


    df = check_edge_list_columns(edge_list_df=df)
    args.original_title = args.args_title

    # TODO: check that source is not contained in target and viceversa
    # Convert columns to sets
    source_set = set(df['source'])
    target_set = set(df['target'])

    # Maximum value for each set
    max_source = max(source_set)
    max_target = max(target_set)

    # Check if sets go from 0 to N
    source_sequence_check = source_set == set(range(max_source + 1))
    target_sequence_check = target_set == set(range(max_target + 1))

    # Intersection
    intersection = source_set.intersection(target_set)

    # Percentage of intersection
    percentage_source = (len(intersection) / len(source_set)) * 100
    percentage_target = (len(intersection) / len(target_set)) * 100

    print((source_sequence_check, target_sequence_check, len(intersection), percentage_source, percentage_target))

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
        # TODO: bipartite stuff
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
        largest_component, component_node_ids = get_largest_component_sparse(args, sparse_graph, original_node_ids)
        # Compute average degree
        degrees = largest_component.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        args.num_points = largest_component.shape[0]
        args.component_node_ids = component_node_ids

        return largest_component, component_node_ids

    elif load_mode == "sparse_weighted":
        largest_component, component_node_ids = load_graph_sparse_weighted(args, df=df)
        args.component_node_ids = component_node_ids
        return largest_component, component_node_ids

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



def add_random_edges_to_csrgraph(csr_graph, num_edges_to_add):
    """
    Add a specified number of random edges to a graph in CSR format.

    :param csr_graph: Graph in CSR format.
    :param num_edges_to_add: Number of random edges to add.
    :param max_weight: Maximum weight of the edges to be added.
    :return: Graph in CSR format with added edges.
    """
    lil_graph = csr_graph.tolil()  # Convert to LIL format for easier modifications
    num_nodes = lil_graph.shape[0]

    for _ in range(num_edges_to_add):
        # Randomly select two different nodes
        node_a, node_b = np.random.choice(num_nodes, 2, replace=False)

        # Add an edge between these nodes with a random weight

        lil_graph[node_a, node_b] = 1
        lil_graph[node_b, node_a] = 1

    # Convert back to CSR format
    return lil_graph.tocsr()

def add_specific_random_edges_to_csrgraph(csr_graph, false_edges_ids, num_edges_to_add):
    """
    Add a specified number of random edges to a graph in CSR format.

    :param csr_graph: Graph in CSR format.
    :param num_edges_to_add: Number of random edges to add.
    :param max_weight: Maximum weight of the edges to be added.
    :return: Graph in CSR format with added edges.
    """
    lil_graph = csr_graph.tolil()  # Convert to LIL format for easier modifications
    num_nodes = lil_graph.shape[0]
    edges_to_add = false_edges_ids[:num_edges_to_add]


    for edge in edges_to_add:
        # Randomly select two different nodes
        node_a, node_b = edge[0], edge[1]

        # Add an edge between these nodes with a random weight

        lil_graph[node_a, node_b] = 1
        lil_graph[node_b, node_a] = 1

    # Convert back to CSR format
    return lil_graph.tocsr()


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
    largest_component, component_node_ids = get_largest_component_sparse(args, sparse_graph, unique_nodes)

    # Compute average degree: sum of weights divided by number of nodes
    degrees = largest_component.sum(axis=0).A1  # Sum of non-zero entries in each column (or row)
    average_degree = np.mean(degrees) / 2  # Adjust for symmetric duplication
    args.average_degree = average_degree
    args.num_points = largest_component.shape[0]
    args.component_node_ids = component_node_ids

    return largest_component, component_node_ids

