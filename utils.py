import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import igraph as ig
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
import scienceplots
plt.style.use(['science', 'ieee'])
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

def get_largest_component_igraph(igraph_graph):
    components = igraph_graph.clusters()
    if len(components) > 1:
        print("Disconnected Graph!")
        largest = components.giant()
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

def load_graph(args, load_mode='igraph'):
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


    # TODO: update edge list if graph is disconnected!
    file_path = f"{args.directory_map['edge_lists']}/edge_list_{args.args_title}.csv"
    df = pd.read_csv(file_path)


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
        largest_component = get_largest_component_igraph(igraph_graph)
        degrees = largest_component.degree()
        average_degree = np.mean(degrees)
        args.average_degree = average_degree
        args.num_points = len(largest_component.vs)
        print("average degree", average_degree)
        print("num points", args.num_points)
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


class CurveFitting:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.popt = None
        self.pcov = None
        self.sigma = None
        self.fitError = None
        self.sorted_x = None
        self.sorted_y = None

    def neg_exponential_model(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def power_model(self, x, a, b):
        return a * np.power(x, b)
    def power_model_w_constant(self, x, a, b, c):
        return a * np.power(x, b) + c
    def logarithmic_model(self, x, a, b, c):
        return a * np.log(b * x) + c

    def spatial_constant_dim2(self, x, a, b):
        return a * np.power(x, 1/2) + b

    def spatial_constant_dim3(self, x, a, b):
        return a * np.power(x, 1/3) + b

    def small_world_model(self, x, a, b):
        return a * np.log(x) + b

    def get_equation_string(self, model_func):
        if model_func == self.neg_exponential_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \exp(-{b:.4f} x) + {c:.4f}$'
        elif model_func == self.power_model:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{{b:.4f}}}$'
        elif model_func == self.power_model_w_constant:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot x^{{{b:.4f}}} + {c: .4f}$'
        elif model_func == self.logarithmic_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot \log({b:.4f} x) + {c:.4f}$'
        elif model_func == self.spatial_constant_dim2:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/2}} + {b:.4f}$'

        elif model_func == self.spatial_constant_dim3:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/3}} + {b:.4f}$'

        elif model_func == self.small_world_model:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot \log(x) + {b:.4f}$'
        else:
            return 'Unknown model'

    def perform_curve_fitting(self, model_func):
        # Sort the x values while keeping y values matched
        sorted_indices = np.argsort(self.x_data)
        self.sorted_x = self.x_data[sorted_indices]
        self.sorted_y = self.y_data[sorted_indices]

        # Perform curve fitting
        self.popt, self.pcov = curve_fit(model_func, self.sorted_x, self.sorted_y)
        self.sigma = np.sqrt(np.diag(self.pcov))

        # Calculate standard deviation of fit values
        param_combinations = list(product(*[(1, -1)]*len(self.sigma)))
        values = np.array([model_func(self.sorted_x, *(self.popt + np.array(comb) * self.sigma)) for comb in param_combinations])
        self.fitError = np.std(values, axis=0)

    def plot_fit_with_uncertainty(self, model_func, xlabel, ylabel, title, save_path):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        curveFit = model_func(self.sorted_x, *self.popt)

        # Plot data and fit
        plt.scatter(self.sorted_x, self.sorted_y, label='Data', alpha=0.5, edgecolors='w', zorder=3)
        plt.plot(self.sorted_x, curveFit, linewidth=2.5, color='green', alpha=0.9, label='Fit', zorder=2)

        # Uncertainty areas
        plt.fill_between(self.sorted_x, curveFit - self.fitError, curveFit + self.fitError, color='red', alpha=0.2, label=r'$\pm 1\sigma$ uncertainty')
        plt.fill_between(self.sorted_x, curveFit - 3*self.fitError, curveFit + 3*self.fitError, color='blue', alpha=0.1, label=r'$\pm 3\sigma$ uncertainty')

        # Equation annotation
        equation = self.get_equation_string(model_func)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.7, 0.05, equation, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')

        # Labels and title
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.title(title, fontsize=28, color='k')
        ax.legend(fontsize=18, loc='best')

        plt.savefig(save_path)

    def plot_fit_with_uncertainty_for_dataset(self, x, y, model_func, ax, label_prefix, color, y_position):
        # Perform curve fitting
        self.perform_curve_fitting(model_func)

        # Plot data and fit
        ax.scatter(x, y, label=f'{label_prefix} Data', alpha=0.5, edgecolors='w', zorder=3, color=color)
        curve_fit = model_func(x, *self.popt)
        ax.plot(x, curve_fit, label=f'Fit {label_prefix}', linestyle='--', color=color, zorder=2)

        # Uncertainty areas
        ax.fill_between(x, curve_fit - self.fitError, curve_fit + self.fitError, alpha=0.2, color=color, label=f'Uncertainty {label_prefix}')

        # Equation annotation
        equation = self.get_equation_string(model_func)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, y_position, equation, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')



