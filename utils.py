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

def load_graph(args, load_mode='igraph', input_file_type='edge_list'):
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
    file_path = f"{args.directory_map['edge_lists']}/{args.edge_list_title}"
    df = pd.read_csv(file_path)
    args.original_title = args.args_title


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
        self.sorted_y_errors = None
        self.reduced_chi_squared = None
        self.r_squared = None

    def neg_exponential_model(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def power_model(self, x, a, b):
        return a * np.power(x, b)
    def power_model_2d_Sconstant(self, x, a):
        s = 1.10
        return s * np.power(x, a)
    def power_model_2d_bi_Sconstant(self, x, a):
        s = 0.9
        return s * np.power(x, a)
    def power_model_3d_Sconstant(self, x, a):
        s = 1.4
        return s * np.power(x, a)
    def power_model_3d_bi_Sconstant(self, x, a):
        s = 1.3
        return s * np.power(x, a)

    def power_model_w_constant(self, x, a, b, c):
        return a * np.power(x, b) + c
    def logarithmic_model(self, x, a, b, c):
        return a * np.log(b * x) + c

    def spatial_constant_dim2(self, x, a, b):
        return a * np.power(x, 1/2) + b

    def spatial_constant_dim2_linearterm(self, x, a):
        return a * np.power(x, 1/2)

    def spatial_constant_dim3(self, x, a, b):
        return a * np.power(x, 1/3) + b

    def spatial_constant_dim3_linearterm(self, x, a):
        return a * np.power(x, 1/3)

    def small_world_model(self, x, a, b):
        return a * np.log(x) + b

    def get_equation_string(self, model_func):
        if model_func == self.neg_exponential_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \exp(-{b:.4f} x) + {c:.4f}$'
        elif model_func == self.power_model:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_2d_Sconstant:
            b = self.popt[0]
            return f'$y = 1.10 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_3d_Sconstant:
            b = self.popt[0]
            return f'$y = 1.4 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_2d_bi_Sconstant:
            b = self.popt[0]
            return f'$y = 0.9 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_3d_bi_Sconstant:
            b = self.popt[0]
            return f'$y = 1.16 \cdot x^{{{b:.4f}}}$'


        elif model_func == self.power_model_w_constant:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot x^{{{b:.4f}}} + {c: .4f}$'
        elif model_func == self.logarithmic_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot \log({b:.4f} x) + {c:.4f}$'

        elif model_func == self.spatial_constant_dim2:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/2}} + {b:.4f}$'

        elif model_func == self.spatial_constant_dim2_linearterm:
            a = self.popt[0]

            return f'$y = {a:.4f} \cdot x^{{1/2}} $'

        elif model_func == self.spatial_constant_dim3:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/3}} + {b:.4f}$'

        elif model_func == self.spatial_constant_dim3_linearterm:
            a = self.popt[0]
            return f'$y = {a:.4f} \cdot x^{{1/3}} $'

        elif model_func == self.small_world_model:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot \log(x) + {b:.4f}$'
        else:
            return 'Unknown model'

    def perform_curve_fitting(self, model_func, constant_error=None):
        # Sort the x values while keeping y values matched
        sorted_indices = np.argsort(self.x_data)
        self.sorted_x = self.x_data[sorted_indices]
        self.sorted_y = self.y_data[sorted_indices]
        self.sorted_y_errors = np.full_like(self.y_data, constant_error if constant_error is not None else 1.0)  #TODO: careful with this! Only if we don't have errors

        # Perform curve fitting
        self.popt, self.pcov = curve_fit(model_func, self.sorted_x, self.sorted_y)
        self.sigma = np.sqrt(np.diag(self.pcov))

        # Calculate standard deviation of fit values
        param_combinations = list(product(*[(1, -1)]*len(self.sigma)))
        values = np.array([model_func(self.sorted_x, *(self.popt + np.array(comb) * self.sigma)) for comb in param_combinations])
        self.fitError = np.std(values, axis=0)


        # Calculate residuals and reduced chi-squared  #TODO: not working for now, Is there a meaningful way to associate errors?
        y_fit = model_func(self.sorted_x, *self.popt)
        residuals = self.sorted_y - y_fit
        chi_squared = np.sum((residuals / self.sorted_y_errors) ** 2)
        degrees_of_freedom = len(self.sorted_y) - len(self.popt)
        self.reduced_chi_squared = chi_squared / degrees_of_freedom
        print("chi squared", self.reduced_chi_squared)

        # For R squared
        mean_y = np.mean(self.sorted_y)
        sst = np.sum((self.sorted_y - mean_y) ** 2)
        ssr = np.sum(residuals ** 2)
        self.r_squared = 1 - (ssr / sst)
        print("R-squared:", self.r_squared)

        # KS test
        ks_statistic, p_value = scipy.stats.kstest(self.sorted_y, lambda x: model_func(x, *self.popt))
        print("ks stat", ks_statistic, "p-value", p_value)

        # Perform the Anderson-Darling test on the residuals
        ad_result = scipy.stats.anderson(residuals)

        # Store the results
        self.ad_statistic = ad_result.statistic
        self.ad_critical_values = ad_result.critical_values
        self.ad_significance_levels = ad_result.significance_level

        # Output results
        print("Anderson-Darling Statistic:", self.ad_statistic)
        for i in range(len(self.ad_critical_values)):
            sl, cv = self.ad_significance_levels[i], self.ad_critical_values[i]
            print(f"Significance Level {sl}%: Critical Value {cv}")

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

        # Equation and reduced chi-squared annotation
        equation = self.get_equation_string(model_func)
        r_squared_text = f'R2: {self.r_squared:.2f}'
        annotation_text = f"{equation}\n{r_squared_text}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.7, 0.05, annotation_text, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')

        # Labels and title
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.title(title, fontsize=28, color='k')
        ax.legend(fontsize=18, loc='best')

        # Save the plot
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



