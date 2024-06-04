import sys
from pathlib import Path
# This is so the script works as a stand alone and as a package
package_root = Path(__file__).parent
if str(package_root) not in sys.path:
    sys.path.append(str(package_root))

import os
import importlib
import config as default_config
import pprint
import shutil
import importlib.resources as pkg_resources
import math
def create_project_structure(target_dir=None):
    """
    Create the project directory structure and return a dictionary mapping directory names to their corresponding paths.
    The user can choose the directory where the project should be created, otherwise it defaults to the current working directory.
    """
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_script_dir)
    add_src = False
    if target_dir is None:
        if os.path.exists(os.path.join(parent_dir, "do_not_delete.txt")):  # this is to distinguish case where user and developer
            # Running as a script within the src directory
            project_root = os.path.dirname(os.path.dirname(current_script_dir))
            add_src = True
        else:
            # Running as installed package or no clear indication of being within src
            project_root = os.getcwd()  # Default to current working directory
    else:
        # User specified target directory
        project_root = target_dir

    # current_script_dir = os.path.dirname(os.path.realpath(__file__))
    # project_root = os.path.dirname(current_script_dir)

    directory_map = {
        'edge_lists': f'{project_root}/data/edge_lists',
        'original_positions': f'{project_root}/data/original_positions',
        'reconstructed_positions': f'{project_root}/data/reconstructed_positions',
        'pixelgen_data': f'{project_root}/data/pixelgen_data',
        'slidetag_data': f'{project_root}/data/slidetag_data',
        'colorfolder': f'{project_root}/data/colorcode',
        'us_counties': f'{project_root}/data/us_counties',
        'weinstein': f'{project_root}/data/weinstein',
        'json': f'{project_root}/data/network_json_format',

        's_constant_results': f'{project_root}/results/individual_spatial_constant_results',
        'output_dataframe': f'{project_root}/results/output_dataframe',
        'plots': f'{project_root}/results/plots',
        'plots_original_image': f'{project_root}/results/plots/original_image',
        'plots_reconstructed_image': f'{project_root}/results/plots/reconstructed_image',

        'reconstructed_positions_subgraphs': f'{project_root}/results/all_subgraphs_reconstruction',
        'rec_positions_subgraphs': f'{project_root}/results/all_subgraphs_reconstruction/reconstructed_positions_subgraphs',
        'rec_images_subgraphs': f'{project_root}/results/all_subgraphs_reconstruction/reconstructed_images_subgraphs',

        'plots_shortest_path_heatmap': f'{project_root}/results/plots/shortest_path_heatmap',
        'distance_decay': f'{project_root}/results/plots/distance_decay',
        'plots_mst_image': f'{project_root}/results/plots/mst_image',
        'plots_euclidean_sp': f'{project_root}/results/plots/correlation_euclidean_sp',
        'spatial_coherence': f'{project_root}/results/plots/main_spatial_coherence_results',  #TODO: Main folder
        'dataframes': f'{project_root}/results/plots/multiple_runs_dataframes',


        'plots_spatial_constant': f'{project_root}/results/plots/spatial_constant',
        'plots_spatial_constant_gg': f'{project_root}/results/plots/spatial_constant/graph_growth',
        'plots_spatial_constant_subgraph_sampling': f'{project_root}/results/plots/spatial_constant/subgraph_sampling',
        'plots_spatial_constant_variation': f'{project_root}/results/plots/spatial_constant/variation_analysis',
        'plots_spatial_constant_weighted_threshold': f'{project_root}/results/plots/spatial_constant/weighted_threshold',
        'plots_spatial_constant_false_edge_difference': f'{project_root}/results/plots/spatial_constant/false_edge_difference',
        'plots_spatial_constant_false_edge_difference_fits': f'{project_root}/results/plots/spatial_constant/false_edge_difference/fits',
        # 'plots_spatial_constant_variation_N': f'{project_root}/results/plots/spatial_constant/variation_analysis/N',
        # 'plots_spatial_constant_variation_prox_mode': f'{project_root}/results/plots/spatial_constant/variation_analysis/prox_mode',
        # 'plots_spatial_constant_variation_degree': f'{project_root}/results/plots/spatial_constant/variation_analysis/degree',

        'plots_predicted_dimension': f'{project_root}/results/plots/predicted_dimension',
        'local_dimension': f'{project_root}/results/plots/predicted_dimension/local_dimension',
        'heatmap_local': f'{project_root}/results/plots/predicted_dimension/heatmap_local_dimension',
        'dimension_prediction_iterations': f'{project_root}/results/plots/predicted_dimension/several_predictions',
        'centered_msp': f'{project_root}/results/plots/predicted_dimension/centered_msp',
        'mds_dim': f'{project_root}/results/plots/predicted_dimension/MDS_dimension',
        'comparative_plots': f'{project_root}/results/plots/comparative_plots',
        'euc_vs_net': f'{project_root}/results/plots/euclidean_vs_network',

        'plots_clustering_coefficient': f'{project_root}/results/plots/clustering_coefficient',
        'plots_degree_distribution': f'{project_root}/results/plots/degree_distribution',
        'plots_shortest_path_distribution': f'{project_root}/results/plots/shortest_path_distribution',
        'plots_weight_distribution': f'{project_root}/results/plots/weight_distribution',

        # Miscellaneous
        'plots_pixelgen': f'{project_root}/results/plots/pixelgen_quality_plots',
        'final_project': f'{project_root}/results/plots/statistical_methods_in_physics_project',
        'animation_output': f'{project_root}/results/bfs_animation/statistical_methods_in_physics_project',
        'profiler': f'{project_root}/results/plots/profiler',

    }

    if add_src:
        directory_map['source_code'] = f'{project_root}/src'
    for key, relative_path in directory_map.items():
        full_path = os.path.join(project_root, relative_path)
        os.makedirs(full_path, exist_ok=True)

        # if key == 'source_code':
        #     with open(os.path.join(full_path, '__init__.py'), 'w') as init_file:
        #         init_file.write("# Init file for src package\n")

    # with open(os.path.join(project_root, 'README.md'), 'w') as readme_file:
    #     readme_file.write("# Project: Spatial Constant Analysis\n")

    files_to_copy = {
        'example_edge_list.pickle': 'data/edge_lists',
        'dna_cool2.png': 'data/colorcode',
    }

    if not add_src:
        ### Move example files for the user to have in the package
        for filename, rel_dest in files_to_copy.items():
            # Use importlib.resources to access the package files
            with pkg_resources.path('network_spatial_coherence', filename) as source_path:
                # Calculate the destination path
                dest_path = Path(project_root) / rel_dest / filename

                # Copy file
                shutil.copy2(source_path, dest_path)
                print(f"Copied {filename} to {dest_path}")

    print(f"Project structure created under '{project_root}'")
    return directory_map

class GraphArgs:
    """
    A class to manage graph arguments and configurations.
    It is automatically initialized based on the configuration file, which has the default name: "config.py"

    Attributes:

        # Directories
        directory_map (type): Simple reference for directory mapping, created by `create_project_structure`.

        # Base Attributes
        proximity_mode (str): The mode of determining proximity among nodes, defaults to "knn" (k-nearest neighbors) for simulations. Can be set to 'experimental', which requires an edge list to be provided.
        edge_list_title (str): The title or name for the edge list, None by default, to be updated with graph parameters.
        dim (int): The dimensionality of the Euclidean point cloud.
        plot_graph_properties (bool): A flag indicating whether to plot graph properties, defaults to False. Properties: degree, clustering and shortest path distribution.
        reconstruct (bool): Indicates whether graph reconstruction is to be performed, defaults to False.
        reconstruction_mode (str): The mode of graph reconstruction, only relevant if `reconstruct` is True.
        large_graph_subsampling (bool): Indicates whether subsampling of large graphs is enabled, defaults to False.
        max_subgraph_size (int): The maximum size for subgraphs, defaults to 3000, relevant if subsampling is enabled.
        weighted (bool): Indicates if the graph is weighted, defaults to False.
        weighted_threshold (float): The threshold for weights in a weighted graph, relevant if `weighted` is True.
        false_edges_count (int): The count of false edges to simulate, if any, defaults to 0.
        false_edge_ids (list): Stores ids of false edges, if needed.

        # Image Coloring
        colorfile (str): Path to a file defining colors for nodes or edges, if applicable.
        node_ids_map_old_to_new (type): Maps old node IDs to new ones, None by default.
        colorcode (dict): Maps node states to colors, with default mappings provided.


        # Simulation Attributes
        _num_points (int): The number of points (or nodes) in the graph, defaults to 300.
        L (int): Length of the square/cube.
        plot_original_image (bool): Flag to plot the original image, relevant for simulations, defaults to False.
        _intended_av_degree (int): The intended average degree of nodes in the graph, defaults to 6.
        id_to_color_simulation (type): Maps node IDs to colors for simulation purposes, None by default.


        # Graph Attributes
        sparse_graph (type): The sparse representation of the graph, None by default.
        igraph_graph (type): The graph represented using the igraph library, None by default.
        shortest_path_matrix (numpy.ndarray): A matrix storing the shortest paths between nodes, None by default.
        mean_shortest_path (type): The mean shortest path length in the graph, None by default.
        is_bipartite (bool): Indicates if the graph is bipartite, defaults to False.
        bipartite_sets (type): The sets of nodes in a bipartite graph, None by default.
        average_degree (int): The average degree of nodes in the graph, initialized to -1.
        mean_clustering_coefficient (type): The mean clustering coefficient of the graph, None by default.

    Methods:
        update_proximity_mode(): Updates the proximity mode based on the configuration.
        update_args_title(): Updates the generic file title to define each individual graph based on graph attributes
        create_project_structure(): Creates and returns a mapping of directory structures, for file organization.
    """

    def __init__(self, override_config_path=None, data_dir=None):

        # Loading Args from configuration file
        self.code_folder = os.getcwd()

        self.override_config_path = override_config_path
        self.data_dir = data_dir
        self.unsorted_config = self.load_config(override_config_path)
        config = self.get_config(config_module=self.unsorted_config)
        self.config = config

        self.verbose = config.get('verbose', True)
        self.show_plots = config.get('show_plots', False)
        self.edge_list_title = None
        self.original_edge_list_title = None
        self._num_points = config.get('num_points', 300)
        self.L = config.get('L', 1)
        self._dim = config.get('dim', 2)
        self._base_proximity_mode = config.get('proximity_mode', "knn")
        self.weighted = config.get('weighted', False)
        self._false_edges_count = config.get('false_edges_count', 0)  # TODO: is this simulation specific?
        self.true_edges_deletion_ratio = config.get('true_edges_deletion_ratio', 0)
        self.point_mode = None  # circle or square
        self.colorfile = config.get('colorfile')
        self.plot_graph_properties = config.get('plot_graph_properties', False)
        self.large_graph_subsampling = config.get('large_graph_subsampling', False)
        self.max_subgraph_size = config.get('max_subgraph_size', 3000)
        self.network_name = ''

        self.format_plots = config.get('format_plots', 'svg')

        self.handle_all_subgraphs = config.get('handle_all_subgraphs', False)
        self.spatial_coherence_validation = config.get('spatial_coherence_validation', False)
        self.community_detection = config.get('community_detection', False)
        self.reconstruct = config.get('reconstruct', False)
        self.original_positions_available = config.get('original_positions_available', False)
        self.plot_original_image = config.get('plot_original_image', False)
        self.plot_reconstructed_image = config.get('plot_reconstructed_image', False)
        self.spatial_coherence_quantiative_dict = {}


        self.weight_to_distance = config.get('weight_to_distance', False)
        self.weight_to_distance_fun = config.get('weight_to_distance_fun', 'exp')
        self.weight_converter = self.WeightToDistance()

        if self.reconstruct:
            self.reconstruction_mode = config.get('reconstruction_mode')

        self._intended_av_degree = config.get('intended_av_degree', 6)
        self.distance_decay_quantile = None # Optional attribute for distance decay proximity mode

        self.update_proximity_mode()

        ### Set edge list title
        if self.proximity_mode == "experimental":
            initial_title = config.get('edge_list_title', None)
            if initial_title:
                self.set_edge_list_title(initial_title)

        if self.weighted:
            self.weight_threshold = config.get('weight_threshold', 0)
            self.weighted_threshold = config.get('weight_threshold', 0)

            self.distance_to_weight_fun = config.get('distance_to_weight', False)








        if self.proximity_mode == "experimental":
            ### Experiment specific
            self.title_experimental = config.get('title_experimental', None)

        else:
            ### Simulation specific

            self._intended_av_degree = config.get('intended_av_degree', 6)

        # self.update_proximity_mode()
        self.update_args_title()



        # Initialize additional properties to their defaults or based on other computed attributes
        self.false_edge_ids = []  # To store false edges if needed
        self.is_bipartite = False
        self.bipartite_sets = None
        self.average_degree = -1
        self.mean_clustering_coefficient = None
        self.directory_map = create_project_structure(target_dir=data_dir)
        self.original_title = None
        self.node_ids_map_old_to_new = None
        self.colorcode = {-1: "gray", 0: "gray", 1: "green", 2: "red"}
        self.id_to_color_simulation = None


        # Store graph representations  #TODO: loading the 2 will be inefficient / memory intensive
        # Maybe add all graph propreties: nodes, edges, average degree...
        self.sparse_graph = None
        self.igraph_graph = None
        self.shortest_path_matrix = None   # numpy matrix storing the shortest paths
        self.mean_shortest_path = None


        #
        # self.edge_list_title = edge_list_title
        # self.title_experimental = title_experimental
        # self.code_folder = code_folder
        # self._num_points = num_points
        # self.L = L
        # self._intended_av_degree = intended_av_degree
        # self._base_proximity_mode = proximity_mode
        # self._false_edges_count = false_edges_count
        # self.false_edge_ids = []  # list of tuples containing the false edges added   # TODO: store them properly?
        # self.update_proximity_mode()
        # self._dim = dim
        #
        # # Graph properties
        # self.is_bipartite = False
        # self.bipartite_sets = None  # Adding this attribute
        # self.average_degree = average_degree
        # self.mean_clustering_coefficient = None
        #
        # # Shortest Path Matrix -- It is reused a lot, maybe store here. Maybe have a "graph object"
        # self.shortest_path_matrix = None
        #
        # # Directory map
        # self.directory_map = create_project_structure()
        #
        # self.plot_original = plot_original  #TODO: implement this as a true false event
        #
        # # auxiliary title (original, when graph is well connected and we don't have to grab largest component)
        # self.original_title = None
        #
        #
        # self.node_ids_map_old_to_new = None
        # self.colorfile = None  # filename where the color ids are stored. It is a dataframe with Node_ID, color in columns
        # self.colorcode = {-1: "gray", 0: "gray", 1: "green", 2: "red"}  # what colors to plot. This is based on weinstein ploting
        # self.id_to_color_simulation = None  # for colored simulations

    # def load_config(self, config_filename, code_folder):
    #     """
    #     Loads configuration from a Python file specified by combining the folder path and file name.
    #
    #     Parameters:
    #         config_filename (str): The name of the configuration file.
    #         code_folder (str): The folder where the configuration file is located.
    #
    #     Returns:
    #         module: A module object containing the configurations.
    #     """
    #     # Combine the folder and filename to create the full path to the config file
    #     config_path = os.path.join(code_folder, config_filename)
    #
    #     # Dynamically load the configuration module from the constructed path
    #     spec = importlib.util.spec_from_file_location(config_filename, config_path)
    #     config = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(config)
    #     # Convert the module to a dictionary
    #     config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
    #     return config_dict

    def load_config(self, override_config_path=None):
        config = default_config  # Start with the default config
        if override_config_path:
            # Dynamically load the override configuration module from the specified path
            spec = importlib.util.spec_from_file_location("override_config", override_config_path)
            override_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(override_config)

            # Merge the override config attributes with the default config
            for attr in dir(override_config):
                if not attr.startswith("__"):
                    setattr(config, attr, getattr(override_config, attr))

        # Convert the config module to a dictionary for compatibility with your existing code
        config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__')}
        return config_dict



    def get_config(self, config_module):
        """
        Determines the scenario based on the 'proximity_mode' in the configuration and merges configurations accordingly.

        Parameters:
            config_module (module): The configuration module containing the 'base', 'experiment', and 'simulation' configurations.

        Returns:
            dict: A dictionary of merged settings.
        """
        # Use getattr to safely get configurations from the module, defaulting to {} if not found
        base = config_module.get('base', {})
        experiment = config_module.get('experiment', {})
        simulation = config_module.get('simulation', {})

        # Determine which configuration to use based on the 'proximity_mode' in base
        if base.get("proximity_mode") == "experimental":
            # Merge base with experiment settings
            return {**base, **experiment}
        else:
            # Merge base with simulation settings
            return {**base, **simulation}


    def update_args(self, **kwargs):
        """updates attributes of GraphArgs objects given a dictionary"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_edge_list_title(self, title):
        if self.original_edge_list_title is None:  # Only set the original title once or when explicitly needed
            self.original_edge_list_title = title
            self.network_name = os.path.splitext(title)[0]   # TODO: make sure this doesn't conflict with custom network names
        self.edge_list_title = title

    def manual_change_edge_list_title(self, new_title):
        self.set_edge_list_title(new_title)
    def update_args_title(self):
        if self.original_edge_list_title is None:
            self.original_edge_list_title = self.edge_list_title
        if "experimental" in self._proximity_mode:
            if self.edge_list_title is not None:
                # print(self._num_points, self._dim, self._proximity_mode, self.original_edge_list_title)
                self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.original_edge_list_title)[0]}"
                self.extra_info = "_" + os.path.splitext(self.edge_list_title)[0]

                # self.network_name = os.path.splitext(self.edge_list_title)[0]
                # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"

            # else:
            #     self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"


            # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"
        else:
            ### Old setup for args and edge title
            # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"
            # self.edge_list_title = f"edge_list_{self.args_title}.csv"

            if self.distance_decay_quantile is not None:
                base_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}_q={self.distance_decay_quantile}"
            else:
                base_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"

            prefix = "edge_list_"
            base_with_prefix = f"{prefix}{base_title}"
            if self.edge_list_title is None:
                self.edge_list_title = f"edge_list_{base_title}.csv"

            if self.edge_list_title.startswith(base_with_prefix):

                extra_info = os.path.splitext(self.edge_list_title[len(base_with_prefix):])[0]  # Extract extra info
                if extra_info == ".csv":
                    extra_info = ""
                # print("EXTRA INFO", extra_info)
                self.extra_info = extra_info
                self.args_title = f"{base_title}{extra_info}"  # Reconstruct with base and extra_info}"  # Reconstruct with possible new base
            else:
                self.args_title = f"{base_title}"
            self.edge_list_title = f"edge_list_{self.args_title}.csv"



    @property
    def num_points(self):
        return self._num_points

    @num_points.setter
    def num_points(self, value):
        self._num_points = value
        self.update_args_title()

    @property
    def intended_av_degree(self):
        return self._intended_av_degree

    @intended_av_degree.setter
    def intended_av_degree(self, value):
        self._intended_av_degree = value
        self.update_args_title()

    @property
    def proximity_mode(self):
        return self._proximity_mode

    @proximity_mode.setter
    def proximity_mode(self, value):
        self._base_proximity_mode = value  # Update the base proximity mode
        self.update_proximity_mode()
        self.update_args_title()
    @property
    def false_edges_count(self):
        return self._false_edges_count

    @false_edges_count.setter
    def false_edges_count(self, value):
        self._false_edges_count = value
        self.update_proximity_mode()

    def update_proximity_mode(self):
        if self._false_edges_count:
            self._proximity_mode = self._base_proximity_mode + f"_with_false_edges={self._false_edges_count}"
        else:
            self._proximity_mode = self._base_proximity_mode
        if self.proximity_mode == "experimental":
            self.dim = 2          # TODO: change this if we ever have 3D experiments
            if self.verbose:
                print("Setting dimension to 2 for experimental settings...")

        self.update_args_title()


    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.update_args_title()



    class WeightToDistance():
        def __init__(self, decay_rate=0.1, max_weight=100, inverse_power=2):
            # TODO: make sure that decay rate makes sense, I think it has to be in the "distance scale" (maybe take it as the median)

            self.decay_rate = decay_rate  # Parameter for the exponential decay model
            self.max_weight = max_weight                        # Scaling factor for both models --> for exponential model it is the max weight
            self.inverse_power = inverse_power                # Exponent for the inverse power law model


        def return_weight_exponential_model(self, d):
            """Calculate the interaction weight using the negative exponential model."""
            # weight = math.exp(-self.decay_rate * d)   #TODO: this might need to be squared to have a "Gaussian smoorthing" effect
            weight = math.exp(-(self.decay_rate * d)**2)
            return round(self.max_weight * weight)   # TODO: add rounding for better modeling

        def return_distance_exponential_model(self, w):
            """Calculate the distance given a weight using the inverse of the exponential model."""

            # ### Without gaussian smoothing
            # if w <= 0:
            #     return float('inf')  # Handle case where weight is zero or negative
            # d = -math.log(w / self.max_weight) / self.decay_rate
            # return max(d, 0)  # Ensure that distance cannot be negative

            ### With Gaussian smoothing (square in the exponent)
            if w <= 0:
                return float('inf')  # Handle case where weight is zero or negative

            if w >= self.max_weight:
                w = self.max_weight-1
            # Compute the distance from the weight
            argument = -math.log(w / self.max_weight)
            if argument < 0:
                raise ValueError("Negative argument is about to be sqrooted")
            d = math.sqrt(argument) / self.decay_rate
            if d <= 0:
                print("argument", argument)
                print("distance", d)
                print("w", w)
                print("max weight", self.max_weight)
                print(math.log(w / self.max_weight))
                raise ValueError("Negative or zero distance. Weight:", w, "Max weight:", self.max_weight, "Decay rate:", self.decay_rate)
            return d




        def return_weight_power_law_model(self, d):
            """Calculate the interaction weight using the inverse power law model."""
            if d == 0:
                return float('inf')  # Avoid division by zero, handling infinite weight
            weight = self.max_weight / (d ** self.inverse_power)
            return round(weight)

        def return_distance_power_law_model(self, w):
            """Calculate the distance given a weight using the inverse of the power law model."""
            if w <= 0:
                return float('inf')  # Handle case where weight is zero or negative
            d = (self.max_weight / w) ** (1 / self.inverse_power)
            return d

    def export_default_config(filepath='default_config.py'):
        from config import base, simulation, experiment  # Adjust this import as needed

        pp = pprint.PrettyPrinter(indent=4)

        with open(filepath, 'w') as f:
            f.write("# Default configuration template\n\n")

            f.write("# Base settings common to all scenarios\n")
            f.write("base = ")
            f.write(pp.pformat(base) + "\n\n")

            f.write("# Settings specific to simulation scenarios\n")
            f.write("simulation = ")
            f.write(pp.pformat(simulation) + "\n\n")

            f.write("# Settings specific to experimental scenarios\n")
            f.write("experiment = ")
            f.write(pp.pformat(experiment) + "\n")

        print(f"Default configuration template written to {filepath}")