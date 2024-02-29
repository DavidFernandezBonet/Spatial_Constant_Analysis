import os

def create_project_structure():


    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(current_script_dir)

    directory_map = {
        'source_code': f'{project_root}/src',
        'edge_lists': f'{project_root}/data/edge_lists',
        'original_positions': f'{project_root}/data/original_positions',
        'reconstructed_positions': f'{project_root}/data/reconstructed_positions',
        'pixelgen_data': f'{project_root}/data/pixelgen_data',
        'slidetag_data': f'{project_root}/data/slidetag_data',
        'colorfolder': f'{project_root}/data/colorcode',

        's_constant_results': f'{project_root}/results/individual_spatial_constant_results',
        'plots': f'{project_root}/results/plots',
        'plots_original_image': f'{project_root}/results/plots/original_image',
        'plots_reconstructed_image': f'{project_root}/results/plots/reconstructed_image',
        'plots_shortest_path_heatmap': f'{project_root}/results/plots/shortest_path_heatmap',
        'plots_mst_image': f'{project_root}/results/plots/mst_image',
        'plots_euclidean_sp': f'{project_root}/results/plots/correlation_euclidean_sp',


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
        'dimension_prediction_iterations': f'{project_root}/results/plots/predicted_dimension/several_predictions',
        'centered_msp': f'{project_root}/results/plots/predicted_dimension/centered_msp',
        'mds_dim': f'{project_root}/results/plots/predicted_dimension/MDS_dimension',

        'plots_clustering_coefficient': f'{project_root}/results/plots/clustering_coefficient',
        'plots_degree_distribution': f'{project_root}/results/plots/degree_distribution',
        'plots_shortest_path_distribution': f'{project_root}/results/plots/shortest_path_distribution',
        'plots_weight_distribution': f'{project_root}/results/plots/weight_distribution',

        # Miscellaneous
        'plots_pixelgen': f'{project_root}/results/plots/pixelgen_quality_plots',
        'final_project': f'{project_root}/results/plots/statistical_methods_in_physics_project',
        'animation_output': f'{project_root}/results/bfs_animation/statistical_methods_in_physics_project'

    }

    for key, relative_path in directory_map.items():
        full_path = os.path.join(project_root, relative_path)
        os.makedirs(full_path, exist_ok=True)

        if key == 'source_code':
            with open(os.path.join(full_path, '__init__.py'), 'w') as init_file:
                init_file.write("# Init file for src package\n")

    # with open(os.path.join(project_root, 'README.md'), 'w') as readme_file:
    #     readme_file.write("# Project: Spatial Constant Analysis\n")

    print(f"Project structure created under '{project_root}'")
    return directory_map

class GraphArgs:
    # def __init__(self, code_folder=os.getcwd(), num_points=300, L=1, intended_av_degree=6,
    #              dim=2, proximity_mode="knn", directory_map=None, average_degree=-1,
    #              edge_list_title=None, false_edges_count=0, plot_original=False, title_experimental=None):
    def __init__(self, config=None):

        print(config)
        # Default values if config is None or does not provide some settings
        self.code_folder = os.getcwd()
        #self.edge_list_title = config.get('edge_list_title', None)
        self.edge_list_title = None
        self.original_edge_list_title = None
        self._num_points = config.get('num_points', 300)
        self.L = config.get('L', 1)
        self._dim = config.get('dim', 2)
        self._base_proximity_mode = config.get('proximity_mode', "knn")
        self._false_edges_count = config.get('false_edges_count', 0)  # TODO: is this simulation specific?
        self.colorfile = config.get('colorfile')
        self.plot_graph_properties = config.get('plot_graph_properties', False)
        self.large_graph_subsampling = config.get('large_graph_subsampling', False)
        self.max_subgraph_size = config.get('max_subgraph_size', 3000)


        self.reconstruct = config.get('reconstruct', False)
        if self.reconstruct:
            self.reconstruction_mode = config.get('reconstruction_mode')


        self.update_proximity_mode()

        if self.proximity_mode == "experimental":
            initial_title = config.get('edge_list_title', None)
            if initial_title:
                self.set_edge_list_title(initial_title)

        if self.proximity_mode == "experimental":
            ### Experiment specific
            self.title_experimental = config.get('title_experimental', None)
            self.weighted = config.get('weighted', False)
            if self.weighted:
                self.weighted_threshold = config.get('weight_threshold', 0)
        else:
            ### Simulation specific
            self.plot_original_image = config.get('plot_original_image', False)
            self._intended_av_degree = config.get('intended_av_degree', 6)

        # self.update_proximity_mode()
        self.update_args_title()







        # Initialize additional properties to their defaults or based on other computed attributes
        self.false_edge_ids = []  # To store false edges if needed
        self.is_bipartite = False
        self.bipartite_sets = None
        self.average_degree = -1
        self.mean_clustering_coefficient = None
        self.directory_map = create_project_structure()
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

    def set_edge_list_title(self, title):
        if self.original_edge_list_title is None:  # Only set the original title once or when explicitly needed
            self.original_edge_list_title = title
        self.edge_list_title = title

    def update_args_title(self):

        if "experimental" in self._proximity_mode:
            if self.edge_list_title is not None:
                self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.original_edge_list_title)[0]}"
                # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"

            # else:
            #     self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"


            # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"
        else:

            self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"


            # if self.edge_list_title is None:
            #     self.edge_list_title = f"edge_list_{self.args_title}.csv"
            # else:
            #     self.edge_list_title = f"{self.edge_list_title}"

            ## For now just update the edge list every time
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


    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.update_args_title()
