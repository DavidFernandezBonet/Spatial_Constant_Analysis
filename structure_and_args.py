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
        'colorfolder': f'{project_root}/data/colorcode',

        's_constant_results': f'{project_root}/results/individual_spatial_constant_results',
        'plots': f'{project_root}/results/plots',
        'plots_original_image': f'{project_root}/results/plots/original_image',
        'plots_reconstructed_image': f'{project_root}/results/plots/reconstructed_image',
        'plots_mst_image': f'{project_root}/results/plots/mst_image',


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
    def __init__(self, code_folder=os.getcwd(), num_points=300, L=1, intended_av_degree=6,
                 dim=2, proximity_mode="knn", directory_map=None, average_degree=-1,
                 edge_list_title=None, false_edges_count=0, plot_original=False, title_experimental=None):

        self.edge_list_title = edge_list_title
        self.title_experimental = title_experimental
        self.code_folder = code_folder
        self._num_points = num_points
        self.L = L
        self._intended_av_degree = intended_av_degree
        self._base_proximity_mode = proximity_mode
        self._false_edges_count = false_edges_count
        self.false_edge_ids = []  # list of tuples containing the false edges added   # TODO: store them properly?
        self.update_proximity_mode()
        self._dim = dim

        # Graph properties
        self.is_bipartite = False
        self.bipartite_sets = None  # Adding this attribute
        self.average_degree = average_degree
        self.mean_clustering_coefficient = None

        # Directory map
        self.directory_map = create_project_structure()

        self.plot_original = plot_original  #TODO: implement this as a true false event

        # auxiliary title (original, when graph is well connected and we don't have to grab largest component)
        self.original_title = None

        self.node_ids_map_old_to_new = None
        self.colorfile = None  # filename where the color ids are stored. It is a dataframe with Node_ID, color in columns
        self.colorcode = {0: "black", 1: "green", 2: "red"}  # what colors to plot



    def update_args_title(self):
        if self._proximity_mode == "experimental":
            if self.edge_list_title is not None:
                self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"
            else:
                self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"
            # self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_{os.path.splitext(self.edge_list_title)[0]}"
        else:
            self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"
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
