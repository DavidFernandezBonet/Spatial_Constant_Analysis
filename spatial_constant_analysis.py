import sys
import os
import matplotlib.pyplot as plt
script_dir = "/home/david/PycharmProjects/Spatial_Graph_Denoising"
# Add this directory to the Python path
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now you can import your module (assuming the file is named your_script.py)
import create_proximity_graph

from algorithms import *
from utils import *
from data_analysis import *
from plots import *
from metrics import *

import scienceplots
plt.style.use(['science', 'ieee'])

def create_project_structure():


    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(current_script_dir)

    directory_map = {
        'source_code': f'{project_root}/src',
        'edge_lists': f'{project_root}/data/edge_lists',
        'original_positions': f'{project_root}/data/original_positions',
        'plots': f'{project_root}/results/plots',
        'plots_spatial_constant': f'{project_root}/results/plots/spatial_constant',
    }

    for key, relative_path in directory_map.items():
        full_path = os.path.join(project_root, relative_path)
        os.makedirs(full_path, exist_ok=True)

        if key == 'source_code':
            with open(os.path.join(full_path, '__init__.py'), 'w') as init_file:
                init_file.write("# Init file for src package\n")

    with open(os.path.join(project_root, 'README.md'), 'w') as readme_file:
        readme_file.write("# Project: Spatial Constant Analysis\n")

    print(f"Project structure created under '{project_root}'")
    return directory_map

class GraphArgs:
    def __init__(self, code_folder=os.getcwd(), num_points=300, L=1, intended_av_degree=6,
                 dim=2, proximity_mode="knn_bipartite", directory_map=None, average_degree=None):
        self.code_folder = code_folder
        self._num_points = num_points
        self.L = L
        self._intended_av_degree = intended_av_degree
        self._proximity_mode = proximity_mode
        self._dim = dim
        self.is_bipartite = False
        self.average_degree = average_degree
        self.directory_map = directory_map

    def update_args_title(self):
        self.args_title = f"N={self._num_points}_dim={self._dim}_{self._proximity_mode}_k={self._intended_av_degree}"

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
        self._proximity_mode = value
        self.update_args_title()

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value
        self.update_args_title()


def run_simulation_false_edges(args, max_edges_to_add=10):
    results = []

    # Load the initial graph
    igraph_graph = load_graph(args, load_mode='igraph')

    for i in range(1, max_edges_to_add + 1):
        # Add i random edges
        igraph_graph = add_random_edges_igraph(igraph_graph, i)

        # Compute mean shortest path and other results
        mean_shortest_path = get_mean_shortest_path(igraph_graph)
        spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path)
        spatial_constant_results['number_of_random_edges'] = i

        # Append the results to the DataFrame
        results.append(spatial_constant_results)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/spatial_constant_change_with_false_edges_data.csv")
    plot_spatial_constant(args, results_df)
    plot_average_shortest_path(args, results_df)
    return results_df

def run_simulation_graph_growth(args, start_n_nodes=50, n_graphs=10, num_random_edges=0, model_func="spatial_constant_dim=2"):
    results = []
    # Load the initial graph
    igraph_graph = load_graph(args, load_mode='igraph')

    # Add random edges if specified
    if num_random_edges > 0:
        igraph_graph = add_random_edges_igraph(igraph_graph, num_random_edges)

    # Generate subgraphs with BFS
    subgraphs = grow_graph_bfs(igraph_graph, nodes_start=start_n_nodes, nodes_finish=args.num_points, n_graphs=n_graphs)

    for subgraph in subgraphs:
        args.num_points = subgraph.vcount()
        # Compute mean shortest path and other results
        mean_shortest_path = get_mean_shortest_path(subgraph)
        spatial_constant_results = get_spatial_constant_results(args, mean_shortest_path)
        # Append the results to the DataFrame
        results.append(spatial_constant_results)

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Filename based on whether random edges were added
    filename_suffix = f"_random_edges_{num_random_edges}" if num_random_edges > 0 else "_no_random_edges"
    csv_filename = f"spatial_constant_change_with_graph_growth_data{filename_suffix}.csv"
    plot_filename = f"mean_sp_vs_graph_growth{filename_suffix}.png"

    # Save the DataFrame and plot
    results_df.to_csv(f"{args.directory_map['plots_spatial_constant']}/{csv_filename}")
    plot_mean_sp_with_graph_growth(args, results_df, plot_filename, model=model_func)
    return results_df


def run_simulation_comparison_large_and_small_world(args, start_n_nodes=50, end_n_nodes=1000, n_graphs=10, num_random_edges_ratio=0.015):
    results = []
    node_counts = np.linspace(start_n_nodes, end_n_nodes, n_graphs, dtype=int)

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


# TODO: args_title is not instantiated if you don't call the parameters (maybe just make a config file with the parameters and call them all)
args = GraphArgs()
args.proximity_mode = "knn"
args.intended_av_degree = 10
args.num_points = 1000
print(args.args_title)
args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args

create_proximity_graph.write_proximity_graph(args)
igraph_graph_original, _ = load_graph(args, load_mode='sparse')
print(type(igraph_graph_original))

reconstruction = ImageReconstruction(graph=igraph_graph_original, dim=2)
reconstructed_points = reconstruction.reconstruct()


edge_list = read_edge_list(args)
original_points = read_position_df(args)

qm = QualityMetrics(original_points, reconstructed_points)
qm.evaluate_metrics()

gta_qm = GTA_Quality_Metrics(edge_list=edge_list, reconstructed_points=reconstructed_points)
gta_qm.evaluate_metrics()





num_random_edges = 0
model_func = "spatial_constant_dim=2"  # small_world
# model_func = "small_world"


# run_simulation_false_edges(args, max_edges_to_add=100)
# run_simulation_graph_growth(args, n_graphs=50, num_random_edges=num_random_edges, model_func=model_func)
# run_simulation_comparison_large_and_small_world(args, start_n_nodes=500, end_n_nodes=5000, n_graphs=10, num_random_edges_ratio=0.015)


