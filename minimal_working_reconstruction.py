from spatial_constant_analysis import *

# Parameters
args = GraphArgs()
# Creates all the necessary project folders
args.directory_map = create_project_structure()  # creates folder and appends the directory map at the args
# Specify parameters
args.dim = 2
args.intended_av_degree = 10
args.num_points = 1000
# Load created graph from the edge list

print(args.intended_av_degree)
# # Option 1 - Simulated graph
create_proximity_graph.write_proximity_graph(args)

# Option 2 - 'Experimental' graph
args.proximity_mode = "experimental"  # define proximity mode before naming (order matters)
args.edge_list_title = "weinstein_data.csv"  # dataframe with ['source', 'target', 'weight'] as columns (weight is optional)

sparse_graph, _ = load_graph(args, load_mode='sparse')
# node_embedding_mode: node2vec, ggvec, landmark_isomap
run_reconstruction(args, sparse_graph=sparse_graph, ground_truth_available=True, node_embedding_mode='landmark_isomap')
