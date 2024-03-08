import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from algorithms import grow_graph_bfs
from algorithms import get_mean_shortest_path
import numpy as np
from matplotlib import gridspec
def get_bfs_subgraph(graph, step_interval, current_step):
    max_nodes = min(current_step * step_interval, len(graph.vs))
    bfs_result = graph.bfs(0)
    bfs_nodes = bfs_result[0][:max_nodes]  # Get the first 'max_nodes' vertices from the BFS result
    return graph.subgraph(bfs_nodes)
def avg_shortest_path_length(subgraph):
    if len(subgraph.vs) > 1:
        return sum(subgraph.shortest_paths_dijkstra()[0]) / len(subgraph.vs)
    return 0
def count_false_edges(subgraph, false_edges):
    count = 0
    for edge in subgraph.es:
        if (edge.source, edge.target) in false_edges or (edge.target, edge.source) in false_edges:
            count += 1
    return count







def main_animation(args, graph, n_graphs, title=""):
    # Load data
    edge_list_folder = args.directory_map["edge_lists"]
    edges_df = pd.read_csv(f"{edge_list_folder}/edge_list_{args.original_title}.csv")
    original_position_folder = args.directory_map["original_positions"]
    positions_df = pd.read_csv(f"{original_position_folder}/positions_{args.original_title}.csv")
    # Define the start and end node counts for BFS growth
    nodes_start = 50  # or any other starting number of nodes
    false_edges = args.false_edge_ids
    nodes_finish = graph.vcount()  # usually the total number of nodes in the graph

    # Setting scale limits
    x_min, x_max = positions_df['x'].min(), positions_df['x'].max()
    y_min, y_max = positions_df['y'].min(), positions_df['y'].max()
    if args.dim == 3:
        z_min, z_max = positions_df['z'].min(), positions_df['z'].max()

    msp_min, msp_max = 0, 30

    # Generate subgraphs using BFS growth
    subgraphs = grow_graph_bfs(graph, nodes_start, nodes_finish, n_graphs)

    # Define the figure with three subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1])
    # Big plot for BFS
    if args.dim == 3:
        ax_graph = plt.subplot(gs[0, :], projection='3d')
    else:
        ax_graph = plt.subplot(gs[0, :])

    # Smaller plots for metrics
    ax_msp = plt.subplot(gs[1, 0])
    ax_spatial_constant= plt.subplot(gs[1, 1])
    ax_diameter = plt.subplot(gs[1, 3])
    ax_false_edges = plt.subplot(gs[1, 2])




    # fig, (ax_graph, ax_msp, ax_false_edges, ax_spatial_constant) = plt.subplots(1, 3, figsize=(20, 6))
    # if args.dim == 3:
    #     ax_graph = fig.add_subplot(131, projection='3d')
    # else:
    #     ax_graph = fig.add_subplot(131)
    #
    # ax_msp = fig.add_subplot(132)
    # ax_false_edges = fig.add_subplot(133)

    mean_shortest_paths = []
    false_edge_counts = []
    subgraph_size_list = []
    spatial_constant_list = []
    diameter_list = []

    def update(i):
        ax_graph.clear()
        ax_msp.clear()
        ax_false_edges.clear()

        # Set limits for graph axes
        ax_graph.set_xlim([x_min, x_max])
        ax_graph.set_ylim([y_min, y_max])
        if args.dim == 3:
            ax_graph.set_zlim([z_min, z_max])

        # Set limits for mean shortest path plot
        ax_msp.set_ylim([msp_min, msp_max])
        ax_msp.set_xlim([0, nodes_finish])

        # Set limits for false edge count plot
        ax_false_edges.set_ylim([0, len(false_edges)])  # Adjust this based on your data
        ax_false_edges.set_xlim([0, nodes_finish])

        # Set limits for spatial constant
        ax_spatial_constant.set_ylim([0, 1.3])  # Adjust this based on your data
        ax_spatial_constant.set_xlim([0, nodes_finish])

        # Set limits for diameter
        ax_diameter.set_ylim([0, 50])  # Adjust this based on your data
        ax_diameter.set_xlim([0, nodes_finish])

        subgraph = subgraphs[i]

        # Calculate mean shortest path for the subgraph
        msp = get_mean_shortest_path(subgraph)
        mean_shortest_paths.append(msp)
        subgraph_size = subgraph.vcount()
        subgraph_size_list.append(subgraph_size)


        # Diameter
        diameter_list.append(subgraph.diameter())

        # Spatial constant
        degrees = subgraph.degree()
        average_degree = np.mean(degrees)
        S_general = msp / ((subgraph_size ** (1 / args.dim)) * (average_degree ** (-1 / args.dim)))
        spatial_constant_list.append(S_general)





        # Draw nodes in graph subplot
        ax_graph.scatter(positions_df['x'], positions_df['y'], facecolors='none', edgecolors='b')

        false_edge_count = 0
        # Draw edges
        for edge in subgraph.es:
            source_name = subgraph.vs[edge.source]['name']
            target_name = subgraph.vs[edge.target]['name']
            if ((source_name, target_name) in false_edges) or (target_name, source_name) in false_edges:
                false_edge_count += 1

            edge_color = 'red' if (source_name, target_name) in false_edges or (
            target_name, source_name) in false_edges else 'k'
            source = positions_df[positions_df['node_ID'] == source_name].iloc[0]
            target = positions_df[positions_df['node_ID'] == target_name].iloc[0]
            if args.dim == 3:
                ax_graph.plot([source['x'], target['x']], [source['y'], target['y']], [source['z'], target['z']],
                              edge_color, linewidth=0.5)
            else:
                ax_graph.plot([source['x'], target['x']], [source['y'], target['y']], edge_color, linewidth=0.5)



        # MSP plot
        ax_msp.plot(subgraph_size_list, mean_shortest_paths, color='blue')
        ax_msp.set_xlabel('Subgraph Size')
        ax_msp.set_ylabel('Mean Shortest Path')

        false_edge_counts.append(false_edge_count)
        # False edge plot
        ax_false_edges.plot(subgraph_size_list, false_edge_counts, color='red')
        ax_false_edges.set_xlabel('Subgraph Size')
        ax_false_edges.set_ylabel('False Edge Count')
        # ax_false_edges.set_title('False Edge Count over BFS steps')


        # Spatial Constant Plot
        ax_spatial_constant.plot(subgraph_size_list, spatial_constant_list, color='blue')
        ax_spatial_constant.set_xlabel('Subgraph Size')
        ax_spatial_constant.set_ylabel('Spatial Constant')
        # ax_spatial_constant.set_title('Spatial Constant over BFS steps')

        # Diameter plot
        ax_diameter.plot(subgraph_size_list, diameter_list, color='blue')
        ax_diameter.set_xlabel('Subgraph Size')
        ax_diameter.set_ylabel('Diameter')


    # Create animation
    anim = FuncAnimation(fig, update, frames=len(subgraphs), interval=500)

    # Save animation
    anim.save(f"{args.directory_map['animation_output']}/bfs_animation_{args.args_title}_{title}.gif")


