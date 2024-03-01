Tutorial
===========================================================

Introduction
------------

This tutorial guides you through the process of loading, processing, analyzing, and reconstructing graphs. You'll learn how to configure the analysis, visualize graph properties, perform spatial coherence validation, and reconstruct the graph based on your settings.

Configuration
-------------

Start by setting up your configuration in a Python file or dictionary. This includes specifying the proximity mode, dimensionality, reconstruction mode, and other parameters relevant to your analysis. For the complete parameter documentation, please see :ref:`args`.


.. code-block:: python

    # Base settings common to all scenarios
    base = {
        "proximity_mode": "delaunay_corrected",
        "dim": 2,
        "false_edges_count": 0,
        "colorfile": 'colored_squares.png',
        "plot_graph_properties": False,
        "large_graph_subsampling": False,
        "max_subgraph_size": 3000,
        "reconstruct": True,
        "reconstruction_mode": "landmark_isomap"
    }

    # Additional settings for simulation
    simulation = {
        "num_points": 1000,
        "intended_av_degree": 5,
        'plot_original_image': True
    }

    # Settings for experimental scenarios
    experiment = {
        "edge_list_title": "subgraph_0_nodes_2053_edges_2646_degree_2.58.pickle",
        "weighted": False,
        "weight_threshold": 0,
    }

Ensure to adjust these settings according to your analysis needs.

Loading and Processing the Graph
--------------------------------

The script starts by creating the necessary project structure, loading the graph based on the specified settings, and subsampling the graph if necessary to manage large datasets efficiently.

.. code-block:: python

    create_project_structure()
    graph, args = load_and_initialize_graph()
    graph = subsample_graph_if_necessary(graph, args)

Graph Analysis and Visualization
--------------------------------

An optional step is to plot the graph's properties and the positions of the original image (if in a simulated setting where ground truth is known):

.. code-block:: python

    plot_and_analyze_graph(graph, args)
    


**Original Graph Visualization**: Here's how the original graph looks like. We plot the edges according to the graph structure and color the nodes according to a sample image.

.. image:: _static/images/original_image_N=10000_dim=2_delaunay_corrected_k=15
   :alt: Original Graph

**Graph Property Analysis**: After analyzing the graph properties, you can visualize aspects like degree distribution and clustering coefficient.

   
.. raw:: html

    <div style="display: flex; justify-content: space-between;">
        <img src="_static/images/degree_dist_N=10000_dim=2_delaunay_corrected_k=15" alt="Degree Distribution" style="width: 48%;" />
        <img src="_static/images/plots_shortest_path_distribution_N=10000_dim=2_delaunay_corrected_k=15" alt="Shortest Path Distribution" style="width: 48%;" />
    </div>

Spatial Coherence Validation and Reconstruction
-----------------------------------------------

The script further validates the network spatial coherence. This is done in 3 steps: spatial constant, correlation dimension and analyzing the rank of the gram matrix (from the shortest path matrix). 

.. code-block:: python

    compute_shortest_paths(graph, args)
    spatial_constant_analysis(graph, args)
    network_correlation_dimension(args)
    rank_matrix_analysis(args)

**Spatial Analysis Plot**: The Spatial Constant remains constant as the graph grows. This when there are no false edges, i.e., edges that connect distant regions in the original image. If we artificially inject false edges, we can see how the Spatial Constant is affected by them. The more edges, the bigger the drop.

.. image:: _static/images/mean_s_general_vs_intended_size_N=3000_dim=2_delaunay_corrected_k=15_false_edge_version.svg
   :alt: Spatial Analysis

**Network Correlation Dimension**: The correlation dimension is obtained by performing BFS from a central node and observing the relationship between the shortest path distance and the number of discovered nodes. In an Euclidean point cloud, this relationship is a power-law with the Euclidean dimension as the power. We expect a similar behavior from networks.

.. image:: _static/images/dimension_prediction_by_node_count_LINEAR_N=10000_dim=2_delaunay_corrected_k=15.svg
   :alt: Network Correlation Dimension

**Rank Matrix Analysis**: We expect the shortest path distance matrix to be a low-rank matrix, or at least a good approximation of a low-rank matrix. This is because such distances come originally from Euclidean space, and Euclidean Distance Matrices (EDMs) have rank at most d+2, where d is the dimension. In fact, the Gram "dot product" matrix obtained from EDMs have rank at most d. Therefore, we can inspect the Gram matrix obtained from our shortest path distance matrix and see if the "d" largest eigenvalues account for most of the eigenvalue contribution.

.. image:: _static/images/mds_cumulative_singular_values_N=10000_dim=2_delaunay_corrected_k=15_sp_matrix.svg
   :alt: Rank Matrix Analysis

**Graph Reconstruction**: Finally, we can reconstruct the original image using, for example, the STRND algorithm.

.. image:: _static/images/reconstructed_image_N=10000_dim=2_delaunay_corrected_k=15_node2vec
   :alt: Graph Reconstruction

Note: Large Graphs
------------------

For large graphs, computations take too long or run out of memory. This can be solved by subsampling the graph, under the (strong) assumption that the behavior in the subgraph will be the same as in the whole graph. Just set the configuration parameters "large_graph_subsampling =True" and "max_subgraph_size" according to your needs.

.. code-block:: python

    # Base settings common to all scenarios
    base = {
        "proximity_mode": "delaunay_corrected",
        "dim": 2,
        "false_edges_count": 0,
        "colorfile": 'colored_squares.png',
        "plot_graph_properties": False,
        "large_graph_subsampling": True,
        "max_subgraph_size": 3000,
        "reconstruct": True,
        "reconstruction_mode": "landmark_isomap"
    }




