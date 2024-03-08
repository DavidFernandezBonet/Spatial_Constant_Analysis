from euclidean_distance_vs_shortest_path_correlation import make_euclidean_sp_correlation_plot
from gram_matrix_analysis import make_comparative_gram_matrix_plot_euc_sp
from dimension_prediction import make_dimension_prediction_plot
from dimension_prediction import make_euclidean_network_dim_pred_comparison_plot
from spatial_constant_analysis import make_spatial_constant_euc_vs_network

# # # Euclidean and Shortest Path correlation plot
# make_euclidean_sp_correlation_plot(single_series=True, multiple_series=False)
# make_euclidean_sp_correlation_plot(single_series=False, multiple_series=True)

# TODO: add shortest path from central node visualization with false edges (hetmap from central node)

# # # Gram Matrix
# make_comparative_gram_matrix_plot_euc_sp()

## Dimension Prediction
# # (Euc vs network)
# make_euclidean_network_dim_pred_comparison_plot()

# # (False edges)
# make_dimension_prediction_plot(num_central_nodes=1)   # num central nodes averages the prediction between several central nodes


# ## Spatial Constant
# # # (Euc vs network)
make_spatial_constant_euc_vs_network()

## Spatial Constant --> Run spatial constant pipeline: false edges [0,20,40,60,80], knn, N=1000

## Reconstructed Images --> Run reconstruction, landmark isomap, N=10000, delaunay_corrected