from euclidean_distance_vs_shortest_path_correlation import make_euclidean_sp_correlation_plot
from gram_matrix_analysis import make_comparative_gram_matrix_plot_euc_sp
from dimension_prediction import make_dimension_prediction_plot
from dimension_prediction import make_euclidean_network_dim_pred_comparison_plot
from spatial_constant_analysis import make_spatial_constant_euc_vs_network
import matplotlib.pyplot as plt
from check_latex_installation import check_latex_installed
from structure_and_args import GraphArgs
import numpy as np
import random

is_latex_in_os = check_latex_installed()
if is_latex_in_os:
    plt.style.use(['nature'])
else:
    plt.style.use(['no-latex', 'nature'])
plt.style.use(['no-latex', 'nature'])
# font_size = 24
# plt.rcParams.update({'font.size': font_size})
# plt.rcParams['axes.labelsize'] = font_size
# plt.rcParams['axes.titlesize'] = font_size + 6
# plt.rcParams['xtick.labelsize'] = font_size
# plt.rcParams['ytick.labelsize'] = font_size
# plt.rcParams['legend.fontsize'] = font_size - 10


base_figsize = (6, 4.5)  # Width, Height in inches
base_fontsize = 18
plt.rcParams.update({
    'figure.figsize': base_figsize,  # Set the default figure size
    'figure.dpi': 300,  # Set the figure DPI for high-resolution images
    'savefig.dpi': 300,  # DPI for saved figures
    'font.size': base_fontsize,  # Base font size
    'axes.labelsize': base_fontsize ,  # Font size for axis labels
    'axes.titlesize': base_fontsize + 2,  # Font size for subplot titles
    'xtick.labelsize': base_fontsize,  # Font size for X-axis tick labels
    'ytick.labelsize': base_fontsize,  # Font size for Y-axis tick labels
    'legend.fontsize': base_fontsize - 6,  # Font size for legends
    'lines.linewidth': 2,  # Line width for plot lines
    'lines.markersize': 6,  # Marker size for plot markers
    'figure.autolayout': True,  # Automatically adjust subplot params to fit the figure
    'text.usetex': False,  # Use LaTeX for text rendering (set to True if LaTeX is installed)
})


plot_folder = GraphArgs().directory_map['euc_vs_net']
np.random.seed(42)
random.seed(42)


# # # # Euclidean and Shortest Path correlation plot
# make_euclidean_sp_correlation_plot(single_series=True, multiple_series=False, useful_plot_folder=plot_folder)
# make_euclidean_sp_correlation_plot(single_series=False, multiple_series=True, useful_plot_folder=plot_folder)
#
# # TODO: add shortest path from central node visualization with false edges (hetmap from central node)
#
# # # # Gram Matrix
# make_comparative_gram_matrix_plot_euc_sp(useful_plot_folder=plot_folder)
#
# ## Dimension Prediction
# # # (Euc vs network)
# make_euclidean_network_dim_pred_comparison_plot(useful_plot_folder=plot_folder)
#
# # # (False edges)
# # make_dimension_prediction_plot(num_central_nodes=1)   # num central nodes averages the prediction between several central nodes


# ## Spatial Constant
# # # (Euc vs network)
make_spatial_constant_euc_vs_network(useful_plot_folder=plot_folder)


## Reconstructed Images --> Run reconstruction, landmark isomap, N=10000, delaunay_corrected