import os

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from utils import CurveFitting
import pandas as pd
import seaborn as sns
import scienceplots
plt.style.use(['science', 'nature'])


def generate_random_points(number_of_points, density, dimensions):
    """
    Generate random points within a square (2D) or cube (3D) with a characteristic length L.
    The characteristic length L is derived from the density and the number of points.

    :param number_of_points: The total number of points to be generated.
    :param density: The number of points per unit area (2D) or volume (3D).
    :param dimensions: 2 for 2D space, 3 for 3D space.
    :return: Numpy array of points.
    """
    if dimensions == 2:
        area = number_of_points / density
        L = np.sqrt(area)  # Characteristic length for a square
    elif dimensions == 3:
        volume = number_of_points / density
        L = np.cbrt(volume)  # Characteristic length for a cube
    else:
        raise ValueError("Dimensions must be either 2 or 3.")

    # Generate points within a square or cube of side length L
    return np.random.rand(number_of_points, dimensions) * L

def average_pairwise_distance(points, dimensions, density):
    """
    Compute the average pairwise distance among a set of points.

    :param points: Numpy array of points.
    :return: Average pairwise distance.
    """
    num_points = len(points)
    dist_matrix = distance_matrix(points, points)
    tri_dist_matrix = np.triu(dist_matrix, 1)  # Upper triangular part to avoid double counting
    total_distances = tri_dist_matrix.sum()
    num_distances = len(points) * (len(points) - 1) / 2  # Number of unique pairs
    average_distance = total_distances / num_distances if num_distances else 0
    spatial_constant = average_distance / ((num_points)**(1/dimensions))
    super_spatial_constant = average_distance / (((num_points) ** (1 / dimensions)) * (density**(-1/dimensions)))
    return average_distance, spatial_constant, super_spatial_constant



# # Parameters
# density = 1  # Density of points
# dimensions = 2  # 2 for 2D, 3 for 3D
#
#
# points_range = range(100, 2001, 100)  # Range of number of points
# print(list(points_range))
#
# # Calculate average pairwise distances for different numbers of points
# avg_distances = [average_pairwise_distance(generate_random_points(num_points, density, dimensions))
#                  for num_points in points_range]
#
# curve_fit = CurveFitting(np.array(points_range), np.array(avg_distances))
#
# if dimensions == 2:
#     fun_fit = curve_fit.spatial_constant_dim2_linearterm
# elif dimensions == 3:
#     fun_fit = curve_fit.spatial_constant_dim3_linearterm
# curve_fit.perform_curve_fitting(model_func=fun_fit)
# # Set labels, title, and save path
# xlabel = 'Number of Points'
# ylabel = 'Average Distance'
# title = 'Average Distance vs. Number of Points'
# save_path = f'{os.getcwd()}/avg_distance_euclidean_points.png'
# print(save_path)
#
# curve_fit.plot_fit_with_uncertainty(fun_fit, xlabel, ylabel, title, save_path)


# #### Density
# point_range = range(100, 2001, 500)
# density_range = range(1, 201, 2)
# dimensions = 3  # 2 for 2D, 3 for 3D
# num_points = 1000
#
#
# # Calculate average pairwise distances for different numbers of points
# data = [(density, average_pairwise_distance(generate_random_points(num_points, density, dimensions), dimensions, density))
#                  for density in density_range]
#
# density_values, data2 = zip(*data)
# avg_distances, spatial_constant_values, super_spatial_constant_values = zip(*data2)
#
# # curve_fit = CurveFitting(np.array(density_values), np.array(avg_distances))
# curve_fit = CurveFitting(np.array(density_values), np.array(spatial_constant_values))
#
#
# if dimensions == 2:
#     fun_fit = curve_fit.power_model
# elif dimensions == 3:
#     fun_fit = curve_fit.power_model
# curve_fit.perform_curve_fitting(model_func=fun_fit)
# # Set labels, title, and save path
# xlabel = 'Density'
# ylabel = 'Average Distance'
# title = 'Average Distance vs. Density'
# save_path = f'{os.getcwd()}/avg_density_euclidean_points.png'
# print(save_path)
#
# curve_fit.plot_fit_with_uncertainty(fun_fit, xlabel, ylabel, title, save_path)


#### Try to predict
density = 6
dimensions = 2  # 2 for 2D, 3 for 3D
num_points = 1000
constant_scaler = np.sqrt(4.5)
bipartite_correction = 1/1.2  # 1/np.sqrt(2)   #TODO: i don't know exactly how the correction should look like!
super_spatial_constant_3d = 0.66 * constant_scaler
# super_spatial_constant_2d = 0.517
super_spatial_constant_2d = 0.517 * constant_scaler
avg_distance, _, _ = average_pairwise_distance(generate_random_points(num_points, density, dimensions), dimensions, density)

super_spatial_constant_2d_bipartite = super_spatial_constant_2d * bipartite_correction
super_spatial_constant_3d_bipartite = super_spatial_constant_3d * bipartite_correction



if dimensions == 3:
    predicted_avg_dist = super_spatial_constant_3d * ((num_points)**(1/dimensions)) * (density**(-1/dimensions))
elif dimensions == 2:
    predicted_avg_dist = super_spatial_constant_2d * ((num_points) ** (1 / dimensions)) * (density ** (-1 / dimensions))
print("True", avg_distance)
print("Predicted", predicted_avg_dist)


### Try a dataframe
df = pd.read_csv("spatial_constant_data_bipartite_2_and_experimental.csv")
#  mean_shortest_path / ((num_nodes ** (1 / dim)) * (average_degree ** (-1 / dim)))
df['S_general'] = df['mean_shortest_path'] / ( (df['num_nodes']) ** (1 / df['dim']) *
                                               (df['average_degree'] ** (-1 / df['dim'])))


# TODO: justification for these values????
df['Predicted_S'] = np.where(df['dim'] == 2, 0.517 * np.sqrt(4.5),
                             np.where(df['dim'] == 3, 0.66 * np.sqrt(4.5), np.nan))
df['Predicted_S_bipartite'] = np.where(df['dim'] == 2, 0.517 * np.sqrt(4.5) * (1/np.sqrt(2)),
                             np.where(df['dim'] == 3, 0.66 * np.sqrt(4.5) * (1/np.cbrt(2)), np.nan))

# Choosing the correct predicted value based on 'prox_mode'
df['Corrected_Predicted_S'] = np.where(df['proximity_mode'].str.contains('bipartite'),
                                       df['Predicted_S_bipartite'],
                                       df['Predicted_S'])


df.to_csv("spatial_constant_completed.csv")

# fixed_num_nodes = 3000
nodes_to_include = [500, 1000, 2000, 3000]  # well behaved nodes

# Filtering the DataFrame
filtered_df = df[df['num_nodes'].isin(nodes_to_include)]


# filtered_df = df[df['num_nodes'] == fixed_num_nodes]

# filtered_df = filtered_df[(filtered_df['proximity_mode'] != 'epsilon_bipartite')]
# filtered_df = filtered_df[(filtered_df['proximity_mode'] != 'delaunay')]
# filtered_df = filtered_df[(filtered_df['proximity_mode'] != 'delaunay_corrected')]
# filtered_df = filtered_df[(filtered_df['dim'] != 3)]

predicted_avg_dist = super_spatial_constant_2d * (filtered_df['num_nodes']** (1 / dimensions)) * (filtered_df['average_degree'] ** (-1 / dimensions))
# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(filtered_df['average_degree'], filtered_df['mean_shortest_path'], label='Actual Mean Shortest Path')
plt.scatter(filtered_df['average_degree'], predicted_avg_dist, color='red', label='Predicted Avg Distance')

# Drawing vertical lines for residuals
for index, row in filtered_df.iterrows():
    plt.vlines(x=row['average_degree'], ymin=row['mean_shortest_path'], ymax=predicted_avg_dist[index], color='green', linestyles='dashed')

plt.title('Mean Shortest Path and Predicted Shortest Path vs Average Degree with Residuals')
plt.xlabel('Average Degree')
plt.ylabel('Mean Shortest Path')
plt.legend()

plt.savefig("test_avdegree_behavior.png")


predicted_medians = [super_spatial_constant_2d, super_spatial_constant_3d, super_spatial_constant_2d_bipartite, super_spatial_constant_3d_bipartite]  # Replace with your actual values
#### Boxplot S plot
# Define a new column for grouping in the boxplot
filtered_df['Group'] = np.nan

# Assign groups based on conditions
filtered_df.loc[(filtered_df['dim'] == 2) & (filtered_df['proximity_mode'].str.contains('bipartite')), 'Group'] = 'Dim 2, Bipartite'
filtered_df.loc[(filtered_df['dim'] == 2) & (~filtered_df['proximity_mode'].str.contains('bipartite')), 'Group'] = 'Dim 2, Non-Bipartite'
filtered_df.loc[(filtered_df['dim'] == 3) & (filtered_df['proximity_mode'].str.contains('bipartite')), 'Group'] = 'Dim 3, Bipartite'
filtered_df.loc[(df['dim'] == 3) & (~filtered_df['proximity_mode'].str.contains('bipartite')), 'Group'] = 'Dim 3, Non-Bipartite'

# Violin plot
plt.figure(figsize=(12, 8))
groups = filtered_df['Group'].unique()  # This gets the unique group names

# Create the violin plot
sns.violinplot(x='Group', y='S_general', data=filtered_df, inner='quartile')

# Add predicted medians as scatter points
for i, group in enumerate(groups):
    plt.scatter(x=i, y=predicted_medians[i], color='red', zorder=3)

plt.title('Violin Plot of S_general for Different Groups')
plt.ylabel('S_general')
plt.xlabel('Group')
plt.xticks(rotation=45)

plt.savefig("s_general_violinplot.png")



# Interactive plot
import plotly.express as px

# Assuming df is your DataFrame
fig = px.violin(filtered_df, x='Group', y='S_general', box=False, points='all', hover_data=filtered_df.columns)

# Add predicted medians as scatter points
for i, group in enumerate(groups):
    fig.add_scatter(x=[group], y=[predicted_medians[i]], mode='markers', marker=dict(color='red'))
fig.update_layout(title='Interactive Violin Plot of S_general for Different Groups',
                  xaxis_title='Group',
                  yaxis_title='S_general')
fig.show()
fig.write_html("spatial_constant_interactive" + '.html')