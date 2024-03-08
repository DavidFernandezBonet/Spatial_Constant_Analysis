import numpy as np
import matplotlib.pyplot as plt

from metrics import QualityMetrics
from sklearn.manifold import Isomap

import scienceplots
plt.style.use(['science', 'ieee'])

# Step 1: Create 2D Spiral Point Cloud
def create_2d_spiral(length=1000, turns=3, spacing=0.05, thickness=0.001):
    # Arm Spacing
    theta = np.linspace(0, 2 * np.pi * turns, length)
    r = spacing * theta / (2 * np.pi)

    # Main Spiral Arm with Thickness
    noise_x = np.random.normal(0, thickness, length)
    noise_y = np.random.normal(0, thickness, length)
    x = r * np.cos(theta) + noise_x
    y = r * np.sin(theta) + noise_y
    spiral_with_thickness = np.column_stack((x, y))

    return spiral_with_thickness

# Step 2: Add White Noise

def add_variable_noise(points, min_noise_level=0, max_noise_level=0.05):
    # Calculate noise levels for each point
    # Using a skewed distribution towards min_noise_level
    points_noise = points.copy()
    noise_levels = np.random.triangular(min_noise_level, min_noise_level, max_noise_level, size=len(points))
    # Apply the calculated noise levels to each point
    for i, noise_level in enumerate(noise_levels):
        noise = np.random.normal(0, noise_level, points.shape[1])
        points_noise[i] += noise
    return points_noise

# Step 3: Transform into a Line
def to_line(points):
    points_line = points.copy()
    y_range = 0
    points_line[:, 0] = np.linspace(0, 1, len(points_line))  # Modify x-coordinate
    points_line[:, 1] = np.random.uniform(-y_range, y_range, len(points_line))  # Set y-coordinate to a low random value
    return points_line

# Define the function to apply Isomap for dimensionality reduction
def apply_isomap(points, n_neighbors=10, n_components=2):
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    transformed_points = isomap.fit_transform(points)
    return transformed_points


def plot_points_2d(points, title, metrics_dict=None):
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=1, s=0.1)

    if metrics_dict != None:
        # Annotate metrics on the plot
        textstr = '\n'.join([f'{key}: {value:.3f}' for key, value in metrics_dict.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Place a text box in upper left in axes coords
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=6,
                       verticalalignment='top', bbox=props)

    plt.title(title)
    plt.axis('equal')
    plt.savefig(f'{title}.pdf')

    plt.close()


# Generate the point clouds
original_points = create_2d_spiral()
plot_points_2d(original_points, 'Spiral')

noisy_points = add_variable_noise(original_points)

line_points = apply_isomap(original_points)

# Plotting with Viridis color gradient

qm_noisy = QualityMetrics(original_points, noisy_points)
noisy_metrics = qm_noisy.evaluate_metrics()

qm_line = QualityMetrics(original_points, line_points)
line_metrics = qm_line.evaluate_metrics()

plot_points_2d(noisy_points, 'Spiral with Noise', noisy_metrics)
plot_points_2d(line_points, 'Spiral reconstructed with Isomap', line_metrics)


