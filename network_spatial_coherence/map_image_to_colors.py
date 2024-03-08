import pandas as pd
from PIL import Image
import numpy as np
from structure_and_args import GraphArgs

def load_image(image_path):
    return Image.open(image_path)


def map_points_to_colors(original_positions_df, filename, args):
    # Load the image

    image_folder = args.directory_map["colorfolder"]
    image_path = f'{image_folder}/{filename}'  # Update this to the path of your .jpg image

    image = load_image(image_path)
    image = image.convert("RGB")  # Ensure image is in RGB format
    pixels = np.array(image)

    # Calculate the scaling factors
    image_width, image_height = image.size
    df_min_x, df_max_x = original_positions_df['x'].min(), original_positions_df['x'].max()
    df_min_y, df_max_y = original_positions_df['y'].min(), original_positions_df['y'].max()

    scale_x = image_width / (df_max_x - df_min_x)
    scale_y = image_height / (df_max_y - df_min_y)

    # Dictionary to hold node_ID to color mapping
    node_id_to_color = {}

    # Iterate over each point in the DataFrame
    for index, row in original_positions_df.iterrows():
        # Scale the point's coordinates
        x_scaled = int((row['x'] - df_min_x) * scale_x)
        y_scaled = int((row['y'] - df_min_y) * scale_y)

        node_id = row['node_ID']

        # Ensure the point is within the image bounds
        if x_scaled >= 0 and y_scaled >= 0 and x_scaled < image_width and y_scaled < image_height:
            # Get the color at the corresponding point and normalize it
            color = tuple(pixels[y_scaled, x_scaled] / 255.0)  # Normalize to 0-1 range
            node_id_to_color[node_id] = color
        else:
            node_id_to_color[node_id] = (0, 0, 0, 1)  # Default color (black, fully opaque) if out of bounds

    return node_id_to_color


