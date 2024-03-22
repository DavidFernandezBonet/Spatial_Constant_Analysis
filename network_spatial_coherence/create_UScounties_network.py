from structure_and_args import GraphArgs
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import Point
import scienceplots

plt.style.use(['no-latex', 'nature'])
font_size = 24
plt.rcParams.update({'font.size': font_size})
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['axes.titlesize'] = font_size + 6
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size - 10



def plot_counties(shapefile_path, highlight_geoids=None, exclude_fips=['02', '15', '60', '66', '69', '72', '78']):
    """
    Plots the US counties from a shapefile.

    Parameters:
    - shapefile_path: str, the path to the county shapefile.
    - highlight_geoids: list of str, optional, GEOIDs of counties to highlight.
    """
    # Load the county shapefile
    gdf = gpd.read_file(shapefile_path)
    # Filter out the counties based on the exclude list
    gdf = gdf[~gdf['STATEFP'].isin(exclude_fips)]

    # Plot all counties
    fig, ax = plt.subplots(figsize=(15, 10))
    gdf.plot(ax=ax, color='lightgrey')

    # Highlight specified counties, if any
    if highlight_geoids is not None and len(highlight_geoids) > 0:
        highlight_gdf = gdf[gdf['GEOID'].isin(highlight_geoids)]
        highlight_gdf.plot(ax=ax, color='red')

    ax.set_title("Map of US Counties")
    plt.axis('off')  # Hide axes ticks
    plt.show()


def filter_counties_by_proximity(gdf, central_latitude=39.50, central_longitude=-98.35, radius=None, top_percent=None):

    # Exclude non-mainland counties if needed
    # gdf = gdf[~gdf['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]

    # Define central point and project
    central_point = gpd.GeoSeries([Point(central_longitude, central_latitude)], crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")  # Adjust CRS
    central_point = central_point.to_crs(gdf.crs)

    # Calculate distances
    gdf['distance_to_central'] = gdf.centroid.distance(central_point[0])

    # Filter based on radius or percentage
    if radius is not None:
        radius_in_meters = radius * 1000  # Convert km to meters if needed
        closest_counties = gdf[gdf['distance_to_central'] <= radius_in_meters]
    elif top_percent is not None:
        num_counties = len(gdf)
        count = int(num_counties * top_percent)
        closest_counties = gdf.nsmallest(count, 'distance_to_central')
    else:
        raise ValueError("Either radius or top_percent must be provided")

    return closest_counties
def create_county_adjacency_network(shapefile_path,  data_folder, exclude_fips=['02', '15', '60', '66', '69', '72', '78']):
    # Load the county shapefile
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[~gdf['STATEFP'].isin(exclude_fips)]
    # num_counties = len(gdf)
    # top_10_percent_count = int(num_counties * 0.1)
    # gdf = filter_counties_by_proximity(gdf, top_percent=top_10_percent_count)
    gdf['unique_name'] = gdf.apply(lambda row: f"{row['NAME']} ({row['STATEFP']}{row['COUNTYFP']})", axis=1)

    # Create a mapping for county_name to integer
    name_to_int = {row['unique_name']: index for index, row in gdf.iterrows()}

    # Initialize an empty Graph
    G = nx.Graph()

    # Add nodes to the graph with integer ID
    for name, int_id in name_to_int.items():
        G.add_node(int_id, name=name)
    print("finished adding nodes")

    spatial_index = gdf.sindex

    # Iterate through each county
    for i, county in gdf.iterrows():
        # Find the indexes of all counties that might touch the current county
        # using the spatial index
        possible_matches_index = list(spatial_index.intersection(county['geometry'].bounds))
        possible_matches = gdf.iloc[possible_matches_index]

        # Further filter these candidates to those that actually touch the county
        precise_matches = possible_matches[possible_matches.geometry.touches(county['geometry'])]

        # Add an edge for each matching county
        for _, matching_county in precise_matches.iterrows():
            if county['unique_name'] != matching_county['unique_name']:  # Check to avoid adding an edge to itself
                print("adding edge between", county['unique_name'], "and", matching_county['unique_name'])
                G.add_edge(name_to_int[county['unique_name']], name_to_int[matching_county['unique_name']])

    # Convert the Graph edges to a DataFrame for easier manipulation and export, using integers
    edge_list_df = pd.DataFrame(list(G.edges()), columns=['source', 'target'])

    # Save the edge list and the name-to-integer mapping
    edge_list_df.to_csv(f'{data_folder}/edge_list_us_counties.csv', index=False)
    pd.Series(name_to_int).to_csv(f'{data_folder}/name_to_int_mapping.csv')

    # Visualization (Optional)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, pos=nx.spring_layout(G), node_size=10, edge_color='gray', with_labels=False,
                     labels={node: data['name'] for node, data in G.nodes(data=True)})
    plt.title("US County Adjacency Network")
    plt.axis('off')
    plt.show()

    return G, edge_list_df, name_to_int

    # Basic visualization of the network
    print("finished creating network")
    print("network has", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(G, node_size=10, edge_color='gray', with_labels=False)
    plt.title("US County Adjacency Network")
    plt.axis('off')
    print("finished plotting")
    plt.show()

    return G


def plot_network_from_files(edge_list_path, mapping_path, gdf, data_folder):
    """
    Plots the geographic locations of counties and their adjacency relationships.

    Parameters:
    - edge_list_path: str, the path to the CSV file containing the network's edge list.
    - mapping_path: str, the path to the CSV file containing the mapping from county names to integers.
    - gdf: GeoDataFrame, containing the geographic data of the counties.
    """
    # Load the edge list and the mapping from CSV files
    edge_list_df = pd.read_csv(edge_list_path)
    name_to_int_df = pd.read_csv(mapping_path, header=None, names=['Name', 'ID'])

    # Create a dictionary for name to integer mapping
    name_to_int = pd.Series(name_to_int_df.ID.values, index=name_to_int_df.Name).to_dict()

    # Invert the mapping to facilitate the lookup of county names based on integer identifiers
    int_to_name = {v: k for k, v in name_to_int.items()}

    # Calculate the centroids of each county for plotting
    gdf['centroid'] = gdf.centroid
    centroids = gdf.set_index('unique_name')['centroid'].to_dict()

    # Initialize an empty graph and populate it with edges from the edge list
    G = nx.Graph()
    for _, row in edge_list_df.iterrows():
        source_name = int_to_name[row['source']]
        target_name = int_to_name[row['target']]
        print("source name", source_name, "target name", target_name)
        if source_name in centroids and target_name in centroids:
            G.add_edge(source_name, target_name)

    # Generate a mapping from county names to their centroid coordinates for plotting
    pos = {county: (centroids[county].x, centroids[county].y) for county in G.nodes}
    positions_df = pd.DataFrame([(name_to_int[node_ID], pos[node_ID][0], pos[node_ID][1]) for node_ID in pos],
                                columns=['node_ID', 'x', 'y'])
    positions_df.to_csv(f"{data_folder}/county_positions.csv", index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 15))
    gdf.plot(ax=ax, color='whitesmoke', edgecolor='black')


    # Draw edges with specific alpha
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='blue', ax=ax)
    # nx.draw_networkx_nodes(G, pos, node_size=10, node_color='black', ax=ax, edgecolors='black', linewidths=2)
    nx.draw_networkx_nodes(G, pos, node_size=3, node_color='blue', ax=ax, alpha=1)


    plt.title("US Counties Network")
    plt.axis('off')
    plt.savefig(f"{data_folder}/county_network_with_edges.png", dpi=300, bbox_inches='tight')

    plt.show()

args = GraphArgs()
data_folder = args.directory_map['us_counties']
filename = f"{data_folder}/tl_2023_us_county.shp"
# plot_counties(filename, highlight_geoids=['36061', '06037']) # Just plots the original image counties
# create_county_adjacency_network(filename, data_folder=data_folder)  # Creates the network

## Filter to mainland counties
exclude_fips=['02', '15', '60', '66', '69', '72', '78']
gdf = gpd.read_file(filename)
gdf = gdf[~gdf['STATEFP'].isin(exclude_fips)]
gdf['unique_name'] = gdf.apply(lambda row: f"{row['NAME']} ({row['STATEFP']}{row['COUNTYFP']})", axis=1)

## Plot Network Edges

plot_network_from_files(f"{data_folder}/edge_list_us_counties.csv", f"{data_folder}/name_to_int_mapping.csv",
                        gdf, data_folder=data_folder)
