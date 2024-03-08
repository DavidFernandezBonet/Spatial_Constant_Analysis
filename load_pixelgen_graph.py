import igraph
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from pixelator import read, simple_aggregate
import seaborn as sns
import networkx as nx
from structure_and_args import GraphArgs

args = GraphArgs()


DATA_DIR = args.directory_map['pixelgen_data']

data_files = [
    f"{DATA_DIR}/Uropod_control.dataset.pxl",
    f"{DATA_DIR}/Uropod_CD54_fixed_RANTES_stimulated.dataset.pxl",
]

pg_data_combined = simple_aggregate(
    ["Control", "Stimulated"], [read(path) for path in data_files]
)


components_to_keep = pg_data_combined.adata[
    (pg_data_combined.adata.obs["edges"] >= 8000)
    & (pg_data_combined.adata.obs["tau_type"] == "normal")
].obs.index

pg_data_combined = pg_data_combined.filter(
    components=components_to_keep
)


example_graph = pg_data_combined.graph(
    "RCVCMP0000318_Control", simplify=True, use_full_bipartite=True
)
raw_nx_graph = example_graph.raw
print("writing edge list...")
nx.write_edgelist(raw_nx_graph, f"{DATA_DIR}/raw_nx_graph.csv", data=False, delimiter=",")
print("edge list written...")
print("drawing graph...")
colors = {"red": "#eb4034", "blue": "#323ea8"}
node_colors = list(
    map(
        lambda x: colors["red"] if x == "A" else colors["blue"],
        example_graph.vs.get_attribute("pixel_type"),
    )
)
nx.draw(raw_nx_graph, node_color=node_colors, node_size=30)
plt.savefig(f"{DATA_DIR}/pixelgen_example_graph.svg")