# Network Spatial Coherence
Python library to validate the spatial coherence of a network. It offers tools to analyze network properties, check how "Euclidean" the network is (spatial coherence), and to reconstruct the network. Networks can be both simulated (e.g. a KNN network) or imported.

## Features
- Analyze the spatial coherence of a network
- Reconstruction images from purely network information
- Efficient graph loading and processing (using sparse matrices or getting a graph sample)


## Install

```bash
pip install git+https://github.com/DavidFernandezBonet/Spatial_Constant_Analysis.git

```

## Usage
For a detailed tutorial on using this toolkit, refer to our [Jupyter Notebook Tutorial](./network_spatial_coherence/network_spatial_coherence_tutorial.ipynb) in this repository.

1. Access documentation for detailed API usage:

```python
from network_spatial_coherence.docs_util import access_docs
access_docs()
```

2. Example analysis workflow:

```python
from network_spatial_coherence import nsc_pipeline
args = nsc_pipeline.default_args()
graph = nsc_pipeline.load_graph(args)
nsc_pipeline.analyze_graph(graph, args)
```



## Contact
[dfb@kth.se]

