# Network Spatial Coherence
Python library to validate the spatial coherence of a network. It offers tools to analyze network properties, check how "Euclidean" the network is (spatial coherence), and to reconstruct the network. Networks can be both simulated (e.g. a KNN network) or imported.

## Features

- Configurable, with solid default parameters but also tunable.
- Efficient graph loading and processing (using sparse matrices).
- Analyze the spatial coherence
- Visualization tools
- Reconstruction algorithms


## Setup

```bash
# Clone repo
git clone https://github.com/YourRepo/YourProject.git
```

# Install
pip install git+https://github.com/DavidFernandezBonet/Spatial_Constant_Analysis.git

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


## Contribute

Report issues, suggest features, or contribute code on GitHub.

## License

[Your License Here]

## Contact

[Your Contact Information Here]

