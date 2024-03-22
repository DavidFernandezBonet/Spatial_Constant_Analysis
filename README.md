# Network Spatial Coherence
Python library to validate the spatial coherence of a network. It offers tools to analyze network properties, check how "Euclidean" the network is (spatial coherence), and to reconstruct the network. Networks can be both simulated (e.g. a KNN network) or imported.

## Features
- Analyze the spatial coherence of a network
- Reconstruction images from purely network information
- Efficient graph loading and processing (using sparse matrices or getting a graph sample)


## Install
Python 3.11 is reccomended, although older versions should work.

```bash
pip install git+https://github.com/DavidFernandezBonet/Spatial_Constant_Analysis.git
```
If you require authentication you can use a PAT (a github token) instead. Go to Developer settings > Personal access tokens > Generate new token and then save the token because it will not be displayed again. You should input it in this line of code
```bash
pip install git+https://<token>:x-oauth-basic@github.com/DavidFernandezBonet/Spatial_Constant_Analysis.git
```
## Usage
For a detailed tutorial, see the [Jupyter Notebook Tutorial](./network_spatial_coherence/network_spatial_coherence_tutorial.ipynb) in this repository.

1. Access documentation for detailed API usage:

```python
from network_spatial_coherence.docs_util import access_docs
access_docs()
```

2. Minimum working example

```python
from network_spatial_coherence import nsc_pipeline
from network_spatial_coherence import structure_and_args
structure_and_args.create_project_structure()
graph, args = nsc_pipeline.load_and_initialize_graph()
nsc_pipeline.run_pipeline(graph, args)
```



## Contact
[dfb@kth.se]

