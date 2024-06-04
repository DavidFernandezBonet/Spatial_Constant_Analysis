


import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from scipy.sparse import csr_matrix
from pygod.detector import DOMINANT
from pygod.metric import eval_roc_auc
from pygod.utils import to_edge_score

def convert_sparse_to_torch_graph(sparse_matrix):
    """
    Converts a sparse matrix representation of a graph to PyTorch Geometric Data format.

    Args:
        sparse_matrix (csr_matrix): The sparse matrix representing the graph.

    Returns:
        Data: A PyTorch Geometric Data object without node features.
    """
    edge_index = torch.tensor(sparse_matrix.nonzero(), dtype=torch.long)
    return Data(edge_index=edge_index)


def add_outlier_mask_to_torch_graph(data, outlier_edge_ids):
    """
    Adds an outlier mask to a PyTorch Geometric Data object indicating outlier edges.

    Args:
        data (torch_geometric.data.Data): The data object containing the graph.
        outlier_edge_ids (list of tuples): List of outlier edge IDs.

    Returns:
        torch_geometric.data.Data: The updated data object with the 'y' attribute for outliers.
    """
    edge_index = data.edge_index
    is_outlier = torch.zeros(edge_index.size(1), dtype=torch.bool)
    edge_list = edge_index.t().tolist()

    # Adding check for both directions of edges, if necessary
    for idx, (u, v) in enumerate(edge_list):
        if (u, v) in outlier_edge_ids or (v, u) in outlier_edge_ids:
            is_outlier[idx] = True

    # Assign the outlier mask to the data object
    data.y = is_outlier
    return data

def add_dummy_node_features(data, num_features=1):
    """
    Adds dummy node features to a PyTorch Geometric Data object.

    Args:
        data (torch_geometric.data.Data): The data object containing the graph.
        num_features (int): Number of dummy features to add.

    Returns:
        torch_geometric.data.Data: The updated data object with node features.
    """
    num_nodes = data.edge_index.max().item() + 1  # Calculate the number of nodes
    data.x = torch.ones((num_nodes, num_features), dtype=torch.float)  # Add dummy features
    return data

def train_and_evaluate_pygod_model(data, epochs=100):
    """
    Trains and evaluates a PyGOD model using the DOMINANT algorithm.

    Args:
        data (Data): A PyTorch Geometric Data object containing the graph data.
        epochs (int): Number of training epochs.

    Returns:
        dict: A dictionary containing predictions, raw scores, probabilities, confidence, and AUC score.
    """
    detector = DOMINANT(hid_dim=64, num_layers=4, epoch=epochs)
    detector.fit(data)

    pred, node_score, prob, conf = detector.predict(data, return_pred=True, return_score=True, return_prob=True,
                                               return_conf=True)
    edge_score = to_edge_score(node_score, data.edge_index)
    auc_score = eval_roc_auc(data.y, edge_score)

    results = {
        'Labels': pred,
        'Raw scores': edge_score,
        'Probability': prob,
        'Confidence': conf,
        'AUC Score': auc_score
    }

    return results
