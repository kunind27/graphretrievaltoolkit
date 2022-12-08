from typing import Optional, List
import warnings

import torch
from torch.functional import Tensor

from ..utils.utility import setup_linear_nn

class MLPEncoder(torch.nn.Module):
    r"""
    """
    def __init__(self, node_feature_dim: int, node_hidden_sizes: List[int], 
                edge_feature_dim: Optional[int] = None, edge_hidden_sizes: Optional[List[int]] = None):
        super(MLPEncoder, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.node_hidden_sizes = node_hidden_sizes
        self.edge_hidden_sizes = edge_hidden_sizes
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # TODO: Include error handling for empty MLP layer size lists
        edge_err = (self.edge_feature_dim is not None) ^ (self.edge_hidden_sizes is not None)
        if edge_err:
            raise RuntimeError("One of edge_feature_dim or edge_hidden_sizes not specified")
        self._in = self.node_feature_dim
        self.mlp_node = setup_linear_nn(self._in, self.node_hidden_sizes)
        
        if self.edge_feature_dim is not None:
            self._in = self.edge_feature_dim
            self.mlp_edge = setup_linear_nn(self._in, self.edge_hidden_sizes)
    
    def reset_parameters(self):
        for lin in self.mlp_node:
            lin.reset_parameters()
        for lin in self.mlp_edge:
            lin.reset_parameters()

    def forward(self, node_features: Tensor, edge_features: Optional[Tensor] = None):
        if edge_features is not None and self.edge_feature_dim is None:
            raise RuntimeError("Dimension of edge features not specified while initialising model, \
                                but edge features provided in forward call")
        
        for layer_idx, lin in enumerate(self.mlp_node):
            node_features = lin(node_features)
            node_features = torch.nn.functional.relu(node_features) if layer_idx != len(self.mlp_node) - 1 else node_features
        if edge_features is not None:
            for layer_idx, lin in enumerate(self.mlp_edge):
                edge_features = lin(edge_features)
                edge_features = torch.nn.functional.relu(edge_features) if layer_idx != len(self.mlp_edge) - 1 else edge_features
    
            return node_features, edge_features
        return node_features

    def __repr__(self):
        return ('{}(node_feature_dim={}, edge_feature_dim={}, node_hidden_sizes={}, \
                    edge_hidden_sizes={})').format(self.__class__.__name__, self.node_feature_dim,
                                                   self.edge_feature_dim, self.node_hidden_sizes,  
                                                   self.edge_hidden_sizes)
