from typing import Optional, List
import warnings

import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence

# TODO:

# 1. Encoder should be telling the user what model's encoder it is using
# 2. Till now, simgnn graphsim and gmn are included here

class GraphEncoder(torch.nn.Module):
    """Encoder module that projects node and edge features to embedding space."""
    def __init__(self, input_node_dim, input_edge_dim, node_hidden_sizes = None, 
                 edge_hidden_sizes = None, filters: list = [64, 32, 16], conv_type: str = 'gcn',
                 name: str ='gmn'):
        """
        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()
        self.name = name
        self.input_node_dim = input_node_dim
        self.input_edge_dim = input_edge_dim
        if self.name in ['simgnn', 'graphsim']:
            if edge_hidden_sizes:
                raise TypeError('{} model does not support edge encoding, edge_hidden_sizes \
                    argument should be None Type'.format(self.name))
        else:
            self._edge_hidden_sizes = edge_hidden_sizes
        self._node_hidden_sizes = node_hidden_sizes
        if self.name == 'gmn':
            self.build_gmn_model()
        elif self.name == 'simgnn':
            self.conv_type = conv_type
            self.conv_filter_list = filters
            self.build_simgnn_model()

    def build_gmn_model(self):
        # Build as LRLRL...
        if self._node_hidden_sizes:
            layer = []
            layer.append(torch.nn.Linear(self.input_node_dim, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(torch.nn.ReLU())
                layer.append(torch.nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP1 = torch.nn.Sequential(*layer)
        else:
            self.MLP1 = None

        if self._edge_hidden_sizes:
            # unneeded?
            if not self.input_edge_dim:
                raise ValueError("Edge transformation applied but 0 edge dimension given")
            layer = []
            layer.append(torch.nn.Linear(self.input_edge_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(torch.nn.ReLU())
                layer.append(torch.nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = torch.nn.Sequential(*layer)
        else:
            self.MLP2 = None
    
    def build_simgnn_model(self):
        # Make this custom length
        if(len(self.conv_filter_list) == 3):
            conv_methods = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
            _conv = conv_methods[self.conv_type]
            self.conv1 = _conv(self.input_node_dim, self.conv_filter_list[0])
            self.conv2 = _conv(self.conv_filter_list[0], self.conv_filter_list[1])
            self.conv3 = _conv(self.conv_filter_list[1], self.conv_filter_list[2])
        else:
            raise RuntimeError(
                f"Number of Convolutional layers "
                f"'{len(self.conv_filter_list)}' should be 3")

    def forward(self, node_features, edge_features, dropout: float = 0):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self.name == 'gmn':
            if not self.MLP1:
                node_outputs = node_features
            else:
                node_outputs = self.MLP1(node_features)
            if edge_features is None or self._edge_hidden_sizes is None:
                edge_outputs = edge_features
            else:
                edge_outputs = self.MLP2(node_features)

            return node_outputs, edge_outputs

        elif self.name == 'simgnn':
            features = self.conv1(node_features, edge_features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

            features = self.conv2(features, edge_features)
            features = torch.nn.functional.relu(features)
            features = torch.nn.functional.dropout(features, p = dropout, training = self.training)

            features = self.conv3(features, edge_features)

            return features

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

    def setup_layers(self):
        # TODO: Include error handling for empty MLP layer size lists
        edge_err = (edge_feature_dim is not None) ^ (edge_hidden_sizes is not None)
        if edge_err:
            raise RuntimeError("One of edge_feature_dim or edge_hidden_sizes not specified")

        self.mlp_node = torch.nn.ModuleList()
        num_layers = len(self.node_hidden_sizes)
        self._in = self.node_feature_dim
        for i in range(num_layers):
            self._out = self.node_hidden_sizes[i]
            self.mlp_node.append(torch.nn.Linear(self._in, self._out))
            self._in = self._out
        
        if edge_feature_dim is not None:
            self.mlp_edge = torch.nn.ModuleList()
            num_layers = len(self.edge_hidden_sizes)
            self._in = self.edge_feature_dim
            for i in range(num_layers):
                self._out = self.edge_hidden_sizes[i]
                self.mlp_edge.append(torch.nn.Linear(self._in, self._out))
                self._in = self._out
            
    def forward(self, node_features: Tensor, edge_features: Optional[Tensor] = None):
        if edge_features is not None and edge_feature_dim is None:
            raise RuntimeError("Dimension of edge features not specified while initialising model, \
                                but edge features provided in forward call")
        
        node_features = self.mlp_node(node_features)
        if edge_features is not None:
            edge_features = self.mlp_edge(edge_features)
            return node_features, edge_features
        return node_features

    def __repr__(self):
        return ('{}(node_feature_dim={}, edge_feature_dim={}, node_hidden_sizes={}, \
                    edge_hidden_sizes={})').format(self.__class__.__name__, self.node_feature_dim,
                                                   self.edge_feature_dim, self.node_hidden_sizes,  
                                                   self.edge_hidden_sizes)
