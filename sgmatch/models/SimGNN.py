from typing import Optional, List

import torch
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv

from ..modules.attention import GlobalContextAttention
from ..modules.ntn import NeuralTensorNetwork

class SimGNN(torch.nn.Module):
    r"""
    End to end implementation of SimGNN from the `"SimGNN: A Neural Network Approach
    to Fast Graph Similarity Computation" <https://arxiv.org/pdf/1808.05689.pdf>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any
    """
    def __init__(self, input_dim: int, ntn_slices: int = 16, filters: list = [64, 32, 16],
                 mlp_neurons: List[int] = [32,16,8,4], hist_bins: int = 16, conv: str = "gcn", 
                 activation = "tanh", include_histogram = False):
        # TODO: give a better name to the include_histogram flag 
        super(SimGNN, self).__init__()
        self.input_dim = input_dim
        self.conv_type = conv
        self.filters = filters
        self.activation = activation
        self.mlp_neurons = mlp_neurons
        
        # Hyperparameters
        self.ntn_slices = ntn_slices
        self.hist_bins = hist_bins
    
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # XXX: Should MLP and GNNs be defined as separate classes to avoid clutter?

        # Convolutional GNN layer
        self.convs = torch.nn.ModuleList()
        conv_methods = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
        _conv = conv_methods[self.conv_type]
        num_layers = len(self.filters)
        self._in = self.input_dim
        for i in range(num_layers):
            self._out = self.filters[i]
            self.convs.append(_conv(in_channels=self._in, out_channels=self._out))
            self._in = self._out

        # Global self attention layer
        self.attention_layer = GlobalContextAttention(self.input_dim, activation = self.activation)
        # Neural Tensor Network module
        self.ntn_layer = NeuralTensorNetwork(self.input_dim, slices = self.ntn_slices, activation = self.activation)
        
        # MLP layer
        self.mlp = torch.nn.ModuleList()
        num_layers = len(mlp_neurons)
        if self.include_histogram:
            self._in = self.ntn_slices + self.hist_bins
        else: 
            self._in = self.ntn_slices
        for i in range(num_layers):
            self._out = mlp_neurons[i]
            self.mlp.append(torch.nn.Linear(self._in, self._out))
            self._in = self._out
        self.scoring_layer = torch.nn.Linear(mlp_neurons[-1], 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.attention_layer.reset_parameters()
        self.ntn_layer.reset_parameters()
        for lin in self.mlp:
            lin.reset_parameters()
        
    def forward(self, x_i: Tensor, edge_index_i: Tensor, x_j: Tensor, edge_index_j: Tensor,
                conv_dropout: int = 0):
        r"""
        """
        # Strategy One: Graph-Level Embedding Interaction
        for filter_idx, conv in enumerate(self.convs):
            x_i = conv(x_i, edge_index_i)
            x_j = conv(x_j, edge_index_j)
            
            if filter_idx == len(self.convs) - 1:
                break
            x_i = torch.nn.functional.relu(x_i)
            x_i = torch.nn.functional.dropout(x_i, p = conv_dropout, training = self.training)
            x_j = torch.nn.functional.relu(x_j)
            x_j = torch.nn.functional.dropout(x_j, p = conv_dropout, training = self.training)

        h_i = self.attention_layer(x_i)
        h_j = self.attention_layer(x_j)

        interaction = self.ntn_layer(h_i, h_j) 
        
        # Strategy Two: Pairwise Node Comparison
        if include_histogram:
            sim_matrix = torch.matmul(h_i, h_j.transpose(-1,-2)).detach()
            sim_matrix = torch.sigmoid(sim_matrix)
            # XXX: is this if statement necessary? Can writing the histogram operation as a single 
            # tensor operation not accomodate batching?
            if len(sim_matrix.shape) == 3:
                scores = sim_matrix.view(sim_matrix.shape[0], -1, 1)
                hist = torch.cat([torch.histc(x, bins = self.hist_bins).unsqueeze(0) for x in scores], dim=0)
            else:
                scores = sim_matrix.view(-1, 1)
                hist = torch.histc(scores, bins = self.hist_bins)
            hist = hist.unsqueeze(-1)
            interaction = torch.cat((interaction, hist), dim = -2)
        
        # Final interaction score prediction
        for layer_idx, lin in enumerate(self.mlp):
            interaction = lin(interaction)
            interaction = torch.nn.functional.relu(interaction)
        # XXX: torch.sigmoid used for normalization, appropriate?
        interaction = torch.sigmoid(self.scoring_layer(interaction))
        
        return interaction