from typing import Optional, List, Type
from sgmatch.utils.utility import Namespace

import torch
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv

from ..modules.attention import GlobalContextAttention
from ..modules.scoring import NeuralTensorNetwork
from ..utils.utility import setup_linear_nn, setup_conv_layers

class SimGNN(torch.nn.Module):
    r"""
    End to end implementation of SimGNN from the `"SimGNN: A Neural Network Approach
    to Fast Graph Similarity Computation" <https://arxiv.org/pdf/1808.05689.pdf>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any

    Args:
        input_dim (int): Input dimension of node feature embedding vectors
        ntn_slices (int): Hyperparameter for the number of tensor slices in the
            Neural Tensor Network. In this domain, it denotes the number of interaction 
            (similarity) scores produced by the model for each graph embedding pair.
        filters ([int]): Number of filters per convolutional layer in the graph 
            convolutional encoder model. (default: :obj:`[64, 32, 16]`)
        mlp_neurons ([int]): Number of hidden neurons in each linear layer of 
            MLP for reducing dimensionality of concatenated output of neural 
            tensor network and histogram features. Note that the final scoring 
            weight tensor of size :obj:`[mlp_neurons[-1], 1]` is kept separate
            from the MLP, therefore specifying only the hidden layer sizes will
            suffice. (default: :obj:`[32,16,8,4]`)
        hist_bins (int): Hyperparameter controlling the number of bins in the node 
            ordering histogram scheme. (default: :obj:`16`)
        conv (str): Type of graph convolutional architecture to be used for encoding
            (:obj:`'GCN'` or :obj:`'SAGE'` or :obj:`'GAT'`) (default: :obj:`'GCN'`)
        activation (str): Type of activation used in Attention and NTN modules. 
            (:obj:`'sigmoid'` or :obj:`'relu'` or :obj:`'leaky_relu'` or :obj:`'tanh'`) 
            (default: :obj:`'tanh`)
        activation_slope (float, Optional): Slope of function for leaky_relu activation. 
            (default: :obj:`None`)
        include_histogram (bool): Flag for including Strategy Two: Nodewise comparison
            from SimGNN. (default: :obj:`True`)
    """
    def __init__(self, av: Type[Namespace], input_dim: int, ntn_slices: int = 16, filters: list = [64, 32, 16],
                 mlp_neurons: List[int] = [32,16,8,4], hist_bins: int = 16, conv: str = "GCN", 
                 activation: str = "tanh", activation_slope: Optional[float] = None, 
                 include_histogram: bool = True):
        # TODO: give a better name to the include_histogram flag 
        super(SimGNN, self).__init__()
        self.input_dim = av.input_dim
        self.ntn_slices = av.ntn_slices
        self.filters = av.filters
        self.mlp_neurons = av.mlp_neurons
        self.hist_bins = av.hist_bins
        self.conv_type = av.conv
        self.activation = av.activation
        self.activation_slope = av.activation_slope
        self.include_histogram = av.include_histogram

        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        # XXX: Should MLP and GNNs be defined as separate classes instead of methods?
        # XXX: Use MLPEncoder for MLP model
        # XXX: How to properly separate activations given to attention and NTN? 
        # XXX: What dimensions to use at end/start of each layer?

        # Convolutional GNN layer
        self.convs = setup_conv_layers(self.input_dim, conv_type=self.conv_type, filters=self.filters)

        # Global self attention layer
        self.attention_layer = GlobalContextAttention(self.filters[-1], activation = self.activation, 
                                                      activation_slope=self.activation_slope)
        # Neural Tensor Network module
        self.ntn_layer = NeuralTensorNetwork(self.filters[-1], slices = self.ntn_slices, activation = self.activation)
        
        # MLP layer
        if self.include_histogram:
            self._in = self.ntn_slices + self.hist_bins
        else: 
            self._in = self.ntn_slices
        self.mlp = setup_linear_nn(self._in, self.mlp_neurons)
        self.scoring_layer = torch.nn.Linear(self.mlp_neurons[-1], 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.attention_layer.reset_parameters()
        self.ntn_layer.reset_parameters()
        for lin in self.mlp:
            lin.reset_parameters()
        self.scoring_layer.reset_parameters()
        
    def forward(self, x_i: Tensor, edge_index_i: Tensor, x_j: Tensor, edge_index_j: Tensor,
                conv_dropout: int = 0):
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
        if self.include_histogram:
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
            # TODO: Normalise histogram features
            hist = hist.unsqueeze(-1)
            interaction = torch.cat((interaction, hist), dim = -2)
        
        # Final interaction score prediction
        for _, lin in enumerate(self.mlp):
            interaction = lin(interaction)
            interaction = torch.nn.functional.relu(interaction)
        # XXX: should torch.sigmoid be used for normalization of scores?
        interaction = self.scoring_layer(interaction)
        
        return interaction
    
    def loss(self, sim, gt):
        num_graph_pairs = sim.shape[-1] # Batch size

        batch_loss = torch.div(torch.sum(torch.square(sim-gt), dim=-1), num_graph_pairs)

        return batch_loss