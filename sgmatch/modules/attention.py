from typing import Optional

import torch
from torch.functional import Tensor
from ..utils.constants import ACTIVATIONS

class GlobalContextAttention(torch.nn.Module):
    r"""
    Attention Mechanism layer for the attention operator from the 
    `"SimGNN: A Neural Network Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper

    TODO: Include latex formula for attention computation and aggregation update
    
    Args:
        input_dim: Input Dimension of the Node Embeddings
        activation: The Activation Function to be used for the Attention Layer
        activation_slope: Slope of the -ve part if the activation is Leaky ReLU
    """
    def __init__(self, input_dim: int, activation: str = "tanh", activation_slope: Optional[float] = None):
        super(GlobalContextAttention, self).__init__()
        self.input_dim = input_dim
        self.activation = activation 
        self.activation_slope = activation_slope
        
        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self):
        r"""
        Weight initialization depends upon the activation function used.
        If ReLU/Leaky ReLU : He (Kaiming) Initialization
        If tanh/sigmoid : Xavier Initialization

        TODO: Initialisation methods need justification/reference
        
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim, self.input_dim))

    def reset_parameters(self):
        # BUG: ReLU needs an activation_slope, why? Presumably activation_slope was for leaky relu
        if self.activation == "leaky_relu" or self.activation == "relu":
            if self.activation_slope is None or self.activation_slope <= 0:
                raise ValueError(f"Activation function slope parameter needs to be a positive \
                                value. {self.activation_slope} is invalid")
            
            torch.nn.init.kaiming_normal_(self.weight_matrix, a = self.activation_slope, nonlinearity = self.activation)
        elif self.activation == "tanh" or self.activation == "sigmoid":
            torch.nn.init.xavier_normal_(self.weight_matrix)
        else:
            raise ValueError("Activation can only take values: 'relu', 'leaky_relu', 'sigmoid', 'tanh';\
                            {} is invalid".format(self.activation))

    def forward(self, x: Tensor):
        r""" 
        Args:
            x (torch.Tensor) : Node Embedding Tensor of shape N x D.
        
        Returns:
            representation (torch.Tensor): Global graph representation for input node 
            representation set.
        """
        if x.shape[1] != self.input_dim:
            raise RuntimeError("dim 1 of input tensor does not match dimension of weight matrix")
        # XXX: Have these dicts stored in separate files?
        activations = {"tanh": torch.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                        "relu": torch.relu, "sigmoid": torch.sigmoid}
        if self.activation not in activations.keys():
            raise ValueError(f"Invalid activation function specified: {self.activation}")

        # Generating the global context vector
        global_context = torch.mean(torch.matmul(x, self.weight_matrix), dim = 0)

        # Applying the Non-Linearity over global context vector
        _activation = ACTIVATIONS[self.activation]
        global_context = _activation(global_context)

        # Computing attention weights and att-weight-aggregating node embeddings
        att_weights = torch.sigmoid(torch.matmul(x, global_context.view(-1, 1)))
        representation = torch.sum(x * att_weights, dim = 0)
        
        return representation
    
    def __repr__(self):
        return ('{}(input_dim={})').format(self.__class__.__name__, self.input_dim)

class CrossGraphAttention(torch.nn.Module):
    r"""
    Attention mechanism layer for the cross-graph attention operator
    from the `"Graph Matching Networks for Learning the Similarity of Graph 
    Structured Objects" https://arxiv.org/pdf/1904.12787.pdf`_ paper

    TODO: Include latex formula for attention computation and aggregation update

    Args:
        similarity_metric: Similarity metric to be used to compute attention scoring 
    """
    def __init__(self, similarity_metric: str = "euclidean"):
        super(CrossGraphAttention, self).__init__()
        self.similarity = similarity_metric

    def forward(self, h_i: Tensor, h_j: Tensor):
        r"""
        """
        sim_dict = {"euclidean": torch.cdist, "cosine": torch.nn.functional.cosine_similarity}
        # XXX: Have these dicts stored in separate files?
        if self.similarity not in sim_dict.keys():
            raise ValueError(f"Invalid similarity metric specified: {self.similarity}")
        self._sim = sim_dict[self.similarity]

        # Attention weight calculation
        a = self._sim(h_i, h_j) # [N, M]
        a_i = torch.softmax(a, dim=0) # att. weights for aggregating nodes in, j->i
        h_i -= torch.matmul(a_i, h_j)

        a_j = torch.softmax(a, dim=1) # att. weights for nodes in graph h_j, i->j
        h_j -= torch.matmul(a_j.transpose(-1,0), h_i)

        return h_i, h_j

    def __repr__(self):
        return ('{}()').format(self.__class__.__name__)


        