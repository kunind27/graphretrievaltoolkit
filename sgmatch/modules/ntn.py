from typing import Optional

import torch
from torch.functional import Tensor

class NeuralTensorNetwork(torch.nn.Module):
    r"""
    Neural Tensor Network layer from the `"SimGNN: A Neural Network 
    Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper

    TODO: Include latex formula for NTN interaction score computation
    
    Args:
        input_dim: Input dimension of the graph-level embeddings
        slices: Number of slices (K) the weight tensor possesses. Often 
        interpreted as the number of entity-pair (in this use case - pairwise
        node) relations the data might possess.
        activation: Non-linearity applied on the computed output of the layer
    """
    def __init__(self, input_dim: int, slices: int = 16, activation: str = "tanh"):
        super(NeuralTensorNetwork, self).__init__()
        self.input_dim = input_dim
        self.slices = slices # K: hyperparameter
        self.activation = activation

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self):
        # XXX: Will arranging weight tensor as (k, d, d) cause problems in batching at dim 0?
        self.weight_tensor = torch.nn.Parameter(torch.Tensor(self.slices, self.input_dim, self.input_dim))
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.slices, 2 * self.input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(self.slices, 1))
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_tensor)
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)
    
    def forward(self, h_i: Tensor, h_j: Tensor):
        r"""
        Args:
            h_i: First graph-level embedding
            h_j: Second graph-level embedding
        Returns:
            scores: (K, 1) graph-graph interaction score vector
        """
        import ipdb; ipdb.set_trace()
        scores = torch.matmul(h_i, self.weight_tensor)
        scores = torch.matmul(scores, torch.t(h_j)).squeeze(-1)
        scores += torch.matmul(self.weight_matrix, torch.t(torch.cat([h_i, h_j], dim=-1)))
        scores += self.bias
        
        # TODO: need to remove this from here and include it in a function in utils
        if self.activation == "tanh":
            _activation = torch.nn.functional.tanh
        elif self.activation == "sigmoid":
            _activation = torch.nn.functional.sigmoid
        
        scores = _activation(scores)

        return scores