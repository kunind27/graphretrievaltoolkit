from typing import Optional

import torch
from torch.functional import Tensor

class GlobalContextAttention(torch.nn.Module):
    r"""
    Attention Mechanism layer for the attention operator from the 
    `"SimGNN: A Neural Network Approach to Fast Graph Similarity Computation"
    <https://arxiv.org/pdf/1808.05689.pdf>`_ paper

    TODO: Include latex formula for attention computation and aggregation update

    Args:
        input_dim: 
        type: Type of attention mechanism to be used
        input_dim: Input Dimension of the Node Embeddings
        activation: The Activation Function to be used for the Attention Layer
        a: Slope of the -ve part if the activation is Leaky ReLU
    """
    def __init__(self, input_dim, activation: str = "tanh", activation_slope: Optional[float] = None):
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
        if self.activation == "leaky_relu" or self.activation == "relu":
            if self.activation_slope is None or self.activation_slope <= 0:
                raise ValueError(f"Activation function slope parameter needs to be a positive \
                                value. {} is invalid".format(self.activation_slope))
            
            torch.nn.init.kaiming_normal_(self.weight_matrix, a = self.activation_slope, nonlinearity = self.activation)
        elif self.activation == "tanh" or self.activation == "sigmoid":
            torch.nn.init.xavier_normal_(self.weight_matrix)
        else:
            raise ValueError("Activation can only take values: 'relu', 'leaky_relu', 'sigmoid', 'tanh';\
                            {} is invalid".format(self.activation))

    def forward(self, x: Tensor) -> Tensor:
        r""" 
        Args:
            x (torch.Tensor) : Node Embedding Tensor of shape N x D.
        
        Returns:
            representation (torch.Tensor): Global graph representation for input node 
            representation set.
        """
        if x.shape[1] != self.input_dim:
            raise RuntimeError("dim 1 of input tensor does not match dimension of weight matrix")
        
        activations = {"tanh": torch.nn.functional.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                        "relu": torch.nn.functional.relu, "sigmoid": torch.nn.functional.sigmoid}

        # Generating the global context vector
        global_context = torch.mean(torch.matmul(x, self.weight_matrix), dim = 0)

        # Applying the Non-Linearity over global context vector
        _activation = activations[self.activation]
        global_context = _activation(global_context)

        # Computing attention weights and att-weight-aggregating node embeddings
        att_weights = torch.sigmoid(torch.matmul(x, global_context.view(-1, 1)))
        representation = torch.sum(x * att_weights, dim = 0)
        
        return representation