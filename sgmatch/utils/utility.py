from typing import List

import torch

from .constants import CONVS, ACTIVATIONS
# from torch import cuda

def setup_linear_nn(input_dim: int, hidden_sizes: List[int]):
    r"""
    """
    mlp = torch.nn.ModuleList()
    _in = input_dim
    for i in range(len(hidden_sizes)):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        _in = _out
    
    return mlp

def setup_LRL_nn(input_dim: int, hidden_sizes: List[int], 
                 activation: str = "relu"):
    r"""
    """
    # XXX: Better to leave this up to MLP class?
    mlp = []
    _in = input_dim
    activation = ACTIVATIONS[activation]
    for i in range(len(hidden_sizes) - 1):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        mlp.append(activation())
        _in = _out
    mlp.append(torch.nn.Linear(_in, hidden_sizes[-1]))
    mlp = torch.nn.Sequential(*mlp)
    
    return mlp

def setup_conv_layers(input_dim, conv_type, filters):
    r"""
    """
    convs = torch.nn.ModuleList()
    _conv = CONVS[conv_type]
    num_layers = len(filters)
    _in = input_dim
    for i in range(num_layers):
        _out = filters[i]
        convs.append(_conv(in_channels=_in, out_channels=_out))
        _in = _out

    return convs

# def cudavar(x):
#     return x.cuda() if cuda.is_available() else x