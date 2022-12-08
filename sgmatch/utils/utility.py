from typing import List

import torch
from torch import cuda

def setup_linear_nn(input_dim: int, hidden_sizes: List[int]):
    mlp = torch.nn.ModuleList()
    _in = input_dim
    for i in range(len(hidden_sizes)):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        _in = _out
    
    return mlp

def setup_LRL_nn(input_dim: int, hidden_sizes: List[int], 
                 activation: torch.nn.Module = torch.nn.ReLU):
    mlp = []
    _in = input_dim
    for i in range(len(hidden_sizes) - 1):
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        mlp.append(activation())
        _in = _out
    mlp.append(torch.nn.Linear(_in, hidden_sizes[-1]))
    mlp = torch.nn.Sequential(*mlp)
    
    return mlp

# def cudavar(x):
#     return x.cuda() if cuda.is_available() else x