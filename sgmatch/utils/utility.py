import os

import torch
from torch import cuda

def setup_LRL(input_dim, hidden_sizes):
    mlp = torch.nn.ModuleList()
    _in = input_dim
    for i in range(len(hidden_sizes)):
        if i != 0:
            mlp.append(torch.nn.ReLU)
        _out = hidden_sizes[i]
        mlp.append(torch.nn.Linear(_in, _out))
        _in = _out
    
    return mlp

def cudavar(x):
    return x.cuda() if cuda.is_available() else x