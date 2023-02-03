import torch_geometric.nn as pyg_nn
import torch

# XXX: Why can GINConv not be initialised like the other operators?

CONVS = {"GCN": pyg_nn.GCNConv, 
        "SAGE": pyg_nn.SAGEConv, 
        "GAT": pyg_nn.GATConv, 
        "GIN": lambda i, h: pyg_nn.GINConv(torch.nn.Sequential(
                torch.nn.Linear(i, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h,h)
        )),
        "graph": pyg_nn.GraphConv, 
        "gated": lambda h, num_hidden: pyg_nn.GatedGraphConv(h, num_hidden),
        "Neuro-PNA": pyg_nn.SAGEConv}
    
ACTIVATION_LAYERS = {"sigmoid": torch.nn.Sigmoid, "relu": torch.nn.ReLU,
               "leaky_relu": torch.nn.LeakyReLU, "tanh": torch.nn.Tanh}

ACTIVATIONS = {"tanh": torch.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                "relu": torch.relu, "sigmoid": torch.sigmoid}