from typing import Optional, List

import torch
import torch_geometric.nn as pyg_nn
from torch.functional import Tensor

from utils.utility import setup_LRL_nn
from utils.constants import CONVS

class SkipLastGNN(torch.nn.Module):
    r"""
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                num_layers: int, conv_type: str = "SAGEConv", dropout: float = 0.0,
                skip: str = "learnable"):
        super(SkipLastGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.skip = skip

        # XXX: Does feature preprocessing need to be moved elsewhere and 
        # XXX: even included in the library
        # if len(feature_preprocess.FEATURE_AUGMENT) > 0:
        #     self.feat_preprocess = feature_preprocess.Preprocess(input_dim)
        #     input_dim = self.feat_preprocess.dim_out
        # else:
        #     self.feat_preprocess = None

        # Using setup_LRL_nn over setup_linear_nn for single linear layer to get torch.nn.Sequential behaviour
        self.pre_mlp = setup_LRL_nn(self.input_dim, hidden_sizes=[3*self.hidden_dim] if self.conv_type=="Neuro-PNA" else [self.hidden_dim])
        
        # TODO: Include error checking for invalid convolution type
        self.conv_model = CONVS[self.conv_type]
        if self.conv_type == "Neuro-PNA":
            self.convs_sum = torch.nn.ModuleList()
            self.convs_mean = torch.nn.ModuleList()
            self.convs_max = torch.nn.ModuleList()
        else:
            self.convs = torch.nn.ModuleList()

        if self.skip == 'learnable':
            self.learnable_skip = torch.nn.Parameter(torch.ones(self.num_layers, self.num_layers))

        for layer in range(self.num_layers):
            if self.skip == 'all' or self.skip == 'learnable':
                hidden_input_dim = self.hidden_dim * (layer + 1)
            else:
                hidden_input_dim = self.hidden_dim
            if self.conv_type == "Neuro-PNA":
                self.convs_sum.append(self.conv_model(3*hidden_input_dim, self.hidden_dim))
                self.convs_mean.append(self.conv_model(3*hidden_input_dim, self.hidden_dim))
                self.convs_max.append(self.conv_model(3*hidden_input_dim, self.hidden_dim))
            else:
                self.convs.append(self.conv_model(hidden_input_dim, self.hidden_dim))

        post_input_dim = self.hidden_dim * (self.num_layers + 1)
        if self.conv_type == "PNA":
            post_input_dim *= 3
        
        self.post_mlp = torch.nn.Sequential(
            torch.nn.Linear(post_input_dim, self.hidden_dim),
            torch.nn.Dropout(self.dropout),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 256), # Should this be output_dim
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.hidden_dim))
        #self.batch_norm = torch.nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)

    def forward(self, node_features: Tensor, edge_index: Tensor):
        # if self.feat_preprocess is not None:
        #     if not hasattr(data, "preprocessed"):
        #         data = self.feat_preprocess(data)
        #         data.preprocessed = True
        # x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        node_features = self.pre_mlp(node_features)

        all_emb = node_features.unsqueeze(1)
        emb = node_features
        for i in range(len(self.convs_sum) if self.conv_type=="Neuro-PNA" else len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,:i+1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(node_features.size(0), -1)
                if self.conv_type == "Neuro-PNA":
                    node_features = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                        self.convs_mean[i](curr_emb, edge_index),
                        self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    node_features = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "Neuro-PNA":
                    node_features = torch.cat((self.convs_sum[i](emb, edge_index),
                        self.convs_mean[i](emb, edge_index),
                        self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    node_features = self.convs[i](emb, edge_index)
            else:
                node_features = self.convs[i](node_features, edge_index)
            node_features = torch.nn.functional.relu(node_features)
            node_features = torch.nn.functional.dropout(node_features, p=self.dropout, training=self.training)
            emb = torch.cat((emb, node_features), 1)
            if self.skip == 'learnable':
                all_emb = torch.cat((all_emb, node_features.unsqueeze(1)), 1)

        # node_features = pyg_nn.global_mean_pool(node_features, batch)
        # emb = torch_geometric.nn.global_add_pool(emb, batch)
        emb = torch.sum(emb, dim=-2)
        emb = self.post_mlp(emb)
        #emb = self.batch_norm(emb)   # TODO: test
        #out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        return torch.nn.functional.nll_loss(pred, label)