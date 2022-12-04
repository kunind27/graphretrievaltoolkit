from typing import Optional, List

import torch
from torch.functional import Tensor

from utils.segment import unsorted_segment_sum
from utils.utility import setup_LRL

# TODO: Credit GMN authors and codebase
# TODO: Fill all docstrings to update on RTD

class GraphProp(torch.nn.Module):
    r"""
    """
    def __init__(self, node_feature_dim: int, node_hidden_sizes: List[int], message_hidden_sizes: List[int], 
                edge_feature_dim: = Optional[int] = None, message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, layer_norm: bool = False, 
                prop_type: str = 'embedding'):
        super(GraphProp, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_hidden_sizes = node_hidden_sizes + [node_feature_dim] # output size is node_state_dim
        self.edge_feature_dim = edge_feature_dim
        self.message_hidden_sizes = message_hidden_sizes
        
        self.message_net_init_scale = message_net_init_scale # Unused
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction 
        self.reverse_dir_param_different = reverse_dir_param_different

        self.layer_norm = layer_norm
        self.prop_type = prop_type
        self.setup_layers()

        if self.layer_norm:
            self.layer_norm1 = torch.nn.LayerNorm()
            self.layer_norm2 = torch.nn.LayerNorm()

    def setup_layers(self):
        # Setup f_{message} as MLP
        self._in = self.node_feature_dim*2 + self.edge_feature_dim if self.edge_feature_dim is not None else self.node_feature_dim*2
        self.message_net = setup_LRL(self._in, self.message_hidden_sizes)

        # optionally compute message vectors in the reverse direction
        if self.use_reverse_direction:
            if self.reverse_dir_param_different:
                self.reverse_message_net = setup_LRL(self._in, self.message_hidden_sizes)
            else:
                self.reverse_message_net = self.message_net

        # TODO: Needs to be changed as this is just a simple propagation base class. Or is it?
        if self.node_update_type == 'gru':
            if self.prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self.node_feature_dim * 2, self.node_feature_dim)
            elif self.prop_type == 'matching':
                self.GRU = torch.nn.GRU(self.node_feature_dim * 3, self.node_feature_dim)
        else:
            # TODO: Is the input correct? Should it instead be self.node_feature_dim + self.message_hidden_sizes[-1] and so on
            # Possible BUG: Difficult to see how these input sizes are correct acc to formula in paper
            if self.prop_type == 'embedding':
                self._in = self.node_feature_dim * 3
            elif self.prop_type == 'matching':
                self._in = self.node_feature_dim * 4
            
            self.MLP = setup_LRL(self._in, self.node_hidden_sizes)

    def _compute_messages(self, node_features: Tensor, from_idx: Tensor, to_idx: Tensor
                         message_net: torch.nn.Module, edge_features: Optional[Tensor] = None):
        r"""
        """
        from_features = node_features[from_idx]
        to_features = node_features[to_idx]
        net_inputs = [from_features, to_features]

        if edge_features is not None:
            net_inputs.append(edge_features)

        net_inputs = torch.cat(net_inputs, dim=-1)
        messages = message_net(net_inputs)

        return messages
    
    def _aggregate_messages(self, messages: Tensor, to_idx: Tensor, num_nodes: int):
        r"""
        """
        return unsorted_segment_sum(messages, to_idx, num_nodes)
    
    def _compute_node_update(self, node_features: Tensor, node_inputs: List[Tensor]):
        r"""
        """
        if self.node_update_type in ('mlp', 'residual'):
            node_inputs.append(node_features)

        if len(node_inputs) == 1:
            node_inputs = node_inputs[0]
        else:
            node_inputs = torch.cat(node_inputs, dim=-1)

        if self.node_update_type == 'gru':
            node_inputs = torch.unsqueeze(node_inputs, 0)
            node_features = torch.unsqueeze(node_features, 0)
            _, new_node_features = self.GRU(node_inputs, node_features)
            new_node_features = torch.squeeze(new_node_features)
            return new_node_features
        else:
            mlp_output = self.MLP(node_inputs)
            if self.layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self.node_update_type == 'mlp':
                return mlp_output
            elif self.node_update_type == 'residual':
                return node_features + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self.node_update_type)

    def forward(self, node_features: Tensor, from_idx: Tensor, to_idx: Tensor, node_features_j: Optional[Tensor] = None,
                edge_features = Optional[Tensor] = None, att_module: Optional[torch.nn.Module] = None):
        r"""
        """
        # XXX: extra node_features argument removed from original code to keep it simple for now
        messages = _compute_messages(node_features, from_idx, to_idx, self.message_net, edge_feature=edge_features)
        aggregated_messages = _aggregate_messages(messages, to_idx, node_features.shape[0])

        if self.use_reverse_direction:
            reverse_messages = _compute_messages(node_features, to_idx, from_idx, self.reverse_message_net, edge_feature=edge_features)
            reverse_aggregated_messages = _aggregate_messages(reverse_messages, from_idx, node_features.shape[0])

            aggregated_messages += reverse_aggregated_messages
        node_input_list = [aggregated_messages]

        if att_module is not None:
            if self.prop_type != "matching":
                raise RuntimeError("Cross graph attention module provided but propagation model initialised as embedding module,\
                                    please specify correct value for self.prop_type while initialising")
            assert node_features_j is not None, "Provide second graph's node features to use GMN Match cross-graph attention,\
                                                node_features_j cannot be None"
            att_features = att_module(node_features, node_features_j)
            node_input_list.append(att_features)
        
        return _compute_node_update(node_features, node_input_list)
    
    def __repr__(self):
        # TODO
        pass
        


