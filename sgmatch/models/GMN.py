from typing import Optional

import torch
from torch_geometric.nn.aggr.attention import AttentionalAggregation

from ..modules.encoder import MLPEncoder
from ..modules.propagation import GraphProp
from ..modules.attention import CrossGraphAttention

class GMNEmbed(torch.nn.Module):
    r"""
    """
    # TODO: Provide default arguments for MLP layer sizes
    def __init__(self, node_feature_dim: int, enc_node_hidden_sizes: List[int], 
                prop_node_hidden_sizes: List[int], prop_message_hidden_sizes: List[int],
                aggr_gate_hidden_sizes: List[int], aggr_mlp_hidden_sizes: List[int], 
                edge_feature_dim: Optional[int] = None, enc_edge_hidden_sizes: Optional[List[int]] = None,
                message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, 
                layer_norm: bool = False):
        super(GMNEmbed, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Encoder Module        
        self.enc_node_layers = enc_node_hidden_sizes
        self.enc_edge_layers = enc_edge_hidden_sizes
        
        # Propagation Module
        self.prop_node_layers = prop_node_hidden_sizes
        self.prop_message_layers = prop_message_hidden_sizes
        
        # Aggregation Module
        self.aggr_gate_layers = aggr_gate_hidden_sizes + [node_feature_dim]
        self.aggr_mlp_layers = aggr_mlp_hidden_sizes + [node_feature_dim]

        self.message_net_init_scale = message_net_init_scale # Unused
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_dir_param_different = reverse_dir_param_different

        self.layer_norm = layer_norm
        self.prop_type = "embedding"
        
        self.setup_layers()

    def setup_layers(self):
        self._encoder = MLPEncoder(node_feature_dim, node_hidden_sizes, edge_feature_dim=edge_feature_dim, 
                            edge_hidden_sizes=edge_hidden_sizes)
        self._propagator = GraphProp(self.node_feature_dim, self.prop_node_layers, self.prop_message_layers, 
                               edge_feature_dim=self.edge_feature_dim, message_net_init_scale=self.message_net_init_scale,
                               node_update_type=self.node_update_type, use_reverse_direction=self.use_reverse_direction,
                               reverse_dir_param_different=self.reverse_dir_param_different, layer_norm=self.layer_norm,
                               prop_type=self.prop_type)        
        
        # Setup aggregator MLPs
        self.aggr_gate = setup_LRL(self.node_feature_dim, self.aggr_gate_layers)
        self.aggr_mlp = setup_LRL(self.node_feature_dim, self.aggr_mlp_layers)

        self._aggregator = AttentionalAggregation(self.aggr_gate, self.aggr_mlp)

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_features: Optional[Tensor] = None,
                num_prop: int = 10):
        from_idx = edge_index[:,0] if len(edge_index.shape) == 3 else edge_index[0]
        to_idx = edge_index[:,1] if len(edge_index.shape) == 3 else edge_index[1]

        if edge_features is not None:
            node_features, edge_features = self._encoder(node_features, edge_features)
        else:
            node_features = self._encoder(node_features)
        
        for _ in range(num_prop):
            # TODO: Can include a list keeping track of propagation layer outputs
            node_features = self._propagator(node_features, from_idx, to_idx, edge_features)

        return self._aggregator(node_features)

class GMNMatch(torch.nn.Module):
    r"""
    """
    # TODO: Provide default arguments for MLP layer sizes
    def __init__(self, node_feature_dim: int, enc_node_hidden_sizes: List[int], 
                prop_node_hidden_sizes: List[int], prop_message_hidden_sizes: List[int],
                aggr_gate_hidden_sizes: List[int], aggr_mlp_hidden_sizes: List[int], 
                edge_feature_dim: Optional[int] = None, enc_edge_hidden_sizes: Optional[List[int]] = None,
                message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, 
                attention_sim_metric: str = "euclidean", layer_norm: bool = False):
        super(GMNEmbed, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim

        # Encoder Module        
        self.enc_node_layers = enc_node_hidden_sizes
        self.enc_edge_layers = enc_edge_hidden_sizes
        
        # Propagation Module
        self.prop_node_layers = prop_node_hidden_sizes
        self.prop_message_layers = prop_message_hidden_sizes
        
        # Aggregation Module
        self.aggr_gate_layers = aggr_gate_hidden_sizes + [node_feature_dim]
        self.aggr_mlp_layers = aggr_mlp_hidden_sizes + [node_feature_dim]

        self.message_net_init_scale = message_net_init_scale # Unused
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_dir_param_different = reverse_dir_param_different

        self.attention_sim_metric = attention_sim_metric
        self.layer_norm = layer_norm
        self.prop_type = "matching"
        
        self.setup_layers()

    def setup_layers(self):
        self._encoder = MLPEncoder(node_feature_dim, node_hidden_sizes, edge_feature_dim=edge_feature_dim, 
                            edge_hidden_sizes=edge_hidden_sizes)
        self._attention = CrossGraphAttention(similarity_metric=self.attention_sim_metric)
        self._propagator = GraphProp(self.node_feature_dim, self.prop_node_layers, self.prop_message_layers, 
                               edge_feature_dim=self.edge_feature_dim, message_net_init_scale=self.message_net_init_scale,
                               node_update_type=self.node_update_type, use_reverse_direction=self.use_reverse_direction,
                               reverse_dir_param_different=self.reverse_dir_param_different, layer_norm=self.layer_norm,
                               prop_type=self.prop_type)        

        # Setup aggregator MLPs
        self.aggr_gate = setup_LRL(self.node_feature_dim, self.aggr_gate_layers)
        self.aggr_mlp = setup_LRL(self.node_feature_dim, self.aggr_mlp_layers)

        self._aggregator = AttentionalAggregation(self.aggr_gate, self.aggr_mlp)

    def forward(self, node_features_i: Tensor, node_features_j: Tensor, edge_index_i: Tensor, edge_index_j: Tensor, 
                edge_features_i: Optional[Tensor] = None, num_prop: int = 10):
        r"""
        """
        from_idx = edge_index[:,0] if len(edge_index.shape) == 3 else edge_index[0]
        to_idx = edge_index[:,1] if len(edge_index.shape) == 3 else edge_index[1]

        if edge_features is not None:
            node_features, edge_features = self._encoder(node_features, edge_features)
        else:
            node_features = self._encoder(node_features)
        
        for _ in range(num_prop):
            # TODO: Can include a list keeping track of propagation layer outputs
            node_features = self._propagator(node_features, from_idx, to_idx, node_features_j, edge_features)

        return self._aggregator(node_features)