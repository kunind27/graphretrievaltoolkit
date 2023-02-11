from typing import Optional, List, Type
from sgmatch.utils.utility import Namespace

import torch
from torch_geometric.nn.aggr.attention import AttentionalAggregation
from torch.functional import Tensor
from torch_geometric.nn.inits import reset

from ..modules.encoder import MLPEncoder
from ..modules.propagation import GraphProp
from ..modules.attention import CrossGraphAttention
from ..utils.utility import setup_linear_nn, setup_LRL_nn

class GMNEmbed(torch.nn.Module):
     r"""
    End to end implementation of Graph Matching Networks - Embed from the `"Graph Matching Networks for Learning the Similarity
    of Graph Structured Objects" <https://arxiv.org/abs/1904.12787>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any

    Args:
        node_feature_dim (int): Input dimension of node feature embedding vectors
        enc_node_hidden_sizes ([int]): Hyperparameter for the number of tensor slices in the
            Neural Tensor Network. In this domain, it denotes the number of interaction 
            (similarity) scores produced by the model for each graph embedding pair.
        prop_node_hidden_sizes ([int]): Number of filters per convolutional layer in the graph 
            convolutional encoder model.
        prop_message_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            MLP for reducing dimensionality of concatenated output of neural 
            tensor network and histogram features. Note that the final scoring 
            weight tensor of size :obj:`[mlp_neurons[-1], 1]` is kept separate
            from the MLP, therefore specifying only the hidden layer sizes will
            suffice.
        aggr_gate_hidden_sizes ([int]): Hyperparameter controlling the number of bins in the node 
            ordering histogram scheme.
        aggr_mlp_hidden_sizes ([int]): Type of graph convolutional architecture to be used for encoding
            (:obj:`'GCN'` or :obj:`'SAGE'` or :obj:`'GAT'`)
        edge_feature_dim (int, Optional): Type of activation used in Attention and NTN modules. 
            (:obj:`'sigmoid'` or :obj:`'relu'` or :obj:`'leaky_relu'` or :obj:`'tanh'`) 
            (default: :obj:`None`)
        enc_edge_hidden_sizes ([int], Optional): Slope of function for leaky_relu activation. 
            (default: :obj:`None`)
        message_net_init_scale (float): Flag for including Strategy Two: Nodewise comparison
            from SimGNN. (default: :obj:`0.1`)
        node_update_type (str): Slope of function for leaky_relu activation. 
            (default: :obj:`'residual'`)
        use_reverse_direction (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
        reverse_dir_param_different (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
        layer_norm (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
    """
    # TODO: Provide default arguments for MLP layer sizes
    def __init__(self, av: Type[Namespace], node_feature_dim: int, enc_node_hidden_sizes: List[int], 
                prop_node_hidden_sizes: List[int], prop_message_hidden_sizes: List[int],
                aggr_gate_hidden_sizes: List[int], aggr_mlp_hidden_sizes: List[int],
                edge_feature_dim: Optional[int] = None, enc_edge_hidden_sizes: Optional[List[int]] = None,
                message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, 
                layer_norm: bool = False):
        super(GMNEmbed, self).__init__()
        self.node_feature_dim = av.node_feature_dim
        self.edge_feature_dim = av.edge_feature_dim

        # Encoder Module        
        self.enc_node_layers = av.enc_node_hidden_sizes
        self.enc_edge_layers = av.enc_edge_hidden_sizes
        
        # Propagation Module
        self.prop_node_layers = av.prop_node_hidden_sizes
        self.prop_message_layers = av.prop_message_hidden_sizes
        
        # Aggregation Module
        self.aggr_gate_layers = av.aggr_gate_hidden_sizes
        self.aggr_mlp_layers = av.aggr_mlp_hidden_sizes

        self.message_net_init_scale = av.message_net_init_scale # Unused
        self.node_update_type = av.node_update_type
        self.use_reverse_direction = av.use_reverse_direction
        self.reverse_dir_param_different = av.reverse_dir_param_different

        self.layer_norm = av.layer_norm
        self.prop_type = "embedding"

        # TODO: Include assertion method to ensure correct dimensionality for mlp outputs
        
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self._encoder = MLPEncoder(self.node_feature_dim, self.enc_node_layers, edge_feature_dim=self.edge_feature_dim, 
                                   edge_hidden_sizes=self.enc_edge_layers)
        self._propagator = GraphProp(self.enc_node_layers[-1], self.prop_node_layers, self.prop_message_layers, 
                               edge_feature_dim=self.edge_feature_dim, message_net_init_scale=self.message_net_init_scale,
                               node_update_type=self.node_update_type, use_reverse_direction=self.use_reverse_direction,
                               reverse_dir_param_different=self.reverse_dir_param_different, layer_norm=self.layer_norm,
                               prop_type=self.prop_type)        
        
        # Setup aggregator MLPs
        self.aggr_gate = setup_LRL_nn(self.prop_node_layers[-1], self.aggr_gate_layers)
        self.aggr_mlp = setup_LRL_nn(self.prop_node_layers[-1], self.aggr_mlp_layers)

        self._aggregator = AttentionalAggregation(self.aggr_gate, self.aggr_mlp)

    def reset_parameters(self):
        self._encoder.reset_parameters()
        self._propagator.reset_parameters()
        self._aggregator.reset_parameters()

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

        return self._aggregator(node_features, index=None)

    def __repr__(self):
        # TODO
        pass

class GMNMatch(torch.nn.Module):
    r"""
    End to end implementation of Graph Matching Networks - Match from the `"Graph Matching Networks for Learning the Similarity
    of Graph Structured Objects" <https://arxiv.org/abs/1904.12787>`_ paper.
    
    TODO: Provide description of implementation and differences from paper if any

    Args:
        node_feature_dim (int): Input dimension of node feature embedding vectors
        enc_node_hidden_sizes ([int]): Hyperparameter for the number of tensor slices in the
            Neural Tensor Network. In this domain, it denotes the number of interaction 
            (similarity) scores produced by the model for each graph embedding pair.
        prop_node_hidden_sizes ([int]): Number of filters per convolutional layer in the graph 
            convolutional encoder model.
        prop_message_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            MLP for reducing dimensionality of concatenated output of neural 
            tensor network and histogram features. Note that the final scoring 
            weight tensor of size :obj:`[mlp_neurons[-1], 1]` is kept separate
            from the MLP, therefore specifying only the hidden layer sizes will
            suffice.
        aggr_gate_hidden_sizes ([int]): Hyperparameter controlling the number of bins in the node 
            ordering histogram scheme.
        aggr_mlp_hidden_sizes ([int]): Type of graph convolutional architecture to be used for encoding
            (:obj:`'GCN'` or :obj:`'SAGE'` or :obj:`'GAT'`)
        edge_feature_dim (int, Optional): Type of activation used in Attention and NTN modules. 
            (:obj:`'sigmoid'` or :obj:`'relu'` or :obj:`'leaky_relu'` or :obj:`'tanh'`) 
            (default: :obj:`None`)
        enc_edge_hidden_sizes ([int], Optional): Slope of function for leaky_relu activation. 
            (default: :obj:`None`)
        message_net_init_scale (float): Flag for including Strategy Two: Nodewise comparison
            from SimGNN. (default: :obj:`0.1`)
        node_update_type (str): Slope of function for leaky_relu activation. 
            (default: :obj:`'residual'`)
        use_reverse_direction (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
        reverse_dir_param_different (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
        attention_sim_metric (str): Slope of function for leaky_relu activation. 
            (default: :obj:`'euclidean'`) 
        layer_norm (bool): Slope of function for leaky_relu activation. 
            (default: :obj:`True`)
    """
    # TODO: Provide default arguments for MLP layer sizes
    def __init__(self, av: Type[Namespace], node_feature_dim: int, enc_node_hidden_sizes: List[int], 
                prop_node_hidden_sizes: List[int], prop_message_hidden_sizes: List[int],
                aggr_gate_hidden_sizes: List[int], aggr_mlp_hidden_sizes: List[int],
                edge_feature_dim: Optional[int] = None, enc_edge_hidden_sizes: Optional[List[int]] = None,
                message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = True, reverse_dir_param_different: bool = True, 
                attention_sim_metric: str = "euclidean", layer_norm: bool = False):
        super(GMNMatch, self).__init__()
        self.node_feature_dim = av.node_feature_dim
        self.edge_feature_dim = av.edge_feature_dim

        # Encoder Module        
        self.enc_node_layers = av.enc_node_hidden_sizes
        self.enc_edge_layers = av.enc_edge_hidden_sizes
        
        # Propagation Module
        self.prop_node_layers = av.prop_node_hidden_sizes
        self.prop_message_layers = av.prop_message_hidden_sizes
        
        # Aggregation Module
        self.aggr_gate_layers = av.aggr_gate_hidden_sizes + [av.node_feature_dim]
        self.aggr_mlp_layers = av.aggr_mlp_hidden_sizes + [av.node_feature_dim]

        self.message_net_init_scale = av.message_net_init_scale # Unused
        self.node_update_type = av.node_update_type
        self.use_reverse_direction = av.use_reverse_direction
        self.reverse_dir_param_different = av.reverse_dir_param_different

        self.attention_sim_metric = av.attention_sim_metric
        self.layer_norm = av.layer_norm
        self.prop_type = "matching"
        
        self.setup_layers()
        self.reset_parameters()

    def setup_layers(self):
        self._encoder = MLPEncoder(self.node_feature_dim, self.enc_node_layers, edge_feature_dim=self.edge_feature_dim, 
                            edge_hidden_sizes=self.enc_edge_layers)
        self._attention = CrossGraphAttention(similarity_metric=self.attention_sim_metric)
        self._propagator = GraphProp(self.node_feature_dim, self.prop_node_layers, self.prop_message_layers, 
                               edge_feature_dim=self.edge_feature_dim, message_net_init_scale=self.message_net_init_scale,
                               node_update_type=self.node_update_type, use_reverse_direction=self.use_reverse_direction,
                               reverse_dir_param_different=self.reverse_dir_param_different, layer_norm=self.layer_norm,
                               prop_type=self.prop_type)
        self._attention = CrossGraphAttention(similarity_metric=self.attention_sim_metric)
        
        # Setup aggregator MLPs
        self.aggr_gate = setup_linear_nn(self.node_feature_dim, self.aggr_gate_layers)
        self.aggr_mlp = setup_linear_nn(self.node_feature_dim, self.aggr_mlp_layers)

        self._aggregator = AttentionalAggregation(self.aggr_gate, self.aggr_mlp)

    def reset_parameters(self):
        self._encoder.reset_parameters()
        self._propagator.reset_parameters()
        for lin in self.aggr_gate:
            lin.reset_parameters()
        for lin in self.aggr_mlp:
            lin.reset_parameters()
        self._aggregator.reset_parameters()

    def forward(self, node_features_i: Tensor, node_features_j: Tensor, edge_index_i: Tensor, 
                edge_features_i: Optional[Tensor] = None, num_prop: int = 10):
        r"""
        """
        from_idx = edge_index_i[:,0] if len(edge_index_i.shape) == 3 else edge_index_i[0]
        to_idx = edge_index_i[:,1] if len(edge_index_i.shape) == 3 else edge_index_i[1]

        if edge_features_i is not None:
            node_features_i, edge_features_i = self._encoder(node_features_i, edge_features_i)
        else:
            node_features_i = self._encoder(node_features_i)
        
        for _ in range(num_prop):
            # TODO: Can include a list keeping track of propagation layer outputs
            node_features_i = self._propagator(node_features_i, from_idx, to_idx, node_features_j, edge_features_i, att_module=self._attention)

        return self._aggregator(node_features_i)

    def __repr__(self):
        # TODO
        pass