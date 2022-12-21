from typing import Optional, List

import torch
from torch.functional import Tensor

from ..utils.segment import unsorted_segment_sum
from ..utils.utility import setup_linear_nn
from .attention import CrossGraphAttention

# TODO: Credit GMN authors and codebase
# TODO: Update math for propagation step
# TODO: Add more error checking
# TODO: layer_norm first docstring

class GraphProp(torch.nn.Module):
    r"""
    Implementation of the message-propagation module from the `"Graph 
    Matching Networks for Learning the Similarity of Graph Structured 
    Objects" https://arxiv.org/pdf/1904.12787.pdf`_ paper.

    .. math::

    NOTE: This module only computes one propagation step at a time and 
    needs to be called :obj:`T` times for T propagation steps (step-wise 
    calls need to be defined by user in model training scripts).

    Args:
        node_feature_dim (int): Input dimension of node feature embedding vectors
        node_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            node update MLP :obj:`f_node`. :obj:`node_feature_dim` is appended as
            the size of the final linear layer to maintain node embedding dimensionality
        message_hidden_sizes ([int]): Number of hidden neurons in each linear layer of 
            message computation MLP :obj:`f_node`. Note that the message vector dimensionality 
            (:obj:`message_hidden_sizes[-1]`) may not be equal to :obj:`node_feature_dim`.
        edge_feature_dim ([int], Optional):  Input dimension of node feature embedding 
            vectors. (default: :obj:`None`)
        message_net_init_scale (float): Initialisation scale for the message net output 
            vectors. (default: :obj:`0.1`)
        node_update_type (str): Type of update applied to node feature vectors (:obj:`"GRU"` or 
            :obj:`"MLP"` or :obj:`"residual`) (default: :obj:`"residual"`)
        use_reverse_direction (bool): Specifies whether or not to use the reverse message 
            aggregation for propagation step. (default: :obj:`False`)
        reverse_dir_param_different (bool): Specifies whether or not message computation model 
            parameters should be shared by forward and reverse messages. (default: :obj:`True`)
        layer_norm (bool): (default: :obj:`False`)
        prop_type (str): Propagation computation type (:obj:`"embedding"` or :obj:`"matching"`)
            (default: :obj:`"embedding"`)
    """
    def __init__(self, node_feature_dim: int, node_hidden_sizes: List[int], message_hidden_sizes: List[int], 
                edge_feature_dim: Optional[int] = None, message_net_init_scale: float = 0.1, node_update_type: str = 'residual', 
                use_reverse_direction: bool = False, reverse_dir_param_different: bool = True, layer_norm: bool = False, 
                prop_type: str = 'embedding'):
        super(GraphProp, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.node_hidden_sizes = node_hidden_sizes + [node_feature_dim]
        self.edge_feature_dim = edge_feature_dim
        self.message_hidden_sizes = message_hidden_sizes
        
        self.message_net_init_scale = message_net_init_scale # Unused
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_dir_param_different = reverse_dir_param_different

        self.layer_norm = layer_norm
        self.prop_type = prop_type
        self.setup_layers()
        self.reset_parameters()

        if self.layer_norm:
            self.layer_norm1 = torch.nn.LayerNorm()
            self.layer_norm2 = torch.nn.LayerNorm()

    def setup_layers(self):
        # Setup f_{message} as MLP
        self._in = self.node_feature_dim*2 + self.edge_feature_dim if self.edge_feature_dim is not None else self.node_feature_dim*2
        self.message_net = setup_linear_nn(self._in, self.message_hidden_sizes)

        # optionally compute message vectors in the reverse direction
        if self.use_reverse_direction:
            if self.reverse_dir_param_different:
                self.reverse_message_net = setup_linear_nn(self._in, self.message_hidden_sizes)
            else:
                self.reverse_message_net = self.message_net

        # TODO: Needs to be changed as this is just a simple propagation base class. Or is it?
        # BUG: Need to put restriction on self.node_update_type values
        if self.node_update_type == 'GRU':
            if self.prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self.node_feature_dim * 2, self.node_feature_dim)
            elif self.prop_type == 'matching':
                self.GRU = torch.nn.GRU(self.node_feature_dim * 3, self.node_feature_dim)
        else:
            # TODO: Is the input correct? Should it instead be self.node_feature_dim + self.message_hidden_sizes[-1] and so on
            # Possible BUG: Difficult to see how these input sizes are correct acc to formula in paper
            if self.prop_type == 'embedding':
                # self._in = self.node_feature_dim * 3
                self._in = self.node_feature_dim * 2
            elif self.prop_type == 'matching':
                # self._in = self.node_feature_dim * 4
                self._in = self.node_feature_dim * 3
            
            self.MLP = setup_linear_nn(self._in, self.node_hidden_sizes)

    def reset_parameters(self):
        for lin in self.message_net:
            lin.reset_parameters()
        if self.use_reverse_direction:
            for lin in self.reverse_message_net:
                lin.reset_parameters()
        if self.node_update_type != "GRU":
            for lin in self.MLP:
                lin.reset_parameters()
        else:
            self.GRU.reset_parameters()

    def _compute_messages(self, node_features: Tensor, from_idx: Tensor, to_idx: Tensor,
                         message_net: torch.nn.Module, edge_features: Optional[Tensor] = None):
        r"""
        Computes messages propagating from nodes indexed by :obj:`from_idx`
        to :obj:`to_idx` in :obj:`node_features`. Optionally extends feature
        vectors if :obj:`edge_features` is not :obj:`None`. Messages are 
        computed using LRL.. network :obj:`message_net`.

        Args:
            node_features (Tensor): Node feature vectors in embedding space of 
                dimensionality :obj:`D`, with shape :obj:`[N, D]`.
            from_idx (Tensor): Indices of the message-origin nodes corresponding
                to :obj:`node_features`. Must be of the shape :obj:`[1, num_messages]`
            to_idx (Tensor): Indices of the message-destination nodes corresponding
                to :obj:`node_features`. Must be of the shape :obj:`[1, num_messages]`
            message_net (torch.nn.Module): Differentiable network for message computation
            edge_features (Tensor, Optional): Edge-wise feature vectors in embedding
                space of dimensionality :obj`E` with shape :obj:`[nC2, E] (default: :obj:`None`)
        
        Returns:
            messages (Tensor): Messages incident from :obj:`from_idx` nodes to :obj:`to_idx` nodes
        """
        from_features = node_features[:,from_idx] if len(node_features.shape) == 3 else node_features[from_idx]
        to_features = node_features[:,to_idx] if len(node_features.shape) == 3 else node_features[to_idx]
        net_inputs = [from_features, to_features]

        if edge_features is not None:
            net_inputs.append(edge_features)

        net_inputs = torch.cat(net_inputs, dim=-1)
        messages = net_inputs
        for lin in message_net:
            messages = lin(messages)
            messages = torch.nn.functional.relu(messages)

        return messages
    
    def _aggregate_messages(self, messages: Tensor, to_idx: Tensor, num_nodes: int):
        return unsorted_segment_sum(messages, to_idx, num_nodes)
    
    def _compute_node_update(self, node_features: Tensor, node_inputs: List[Tensor]):
        r"""
        Updates :obj:`node_features` with input vectors :obj:`node_inputs`. Note
        that the latter may contain multiple inputs like messages, reverse 
        messages, attention-weighted interactions etc. which are then concatenated
        and fed to the updating differentiable functions (such as MLPs/RNNs).

        Args:
            node_features (Tensor): Node feature vectors in embedding space of 
                dimensionality :obj:`D`, with shape :obj:`[N, D]`.
            node_inputs ([Tensor]): Per-node inputs used to update hidden node
                feature embeddings.
        """
        if self.node_update_type in ('MLP', 'residual'):
            node_inputs.append(node_features)

        if len(node_inputs) == 1:
            node_inputs = node_inputs[0]
        else:
            node_inputs = torch.cat(node_inputs, dim=-1)

        if self.node_update_type == 'GRU':
            node_inputs = torch.unsqueeze(node_inputs, 0)
            node_features = torch.unsqueeze(node_features, 0)
            _, new_node_features = self.GRU(node_inputs, node_features)
            new_node_features = torch.squeeze(new_node_features)
            return new_node_features
        else:
            mlp_output = node_inputs
            for lin in self.MLP:
                mlp_output = lin(mlp_output)
                mlp_output = torch.nn.functional.relu(mlp_output)
            
            if self.layer_norm:
                mlp_output = self.layer_norm2(mlp_output)
            if self.node_update_type == 'MLP':
                return mlp_output
            elif self.node_update_type == 'residual':
                return node_features + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self.node_update_type)

    def forward(self, node_features: Tensor, from_idx: Tensor, to_idx: Tensor, node_features_j: Optional[Tensor] = None,
                edge_features: Optional[Tensor] = None, att_module: Optional[torch.nn.Module | str] = None):
        # TODO: Generalise function to accept edge_index and sparse edge indices
        # TODO: Checking validity of user-provided cross graph attention module
        r"""
        Implementation of the forward call for the propagation scheme.

        Args:
            node_features (Tensor): Node feature vectors in embedding space of 
                dimensionality :obj:`D`, with shape :obj:`[N, D]`.
            from_idx (Tensor): Indices of the message-origin nodes corresponding
                to :obj:`node_features`. Must be of the shape :obj:`[1, num_messages]`
            to_idx (Tensor): Indices of the message-destination nodes corresponding
                to :obj:`node_features`. Must be of the shape :obj:`[1, num_messages]`
            node_features_j (Tensor, Optional): Node feature vectors of second graph
                for computing cross-graph attention aggregated vectors. Ignored if 
                :obj:`att_module` is :obj:`None` (default: :obj:`None`)
            edge_features (Tensor, Optional): Edge-wise feature vectors in embedding
                space of dimensionality :obj`E` with shape :obj:`[num_edges, E] (default: :obj:`None`)
            att_module (torch.nn.Module, Optional): Cross-graph attention module, can be 
                appropriately user-defined if :obj:`sgmatch.modules.attention.CrossGraphAttention` 
                or :obj:`"default"` are not given as the argument. (default: :obj:`None`)

        Returns:
            Updated node feature vectors for one propagation step
        """
        # XXX: extra node_features argument removed from original code to keep it simple for now
        messages = self._compute_messages(node_features, from_idx, to_idx, self.message_net, edge_features=edge_features)
        aggregated_messages = self._aggregate_messages(messages, to_idx, node_features.shape[0])

        if self.use_reverse_direction:
            reverse_messages = self._compute_messages(node_features, to_idx, from_idx, self.reverse_message_net, edge_features=edge_features)
            reverse_aggregated_messages = self._aggregate_messages(reverse_messages, from_idx, node_features.shape[0])

            aggregated_messages += reverse_aggregated_messages
        node_input_list = [aggregated_messages]

        if att_module is not None:
            if att_module == "default":
                att_module = CrossGraphAttention
            else:
                if isinstance(att_module, str):
                    raise ValueError("Invalid value for att_module, cannot have a string value other than 'default'.")
            if self.prop_type != "matching":
                raise ValueError("Cross graph attention module provided but propagation model initialised as embedding module,\
                                    please specify correct value for self.prop_type while initialising")
            assert node_features_j is not None, "Provide second graph's node features to use GMN Match cross-graph attention,\
                                                node_features_j cannot be None when attention module is provided"
            att_features = att_module(node_features, node_features_j)
            node_input_list.append(att_features)
        
        out = self._compute_node_update(node_features, node_input_list)
        
        return out
    
    def __repr__(self):
        return ('{}(node_feature_dim={}, node_hidden_sizes={}, message_hidden_sizes={}, \
                    edge_feature_dim={})'.format(self.__class__.__name__, self.node_feature_dim,
                                                 self.node_hidden_sizes, self.message_hidden_sizes,
                                                 self.edge_feature_dim))
        


