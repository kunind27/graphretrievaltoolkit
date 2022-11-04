import torch
from torch_geometric.nn.conv import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch
from torch.nn.utils.rnn import pad_sequence
from utils.utility import cudavar
from utils.segment import unsorted_segment_sum

# TODO:

# 1. Should probably rename this file to GEM to separate it from GMN functionality

class GraphEncoder(torch.nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, node_hidden_sizes = None, 
                 edge_hidden_sizes = None, name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        # this also handles the case of an empty list
        self._node_feature_dim = node_feature_dim
        self._edge_feature_dim = edge_feature_dim
        self._node_hidden_sizes = node_hidden_sizes
        self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        # Build as LRLRL...
        if self._node_hidden_sizes:
            layer = []
            layer.append(torch.nn.Linear(self._node_feature_dim, self._node_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(torch.nn.ReLU())
                layer.append(torch.nn.Linear(self._node_hidden_sizes[i - 1], self._node_hidden_sizes[i]))
            self.MLP1 = torch.nn.Sequential(*layer)
        else:
            self.MLP1 = None

        if self._edge_hidden_sizes:
            # unneeded?
            if not self._edge_feature_dim:
                raise ValueError("Edge transformation applied but 0 edge dimension given")
            layer = []
            layer.append(torch.nn.Linear(self._edge_feature_dim, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._node_hidden_sizes)):
                layer.append(torch.nn.ReLU())
                layer.append(torch.nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP2 = torch.nn.Sequential(*layer)
        else:
            self.MLP2 = None

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if not self.MLP1:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = self.MLP2(node_features)

        return node_outputs, edge_outputs

def graph_prop_once(node_states, from_idx, to_idx, message_net, 
                    aggregation_module=None, edge_features=None):
    """One round of propagation (message passing) in a graph.

    Args:
      node_states: [n_nodes, node_state_dim] float tensor, node state vectors, one
        row for each node.
      from_idx: [n_edges] int tensor, index of the from nodes.
      to_idx: [n_edges] int tensor, index of the to nodes.
      message_net: a network that maps concatenated edge inputs to message
        vectors.
      aggregation_module: a module that aggregates messages on edges to aggregated
        messages for each node.  Should be a callable and can be called like the
        following,
        `aggregated_messages = aggregation_module(messages, to_idx, n_nodes)`,
        where messages is [n_edges, edge_message_dim] tensor, to_idx is the index
        of the to nodes, i.e. where each message should go to, and n_nodes is an
        int which is the number of nodes to aggregate into.
      edge_features: if provided, should be a [n_edges, edge_feature_dim] float
        tensor, extra features for each edge.

    Returns:
      aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
        aggregated messages, one row for each node.
    """
    from_states = node_states[from_idx]
    to_states = node_states[to_idx]
    edge_inputs = [from_states, to_states]

    if edge_features is not None:
        edge_inputs.append(edge_features)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    tensor = unsorted_segment_sum(messages, to_idx, node_states.shape[0])
    return tensor