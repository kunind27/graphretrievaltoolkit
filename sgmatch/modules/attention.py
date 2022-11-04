import torch
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
from ..utils.utility import cudavar

class AttentionLayer(torch.nn.Module):
    def __init__(self, input_dim, type: str = 'simgnn', activation: str = "tanh", a = 0.1):
        """
        :param: type: Type of attention mechanism to be used
        :param: input_dim: Input Dimension of the Node Embeddings
        :param: activation: The Activation Function to be used for the Attention Layer
        :param: a: Slope of the -ve part if the activation is Leaky ReLU
        """
        super(AttentionLayer, self).__init__()
        self.type = type
        self.d = input_dim # Input dimension of the node embeddings
        self.activation = activation 
        self.a = a # Slope of the negative part in Leaky-ReLU
        
        self.params()
        self.initialize()
        
    def params(self):
        if(self.type == 'simgnn'):
            self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.d, self.d))

    def initialize(self):
        """
        Weight initialization depends upon the activation function used.
        If ReLU/ Leaky ReLU : He (Kaiming) Initialization
        If tanh/ sigmoid : Xavier Initialization
        """
        if self.activation == "leaky_relu" or self.activation == "relu":
            torch.nn.init.kaiming_normal_(self.weight_matrix, a = self.a, nonlinearity = self.activation)
        elif self.activation == "tanh" or self.activation == "sigmoid":
            torch.nn.init.xavier_normal_(self.weight_matrix)
        else:
            raise ValueError("Activation can only take values: 'relu', 'leaky_relu', 'sigmoid', 'tanh';\
                            {} is invalid".format(self.activation))

    def forward(self, node_embeds: Tensor):
        """ 
        :param: node_embeds : Node Embedding Tensor of shape N x D
        :return: global_graph_embedding for each graph in the batch
        """
        context = torch.mean(torch.matmul(node_embeds, self.weight_matrix), dim = 0)
        activations = {"tanh": torch.nn.functional.tanh, "leaky_relu": torch.nn.functional.leaky_relu,
                        "relu": torch.nn.functional.relu, "sigmoid": torch.nn.functional.sigmoid}
        _activation = activations[self.activation]
        # Applying the Non-Linearity over Weight_matrix*mean(U_i), the default is tanh
        transformed_context = _activation(context)
        sigmoid_scores = torch.sigmoid(torch.mm(node_embeds, transformed_context.view(-1, 1)))
        representation = torch.matmul(torch.t(node_embeds), sigmoid_scores)
        
        return representation