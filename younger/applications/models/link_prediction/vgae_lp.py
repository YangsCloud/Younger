import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GCNConv


class Encoder_LP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, output_dim):
        super(Encoder_LP, self).__init__()
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.conv_1 = GCNConv(node_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, output_dim)
        self.conv_logvar = GCNConv(hidden_dim, output_dim)
        self.initialize_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)