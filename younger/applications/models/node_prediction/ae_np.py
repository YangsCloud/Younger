import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GCNConv


class Encoder_NP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, ae_type):
        super(Encoder_NP, self).__init__()
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.model_type = ae_type
        self.conv_1 = GCNConv(node_dim, 2 * hidden_dim)
        if self.model_type == "GAE":
            self.conv_2 = GCNConv(2 * hidden_dim, hidden_dim)
        elif self.model_type == "VGAE":
            self.conv_mu = GCNConv(2 * hidden_dim, hidden_dim)
            self.conv_logvar = GCNConv(2 * hidden_dim, hidden_dim)
        self.initialize_parameters()

    def forward(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        x = self.conv_1(x, edge_index)
        x = F.relu(x)
        if self.model_type == "GAE":
            return self.conv_2(x, edge_index)
        elif self.model_type == "VGAE":
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)


class LinearCls(nn.Module):

    def __init__(self, hidden_dim, node_dict_size, output_embedding = False):
        super(LinearCls, self).__init__()
        self.output_embedding = output_embedding
        self.linear = nn.Linear(hidden_dim, node_dict_size)

    def forward(self, x, mask_x_position):
        x = self.linear(x)
        if self.output_embedding:
            return torch.mean(x, dim=0).unsqueeze(0)
        return F.log_softmax(x[mask_x_position], dim=1)