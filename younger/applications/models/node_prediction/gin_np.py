import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GINConv


class GIN_NP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, layer_number = 3, output_embedding = False):
        super(GIN_NP, self).__init__()
        self.output_embedding = output_embedding
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(GIN_Conv(node_dim, hidden_dim, hidden_dim))
        for i in range(layer_number):
            self.layers.append(GIN_Conv(hidden_dim, hidden_dim, hidden_dim))

        self.layers.append(GIN_Conv(hidden_dim, hidden_dim, node_dict_size))
        self.initialize_parameters()

    def forward(self, x, edge_index, mask_x_position):
        x = self.node_embedding_layer(x).squeeze(1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)

        if self.output_embedding:
            return torch.mean(x, dim=0).unsqueeze(0)
        return F.log_softmax(x[mask_x_position], dim=1)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)


class GIN_Conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon = 0):
        super(GIN_Conv, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()), eps=epsilon)
        self.lr1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.ac1 = nn.PReLU()
        self.lr2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.ac2 = nn.PReLU()
        if input_dim == output_dim:
            self.res = nn.Identity()
        else:
            self.res = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, edge_index):
        o = self.gnn(x, edge_index)
        o = self.lr1(o)
        o = self.bn1(o)
        o = self.ac1(o)
        o = self.lr2(o)
        o = self.bn2(o)
        o = self.ac2(o)
        o = o + self.res(x)
        return o