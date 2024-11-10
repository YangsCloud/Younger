import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.nn import GINConv


class MAEGIN(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, layer_number = 3):
        super(MAEGIN, self).__init__()
        self.encoder = MAEGINEncoder(node_dict_size, node_dim, hidden_dim, dropout, layer_number)
        self.project = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = MAEGINDecoder(node_dict_size, hidden_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.project(x)
        x = self.decoder(x, edge_index)
        return x


class MAEGINEncoder(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, layer_number = 3):
        super(MAEGINEncoder, self).__init__()
        self.dropout = dropout

        self.node_embedding_layer = Embedding(node_dict_size, node_dim)

        self.layers = nn.ModuleList()
        self.layers.append(MAEGINConv(node_dim, hidden_dim, hidden_dim))
        for i in range(layer_number):
            self.layers.append(MAEGINConv(hidden_dim, hidden_dim, hidden_dim))

        self.layers.append(MAEGINConv(hidden_dim, hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
        return x


class MAEGINDecoder(nn.Module):

    def __init__(self, node_dict_size, hidden_dim):
        super(MAEGINDecoder, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()))
        self.trn = nn.Linear(hidden_dim, hidden_dim)
        self.prd = nn.Linear(hidden_dim, node_dict_size)

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = self.trn(x)
        x = self.prd(x)
        return x


class MAEGINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MAEGINConv, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()))
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