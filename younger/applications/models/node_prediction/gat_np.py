import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GAT_NP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, output_embedding = False):
        super(GAT_NP, self).__init__()
        self.output_embedding = output_embedding
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.dropout = dropout
        self.layer_1 = GATConv(node_dim, hidden_dim, heads=8, dropout=dropout, concat=False)
        self.layer_2 = GATConv(hidden_dim, node_dict_size, heads=8, dropout=dropout, concat=False)
        self.initialize_parameters()

    def forward(self, x, edge_index, mask_x_position):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        if self.output_embedding:
            return torch.mean(x, dim=0).unsqueeze(0)
        return F.log_softmax(x[mask_x_position], dim=1)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)