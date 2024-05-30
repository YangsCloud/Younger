import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class SAGE_NP(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout):
        super(SAGE_NP, self).__init__()
        self.dropout = dropout
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.layer_1 = SAGEConv(node_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, node_dict_size)
        self.initialize_parameters()

    def forward(self, x, edge_index, mask_x_position):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.dropout(x, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.layer_2(x, edge_index)
        return F.log_softmax(x[mask_x_position], dim=1)
    
    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)