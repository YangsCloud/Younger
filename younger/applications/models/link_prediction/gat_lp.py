import torch.nn.functional as F

from torch import nn
from torch.nn import Embedding
from torch_geometric.nn import GATConv


class GAT_LP(nn.Module):
    def __init__(self, node_dict_size, node_dim, hidden_dim, output_dim):
        super(GAT_LP, self).__init__()
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.conv1 = GATConv(node_dim, hidden_dim, heads=8, concat=False)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=8, concat=False)
        self.initialize_parameters()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
