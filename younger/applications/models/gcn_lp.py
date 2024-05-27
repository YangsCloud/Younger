import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch.nn import Embedding


class GCN_LP(nn.Module):
    def __init__(self, node_dict_size, node_dim, hidden_dim, output_dim):
        super(GCN_LP, self).__init__()
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.initialize_parameters()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        # print("x: ", x , x.shape)
        x = self.node_embedding_layer(x).squeeze(1)
        
        # print("x: ", x , x.shape)
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x

    def decode(self, z, edge_label_index):
        # print("edge_label_index[0]: ",edge_label_index[0])
        # print("edge_label_index[1]: ",edge_label_index[1])
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        # print("src:  ", src.shape)
        # print("dst:  ", dst.shape)
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
