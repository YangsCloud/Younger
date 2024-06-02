#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-25 16:13
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

# The Code of GLASS are copied and modified from Official GLASS Repository: https://github.com/Xi-yuanWang/GLASS


import torch

from typing import Literal

from torch.nn import Embedding, ELU, ReLU, Dropout, Linear

from torch_geometric.nn import GraphNorm, GraphSizeNorm, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


def to_sparse_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor, node_number: int, aggr_type: Literal['sum', 'avg', 'gcn'] = 'avg'):
    sparse_adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(node_number, node_number), device=edge_index.device)
    if aggr_type == 'avg':
        i_degree = torch.sparse.sum(sparse_adj, dim=(0, )).to_dense().flatten()
        i_degree[i_degree < 1] += 1
        i_degree = 1.0 / i_degree
        aggr_weight = i_degree[edge_index[0]] * edge_weight

    if aggr_type == "sum":
        aggr_weight = edge_weight

    if aggr_type == "gcn":
        i_degree = torch.sparse.sum(sparse_adj, dim=(0, )).to_dense().flatten()
        i_degree[i_degree < 1] += 1
        i_degree = torch.pow(i_degree, -0.5)

        o_degree = torch.sparse.sum(sparse_adj, dim=(1, )).to_dense().flatten()
        o_degree[o_degree < 1] += 1
        o_degree = torch.pow(o_degree, -0.5)

        aggr_weight = i_degree[edge_index[0]] * edge_weight * o_degree[edge_index[1]]
    return torch.sparse_coo_tensor(edge_index, aggr_weight, size=(node_number, node_number), device=edge_index.device)


class GLASSConv(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        aggr_type: Literal['sum', 'avg', 'gcn'] = 'avg',
        ratio=0.8,
        dropout=0.2,
    ):
        super().__init__()
        self.transform_for_0 = Linear(input_dim, output_dim)
        self.combine_for_0 = Linear(input_dim + output_dim, output_dim)

        self.transform_for_1 = Linear(input_dim, output_dim)
        self.combine_for_1 = Linear(input_dim + output_dim, output_dim)

        self.activation = ELU(inplace=True)
        self.graph_norm = GraphNorm(output_dim)

        self.aggr_type = aggr_type
        self.ratio = ratio
        self.dropout = dropout

        self.reset_parameters()

    def forward(self, x, edge_index, edge_weight, block_mask):
        adj = to_sparse_adj(edge_index, edge_weight, x.shape[0], self.aggr_type)

        origin_x = x

        # transform node features with different parameters individually.
        x0 = self.activation(self.transform_for_0(x))
        x1 = self.activation(self.transform_for_1(x))

        # mix transformed feature.
        x = torch.where(block_mask, self.ratio * x1 + (1 - self.ratio) * x0, self.ratio * x0 + (1 - self.ratio) * x1)

        # pass messages.
        x = adj @ x
        x = self.graph_norm(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, origin_x), dim=-1)

        # transform node features with different parameters individually.
        x0 = self.combine_for_0(x)
        x1 = self.combine_for_1(x)

        # mix transformed feature.
        x = torch.where(block_mask, self.ratio * x1 + (1 - self.ratio) * x0, self.ratio * x0 + (1 - self.ratio) * x1)
        return x

    def reset_parameters(self):
        self.transform_for_0.reset_parameters()
        self.transform_for_1.reset_parameters()
        self.combine_for_0.reset_parameters()
        self.combine_for_1.reset_parameters()
        self.graph_norm.reset_parameters()


class GLASS(torch.nn.Module):
    def __init__(
        self,
        node_dict_size: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        pool_type: Literal['sum', 'max', 'mean', 'size'] = 'mean',
        dropout: float = 0.2,

        aggr_type: Literal['sum', 'avg', 'gcn'] = 'avg',
        ratio: float = 0.8,
    ):
        super().__init__()
        self.node_emb = Embedding(node_dict_size, hidden_dim)

        self.linear_0 = Linear(hidden_dim, hidden_dim)
        self.relu = ReLU(inplace=True)

        self.glass_conv_0 = GCNConv(hidden_dim, hidden_dim)
        self.graph_norm_0 = GraphNorm(hidden_dim)

        self.linear_1 = Linear(hidden_dim + hidden_dim, hidden_dim)

        self.glass_conv_1 = GLASSConv(input_dim=hidden_dim, output_dim=hidden_dim, aggr_type=aggr_type, ratio=ratio, dropout=dropout)
        self.graph_norm_1 = GraphNorm(hidden_dim)

        self.glass_conv_2 = GLASSConv(input_dim=hidden_dim, output_dim=hidden_dim, aggr_type=aggr_type, ratio=ratio, dropout=dropout)
        self.graph_norm_2 = GraphNorm(hidden_dim + hidden_dim)

        self.activation = ELU(inplace=True)
        self.pool = Pool(pool_type)
        self.dropout = dropout

        self.output = Linear(hidden_dim + hidden_dim, output_dim)

        self.reset_parameters()

    def forward(self, x, edge_index, edge_weight, block_mask, batch):
        # x
        # total_node_number = sum(node_number_of_graph_{1}, ..., node_number_of_graph_{batch_size})
        # [ total_node_number X num_node_features ] (Current Version: num_node_features=1)
        # block_mask
        # [ total_node_number X 1]
        assert x.shape[0] == block_mask.shape[0], f'Wrong shape of input \'x\'({x.shape[0]}) and \'block_mask\'({block_mask.shape[0]})'
        block_mask = block_mask.reshape(block_mask.shape[0], -1)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=edge_index.device)

        main_feature = x[:, 0]
        # [ total_node_number X 1 ]

        x = self.node_emb(main_feature)
        jumping_knowledge = x

        x = self.linear_0(x)
        x = self.relu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.glass_conv_0(x, edge_index)
        x = self.graph_norm_0(x)

        x = torch.cat([x, jumping_knowledge], dim=-1)
        x = self.linear_1(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        block_mask = (block_mask > 0)
        # pass messages at each layer.
        x = self.glass_conv_1(x, edge_index, edge_weight, block_mask)
        jumping_knowledge = x
        x = self.graph_norm_1(x)
        x = self.activation(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)

        x = self.glass_conv_2(x, edge_index, edge_weight, block_mask)
        x = torch.cat([jumping_knowledge, x], dim=-1)
        x = self.graph_norm_2(x)

        x = torch.where(block_mask, x, 0)
        x = self.pool(x, batch)
        return self.output(x)

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        self.linear_0.reset_parameters()
        self.glass_conv_0.reset_parameters()
        
        self.graph_norm_0.reset_parameters()
        self.linear_1.reset_parameters()

        self.glass_conv_1.reset_parameters()
        self.graph_norm_1.reset_parameters()

        self.glass_conv_2.reset_parameters()
        self.graph_norm_2.reset_parameters()


class BasePool(torch.nn.Module):
    def __init__(self, pool_method):
        super().__init__()
        self.pool_method = pool_method

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        return self.pool_method(x, batch)


class AddPool(BasePool):
    def __init__(self):
        super().__init__(global_add_pool)


class MaxPool(BasePool):
    def __init__(self):
        super().__init__(global_max_pool)


class MeanPool(BasePool):
    def __init__(self):
        super().__init__(global_mean_pool)


class SizePool(AddPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class Pool(torch.nn.Module):
    def __init__(self, pool_type: Literal['sum', 'max', 'mean', 'size'] = 'mean'):
        super().__init__()
        mapping = dict(
            max = MaxPool,
            sum = AddPool,
            mean = MeanPool,
            size = SizePool,
        )
        self.pool = mapping[pool_type]()

    def forward(self, x, batch):
        return self.pool(x, batch)