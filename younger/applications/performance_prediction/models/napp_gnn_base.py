#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 16:53
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import DenseGraphConv, DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_num: int, embedding_dim: int = 512) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class NAPPGNNBase(torch.nn.Module):
    def __init__(
        self,
        node_dict: dict,
        metric_dict: dict,
        node_dim: int = 512,
        metric_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # GNN Layer
        self.node_embedding_layer = EmbeddingLayer(len(node_dict), node_dim)

        self.conv1 = DenseGraphConv(node_dim, hidden_dim)
        self.pool1 = DMoNPooling([hidden_dim, hidden_dim], 16)

        self.conv1 = DenseGraphConv(node_dim, hidden_dim)
        self.pool1 = DMoNPooling([hidden_dim, hidden_dim], 8)

        self.lin1 = Linear(hidden_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)

        # Fuse Layer
        self.metric_embedding_layer = EmbeddingLayer(len(metric_dict), metric_dim)

        self.lin3 = Linear(metric_dim, hidden_dim)
        self.fuse = Linear(hidden_dim + hidden_dim, hidden_dim)
        self.output = Linear(hidden_dim, 1)

    def forward(self, x, x_mask, edge_index, metric):

        x = self.node_embedding_layer(x)

        x = self.conv1(x, edge_index).relu()
        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, x_mask)

        x = self.conv2(x, edge_index).relu()
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj, x_mask)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        m = self.metric_embedding_layer(metric)
        m = self.lin3(m)

        out = self.output(self.fuse(torch.concat(x, m)))
        return out, sp1 + sp2 + o1 + o2 + c1 + c2