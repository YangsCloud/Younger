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


import torch

from typing import Literal

from torch.nn import Embedding

from torch_geometric.nn import resolver, MLP, DMoNPooling, GINConv, DenseGINConv, GraphConv, DenseGraphConv, GCNConv, DenseGCNConv, SAGEConv, DenseSAGEConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class NAPPGNNBase(torch.nn.Module):
    # Neural Architecture Performance Prediction - GNN - Base Model
    def __init__(
        self,
        node_dict: dict,
        metric_dict: dict,
        node_dim: int = 512,
        metric_dim: int = 512,
        hidden_dim: int = 512,
        readout_dim: int = 256,
        cluster_num: int = 16,
        mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised'
    ):
        super().__init__()

        assert mode in ['Supervised', 'Unsupervised']
        self.mode = mode

        self.activation_layer = resolver.activation_resolver('ELU')

        self.node_embedding_layer = Embedding(len(node_dict), node_dim)
        self.metric_embedding_layer = Embedding(len(metric_dict), metric_dim)

        # GNN Layers
        self.gnn_head_mp_layer = GINConv(
            nn=MLP(
                channel_list=[node_dim, hidden_dim, hidden_dim],
                act='ELU',
            ),
            eps=0,
            train_eps=False
        )

        self.gnn_pooling_layer = DMoNPooling(hidden_dim, cluster_num)

        if self.mode == 'Unsupervised':
            return

        self.gnn_tail_mp_layer = DenseGINConv(
            nn=MLP(
                channel_list=[hidden_dim, hidden_dim, hidden_dim],
                act='ELU',
                norm=None,
            ),
            eps=0,
            train_eps=False
        )

        self.gnn_readout_layer = MLP(
            channel_list=[hidden_dim, hidden_dim, readout_dim],
            act='ELU',
            norm=None,
            dropout=0.5
        )

        # Metric Layers
        self.metric_layer = MLP(
            channel_list=[metric_dim, hidden_dim, hidden_dim],
            act='RELU',
        )

        # Fuse Layers
        self.fuse_layer = MLP(
            channel_list=[readout_dim + hidden_dim, hidden_dim, hidden_dim],
            act='ELU'
        )
        # Output Layer
        self.output_layer = MLP(
            channel_list=[hidden_dim, 1],
            act=None,
            norm=None,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, metric: torch.Tensor, batch: torch.Tensor):
        x = self.node_embedding_layer(x)

        x = self.gnn_head_mp_layer(x, edge_index)
        x = self.activation_layer(x)

        (x, mask), adj = to_dense_batch(x, batch), to_dense_adj(edge_index, batch)

        _, x, adj, spectral_loss, orthogonality_loss, cluster_loss = self.gnn_pooling_layer(x, adj, mask)

        gnn_pooling_loss = spectral_loss + orthogonality_loss + cluster_loss

        if self.mode == 'Unsupervised':
            return gnn_pooling_loss

        x = self.gnn_tail_mp_layer(x, adj)
        x = self.activation_layer(x)

        metric = self.metric_embedding_layer(metric)
        metric = self.metric_layer(metric)

        fused_information = self.fuse_layer(torch.concat([x.sum(dim=1), metric], dim=-1))

        output = self.output_layer(fused_information)

        return output, gnn_pooling_loss