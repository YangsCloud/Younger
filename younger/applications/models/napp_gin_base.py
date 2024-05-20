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


class NAPPGINBase(torch.nn.Module):
    # Neural Architecture Performance Prediction - GNN - Base Model
    def __init__(
        self,
        meta: dict,
        node_dim: int = 512,
        hidden_dim: int = 512,
        readout_dim: int = 256,
        cluster_num: int = 16,
        mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised'
    ):
        super().__init__()

        assert mode in {'Supervised', 'Unsupervised'}
        self.mode = mode

        self.activation_layer = resolver.activation_resolver('ELU')

        # GNN Layers
        self.node_embedding_layer = Embedding(len(meta['o2i']), node_dim)

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

        # Output Layer
        # Now Try Classification
        # Task: t2i
        # Dataset: d2i
        # Metric: m2i
        self.cls_output_layer = MLP(
            channel_list=[readout_dim, len(meta['m2i'])],
            act=None,
            norm=None,
        )
        # Now Try Regression
        self.reg_output_layer = MLP(
            channel_list=[readout_dim, 1],
            act=None,
            norm=None,
        )
        self.initialize_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        # x - [ batch_size * max(node_number_of_graphs) X num_node_features ] (Current Version: num_node_features=1)
        main_feature = x[:, 0]
        x = self.node_embedding_layer(main_feature)
        # x - [ batch_size * max(node_number_of_graphs) X node_dim ]

        x = self.gnn_head_mp_layer(x, edge_index)
        x = self.activation_layer(x)

        # x - [ batch_size * max(node_number_of_graphs) X hidden_dim ]
        (x, mask), adj = to_dense_batch(x, batch), to_dense_adj(edge_index, batch)
        # x - [ batch_size X max(node_number_of_graphs) X hidden_dim ]

        _, x, adj, spectral_loss, orthogonality_loss, cluster_loss = self.gnn_pooling_layer(x, adj, mask)

        gnn_pooling_loss = spectral_loss + orthogonality_loss + cluster_loss

        if self.mode == 'Unsupervised':
            return gnn_pooling_loss

        x = self.gnn_tail_mp_layer(x, adj)
        x = self.activation_layer(x)

        # x - [ batch_size X max(node_number_of_graphs) X hidden_dim ]
        x = self.gnn_readout_layer(x)
        # x - [ batch_size X max(node_number_of_graphs) X readout_dim ]
        x = x.sum(dim=1)
        # x - [ batch_size X readout_dim ]

        cls_output = self.cls_output_layer(x)
        # cls_output - [ batch_size X _ ]

        reg_output = self.reg_output_layer(x)
        # reg_output - [ batch_size X 1 ]

        return cls_output, reg_output, gnn_pooling_loss

    def initialize_parameters(self):
        torch.nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
