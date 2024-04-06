#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-06 10:12
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import torch
import pathlib

import os.path as osp

from typing import Callable, Literal, Any
from torch_geometric.data import Data, Dataset, download_url

from younger.datasets.modules import Instance

from younger.datasets.utils.io import load_json, tar_extract
from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType


class YoungerDataset(Dataset):
    def __init__(
        self,
        root: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
        mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised',
        x_encode_method: Literal['OnlyOp'] = 'OnlyOp',
        y_encode_method: Literal['OnlyMt'] = 'OnlyMt'
    ):
        assert mode in ['Supervised', 'Unsupervised'], f'Dataset Mode Not Support - {mode}!'

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        self.mode = mode
        self.x_encode_method = x_encode_method
        self.y_encode_method = y_encode_method
    
    @property
    def node_type_indices(self) -> dict[str, int]:
        return self._node_type_indices

    @property
    def metric_indices(self) -> dict[str, int]:
        return self._metric_indices

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return osp.join(self.root, name, self.mode)

    @property
    def processed_dir(self) -> str:
        name = f'processed'
        return osp.join(self.root, name, self.mode)

    @property
    def raw_file_names(self):
        onnx_operators_path = osp.join(self.raw_dir, 'onnx_operators.json')
        onnx_operators: list[str] = load_json(onnx_operators_path)
        node_types = [f'__{node_type}__' for node_type in YoungerDatasetNodeType.attributes] + onnx_operators
        self._node_type_indices = {node_type: index for index, node_type in enumerate(node_types)}

        metrics_path = osp.join(self.raw_dir, 'metrics.json')
        metrics: list[str] = load_json(metrics_path)
        self._metric_indices = {metric: index for index, metric in enumerate(metrics)}

        indicator_path = osp.join(self.raw_dir, 'indicator.json')
        assert osp.exists(indicator_path)
        indicator: dict[str, Any] = load_json(indicator_path)
        tar_filenames = indicator['tar_filenames'] # without suffix '.tar.gz'
        return tar_filenames

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.total_processed_data)]

    def download(self):
        if self.mode == 'Supervised':
            raw_dataset_tar_path = download_url(YoungerDatasetAddress.SUPERVISED, self.raw_dir)

        if self.mode == 'Unsupervised':
            raw_dataset_tar_path = download_url(YoungerDatasetAddress.UNSUPERVISED, self.raw_dir)

        tar_extract(raw_dataset_tar_path, self.mode)
    
    def encode_node_features(self, node_labels: dict) -> list:
        node_type: str = node_labels['type']
        node_feature = list()
        if node_type == 'operator':
            node_feature.append(self.node_type_indices[node_labels['operator']['op_type']])
        else:
            node_feature.append(self.node_type_indices[f'__{node_type.upper()}__'])
        return node_feature

    def get_x(self, instance: Instance) -> torch.Tensor:
        node_indices = list(instance.network.graph.nodes)
        node_features = list()
        for node_index in node_indices:
            node_feature = self.encode_node_feature(instance.network.graph.nodes[node_index])
            node_features.append(node_feature)
        return torch.tensor(node_features)

    def get_edge_index(self, instance: Instance) -> torch.Tensor:
        edges = list(instance.network.graph.edges)
        src = [int(edge[0]) for edge in edges]
        dst = [int(edge[1]) for edge in edges]
        edge_index = torch.tensor([src, dst])
        return edge_index
    
    def encode_graph_feature(self, graph_labels: dict) -> list:
        task_name = graph_labels['task_name']
        dataset_name = graph_labels['dataset_name']
        metric_name = graph_labels['metric_name']
        metric_value = graph_labels['metric_value']

        if self.y_encode_method == 'OnlyMt':
            graph_feature = [self.metric_indices[metric_name], metric_value]
        
        return graph_feature

    def get_y(self, instance: Instance) -> torch.Tensor | None:
        if self.mode == 'Supervised':
            graph_feature = self.encode_graph_feature(instance.labels)
            return torch.tensor(graph_feature)
        if self.mode == 'Unsupervised':
            return None

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            tar_extract(raw_path+'.tar.gz', raw_path)

            raw_path = pathlib.Path(raw_path)
            instances = list()
            for path in raw_path.iterdir():
                if path.is_dir():
                    instance = Instance()
                    instance.load(path)
                    instances.append(instance)

            for instance in instances:
                x = self.get_x(instance)
                edge_index = self.get_edge_index(instance)
                y = self.get_y(instance)
                data = Data(x=x, edge_index=edge_index, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
        
        self.total_processed_data = idx

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data