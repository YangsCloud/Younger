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


import tqdm
import torch
import pathlib

import os.path as osp

from typing import Callable, Literal, Any
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, tar_extract

from younger.datasets.modules import Instance
from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType
from younger.datasets.utils.metric_cleaner import clean_metric


def download_aux_file(aux_filepath, url, folder):
    if osp.exists(aux_filepath):
        return aux_filepath
    else:
        return download_url(url, folder)


class YoungerDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
        mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised',
        split: Literal['Train', 'Valid', 'Test'] = 'Train',
        x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
        y_feature_get_type: Literal['OnlyMt'] = 'OnlyMt'
    ):
        assert mode in {'Supervised', 'Unsupervised'}, f'Dataset Mode Not Support - {mode}!'
        assert split in {'Train', 'Valid', 'Test'}, f'Dataset Split Not Support - {split}!'

        self.root = osp.expanduser(fs.normpath(root))

        self.mode = mode
        self.split = split
        self.x_feature_get_type = x_feature_get_type
        self.y_feature_get_type = y_feature_get_type

        self.split_name = f'{self.mode}{f"_{self.split}" if self.mode == "Supervised" else ""}'

        self.indicators_filename = 'indicators.json'
        self.onnx_operators_filename = 'onnx_operators.json'
        self.metrics_filename = 'metrics.json'

        indicators_filepath = download_aux_file(osp.join(self.root, self.indicators_filename), YoungerDatasetAddress.INDICATORS, self.root)
        self.indicators: dict[str, Any] = load_json(indicators_filepath)

        onnx_operators_filepath = download_aux_file(osp.join(self.root, self.onnx_operators_filename), YoungerDatasetAddress.ONNX_OPERATORS, self.root)
        self._node_dict = self.__class__.load_node_dict(onnx_operators_filepath)

        metrics_filepath = download_aux_file(osp.join(self.root, self.metrics_filename), YoungerDatasetAddress.METRICS, self.root)
        self._metric_dict = self.__class__.load_metric_dict(metrics_filepath)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    @property
    def node_dict(self) -> dict[str, int]:
        return self._node_dict

    @property
    def metric_dict(self) -> dict[str, int]:
        return self._metric_dict

    @property
    def raw_dir(self) -> str:
        name = f'younger_raw'
        return osp.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self):
        return [self.split_name+'.tar.gz']

    @property
    def processed_file_names(self):
        return [osp.join(self.split_name, f'data_{i}.pt') for i in range(self.indicators[self.split_name]['instances_number'])]

    def download(self):
        # {self.raw_dir}/
        main_url = getattr(YoungerDatasetAddress, f'{self.split_name.upper()}')
        # {self.raw_dir}/{self.split_name}.tar.gz
        raw_dataset_tar_path = download_url(main_url, self.raw_dir)
        # {self.raw_dir}/{self.split_name}/
        tar_extract(raw_dataset_tar_path, self.raw_dir)

        # {self.raw_dir}/{self.split_name}/
        split_dirpath = osp.join(self.raw_dir, self.split_name)
        # {self.raw_dir}/{self.split_name}/{tar_filename}
        for tar_filename in self.indicators[self.split_name]['tar_filenames']:
            # {self.raw_dir}/{self.split_name}/{instances_dirname}
            tar_extract(osp.join(split_dirpath, tar_filename+'.tar.gz'), split_dirpath)
        total_instances = 0
        instances_dirpath = pathlib.Path(osp.join(split_dirpath, self.indicators[self.split_name]['instances_dirname']))
        for path in instances_dirpath.iterdir():
                if path.is_dir():
                    total_instances += 1
        assert total_instances == self.indicators[self.split_name]['instances_number']

    def process(self):
        fs.makedirs(osp.join(self.processed_dir, self.split_name), exist_ok=True)
        raw_instances_dirpath = pathlib.Path(osp.join(self.raw_dir, self.split_name, self.indicators[self.split_name]['instances_dirname']))
        for index, raw_instance_dirpath in tqdm.tqdm(enumerate(raw_instances_dirpath.iterdir()), total=self.indicators[self.split_name]['instances_number']):
            if raw_instance_dirpath.is_dir():
                instance = Instance()
                instance.load(raw_instance_dirpath)

                x = self.__class__.get_x(instance, self.node_dict, self.x_feature_get_type)
                edge_index = self.__class__.get_edge_index(instance)
                if self.mode == 'Supervised':
                    y = self.__class__.get_y(instance, self.metric_dict, self.y_feature_get_type)
                    data = Data(x=x, edge_index=edge_index, y=y)
                else:
                    data = Data(x=x, edge_index=edge_index)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, self.split_name, f'data_{index}.pt'))

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, index: int):
        return torch.load(osp.join(self.processed_dir, self.split_name, f'data_{index}.pt'))
    
    @classmethod
    def load_node_dict(cls, node_dict_filepath: str) -> dict[str, int]:
        onnx_operators: list[str] = load_json(node_dict_filepath)
        node_types = [
            YoungerDatasetNodeType.UNK,
            YoungerDatasetNodeType.OUTER,
            YoungerDatasetNodeType.INPUT,
            YoungerDatasetNodeType.OUTPUT,
            YoungerDatasetNodeType.CONSTANT
        ] + onnx_operators
        return {node_type: index for index, node_type in enumerate(node_types)}

    @classmethod
    def load_metric_dict(cls, metric_dict_filepath: str) -> dict[str, int]:
        metrics: list[str] = load_json(metric_dict_filepath)
        return {metric: index for index, metric in enumerate(metrics)}

    @classmethod
    def get_node_feature(cls, node_labels: dict, node_dict: dict[str, int], x_feature_get_type: Literal['OnlyOp']) -> list:
        node_type: str = node_labels['type']
        node_feature = list()
        if x_feature_get_type == 'OnlyOp':
            if node_type == 'operator':
                node_feature.append(node_dict.get(node_labels['operator']['op_type'], node_dict[YoungerDatasetNodeType.UNK]))
            else:
                node_feature.append(node_dict[getattr(YoungerDatasetNodeType, node_type.upper())])
        return node_feature

    @classmethod
    def get_x(cls, instance: Instance, node_dict: dict[str, int], x_feature_get_type: Literal['OnlyOp']) -> torch.Tensor:
        node_indices = list(instance.network.graph.nodes)
        node_features = list()
        for node_index in node_indices:
            node_feature = cls.get_node_feature(instance.network.graph.nodes[node_index], node_dict, x_feature_get_type)
            node_features.append(node_feature)
        return torch.tensor(node_features)

    @classmethod
    def get_edge_index(cls, instance: Instance) -> torch.Tensor:
        edges = list(instance.network.graph.edges)
        src = [int(edge[0]) for edge in edges]
        dst = [int(edge[1]) for edge in edges]
        edge_index = torch.tensor([src, dst])
        return edge_index

    @classmethod
    def get_graph_feature(cls, graph_labels: dict, metric_dict: dict[str, int], y_feature_get_type: Literal['OnlyMt']) -> list:
        #task_name = graph_labels['task_name']
        #dataset_name = graph_labels['dataset_name']
        metrics: list[dict] = graph_labels['metrics']

        if y_feature_get_type == 'OnlyMt':
            graph_feature = [metric_dict[clean_metric(metrics[0]['metric_type'], metrics[0]['metric_name'])], float(metrics[0]['metric_value'])]

        return graph_feature

    @classmethod
    def get_y(cls, instance: Instance, metric_dict: dict[str, int], y_feature_get_type: Literal['OnlyMt']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(instance.labels, metric_dict, y_feature_get_type)
        return torch.tensor(graph_feature)
