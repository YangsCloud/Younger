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


import os
import torch
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, tar_extract

from younger.datasets.modules import Instance
from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType


def download_aux_file(aux_filepath, url, folder):
    if os.path.exists(aux_filepath):
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
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'
        self.meta = self.__class__.load_meta(meta_filepath)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    @property
    def raw_dir(self) -> str:
        name = f'younger_raw'
        return os.path.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed'
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return [instance_name for instance_name in self.instance_names]

    @property
    def processed_file_names(self):
        return [f'{instance_name}.pt' for instance_name in self.instance_names]

    def len(self) -> int:
        return len(self.instance_names)

    def get(self, index: int):
        return torch.load(os.path.join(self.processed_dir, f'{self.instance_names[index]}.pt'))

    def download(self):
        archive_filepath = os.path.join(self.root, self.archive)
        if not fs.exists(archive_filepath):
            download_url(getattr(YoungerDatasetAddress, self.archive), self.root)
        tar_extract(archive_filepath, self.raw_dir)

        for instance_name in self.instance_names:
            assert fs.exists(os.path.join(self.raw_dir, instance_name))

    def process(self):
        with multiprocessing.Pool(self.worker_number) as pool:
            pool.map(self.process_instance, self.instance_names)

    def process_instance(self, instance_name):
        instance_dirpath = os.path.join(self.raw_dir, instance_name)
        instance = Instance()
        instance.load(instance_dirpath)

        x = self.__class__.get_x(instance, self.meta, self.x_feature_get_type)
        edge_index = self.__class__.get_edge_index(instance)
        y = self.__class__.get_y(instance, self.meta, self.y_feature_get_type)

        data = Data(x=x, edge_index=edge_index, y=y)
        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, os.path.join(self.processed_dir, f'{instance_name}.pt'))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

        meta['version'] = loaded_meta['version']
        meta['archive'] = loaded_meta['archive']
        meta['instance_names'] = loaded_meta['instance_names']


        meta['i2o'] = [
            YoungerDatasetNodeType.UNK,
            YoungerDatasetNodeType.OUTER,
            YoungerDatasetNodeType.INPUT,
            YoungerDatasetNodeType.OUTPUT,
            YoungerDatasetNodeType.CONSTANT
        ] + loaded_meta['operators']

        meta['i2t'] = loaded_meta['tasks']
        meta['i2d'] = loaded_meta['datasets']
        meta['i2s'] = loaded_meta['splits']
        meta['i2m'] = loaded_meta['metrics']

        meta['o2i'] = {operator: index for index, operator in meta['i2o']}

        meta['t2i'] = {task: index for index, task in meta['i2t']}
        meta['d2i'] = {dataset: index for index, dataset in meta['i2d']}
        meta['s2i'] = {split: index for index, split in meta['i2s']}
        meta['m2i'] = {metric: index for index, metric in meta['i2m']}

        return meta

    @classmethod
    def get_edge_index(cls, instance: Instance) -> torch.Tensor:
        edges = list(instance.network.graph.edges)
        src = [int(edge[0]) for edge in edges]
        dst = [int(edge[1]) for edge in edges]
        edge_index = torch.tensor([src, dst])
        return edge_index

    @classmethod
    def get_node_feature(cls, node_labels: dict, meta: dict[str, Any], x_feature_get_type: Literal['OnlyOp']) -> list:
        node_type: str = node_labels['type']
        node_feature = list()
        if x_feature_get_type == 'OnlyOp':
            if node_type == 'operator':
                node_feature.append(meta['o2i'].get(str(node_labels['operator']), meta['o2i'][YoungerDatasetNodeType.UNK]))
            else:
                node_feature.append(meta['o2i'][getattr(YoungerDatasetNodeType, node_type.upper())])
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
    def get_graph_feature(cls, graph_labels: dict, meta: dict[str, Any], y_feature_get_type: Literal['OnlyMt']) -> list:
        task: str = graph_labels['labels']['task']
        dataset: str = graph_labels['labels']['dataset'][0]
        split: str = graph_labels['labels']['dataset'][1]
        metric: str = graph_labels['labels']['metric'][0]
        value: float = graph_labels['labels']['metric'][1]

        if y_feature_get_type == 'OnlyMt':
            graph_feature = [meta['m2i'][metric], float(value)]

        return graph_feature

    @classmethod
    def get_y(cls, instance: Instance, meta: dict[str, Any], y_feature_get_type: Literal['OnlyMt']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(instance.labels['labels'], meta, y_feature_get_type)
        return torch.tensor(graph_feature)
