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
import tqdm
import torch
import networkx
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType


class YoungerDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,

        x_feature_get_type: Literal['OnlyOp'] = 'OnlyOp',
        y_feature_get_type: Literal['OnlyMt', 'TkDsMt'] = 'OnlyMt',
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'
        self.meta = self.__class__.load_meta(meta_filepath)

        self.x_feature_get_type = x_feature_get_type
        self.y_feature_get_type = y_feature_get_type

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
        return [f'instance-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'instance-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return self.meta['size']

    def get(self, index: int):
        return torch.load(os.path.join(self.processed_dir, f'instance-{index}.pth'))

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'instance-{index}.pkl'))

    def process(self):
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for _ in pool.imap_unordered(self.process_instance, range(self.meta['size'])):
                    progress_bar.update()

    def process_instance(self, index):
        instance_filepath = os.path.join(self.raw_dir, f'instance-{index}.pkl')
        instance = load_pickle(instance_filepath)

        x = self.__class__.get_x(instance, self.meta, self.x_feature_get_type)
        edge_index = self.__class__.get_edge_index(instance)
        y = self.__class__.get_y(instance, self.meta, self.y_feature_get_type)

        data = Data(x=x, edge_index=edge_index, y=y)
        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, os.path.join(self.processed_dir, f'instance-{index}.pth'))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

        meta['version'] = loaded_meta['version']
        meta['archive'] = loaded_meta['archive']
        meta['size'] = loaded_meta['size']
        meta['url'] = loaded_meta['url']


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

        meta['o2i'] = {operator: index for index, operator in enumerate(meta['i2o'])}

        meta['t2i'] = {task: index for index, task in enumerate(meta['i2t'])}
        meta['d2i'] = {dataset: index for index, dataset in enumerate(meta['i2d'])}
        meta['s2i'] = {split: index for index, split in enumerate(meta['i2s'])}
        meta['m2i'] = {metric: index for index, metric in enumerate(meta['i2m'])}

        return meta

    @classmethod
    def get_edge_index(cls, instance: networkx.DiGraph) -> torch.Tensor:
        edges = list(instance.edges)
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
    def get_x(cls, instance: networkx.DiGraph, meta: dict[str, Any], x_feature_get_type: Literal['OnlyOp']) -> torch.Tensor:
        node_indices = list(instance.nodes)
        node_features = list()
        for node_index in node_indices:
            node_feature = cls.get_node_feature(instance.nodes[node_index], meta, x_feature_get_type)
            node_features.append(node_feature)
        node_features = torch.tensor(node_features)
        return node_features

    @classmethod
    def get_graph_feature(cls, graph_labels: dict, meta: dict[str, Any], y_feature_get_type: Literal['OnlyMt']) -> list:
        task: str = graph_labels['task']
        dataset: str = graph_labels['dataset']
        metric: str = graph_labels['metric']
        metric_value: float = graph_labels['metric_value']

        graph_feature = None
        if y_feature_get_type == 'OnlyMt':
            if metric is not None and metric_value is not None:
                graph_feature = [meta['m2i'][metric], float(metric_value)]
            else:
                None

        if y_feature_get_type == 'TkDsMt':
            if metric is not None and metric_value is not None:
                graph_feature = [meta['m2i'][metric], float(metric_value), meta['t2i'][task], meta['d2i'][dataset]]
            else:
                None

        return graph_feature

    @classmethod
    def get_y(cls, instance: networkx.DiGraph, meta: dict[str, Any], y_feature_get_type: Literal['OnlyMt']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(instance.graph, meta, y_feature_get_type)
        if graph_feature is None:
            graph_feature = None
        else:
            graph_feature = torch.tensor(graph_feature)
        return graph_feature
