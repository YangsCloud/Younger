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
import numpy
import torch
import networkx
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.modules import Network
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

        feature_get_type: Literal['none', 'mean', 'rand'] = 'mean',
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'
        self.meta = self.__class__.load_meta(meta_filepath)

        self.feature_get_type = feature_get_type

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
        return [f'sample-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'sample-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return self.meta['size']

    def get(self, index: int):
        return torch.load(os.path.join(self.processed_dir, f'sample-{index}.pth'))

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{index}.pkl'))

    def process(self):
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for _ in pool.imap_unordered(self.process_instance, range(self.meta['size'])):
                    progress_bar.update()

    def process_instance(self, index):
        sample_filepath = os.path.join(self.raw_dir, f'sample-{index}.pkl')
        sample = load_pickle(sample_filepath)

        data = self.__class__.get_data(sample, self.meta, self.feature_get_type)
        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(data, os.path.join(self.processed_dir, f'sample-{index}.pth'))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

        meta['metric_name'] = loaded_meta['metric_name']
        meta['version'] = loaded_meta['version']
        meta['archive'] = loaded_meta['archive']
        meta['size'] = loaded_meta['size']
        meta['url'] = loaded_meta['url']

        meta['i2t'] = loaded_meta['all_tasks'] + ['__UNK__']
        meta['i2n'] = loaded_meta['all_nodes'] + ['__UNK__']


        meta['t2i'] = {task: index for index, task in enumerate(meta['i2t'])}
        meta['n2i'] = {node: index for index, node in enumerate(meta['i2n'])}

        return meta

    @classmethod
    def get_edge_index(cls, sample: networkx.DiGraph) -> torch.Tensor:
        mapping = dict(zip(sample.nodes(), range(sample.number_of_nodes())))
        edge_index = torch.empty((2, sample.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(sample.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

    @classmethod
    def get_node_feature(cls, node_features: dict, meta: dict[str, Any]) -> list:
        node_identifier: str = Network.standardized_node_identifier(node_features)
        node_feature = list()
        node_feature.append(meta['n2i'].get(node_identifier, meta['n2i']['__UNK__']))
        return node_feature

    @classmethod
    def get_x(cls, sample: networkx.DiGraph, meta: dict[str, Any]) -> torch.Tensor:
        node_indices = list(sample.nodes)
        node_features = list()
        for node_index in node_indices:
            node_feature = cls.get_node_feature(sample.nodes[node_index]['features'], meta)
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)
        return node_features

    @classmethod
    def get_graph_feature(cls, graph_labels: dict, meta: dict[str, Any], feature_get_type: Literal['none', 'mean', 'rand']) -> list:

        if feature_get_type == 'none':
            graph_feature = None
        else:
            downloads: int = graph_labels['downloads']
            likes: int = graph_labels['likes']
            tasks: str = graph_labels['tasks']
            metrics: list[float] = graph_labels['metrics']

            graph_feature = list()
            task = tasks[numpy.random.randint(len(tasks))] if len(tasks) else None
            graph_feature.append(meta['t2i'].get(task, meta['n2i']['__UNK__']))

            if feature_get_type == 'mean':
                graph_feature.append(numpy.mean(metrics))

            if feature_get_type == 'rand':
                graph_feature.append(metrics[numpy.random.randint(len(metrics))])

        return graph_feature

    @classmethod
    def get_y(cls, sample: networkx.DiGraph, meta: dict[str, Any], feature_get_type: Literal['none', 'mean', 'rand']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(sample.graph, meta, feature_get_type)
        if graph_feature is None:
            graph_feature = None
        else:
            graph_feature = torch.tensor(graph_feature, dtype=torch.float32)
        return graph_feature

    @classmethod
    def get_data(cls, sample, meta: dict[str, Any], feature_get_type: Literal['none', 'mean', 'rand']) -> Data:
        x = cls.get_x(sample, meta)
        edge_index = cls.get_edge_index(sample)
        y = cls.get_y(sample, meta, feature_get_type)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data