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
from younger.datasets.utils.constants import YoungerDatasetAddress


class ArchitectureDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,

        task_dict_size: int | None = None,
        node_dict_size: int | None = None,
        metric_feature_get_type: Literal['none', 'mean', 'rand'] = 'none',
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size)
        self.y_dict = self.__class__.get_y_dict(self.meta, task_dict_size)

        self.metric_feature_get_type = metric_feature_get_type

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
                for _ in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    progress_bar.update()

    def process_sample(self, index: int):
        sample_filepath = os.path.join(self.raw_dir, f'sample-{index}.pkl')
        sample: networkx.DiGraph = load_pickle(sample_filepath)

        data = self.__class__.get_data(sample, self.x_dict, self.y_dict, self.metric_feature_get_type)
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
        meta['all_tasks'] = loaded_meta['all_tasks']
        meta['all_nodes'] = loaded_meta['all_nodes']

        meta['split'] = loaded_meta['split']
        meta['archive'] = loaded_meta['archive']
        meta['version'] = loaded_meta['version']
        meta['size'] = loaded_meta['size']
        meta['url'] = loaded_meta['url']

        return meta

    @classmethod
    def get_mapping(cls, sample: networkx.DiGraph):
        return dict(zip(sorted(sample.nodes()), range(sample.number_of_nodes())))

    @classmethod
    def get_x_dict(cls, meta: dict[str, Any], node_dict_size: int | None = None) -> dict[str, list[str] | dict[str, int]]:
        all_nodes = [(node_id, node_count) for node_id, node_count in meta['all_nodes']['onnx'].items()] + [(node_id, node_count) for node_id, node_count in meta['all_nodes']['others'].items()]
        all_nodes = sorted(all_nodes, key=lambda x: x[1])

        node_dict_size = len(all_nodes) if node_dict_size is None else node_dict_size

        x_dict = dict()
        x_dict['i2n'] = ['__UNK__'] + [node_id for node_id, node_count in all_nodes[:node_dict_size]]

        x_dict['n2i'] = {node_id: index for index, node_id in enumerate(x_dict['i2n'])}
        return x_dict

    @classmethod
    def get_y_dict(cls, meta: dict[str, Any], task_dict_size: int | None = None) -> dict[str, list[str] | dict[str, int]]:
        all_tasks = [(task_id, task_count) for task_id, task_count in meta['all_tasks'].items()]
        all_tasks = sorted(all_tasks, key=lambda x: x[1])

        task_dict_size = len(all_tasks) if task_dict_size is None else task_dict_size

        y_dict = dict()
        y_dict['i2t'] = ['__UNK__'] + [task_id for task_id, task_count in all_tasks[:task_dict_size]]

        y_dict['t2i'] = {task_id: index for index, task_id in enumerate(y_dict['i2t'])}
        return y_dict

    @classmethod
    def get_edge_index(cls, sample: networkx.DiGraph) -> torch.Tensor:
        mapping = cls.get_mapping(sample)
        edge_index = torch.empty((2, sample.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(sample.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

    @classmethod
    def get_node_feature(cls, node_features: dict, x_dict: dict[str, Any]) -> list:
        node_identifier: str = Network.get_node_identifier_from_features(node_features)
        node_feature = list()
        node_feature.append(x_dict['n2i'].get(node_identifier, x_dict['n2i']['__UNK__']))
        return node_feature

    @classmethod
    def get_x(cls, sample: networkx.DiGraph, x_dict: dict[str, Any]) -> torch.Tensor:
        node_indices = list(sample.nodes)
        node_features = list()
        for node_index in node_indices:
            node_feature = cls.get_node_feature(sample.nodes[node_index]['features'], x_dict)
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)
        return node_features

    @classmethod
    def get_graph_feature(cls, graph_labels: dict, y_dict: dict[str, list[str] | dict[str, int]], metric_feature_get_type: Literal['none', 'mean', 'rand']) -> list:

        downloads: int = graph_labels['downloads']
        likes: int = graph_labels['likes']
        tasks: str = graph_labels['tasks']
        metrics: list[float] = graph_labels['metrics']

        graph_feature = list()
        task = tasks[numpy.random.randint(len(tasks))] if len(tasks) else None
        graph_feature.append(y_dict['t2i'].get(task, y_dict['t2i']['__UNK__']))

        if metric_feature_get_type == 'mean':
            graph_feature.append(numpy.mean(metrics))

        if metric_feature_get_type == 'rand':
            graph_feature.append(metrics[numpy.random.randint(len(metrics))])

        return graph_feature

    @classmethod
    def get_y(cls, sample: networkx.DiGraph, y_dict: dict[str, Any], metric_feature_get_type: Literal['none', 'mean', 'rand']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(sample.graph, y_dict, metric_feature_get_type)
        if graph_feature is None:
            graph_feature = None
        else:
            graph_feature = torch.tensor(graph_feature, dtype=torch.float32)
        return graph_feature

    @classmethod
    def get_data(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any], y_dict: dict[str, Any],
        metric_feature_get_type: Literal['none', 'mean', 'rand']
    ) -> Data:
        x = cls.get_x(sample, x_dict)
        edge_index = cls.get_edge_index(sample)
        y = cls.get_y(sample, y_dict, metric_feature_get_type)

        data = Data(x=x, edge_index=edge_index, y=y)
        return data