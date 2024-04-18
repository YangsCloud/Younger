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
import multiprocessing

import os.path as osp

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, tar_extract

from younger.datasets.modules import Instance
from younger.datasets.utils.constants import YoungerDatasetAddress, YoungerDatasetNodeType


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
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = osp.join(root, 'meta.json')
        assert osp.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta: dict[str, Any] = load_json(meta_filepath)
        self.instances: list[str] = self.meta['instances']

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

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
        return [instance for instance in self.instances]

    @property
    def processed_file_names(self):
        return [f'{instance}.pt' for instance in self.instances]

    def len(self) -> int:
        return len(self.instances)

    def get(self, index: int):
        return torch.load(f'{self.instances[index]}.pt')

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

    def sub_process(self, sub_process_paths: list[pathlib.Path]):
        for sub_process_path in sub_process_paths:
            if sub_process_path.is_dir():
                instance = Instance()
                instance.load(sub_process_path)

                x = self.__class__.get_x(instance, self.node_dict, self.x_feature_get_type)
                edge_index = self.__class__.get_edge_index(instance)
                if self.mode == 'Supervised':
                    ys = self.__class__.get_y(instance, self.metric_dict, self.y_feature_get_type)
                    for y in ys:
                        data = Data(x=x, edge_index=edge_index, y=y)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    torch.save(data, osp.join(self.processed_dir, self.split_name, f'data_{index}.pt'))

                else:
                    data = Data(x=x, edge_index=edge_index)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, osp.join(self.processed_dir, self.split_name, f'data_{index}.pt'))

    def process(self):
        fs.makedirs(osp.join(self.processed_dir, self.split_name), exist_ok=True)
        raw_instances_dirpath = pathlib.Path(osp.join(self.raw_dir, self.split_name, self.indicators[self.split_name]['instances_dirname']))
        paths = list(raw_instances_dirpath.iterdir())
        quotient, remainder = divmod(len(paths), self.worker_number)
        step = quotient + (remainder != 0)
        sub_process_paths = [paths[index: index+step] for index in range(0, len(paths), step)]
        with multiprocessing.Pool(self.worker_number) as pool:
            pool.map(self.sub_process, sub_process_paths)

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
                node_feature.append(node_dict.get(str(node_labels['operator']), node_dict[YoungerDatasetNodeType.UNK]))
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
        metrics: list[dict] = graph_labels['labels']

        if y_feature_get_type == 'OnlyMt':
            graph_feature = [metric_dict[metrics[0]['metric_type'], metrics[0]['metric_name']], float(metrics[0]['metric_value'])]

        return graph_feature

    @classmethod
    def get_y(cls, instance: Instance, metric_dict: dict[str, int], y_feature_get_type: Literal['OnlyMt']) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(instance.labels, metric_dict, y_feature_get_type)
        return torch.tensor(graph_feature)
