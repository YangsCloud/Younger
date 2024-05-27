#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-21 12:24
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
from torch_geometric.utils import negative_sampling

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.modules import Network
from younger.datasets.utils.constants import YoungerDatasetAddress


class LinkDataset(Dataset):
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
        seed: int | None = None,
        link_get_number: int | None = None,
        worker_number: int = 4,
    ):
        self.seed = seed
        self.worker_number = worker_number
        self.link_get_number = link_get_number
        print(self.link_get_number)

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size)
        self.y_dict = self.__class__.get_y_dict(self.meta, task_dict_size)

        self._link_locations: list[tuple[int, int]] = list()
        self._link_locations_filename = 'link_locations.pt'

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        _link_locations_filepath = os.path.join(self.processed_dir, self._link_locations_filename)

        try:
            self._link_locations = torch.load(_link_locations_filepath)
        except FileExistsError as error:
            print(f'There is no link locations file \'{self._link_locations_filename}\', please remove the processed data directory: {self.processed_dir} and re-run the command.')
            raise error

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
        return len(self._link_locations)

    def get(self, index: int) -> Data:
        sample_index, link_index = self._link_locations[index]
        graph_data_with_links: dict[str, Data | torch.LongTensor] = torch.load(os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        graph_data: Data = graph_data_with_links['graph_data']
        link: torch.LongTensor = graph_data_with_links['links'][:,link_index].view(2,1)
        link_label: int = graph_data_with_links['link_labels'][link_index]
        # print("here!____________:", link)
        # print("here!____________:", link.view(2,1))
        # print("here!____________:", graph_data.edge_index[:,link_index])

        combined_data: Data = Data(x=graph_data.x, edge_index=graph_data.edge_index, y=graph_data.y, link=link, link_label=link_label)
        # print("here!-------: ", combined_data)]
        return combined_data

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for sample_index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{sample_index}.pkl'))

    def process(self):
        unordered_link_records = list()
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, link_number in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    print("link_num: ",link_number)
                    unordered_link_records.append((sample_index, link_number))
                    progress_bar.update()
        sorted_link_records = sorted(unordered_link_records, key=lambda x: x[0])
        for sample_index, link_number in sorted_link_records:
            link_location = [(sample_index, link_index) for link_index in range(link_number)]
            self._link_locations.extend(link_location)
        _link_locations_filepath = os.path.join(self.processed_dir, self._link_locations_filename)
        torch.save(self._link_locations, _link_locations_filepath)

    def process_sample(self, sample_index: int) -> tuple[int, int]:
        sample_filepath = os.path.join(self.raw_dir, f'sample-{sample_index}.pkl')
        sample: networkx.DiGraph = load_pickle(sample_filepath)

        graph_data_with_links = self.__class__.get_graph_data_with_links(sample, self.x_dict, self.y_dict, self.seed, self.link_get_number)
        if self.pre_filter is not None and not self.pre_filter(graph_data_with_links):
            return

        if self.pre_transform is not None:
            graph_data_with_links = self.pre_transform(graph_data_with_links)
        torch.save(graph_data_with_links, os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        return (sample_index, graph_data_with_links['links'].shape[-1])

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
        mapping = dict(zip(sample.nodes(), range(sample.number_of_nodes())))
        edge_index = torch.empty((2, sample.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(sample.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

    @classmethod
    def get_label_edge(cls, edge_index: Data.edge_index, x: Data.x) -> tuple[list, list]:
        neg_edge_index = negative_sampling(
            edge_index=edge_index, num_nodes=len(x),
            num_neg_samples=edge_index.size(1), method='sparse')
        edge_label_index = torch.cat(
            [edge_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            torch.ones(edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ], dim=0)

        return edge_label, edge_label_index

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
    def get_graph_feature(cls, graph_labels: dict, y_dict: dict[str, list[str] | dict[str, int]]) -> list:

        downloads: int = graph_labels['downloads']
        likes: int = graph_labels['likes']
        tasks: str = graph_labels['tasks']
        metrics: list[float] = graph_labels['metrics']

        graph_feature = list()
        task = tasks[numpy.random.randint(len(tasks))] if len(tasks) else None
        graph_feature.append(y_dict['t2i'].get(task, y_dict['t2i']['__UNK__']))
        
        return graph_feature

    @classmethod
    def get_y(cls, sample: networkx.DiGraph, y_dict: dict[str, Any]) -> torch.Tensor:
        graph_feature = cls.get_graph_feature(sample.graph, y_dict)
        if graph_feature is None:
            graph_feature = None
        else:
            graph_feature = torch.tensor(graph_feature, dtype=torch.float32)
        return graph_feature

    @classmethod
    def get_graph_data(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any], y_dict: dict[str, Any]
    ) -> Data:
        x = cls.get_x(sample, x_dict)
        edge_index = cls.get_edge_index(sample)
        y = cls.get_y(sample, y_dict)

        edge_label, edge_label_index = cls.get_label_edge(edge_index, x)

        graph_data = Data(x=x, edge_index=edge_index, y=y, edge_label=edge_label, edge_label_index=edge_label_index)
        return graph_data

    @classmethod
    def get_graph_data_with_links(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any], y_dict: dict[str, Any],
        seed: int | None = None,
        link_get_number: int | None = None
    ) -> dict[str, Data | list[set]]:

        graph_data = cls.get_graph_data(sample, x_dict, y_dict)
        print("link_get_number: ", link_get_number)
        if link_get_number is not None:
            pos_links = graph_data.edge_label_index[:, :link_get_number//2]
            neg_loc = graph_data.edge_label_index.shape[1]//2
            neg_links = graph_data.edge_label_index[:, neg_loc:neg_loc+(link_get_number//2)]
            links = torch.cat([pos_links,neg_links],dim=1)
            link_labels = torch.cat([
                torch.ones(link_get_number//2),
                torch.zeros(link_get_number//2)
            ], dim=0)
            print("full: " , graph_data.edge_label_index)
            print("pos: ", pos_links)
            print("neg: ", neg_links)
            print("links: ", links)
            print("label: ", link_labels)
        else:
            links = graph_data.edge_label_index
            link_labels = graph_data.edge_label

        return dict(
            graph_data = graph_data,
            links = links,
            link_labels = link_labels
        )