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
        split: Literal['train', 'valid', 'test'],
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,

        node_dict_size: int | None = None,
        operator_dict_size: int | None = None,
        encode_type: Literal['node', 'operator'] = 'node',
        seed: int | None = None,
        link_get_number: int | None = None,
        worker_number: int = 4,
    ):
        assert encode_type in {'node', 'operator'}
        self.encode_type = encode_type
        self.seed = seed
        self.worker_number = worker_number
        self.link_get_number = link_get_number

        meta_filepath = os.path.join(root, 'meta.json')
        if not os.path.isfile(meta_filepath):
            print(f'No \'meta.json\' File Provided, It will be downloaded From Official Cite ...')
            if encode_type == 'node':
                download_url(getattr(YoungerDatasetAddress, f'DATAFLOW_{split.upper()}_WA_PAPER'), root, filename='meta.json')
            if encode_type == 'operator':
                download_url(getattr(YoungerDatasetAddress, f'DATAFLOW_{split.upper()}_WOA_PAPER'), root, filename='meta.json')

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size, operator_dict_size)

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
        name = f'younger_raw_{self.encode_type}_lp'
        return os.path.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed_{self.encode_type}_lp'
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return [f'sample-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'sample-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return self.meta['size']

    def get(self, index: int) -> Data:
        graph_data_with_links: dict[str, Data | torch.LongTensor] = torch.load(os.path.join(self.processed_dir, f'sample-{index}.pth'))
        graph_data: Data = graph_data_with_links['graph_data']
        link: torch.LongTensor = graph_data_with_links['links']
        link_label: int = graph_data_with_links['link_labels']
        combined_data: Data = Data(x=graph_data.x, edge_index=graph_data.edge_index, link=link, link_label=link_label)
        # print("index:", index)
        # print("graph_data.edge_index[:,:5]:", graph_data.edge_index[:,:5])
        # print("combined_data: ", combined_data)
        return combined_data

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            print(f'Dataset Archive Not Found.It will be downloaded From Official Cite ...')
            print(f'Begin Download {self.meta["archive"]}')
            archive_filepath = download_url(self.meta['url'], self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for sample_index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{sample_index}.pkl'))

    def process(self):
        unordered_link_records = list()
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, link_number in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
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
        graph_data_with_links = self.__class__.get_graph_data_with_links(sample, self.x_dict, self.encode_type, self.link_get_number)
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
        meta['all_operators'] = loaded_meta['all_operators']

        meta['split'] = loaded_meta['split']
        meta['archive'] = loaded_meta['archive']
        meta['version'] = loaded_meta['version']
        meta['size'] = loaded_meta['size']
        meta['url'] = loaded_meta['url']

        return meta
    
    @classmethod
    def get_mapping(cls, graph: networkx.DiGraph) -> dict[str, int]:
        return dict(zip(sorted(graph.nodes()), range(graph.number_of_nodes())))

    @classmethod
    def get_x_dict(cls, meta: dict[str, Any], node_dict_size: int | None = None, operator_dict_size: int | None = None) -> dict[str, list[str] | dict[str, int]]:
        all_nodes = [(node_id, node_count) for node_id, node_count in meta['all_nodes']['onnx'].items()] + [(node_id, node_count) for node_id, node_count in meta['all_nodes']['others'].items()]
        all_nodes = sorted(all_nodes, key=lambda x: x[1])

        all_operators = [(operator_id, operator_count) for operator_id, operator_count in meta['all_operators']['onnx'].items()] + [(operator_id, operator_count) for operator_id, operator_count in meta['all_operators']['others'].items()]
        all_operators = sorted(all_operators, key=lambda x: x[1])

        node_dict_size = len(all_nodes) if node_dict_size is None else node_dict_size
        operator_dict_size = len(all_operators) if operator_dict_size is None else operator_dict_size

        x_dict = dict()
        x_dict['i2n'] = ['__UNK__'] + ['__MASK__'] + [node_id for node_id, node_count in all_nodes[:node_dict_size]]
        x_dict['n2i'] = {node_id: index for index, node_id in enumerate(x_dict['i2n'])}
        x_dict['i2o'] = ['__UNK__'] + ['__MASK__'] + [operator_id for operator_id, operator_count in all_operators[:operator_dict_size]]
        x_dict['o2i'] = {operator_id: index for index, operator_id in enumerate(x_dict['i2o'])}
        return x_dict

    @classmethod
    def get_edge_index(cls, sample: networkx.DiGraph, mapping: dict[str, int]) -> torch.Tensor:
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
    def get_node_class(cls, node_features: dict, x_dict: dict[str, Any], encode_type: Literal['node', 'operator'] = 'node') -> int:
        if encode_type == 'node':
            node_identifier: str = Network.get_node_identifier_from_features(node_features, mode='full')
            node_class = x_dict['n2i'].get(node_identifier, x_dict['n2i']['__UNK__'])

        if encode_type == 'operator':
            node_identifier: str = Network.get_node_identifier_from_features(node_features, mode='type')
            node_class = x_dict['o2i'].get(node_identifier, x_dict['o2i']['__UNK__'])

        return node_class

    @classmethod
    def get_x(cls, graph: networkx.DiGraph, mapping: dict[str, int], x_dict: dict[str, Any], encode_type: Literal['node', 'operator'] = 'node') -> torch.Tensor:
        node_indices = sorted(list(graph.nodes), key=lambda x: mapping[x])

        node_features = list()
        for node_index in node_indices:
            node_feature = [cls.get_node_class(graph.nodes[node_index]['features'], x_dict, encode_type=encode_type)]
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)

        return node_features

    @classmethod
    def get_graph_data(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any], 
        encode_type: Literal['node', 'operator'] = 'node',
    ) -> Data:
        mapping = cls.get_mapping(sample)
        x = cls.get_x(sample, mapping, x_dict, encode_type)
        edge_index = cls.get_edge_index(sample, mapping)

        edge_label, edge_label_index = cls.get_label_edge(edge_index, x)

        graph_data = Data(x=x, edge_index=edge_index, edge_label=edge_label, edge_label_index=edge_label_index)
        return graph_data

    @classmethod
    def get_graph_data_with_links(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any],
        encode_type: Literal['node', 'operator'] = 'node',
        link_get_number: int | None = None
    ) -> dict[str, Data | list[set]]:
        
        graph_data = cls.get_graph_data(sample, x_dict, encode_type)
        if link_get_number is not None:
            link_get_number = min(link_get_number, len(graph_data.edge_label_index[0]))
            
            pos_links = graph_data.edge_label_index[:, :link_get_number//2]
            neg_loc = graph_data.edge_label_index.shape[1]//2
            neg_links = graph_data.edge_label_index[:, neg_loc:neg_loc+(link_get_number//2)]
            links = torch.cat([pos_links,neg_links],dim=1)
            link_labels = torch.cat([
                torch.ones(link_get_number//2),
                torch.zeros(link_get_number//2)
            ], dim=0)
            print("encode_type: ",encode_type)
            print("link_get_number: ", link_get_number)
            print("full: " , graph_data.edge_label_index.shape)
            print("pos: ", pos_links.shape)
            print("neg: ", neg_links.shape)
            print("links: ", links.shape)
            print("label: ", link_labels.shape)
        else:
            links = graph_data.edge_label_index
            link_labels = graph_data.edge_label

        return dict(
            graph_data = graph_data,
            links = links,
            link_labels = link_labels
        )