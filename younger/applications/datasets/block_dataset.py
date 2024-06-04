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
import random
import networkx
import multiprocessing

from typing import Any, Callable, Literal, Mapping
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.modules import Network
from younger.datasets.utils.constants import YoungerDatasetAddress


class BlockDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,

        node_dict_size: int | None = None,
        operator_dict_size: int | None = None,
        encode_type: Literal['node', 'operator'] = 'node',

        block_get_type: Literal['louvain', 'label', 'asyn_fluidc', 'greedy_modularity_communities'] = 'greedy_modularity_communities',

        block_get_number: int | None = None,

        seed: int | None = None,
        worker_number: int = 4,
    ):
        assert block_get_type in {'louvain', 'label', 'asyn_fluidc', 'greedy_modularity_communities'}
        assert encode_type in {'node', 'operator'}

        self.encode_type = encode_type
        self.block_get_type = block_get_type
        self.block_get_number = block_get_number
        self.seed = seed
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size, operator_dict_size)

        self._community_locations: list[tuple[int, int]] = list()

        self._community_locations_filename = 'community_locations.pt'

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        try:
            _community_locations_filepath = os.path.join(self.processed_dir, self._community_locations_filename)
            self._community_locations = torch.load(_community_locations_filepath)
        except FileNotFoundError as error:
            print(f'There is no community locations file \'{self._community_locations_filename}\', please remove the processed data directory: {self.processed_dir} and re-run the command.')
            raise error

    @property
    def raw_dir(self) -> str:
        name = f'younger_raw_{self.encode_type}_bk'

        return os.path.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed_{self.encode_type}_bk'

        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return [f'sample-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'sample-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return len(self._community_locations)

    def get(self, index: int) -> Data:
        sample_index, community_index = self._community_locations[index]
        graph_data_with_communities: dict[str, Data | list[set]] = torch.load(os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))

        graph_data: Data = graph_data_with_communities['graph_data']
        community, labels = graph_data_with_communities['communities'][community_index]
        block_mask = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
        block_mask[list(community)] = 1
        block_labels = torch.tensor(list(labels), dtype=torch.float)
        block_data: Data = Data(x=graph_data.x, edge_index=graph_data.edge_index, y=graph_data.y, block_mask=block_mask, block_labels=block_labels)
        return block_data

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for sample_index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{sample_index}.pkl'))

    def process(self):
        unordered_community_records = list()
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, community_number in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    print("sample_index / community_number : ",sample_index, " / ",community_number)
                    unordered_community_records.append((sample_index, community_number))
                    progress_bar.update()
        sorted_community_records = sorted(unordered_community_records, key=lambda x: x[0])
        for sample_index, community_number in sorted_community_records:
            community_location = [(sample_index, community_index) for community_index in range(community_number)]
            self._community_locations.extend(community_location)
        _community_locations_filepath = os.path.join(self.processed_dir, self._community_locations_filename)
        torch.save(self._community_locations, _community_locations_filepath)

    def process_sample(self, sample_index: int) -> tuple[int, int]:
        sample_filepath = os.path.join(self.raw_dir, f'sample-{sample_index}.pkl')
        sample: networkx.DiGraph = load_pickle(sample_filepath)

        graph_data_with_communities = self.__class__.get_graph_data_with_communities(sample, self.x_dict, self.encode_type, self.block_get_type, self.block_get_number, self.seed)
        if self.pre_filter is not None and not self.pre_filter(graph_data_with_communities):
            return

        if self.pre_transform is not None:
            graph_data_with_communities = self.pre_transform(graph_data_with_communities)
        torch.save(graph_data_with_communities, os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        return (sample_index, len(graph_data_with_communities['communities']))

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
    def get_mapping(cls, sample: networkx.DiGraph):
        return dict(zip(sorted(sample.nodes()), range(sample.number_of_nodes())))

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
    def get_edge_index(cls, sample: networkx.DiGraph) -> torch.Tensor:
        mapping = cls.get_mapping(sample)
        edge_index = torch.empty((2, sample.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(sample.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

    @classmethod
    def get_node_class(cls, node_features: dict, x_dict: dict[str, Any], encode_type: Literal['node', 'operator'] = 'node') -> list:
        node_identifier: str = Network.get_node_identifier_from_features(node_features)

        if encode_type == 'node':
            node_identifier: str = Network.get_node_identifier_from_features(node_features, mode='full')
            node_class = x_dict['n2i'].get(node_identifier, x_dict['n2i']['__UNK__'])

        if encode_type == 'operator':
            node_identifier: str = Network.get_node_identifier_from_features(node_features, mode='type')
            node_class = x_dict['o2i'].get(node_identifier, x_dict['o2i']['__UNK__'])

        return node_class

    @classmethod
    def get_x(cls, graph: networkx.DiGraph, mapping: dict[str, int],  x_dict: dict[str, Any], encode_type: Literal['node', 'operator'] = 'node') -> torch.Tensor:
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
        edge_index = cls.get_edge_index(sample)

        graph_data = Data(x=x, edge_index=edge_index)
        return graph_data

    @classmethod
    def get_communities(cls, sample: networkx.DiGraph, block_get_type: Literal['louvain', 'label', 'asyn_fluidc', 'greedy_modularity_communities'], block_get_number: int | None = None, **kwargs) -> list[tuple[set, tuple]]:
        mapping = cls.get_mapping(sample)
        if block_get_type == 'greedy_modularity_communities':
            seed = kwargs.get('seed', None)
            resolution = kwargs.get('resolution', 1)
            cutoff = kwargs.get('cutoff', 1)
            communities = list(networkx.community.greedy_modularity_communities(sample, resolution=resolution, cutoff=cutoff))

        if block_get_type == 'asyn_fluidc':
            seed = kwargs.get('seed', None)
            k = len(sample.nodes())//20 if len(sample.nodes()) >= 20 else 1
            communities = list(networkx.community.asyn_fluidc(sample, k=k, seed=seed))

        if block_get_type == 'louvain':
            seed = kwargs.get('seed', None)
            resolution = kwargs.get('resolution', 1)
            communities = list(networkx.community.louvain_communities(sample, resolution=resolution, seed=seed))

        if block_get_type == 'label':
            seed = kwargs.get('seed', None)
            communities = list(networkx.community.asyn_lpa_communities(sample, resolution=resolution, seed=seed))

        if block_get_number is None:
            pass
        else:
            communities = random.sample(communities, min(block_get_number, len(communities)))
        community_with_labels = list()
        for community in communities:
            if len(community) == 0:
                continue
            block: networkx.DiGraph = networkx.subgraph(sample, community).copy()

            # Labels
            density = networkx.density(block)
            coreness = numpy.average(list(networkx.core_number(block).values()))
            if len(community) == sample.number_of_nodes():
                cut_ratio = 0
            else:
                cut_ratio = len(list(networkx.edge_boundary(sample, block.nodes))) / (block.number_of_nodes() * (sample.number_of_nodes() - block.number_of_nodes()))

            community = set([mapping[node] for node in community])
            community_with_labels.append((community, (density, coreness, cut_ratio)))

        # print('block_get_type: ', block_get_type)
        # print('block_get_number, len(communities): ',block_get_number, len(communities))
        # print("block_get_number/len(all_community_with_boundary): ", block_get_number, len(community_with_labels))

        return community_with_labels

    @classmethod
    def get_graph_data_with_communities(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any],
        encode_type: Literal['node', 'operator'] = 'node',
        block_get_type: Literal['louvain', 'label', 'asyn_fluidc', 'greedy_modularity_communities'] = 'greedy_modularity_communities',
        block_get_number: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Data | list[set]]:

        graph_data = cls.get_graph_data(sample, x_dict, encode_type)
        communities: list[set] = list()
        for community in cls.get_communities(sample, block_get_type, block_get_number=block_get_number, seed=seed):
            communities.append(community)

        return dict(
            graph_data = graph_data,
            communities = communities
        )
