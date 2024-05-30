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
import torch
import random
import networkx
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import is_sparse

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.modules import Network
from younger.datasets.utils.constants import YoungerDatasetAddress


class NodeData(Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'mask_x_label': # op type
            return -1
        if key == 'mask_x_position': # index
            return -1

        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'mask_x_label':
            return 0
        if key == 'mask_x_position':
            return self.num_nodes

        if 'batch' in key and isinstance(value, torch.Tensor):
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


class NodeDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,

        node_dict_size: int | None = None,
        block_get_type: Literal['louvain', 'label'] = 'louvain',
        block_get_number: int | None = None,

        seed: int | None = None,
        worker_number: int = 4,
    ):
        assert block_get_type in {'louvain', 'label'}

        self.block_get_type = block_get_type
        self.block_get_number = block_get_number
        self.seed = seed
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size)

        self._block_locations: list[tuple[int, int]] = list()

        self._block_locations_filename = 'block_locations.pt'

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        try:
            _block_locations_filepath = os.path.join(self.processed_dir, self._block_locations_filename)
            self._block_locations = torch.load(_block_locations_filepath)
        except FileNotFoundError as error:
            print(f'There is no block locations file \'{self._block_locations_filename}\', please remove the processed data directory: {self.processed_dir} and re-run the command.')
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
        return len(self._block_locations)

    def get(self, index: int) -> NodeData:
        sample_index, block_index = self._block_locations[index]
        block_data_list: dict[str, NodeData | list[set]] = torch.load(os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        # print("block_data_list: ", block_data_list)
        # print("len(block_data_list)",len(block_data_list))
        block_data = block_data_list[block_index]
        return block_data

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for sample_index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{sample_index}.pkl'))

    def process(self):
        unordered_block_records = list()
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, block_number in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    unordered_block_records.append((sample_index, block_number))
                    progress_bar.update()
        sorted_block_records = sorted(unordered_block_records, key=lambda x: x[0])
        for sample_index, block_number in sorted_block_records:
            block_location = [(sample_index, block_index) for block_index in range(block_number)]
            self._block_locations.extend(block_location)
        _block_locations_filepath = os.path.join(self.processed_dir, self._block_locations_filename)
        torch.save(self._block_locations, _block_locations_filepath)

    def process_sample(self, sample_index: int) -> tuple[int, int]:
        sample_filepath = os.path.join(self.raw_dir, f'sample-{sample_index}.pkl')
        sample: networkx.DiGraph = load_pickle(sample_filepath)
        block_data_list = self.__class__.get_block_data_list(sample, self.x_dict, self.block_get_type, self.block_get_number, self.seed)
        if self.pre_filter is not None and not self.pre_filter(block_data_list):
            return
        if self.pre_transform is not None:
            block_data_list = self.pre_transform(block_data_list)
        torch.save(block_data_list, os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        return (sample_index, len(block_data_list))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

        meta['metric_name'] = loaded_meta['metric_name']
        meta['all_nodes'] = loaded_meta['all_nodes']

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
    def get_x_dict(cls, meta: dict[str, Any], node_dict_size: int | None = None) -> dict[str, list[str] | dict[str, int]]:
        all_nodes = [(node_id, node_count) for node_id, node_count in meta['all_nodes']['onnx'].items()] + [(node_id, node_count) for node_id, node_count in meta['all_nodes']['others'].items()]
        all_nodes = sorted(all_nodes, key=lambda x: x[1])

        node_dict_size = len(all_nodes) if node_dict_size is None else node_dict_size

        x_dict = dict()
        x_dict['i2n'] = ['__UNK__'] + ['__MASK__'] + [node_id for node_id, node_count in all_nodes[:node_dict_size]]

        x_dict['n2i'] = {node_id: index for index, node_id in enumerate(x_dict['i2n'])}
        return x_dict

    @classmethod
    def get_edge_index(cls, graph: networkx.DiGraph, mapping: dict[str, int]) -> torch.Tensor:
        edge_index = torch.empty((2, graph.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(graph.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

    @classmethod
    def get_node_class(cls, node_features: dict, x_dict: dict[str, Any]) -> int:
        node_identifier: str = Network.get_node_identifier_from_features(node_features)
        node_class = x_dict['n2i'].get(node_identifier, x_dict['n2i']['__UNK__'])
        return node_class

    @classmethod
    def get_x(cls, graph: networkx.DiGraph, mapping: dict[str, int], boundary: set[str], x_dict: dict[str, Any]) -> torch.Tensor:
        node_indices = sorted(list(graph.nodes), key=lambda x: mapping[x])
        node_features = list()
        for node_index in node_indices:
            if node_index in boundary:
                node_feature = [x_dict['n2i']['__MASK__']]
            else:
                node_feature = [cls.get_node_class(graph.nodes[node_index]['features'], x_dict)]
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)

        return node_features

    @classmethod
    def get_block_data(
        cls,
        sample: networkx.DiGraph,
        community: set[str],
        boundary: set[str],
        x_dict: dict[str, Any]
    ) -> NodeData:
        subgraph: networkx.DiGraph = networkx.subgraph(sample, community | boundary).copy()
        mapping = cls.get_mapping(subgraph) # dict(zip(sorted(G.nodes()), range(G.number_of_nodes())))
                                            # e.g. >>> print(node_mapping)
                                            # {2: 0, 3: 1, 5: 2, 10: 3}
                                            # >>> print(G.nodes())
                                            # [10, 2, 3, 5]
                                            # >>> print(sorted(G.nodes()))
                                            # [2, 3, 5, 10]

        x = cls.get_x(subgraph, mapping, boundary, x_dict)
        edge_index = cls.get_edge_index(subgraph, mapping)
        mask_x_label = torch.tensor([cls.get_node_class(subgraph.nodes[node]['features'], x_dict) for node in sorted(list(boundary))], dtype=torch.long)
        mask_x_position = torch.tensor([mapping[node] for node in sorted(list(boundary))], dtype=torch.long)

        block_data = NodeData(x=x, edge_index=edge_index, mask_x_label=mask_x_label, mask_x_position=mask_x_position)
        return block_data

    @classmethod
    def get_all_community_with_boundary(cls, sample: networkx.DiGraph, block_get_type: Literal['louvain', 'label'], block_get_number: int | None = None, **kwargs) -> list[tuple[set, set]]:
        if block_get_type == 'louvain':
            seed = kwargs.get('seed', None)
            resolution = kwargs.get('resolution', 1)
            communities: list[set] = list(networkx.community.louvain_communities(sample, resolution=resolution, seed=seed))

        if block_get_type == 'label':
            seed = kwargs.get('seed', None)
            communities: list[set] = list(networkx.community.asyn_lpa_communities(sample, seed=seed))
        if block_get_number is None:
            pass
        else:
            communities: list[set] = random.sample(communities, min(block_get_number, len(communities)))
        all_community_with_boundary = list()
        for community in communities:
            boundary = networkx.node_boundary(sample, community)
            if len(boundary) == 0:
                continue
            all_community_with_boundary.append((community, boundary))
        return all_community_with_boundary

    @classmethod
    def get_block_data_list(
        cls,
        sample: networkx.DiGraph,
        x_dict: dict[str, Any], 
        block_get_type: Literal['louvain', 'label'],
        block_get_number: int | None = None,
        seed: int | None = None,
    ) -> dict[str, NodeData | list[set]]:

        block_data_list = list()
        all_community_with_boundary = cls.get_all_community_with_boundary(sample, block_get_type, block_get_number=block_get_number, seed=seed)
        for (community, boundary) in all_community_with_boundary:
            block_data = cls.get_block_data(sample, community, boundary, x_dict)
            block_data_list.append(block_data)

        return block_data_list