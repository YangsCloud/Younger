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
        operator_dict_size: int | None = None,
        encode_type: Literal['node', 'operator'] = 'node',

        worker_number: int = 4,
    ):
        assert encode_type in {'node', 'operator'}

        self.encode_type = encode_type
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        assert os.path.isfile(meta_filepath), f'Please Download The \'meta.json\' File Of A Specific Version Of The Younger Dataset From Official Website.'

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta, node_dict_size, operator_dict_size)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    @property
    def raw_dir(self) -> str:
        name = f'younger_raw_{self.encode_type}_np'
        return os.path.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed_{self.encode_type}_np'
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return [f'sample-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'sample-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return len(self.meta['size'])

    def get(self, index: int) -> NodeData:
        block_data = torch.load(os.path.join(self.processed_dir, f'sample-{index}.pth'))
        # print("block_data_list: ", block_data_list)
        # print("len(block_data_list)",len(block_data_list))
        return block_data

    def download(self):
        archive_filepath = os.path.join(self.root, self.meta['archive'])
        if not fs.exists(archive_filepath):
            archive_filepath = download_url(getattr(YoungerDatasetAddress, self.meta['url']), self.root, filename=self.meta['archive'])
        tar_extract(archive_filepath, self.raw_dir)

        for sample_index in range(self.meta['size']):
            assert fs.exists(os.path.join(self.raw_dir, f'sample-{sample_index}.pkl'))

    def process(self):
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, _ in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    progress_bar.update()

    def process_sample(self, sample_index: int) -> tuple[int, int]:
        sample_filepath = os.path.join(self.raw_dir, f'sample-{sample_index}.pkl')
        sample: tuple[str, networkx.DiGraph, tuple] = load_pickle(sample_filepath)
        block_data = self.__class__.get_block_data(sample, self.x_dict, self.encode_type)
        if self.pre_filter is not None and not self.pre_filter(block_data):
            return

        if self.pre_transform is not None:
            block_data = self.pre_transform(block_data)
        torch.save(block_data, os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        return (sample_index, len(block_data))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

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
    def get_edge_index(cls, graph: networkx.DiGraph, mapping: dict[str, int]) -> torch.Tensor:
        edge_index = torch.empty((2, graph.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(graph.edges()):
            edge_index[0, index] = mapping[src]
            edge_index[1, index] = mapping[dst]
        return edge_index

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
    def get_x(cls, graph: networkx.DiGraph, mapping: dict[str, int], boundary: set[str], x_dict: dict[str, Any], encode_type: Literal['node', 'operator'] = 'node') -> torch.Tensor:
        node_indices = sorted(list(graph.nodes), key=lambda x: mapping[x])

        if encode_type == 'node':
            dict_key = 'n2i'
        if encode_type == 'operator':
            dict_key = 'o2i'

        node_features = list()
        for node_index in node_indices:
            if node_index in boundary:
                node_feature = [x_dict[dict_key]['__MASK__']]
            else:
                node_feature = [cls.get_node_class(graph.nodes[node_index]['features'], x_dict, encode_type=encode_type)]
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)

        return node_features

    @classmethod
    def get_block_data(
        cls,
        sample: tuple[str, networkx.DiGraph, tuple],
        x_dict: dict[str, Any],
        encode_type: Literal['node', 'operator'] = 'node',
    ) -> NodeData:
        subgraph_hash, (subgraph, boundary) = sample
        mapping = cls.get_mapping(subgraph) # e.g.
                                            # dict(zip(sorted(G.nodes()), range(G.number_of_nodes())))
                                            # >>> print(node_mapping)
                                            # {2: 0, 3: 1, 5: 2, 10: 3}
                                            # >>> print(G.nodes())
                                            # [10, 2, 3, 5]
                                            # >>> print(sorted(G.nodes()))
                                            # [2, 3, 5, 10]

        x = cls.get_x(subgraph, mapping, boundary, x_dict, encode_type)
        edge_index = cls.get_edge_index(subgraph, mapping)
        mask_x_label = torch.tensor([cls.get_node_class(subgraph.nodes[node]['features'], x_dict, encode_type) for node in sorted(list(boundary))], dtype=torch.long)
        mask_x_position = torch.tensor([mapping[node] for node in sorted(list(boundary))], dtype=torch.long)

        block_data = NodeData(x=x, edge_index=edge_index, mask_x_label=mask_x_label, mask_x_position=mask_x_position)
        return block_data