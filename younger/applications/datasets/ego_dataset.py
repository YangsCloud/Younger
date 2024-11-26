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
import networkx
import multiprocessing

from typing import Any, Callable, Literal
from torch_geometric.io import fs
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.utils import is_sparse

from younger.commons.io import load_json, load_pickle, tar_extract

from younger.datasets.modules import Network
from younger.datasets.utils.constants import YoungerDatasetAddress


class EgoData(Data):
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


class EgoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: Literal['train', 'valid', 'test'],
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
        worker_number: int = 4,
    ):
        self.worker_number = worker_number

        meta_filepath = os.path.join(root, 'meta.json')
        if not os.path.isfile(meta_filepath):
            print(f'No \'meta.json\' File Provided, It will be downloaded From Official Cite ...')
            download_url(getattr(YoungerDatasetAddress, f'EGO_{split.upper()}_PAPER'), root, filename='meta.json')

        self.meta = self.__class__.load_meta(meta_filepath)
        self.x_dict = self.__class__.get_x_dict(self.meta)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

    @property
    def raw_dir(self) -> str:
        name = f'younger_raw_ego'
        return os.path.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'younger_processed_ego'
        return os.path.join(self.root, name)

    @property
    def raw_file_names(self):
        return [f'sample-{index}.pkl' for index in range(self.meta['size'])]

    @property
    def processed_file_names(self):
        return [f'sample-{index}.pth' for index in range(self.meta['size'])]

    def len(self) -> int:
        return self.meta['size']

    def get(self, index: int) -> EgoData:
        ego_data = torch.load(os.path.join(self.processed_dir, f'sample-{index}.pth'))
        return ego_data

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
        with multiprocessing.Pool(self.worker_number) as pool:
            with tqdm.tqdm(total=self.meta['size']) as progress_bar:
                for sample_index, _ in pool.imap_unordered(self.process_sample, range(self.meta['size'])):
                    progress_bar.update(1)

    def process_sample(self, sample_index: int) -> tuple[int, int]:
        sample_filepath = os.path.join(self.raw_dir, f'sample-{sample_index}.pkl')
        sample: tuple[str, networkx.DiGraph, tuple] = load_pickle(sample_filepath)
        ego_data = self.__class__.get_ego_data(sample, self.x_dict)
        if self.pre_filter is not None and not self.pre_filter(ego_data):
            return

        if self.pre_transform is not None:
            ego_data = self.pre_transform(ego_data)
        torch.save(ego_data, os.path.join(self.processed_dir, f'sample-{sample_index}.pth'))
        return (sample_index, len(ego_data))

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        loaded_meta: dict[str, Any] = load_json(meta_filepath)
        meta: dict[str, Any] = dict()

        meta['all_operators'] = loaded_meta['all_operators']
        meta['tail_index'] = loaded_meta['tail_index']

        meta['archive'] = loaded_meta['archive']
        meta['size'] = loaded_meta['size']
        meta['url'] = loaded_meta['url']

        return meta

    @classmethod
    def get_mapping(cls, graph: networkx.DiGraph) -> dict[str, int]:
        return dict(zip(sorted(graph.nodes()), range(graph.number_of_nodes())))

    @classmethod
    def get_x_dict(cls, meta: dict[str, Any]) -> dict[str, list[str] | dict[str, int]]:
        all_operators = [(operator_id, operator_count) for operator_id, operator_count in meta['all_operators'].items()]
        all_operators = sorted(all_operators, key=lambda x: (x[1], x[0]))[::-1]

        top_operators = [(op_id, op_count) for (op_id, op_count) in all_operators[:meta['tail_index']]]
        lt_operators = [(op_id, op_count) for (op_id, op_count) in all_operators[meta['tail_index']:]]

        x_dict = dict()
        x_dict['i2o'] = ['__UNK__'] + ['__MASK__'] + ['__TAIL__'] + [operator_id for operator_id, operator_count in top_operators]
        x_dict['o2i'] = {operator_id: index for index, operator_id in enumerate(x_dict['i2o'])}
        x_dict['lto'] = {operator_id for operator_id, operator_count in lt_operators}
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
        node_identifier: str = Network.get_node_identifier_from_features(node_features, mode='type')
        if node_identifier in x_dict['lto']:
            node_class = x_dict['o2i']['__TAIL__']
        else:
            node_class = x_dict['o2i'].get(node_identifier, x_dict['o2i']['__UNK__'])

        return node_class

    @classmethod
    def get_x(cls, graph: networkx.DiGraph, mapping: dict[str, int], focus: str, x_dict: dict[str, Any]) -> torch.Tensor:
        node_indices = sorted(list(graph.nodes), key=lambda x: mapping[x])

        dict_key = 'o2i'

        node_features = list()
        for node_index in node_indices:
            if node_index == focus:
                node_feature = [x_dict[dict_key]['__MASK__']]
            else:
                node_feature = [cls.get_node_class(graph.nodes[node_index]['features'], x_dict)]
            node_features.append(node_feature)
        node_features = torch.tensor(node_features, dtype=torch.long)

        return node_features

    @classmethod
    def get_ego_data(
        cls,
        sample: tuple[str, networkx.DiGraph, tuple],
        x_dict: dict[str, Any]
    ) -> EgoData:
        focus, ego, ego_hash= sample
        mapping = cls.get_mapping(ego) # e.g.
                                            # dict(zip(sorted(G.nodes()), range(G.number_of_nodes())))
                                            # >>> print(node_mapping)
                                            # {2: 0, 3: 1, 5: 2, 10: 3}
                                            # >>> print(G.nodes())
                                            # [10, 2, 3, 5]
                                            # >>> print(sorted(G.nodes()))
                                            # [2, 3, 5, 10]

        x = cls.get_x(ego, mapping, focus, x_dict)
        edge_index = cls.get_edge_index(ego, mapping)
        mask_x_label = torch.tensor([cls.get_node_class(ego.nodes[node_index]['features'], x_dict) for node_index in ego.nodes() if node_index != focus], dtype=torch.long)
        mask_x_position = torch.tensor([mapping[focus]], dtype=torch.long)

        ego_data = EgoData(x=x, edge_index=edge_index, mask_x_label=mask_x_label, mask_x_position=mask_x_position)
        return ego_data