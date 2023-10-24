#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-09-14 12:22
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import json
import pathlib
import networkx

from google.protobuf import json_format
from typing import Set, List, Dict, Tuple, Generator

from youngbench.dataset.modules.node import Node

from youngbench.dataset.utils import hash_strings, read_json, write_json


class Network(object):
    def __init__(self, nn_graph: networkx.DiGraph = networkx.DiGraph(), nn_nodes: Dict[str, Node] = dict(), nn_size: int = int()) -> None:
        assert len(nn_graph) == nn_size
        assert len(nn_nodes) == nn_size

        self._nn_graph_filename = 'nn_graph.json'
        self._nn_graph = nn_graph

        self._nn_nodes_filename = 'nn_nodes.json'
        self._nn_nodes = nn_nodes

        self._nn_size = nn_size

    @property
    def nn_graph(self) -> networkx.DiGraph:
        return self._nn_graph

    @property
    def nn_nodes(self) -> Dict[str, Node]:
        return self._nn_nodes

    @property
    def node_ids(self) -> List[int]:
        node_ids = [int(node_id) for node_id in self.nn_graph.nodes]
        return node_ids

    def get_node(self, node_id: int) -> Dict:
        return self.nn_graph.nodes[str(node_id)]

    @property
    def edge_ids(self) -> List[Tuple[int, int]]:
        edge_ids = [(int(u_node_id), int(v_node_id)) for u_node_id, v_node_id in self.nn_graph.edges]
        return edge_ids

    def get_edge(self, u_node_id: int, v_node_id: int) -> Dict:
        return self.nn_graph.edges[(str(u_node_id), str(v_node_id))]

    @property
    def input_ids(self) -> List[int]:
        input_ids = list()
        for node_id, in_degree in self.nn_graph.in_degree():
            if in_degree == 0:
                input_ids.append(node_id)
        return input_ids

    @property
    def output_ids(self) -> List[int]:
        output_ids = list()
        for node_id, out_degree in self.nn_graph.out_degree():
            if out_degree == 0:
                output_ids.append(node_id)
        return output_ids

    @property
    def bfs_layers(self) -> Generator[List[int], None, None]:
        for bfs_layer in networkx.bfs_layers(self.nn_graph, self.input_ids):
            bfs_layer = [int(node_id) for node_id in bfs_layer]
            yield bfs_layer

    @property
    def op_types(self) -> Set:
        op_types = set()
        for node_id in self.node_ids:
            node = self.get_node(node_id)
            op_types.add((node['type'], node['domain']))

        return op_types

    @property
    def identifier(self) -> str:
        layer_strings = list()
        for index, bfs_layer in enumerate(self.bfs_layers):
            bfs_layer = [self.nn_graph.nodes[str(node_id)] for node_id in bfs_layer]
            bfs_layer = sorted(bfs_layer, key=lambda x: (x['type'], x['domain'], x['in_number'], x['out_number']))
            layer_string = f'{index}'
            for node in bfs_layer:
                layer_string = layer_string + f'|type={node["type"]}'
                layer_string = layer_string + f'|domain={node["domain"]}'
                layer_string = layer_string + f'|in_number={node["in_number"]}'
                layer_string = layer_string + f'|out_number={node["out_number"]}'
            layer_strings.append(layer_string)
        return hash_strings(layer_strings)

    @property
    def dict(self) -> Dict:
        return dict(
            nn_size = self._nn_size,
        )

    def __eq__(self, network: 'Network') -> bool:
        if len(self.node_ids) != len(network.node_ids):
            return False
        if len(self.edge_ids) != len(network.edge_ids):
            return False
        if self.op_types != network.op_types:
            return False
        if self.identifier != network.identifier:
            return False
        return True

    def __len__(self) -> int:
        return self._nn_size

    def load(self, network_filepath: pathlib.Path) -> None:
        assert network_filepath.is_file(), f'There is no \"Network\" can be loaded from the specified directory \"{network_filepath.absolute()}\".'
        self._nn_size = read_json(network_filepath)
        network_dirpath = network_filepath.parent
        nn_garph_filepath = network_dirpath.joinpath(self._nn_graph_filename)
        self._load_nn_graph(nn_garph_filepath)
        nn_nodes_filepath = network_dirpath.joinpath(self._nn_nodes_filename)
        self._load_nn_nodes(nn_nodes_filepath)
        return

    def save(self, network_filepath: pathlib.Path) -> None:
        assert not network_filepath.is_file(), f'\"Network\" can not be saved into the specified directory \"{network_filepath.absolute()}\".'
        write_json(self._nn_size, network_filepath)
        network_dirpath = network_filepath.parent
        nn_garph_filepath = network_dirpath.joinpath(self._nn_graph_filename)
        self._save_nn_graph(nn_garph_filepath)
        nn_nodes_filepath = network_dirpath.joinpath(self._nn_nodes_filename)
        self._save_nn_nodes(nn_nodes_filepath)
        return

    def _load_nn_graph(self, nn_graph_filepath: pathlib.Path) -> None:
        assert nn_graph_filepath.is_file(), f'There is no \"nn_graph\" can be loaded from the specified path \"{nn_graph_filepath.absolute()}\".'
        self._nn_graph = networkx.read_graphml(nn_graph_filepath)
        return

    def _save_nn_graph(self, nn_graph_filepath: pathlib.Path) -> None:
        assert not nn_graph_filepath.is_file(), f'\"nn_graph\" can not be saved into the specified path \"{nn_graph_filepath.absolute()}\".'
        networkx.write_graphml(self._nn_graph, nn_graph_filepath, encoding='utf-8')
        return

    def _load_nn_nodes(self, nn_nodes_filepath: pathlib.Path) -> None:
        assert nn_nodes_filepath.is_file(), f'There is no \"nn_nodes\" can be loaded from the specified path \"{nn_nodes_filepath.absolute()}\".'
        nn_nodes = read_json(nn_nodes_filepath)
        self.nn_nodes = dict()
        for nid, nn_node in nn_nodes.items():
            nn_node = Node(**nn_node)
            self.nn_nodes[nid] = nn_node
        return

    def _save_nn_nodes(self, nn_nodes_filepath: pathlib.Path) -> None:
        assert not nn_nodes_filepath.is_file(), f'\"nn_nodes\" can not be saved into the specified path \"{nn_nodes_filepath.absolute()}\".'
        nn_nodes = dict()
        for nid, nn_node in self.nn_nodes.items():
            nn_node = nn_node.dict
            nn_nodes[nid] = nn_node
        write_json(nn_nodes, nn_nodes_filepath)
        return