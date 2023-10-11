#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-06 10:07
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import onnx
import pathlib
import networkx

from typing import List, Dict, Tuple, Generator

from youngbench.dataset.utils import hash_strings

from youngbench.constants import ONNX


class Prototype(object):
    fields = dict(
        op_type = 'op_type',
        op_domain = 'domain',
        i_num = 'input',
        o_num = 'output'
    )
    def __init__(self, neural_network: networkx.DiGraph = networkx.DiGraph()) -> None:
        assert networkx.is_directed_acyclic_graph(neural_network), f'Parameter neural_network must be a object with type of \"networkx.DiGraph\" instead of \"{type(neural_network)}\"!'
        self._neural_network = neural_network

    def __len__(self):
        return len(self.neural_network)

    def __eq__(self, prototype: 'Prototype') -> bool:
        if len(self.node_ids) != len(prototype.node_ids):
            return False

        if len(self.edge_ids) != len(prototype.edge_ids):
            return False

        if self.op_types != prototype.op_types:
            return False

        if self.identifier != prototype.identifier:
            return False

        return True

    @property
    def op_types(self) -> set:
        op_types = set()
        for node_id in self.node_ids:
            node = self.get_node(node_id)
            op_types.add((node['op_type'], node['op_domain']))

        return op_types

    @property
    def neural_network(self) -> networkx.DiGraph:
        return self._neural_network

    @property
    def node_ids(self) -> List[int]:
        node_ids = [int(node_id) for node_id in self.neural_network.nodes]
        return node_ids

    @property
    def edge_ids(self) -> List[Tuple[int, int]]:
        edge_ids = [(int(u_node_id), int(v_node_id)) for u_node_id, v_node_id in self.neural_network.edges]
        return edge_ids

    @property
    def identifier(self) -> str:
        layer_strings = list()
        for index, bfs_layer in enumerate(self.bfs_layers):
            bfs_layer = [self.neural_network.nodes[str(node_id)] for node_id in bfs_layer]
            bfs_layer = sorted(bfs_layer, key=lambda x: (x['op_type'], x['op_domain'], x['i_num'], x['o_num']))

            layer_string = f'{index}'
            for node in bfs_layer:
                layer_string = layer_string + f'|op_type={node["op_type"]}'
                layer_string = layer_string + f'|op_domain={node["op_domain"]}'
                layer_string = layer_string + f'|i_num={node["i_num"]}'
                layer_string = layer_string + f'|o_num={node["o_num"]}'

            layer_strings.append(layer_string)

        return hash_strings(layer_strings)

    @property
    def input_ids(self) -> List[int]:
        input_ids = list()
        for node_id, in_degree in self.neural_network.in_degree():
            if in_degree == 0:
                input_ids.append(node_id)

        return input_ids

    @property
    def output_ids(self) -> List[int]:
        output_ids = list()
        for node_id, out_degree in self.neural_network.out_degree():
            if out_degree == 0:
                output_ids.append(node_id)

        return output_ids

    @property
    def bfs_layers(self) -> Generator[List[int], None, None]:
        for bfs_layer in networkx.bfs_layers(self.neural_network, self.input_ids):
            bfs_layer = [int(node_id) for node_id in bfs_layer]
            yield bfs_layer

    def get_node(self, node_id: int) -> Dict:
        return self.neural_network.nodes[str(node_id)]

    def get_edge(self, u_node_id: int, v_node_id: int) -> Dict:
        return self.neural_network.edges[(str(u_node_id), str(v_node_id))]

    def from_onnx_model(self, onnx_model: onnx.ModelProto) -> None:
        neural_network = networkx.DiGraph()
        tn2id = dict() # tensor name -> node id
        id2nd = dict() # node id -> node
        for index, node in enumerate(onnx_model.graph.node):
            id2nd[str(index)] = node
            for input in node.input:
                tn2id[input] = str(index)

        for index, node in id2nd.items():
            op_type = node.op_type
            op_domain = node.domain or str(ONNX.OP_DOMAIN)
            i_num = len(node.input)
            o_num = len(node.output)
            neural_network.add_node(str(index), op_type=op_type, op_domain=op_domain, i_num=i_num, o_num=o_num)

            for output in node.output:
                next_node_id = tn2id.get(output, None)
                if next_node_id is None:
                    continue
                else:
                    neural_network.add_edge(str(index), next_node_id)

        assert networkx.is_directed_acyclic_graph(neural_network), f'The \"Prototype\" converted from the \"ONNX Model\" (onnx_model) is not a Directed Acyclic Graph.'

        self._neural_network = neural_network

    def load(self, prototype_filepath: pathlib) -> None:
        self._neural_network = networkx.read_graphml(prototype_filepath)

        return

    def save(self, prototype_filepath: pathlib) -> None:
        networkx.write_graphml(self._neural_network, prototype_filepath, encoding='utf-8')

        return