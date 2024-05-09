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


import onnx
import pathlib
import networkx

from typing import Any

from younger.commons.io import load_json, save_json, load_pickle, save_pickle, create_dir


class Network(object):
    def __init__(
            self,
            graph: networkx.DiGraph | None = None,
    ) -> None:
        graph = graph or networkx.DiGraph()
        # node_attributes:
        #   1. type='operator':
        #     name
        #     doc_string
        #     operator
        #     operands
        #     results
        #     attributes
        #   2. type='input':
        #     graph_inputs
        #   3. type='output':
        #     graph_outputs
        #   4. type='constant':
        #     graph_constants

        # edge_attributes:
        #   head_index
        #   tail_index
        #   emit_index
        #   trap_index
        #   dataflow
        #   default_value

        self._graph_filename = 'graph.pkl'
        self._graph = graph

        ir_version: int = graph.graph.get('ir_version', None)
        opset_import: dict[str, int] = graph.graph.get('opset_import', None)
        producer_name: str | None = graph.graph.get('producer_name', None)
        producer_version: str | None = graph.graph.get('producer_version', None)
        domain: str | None = graph.graph.get('domain', None)
        model_version: int | None = graph.graph.get('model_version', None)
        doc_string: str | None = graph.graph.get('doc_string', None)
        metadata_props: list[dict[str, str]] | None = graph.graph.get('metadata_props', None)

        self._info_filename = 'info.json'
        self._info = dict(
            ir_version = ir_version,
            opset_import = opset_import,
            producer_name = producer_name,
            producer_version = producer_version,
            domain = domain,
            model_version = model_version,
            doc_string = doc_string,
            metadata_props = metadata_props,
        )

    @property
    def graph(self) -> networkx.DiGraph:
        return self._graph

    @property
    def info(self) -> dict:
        return self._info

    @property
    def hash(self) -> str:
        return networkx.weisfeiler_lehman_graph_hash(self._graph, edge_attr='connection', node_attr='operator', iterations=6, digest_size=16)

    def load(self, network_dirpath: pathlib.Path) -> None:
        assert network_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{network_dirpath.absolute()}\".'
        info_filepath = network_dirpath.joinpath(self._info_filename)
        self._load_info(info_filepath)
        graph_filepath = network_dirpath.joinpath(self._graph_filename)
        self._load_graph(graph_filepath)
        return 

    def save(self, network_dirpath: pathlib.Path) -> None:
        assert not network_dirpath.is_dir(), f'\"Network\" can not be saved into the specified directory \"{network_dirpath.absolute()}\".'
        info_filepath = network_dirpath.joinpath(self._info_filename)
        self._save_info(info_filepath)
        graph_filepath = network_dirpath.joinpath(self._graph_filename)
        self._save_graph(graph_filepath)
        return

    def _load_graph(self, graph_filepath: pathlib.Path) -> None:
        assert graph_filepath.is_file(), f'There is no \"graph\" can be loaded from the specified path \"{graph_filepath.absolute()}\".'
        self._graph = load_pickle(graph_filepath)
        return

    def _save_graph(self, graph_filepath: pathlib.Path) -> None:
        assert not graph_filepath.is_file(), f'\"graph\" can not be saved into the specified path \"{graph_filepath.absolute()}\".'
        save_pickle(self._graph, graph_filepath)
        return

    def _load_info(self, info_filepath: pathlib.Path) -> None:
        assert info_filepath.is_file(), f'There is no \"INFO\" can be loaded from the specified path \"{info_filepath.absolute()}\".'
        self._info = load_json(info_filepath)
        return

    def _save_info(self, info_filepath: pathlib.Path) -> None:
        assert not info_filepath.is_file(), f'\"INFO\" can not be saved into the specified path \"{info_filepath.absolute()}\".'
        save_json(self._info, info_filepath)
        return

    @classmethod
    def destringizer(cls, tobe_loaded: str) -> Any:
        if tobe_loaded == '_YoungBench_None_':
            return None
        else:
            return tobe_loaded

    @classmethod
    def stringizer(cls, tobe_saved: Any) -> str:
        if tobe_saved == None:
            return '_YoungBench_None_'
        else:
            return tobe_saved

    @classmethod
    def cleanse(cls, graph: networkx.DiGraph) -> networkx.DiGraph:
        flattened_graph = cls.flatten(graph)
        cleansed_graph = networkx.DiGraph()
        cleansed_graph.add_nodes_from(flattened_graph.nodes())
        cleansed_graph.add_edges_from(flattened_graph.edges())

        for node_index, node_attrs in flattened_graph.nodes.items():
            if node_attrs['type'] != 'operator':
                cleansed_graph.remove_node(node_index)
        
        return cleansed_graph

    @classmethod
    def simplify(cls, graph: networkx.DiGraph, preserve_node_attributes: list[str] | None= None, preserve_edge_attributes: list[str] | None= None) -> networkx.DiGraph:
        preserve_node_attributes = [] if preserve_node_attributes is None else preserve_node_attributes
        preserve_edge_attributes = [] if preserve_edge_attributes is None else preserve_edge_attributes
        flattened_graph = cls.flatten(graph)
        simplified_graph = networkx.DiGraph()

        simplified_graph.add_nodes_from(flattened_graph.nodes())
        simplified_graph.add_edges_from(flattened_graph.edges())

        for preserve_node_attribute in preserve_node_attributes:
            attributes_keyed_by_nodes = networkx.get_node_attributes(flattened_graph, preserve_node_attribute)
            networkx.set_node_attributes(simplified_graph, attributes_keyed_by_nodes, preserve_node_attribute)

        for preserve_edge_attribute in preserve_edge_attributes:
            attributes_keyed_by_edges = networkx.get_edge_attributes(flattened_graph, preserve_edge_attribute)
            networkx.set_edge_attributes(simplified_graph, attributes_keyed_by_edges, preserve_edge_attribute)

        return simplified_graph

    @classmethod
    def flatten(cls, graph: networkx.DiGraph) -> networkx.DiGraph:
        # TODO: All Sub-Graphs Should Be Flattened
        fathers = list()
        sub_graphs = list()
        for node_index, node_attrs in graph.nodes.items():
            if node_attrs['attributes'] is not None:
                for op_attr_name, op_attr_dict in node_attrs['attributes'].items():
                    sub_l = len(sub_graphs)
                    if op_attr_dict['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPH:
                        sub_graphs.append(op_attr_dict['value'])
                    if op_attr_dict['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPHS:
                        sub_graphs.extend(op_attr_dict['value'])
                    sub_r = len(sub_graphs)
                    fathers.extend([node_index for _ in range(sub_l, sub_r)])

        assert len(fathers) == len(sub_graphs)

        flattened_graph = networkx.DiGraph()
        flattened_graph.update(graph)

        for father, sub_graph in zip(fathers, sub_graphs):
            start_node_index = str(flattened_graph.number_of_nodes())
            flattened_sub_graph = cls.flatten(sub_graph)
            flattened_graph.add_edge(father, start_node_index)
            flattened_graph.update(
                networkx.relabel_nodes(
                    flattened_sub_graph,
                    {sub_node_index: str(int(start_node_index) + int(sub_node_index)) for sub_node_index in flattened_sub_graph.nodes}
                )
            )

        return flattened_graph