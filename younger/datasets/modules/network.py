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

from younger.datasets.utils.io import load_json, save_json, load_pickle, save_pickle, create_dir


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

        self._graph_filename = 'detailed_graph.pkl'
        self._graph = graph

        self._simplified_graph_dirname = 'simplified_graph'
        self._simplified_graph: list[networkx.DiGraph] = self.__class__.simplify(graph)

        ir_version: int = graph.graph.get('ir_version', None)
        opset_import: list[dict[str, str | int]] = graph.graph.get('opset_import', None)
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
    def simplified_graph(self) -> list[networkx.DiGraph]:
        return self._simplified_graph

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
        simplified_graph_dirpath = network_dirpath.joinpath(self._simplified_graph_dirname)
        self._load_simplified_graph(simplified_graph_dirpath)
        return 

    def save(self, network_dirpath: pathlib.Path) -> None:
        assert not network_dirpath.is_dir(), f'\"Network\" can not be saved into the specified directory \"{network_dirpath.absolute()}\".'
        info_filepath = network_dirpath.joinpath(self._info_filename)
        self._save_info(info_filepath)
        graph_filepath = network_dirpath.joinpath(self._graph_filename)
        self._save_graph(graph_filepath)
        simplified_graph_dirpath = network_dirpath.joinpath(self._simplified_graph_dirname)
        self._save_simplified_graph(simplified_graph_dirpath)
        return

    def _load_graph(self, graph_filepath: pathlib.Path) -> None:
        assert graph_filepath.is_file(), f'There is no \"graph\" can be loaded from the specified path \"{graph_filepath.absolute()}\".'
        self._graph = load_pickle(graph_filepath)
        return

    def _save_graph(self, graph_filepath: pathlib.Path) -> None:
        assert not graph_filepath.is_file(), f'\"graph\" can not be saved into the specified path \"{graph_filepath.absolute()}\".'
        save_pickle(self._graph, graph_filepath)
        return

    def _load_simplified_graph(self, simplified_graph_dirpath: pathlib.Path) -> None:
        assert simplified_graph_dirpath.is_dir(), f'There is no \"simplified_graph\" can be loaded from the specified directory \"{simplified_graph_dirpath.absolute()}\".'
        simplified_graph_hash_filepath = simplified_graph_dirpath.joinpath('hash.pkl')
        hash_strings: list[str] = load_pickle(simplified_graph_hash_filepath)
        self._simplified_graph = list()
        for hash_string in hash_strings:
            simplified_graph_filepath = simplified_graph_dirpath.joinpath(f'{hash_string}.gml')
            self._simplified_graph.append(networkx.read_gml(simplified_graph_filepath, destringizer=self.__class__.destringizer))
        return

    def _save_simplified_graph(self, simplified_graph_dirpath: pathlib.Path) -> None:
        assert not simplified_graph_dirpath.is_dir(), f'\"simplified_graph\" can not be saved into the specified directory \"{simplified_graph_dirpath.absolute()}\".'
        create_dir(simplified_graph_dirpath)
        simplified_graph_hash_filepath = simplified_graph_dirpath.joinpath('hash.pkl')
        hash_strings: list[str] = list()
        for simplified_graph in self._simplified_graph:
            hash_string = networkx.weisfeiler_lehman_graph_hash(simplified_graph)
            simplified_graph_filepath = simplified_graph_dirpath.joinpath(f'{hash_string}.gml')
            networkx.write_gml(simplified_graph, simplified_graph_filepath, stringizer=self.__class__.stringizer)
            hash_strings.append(hash_string)
        save_pickle(hash_strings, simplified_graph_hash_filepath)
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
    def simplify(cls, graph: networkx.DiGraph) -> list[networkx.DiGraph]:
        simplified_graph: list[networkx.DiGraph] = list()

        def flat_g(g: networkx.DiGraph) -> networkx.DiGraph:
            flatted_g = networkx.DiGraph()

            for node_index, node_attrs in g.nodes.items():
                g_index = None
                if node_attrs['attributes'] is not None:
                    for op_attr_name, op_attr_dict in node_attrs['attributes'].items():
                        if op_attr_dict['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPH:
                            flatted_sub_g = flat_g(op_attr_dict['value'])
                            g_index = len(simplified_graph)
                            simplified_graph.append(flatted_sub_g)
                        if op_attr_dict['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPHS:
                            g_index = list()
                            for sub_g in op_attr_dict['value']:
                                flatted_sub_g = flat_g(sub_g)
                                g_index.append(len(simplified_graph))
                                simplified_graph.append(flatted_sub_g)
                attributes = dict(type=node_attrs['type'])
                if g_index is not None:
                    attributes.update(subgraph_index=g_index)
                flatted_g.add_node(node_index, **attributes)

            for (edge_tail_index, edge_head_index), edge_attrs in g.edges.items():
                flatted_g.add_edge(edge_tail_index, edge_head_index, connection=edge_attrs['connection'])

            return flatted_g

        flatted_graph = flat_g(graph)
        simplified_graph.append(flatted_graph)

        return simplified_graph