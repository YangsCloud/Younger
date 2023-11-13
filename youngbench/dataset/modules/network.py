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
import json
import pathlib
import networkx
import semantic_version

from typing import Set, List, Dict, Tuple, Optional, Generator
from onnx.shape_inference import infer_shapes
from onnx.helper import make_model
from google.protobuf import json_format

from youngbench.dataset.modules.meta import Meta
from youngbench.dataset.modules.stamp import Stamp
from youngbench.dataset.modules.node import Node
from youngbench.dataset.modules.model import Model

from youngbench.dataset.utils.io import hash_strings, read_json, write_json
from youngbench.logging import logger
from youngbench.constants import ONNXOperatorDomain, ONNXAttributeType


class Prototype(object):
    def __init__(
            self,
            nn_graph: Optional[networkx.DiGraph] = None,
            nn_size: int = 0,
            is_sub: bool = False,
            is_fnc: bool = False
    ) -> None:
        nn_graph = nn_graph or networkx.DiGraph()

        assert len(nn_graph) == nn_size

        self._nn_graph_filename = 'nn_graph.json'
        self._nn_graph = nn_graph

        self._nn_size = nn_size
        self._is_sub = is_sub # Whether Subgraph
        self._is_fnc = is_fnc # Whether Function

    @property
    def nn_graph(self) -> networkx.DiGraph:
        return self._nn_graph

    @property
    def is_sub(self) -> bool:
        return self._is_sub

    @property
    def is_fnc(self) -> bool:
        return self._is_fnc

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
            if not node['is_custom']:
                op_types.add((node['type'], node['domain']))
        return op_types

    @property
    def identifier(self) -> str:
        layer_strings = list()
        for index, bfs_layer in enumerate(self.bfs_layers):
            bfs_layer = [self.nn_graph.nodes[str(node_id)] for node_id in bfs_layer]
            bfs_layer = sorted(bfs_layer, key=lambda x: (x['is_first'], x['is_last'], x['is_custom'], x['has_subgraph'], x['type'], x['domain'], x['in_number'], x['out_number']))
            layer_string = f'[{index}:'
            for i, node in enumerate(bfs_layer):
                layer_string = layer_string + f'={i}'
                if node['is_custom']:
                    layer_string = layer_string + f'|in_number={node["in_number"]}'
                    layer_string = layer_string + f'|out_number={node["out_number"]}'
                else:
                    layer_string = layer_string + f'|type={node["type"]}'
                    layer_string = layer_string + f'|domain={node["domain"]}'
                    layer_string = layer_string + f'|in_number={node["in_number"]}'
                    layer_string = layer_string + f'|out_number={node["out_number"]}'
                layer_string = layer_string + f'|='
            layer_string = layer_string + ']'
            layer_strings.append(layer_string)
        return hash_strings(layer_strings)

    def dict2meta(self, dict) -> None:
        self._nn_size = dict['nn_size']
        self._is_sub = dict['is_sub']
        self._is_fnc = dict['is_fnc']

    def meta2dict(self) -> Dict:
        return dict(
            nn_size = self._nn_size,
            is_sub = self._is_sub,
            is_fnc = self._is_fnc,
        )

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

    def __len__(self) -> int:
        return self._nn_size

    def load(self, prototype_filepath: pathlib.Path) -> None:
        assert prototype_filepath.is_file(), f'There is no \"Prototype\" can be loaded from the specified directory \"{prototype_filepath.absolute()}\".'
        info = read_json(prototype_filepath)
        self.dict2meta(info)
        prototype_dirpath = prototype_filepath.parent
        nn_garph_filepath = prototype_dirpath.joinpath(self._nn_graph_filename)
        self._load_nn_graph(nn_garph_filepath)
        return

    def save(self, prototype_filepath: pathlib.Path) -> None:
        assert not prototype_filepath.is_file(), f'\"Prototype\" can not be saved into the specified directory \"{prototype_filepath.absolute()}\".'
        info = self.meta2dict()
        write_json(info, prototype_filepath)
        prototype_dirpath = prototype_filepath.parent
        nn_garph_filepath = prototype_dirpath.joinpath(self._nn_graph_filename)
        self._save_nn_graph(nn_garph_filepath)
        return

    def _load_nn_graph(self, nn_graph_filepath: pathlib.Path) -> None:
        assert nn_graph_filepath.is_file(), f'There is no \"nn_graph\" can be loaded from the specified path \"{nn_graph_filepath.absolute()}\".'
        self._nn_graph = networkx.read_gml(nn_graph_filepath)
        return

    def _save_nn_graph(self, nn_graph_filepath: pathlib.Path) -> None:
        assert not nn_graph_filepath.is_file(), f'\"nn_graph\" can not be saved into the specified path \"{nn_graph_filepath.absolute()}\".'
        networkx.write_gml(self._nn_graph, nn_graph_filepath)
        return


class Network(Prototype):
    def __init__(
            self,
            nn_graph: Optional[networkx.DiGraph] = None,
            nn_nodes: Optional[Dict[str, Node]] = None,
            nn_size: int = 0,
            is_sub: bool = False,
            is_fnc: bool = False,
            models: Optional[List[Model]] = None,
            version: semantic_version.Version = semantic_version.Version('0.0.0')
    ) -> None:
        super(Network, self).__init__(nn_graph, nn_size, is_sub, is_fnc)
        nn_nodes = nn_nodes or dict()
        models = models or list()

        assert len(nn_nodes) == nn_size
        self._meta_filename = 'meta.json'
        self._meta = Meta()

        self._prototype_filename = 'prototype.json'

        self._nn_nodes_filename = 'nn_nodes.json'
        self._nn_nodes = nn_nodes

        self._stamps_filename = 'stamps.json'
        self._stamps = set()

        self._uniques_filename = 'uniques.json'
        self._uniques = list()

        self._models_dirname = 'models'
        self._models = dict()

        self._mode = None

        self._legacy = False

        self.insert_models(models)
        self.release(version)

    @property
    def nn_nodes(self) -> Dict[str, Node]:
        return self._nn_nodes

    @property
    def meta(self) -> Meta:
        return self._meta

    @property
    def stamps(self) -> Set[Stamp]:
        return self._stamps

    @property
    def uniques(self) -> List[str]:
        return self._uniques

    @property
    def models(self) -> Dict[str, Model]:
        return self._models

    @property
    def mode(self):
        return self._mode

    @property
    def mode_open(self) -> int:
        return 0B1

    @property
    def mode_close(self) -> int:
        return 0B0

    @property
    def latest_version(self) -> semantic_version.Version:
        latest_version = semantic_version.Version('0.0.0')
        for stamp in self.stamps:
            latest_version = max(latest_version, stamp.version)
        return latest_version

    @property
    def checksum(self) -> str:
        ids = list()
        ids.append(self.identifier)
        for identifier in self.uniques:
            model = self.models[identifier]
            if model.is_release:
                ids.append(model.identifier)
        return hash_strings(ids)

    @property
    def identifier(self) -> str:
        layer_strings = list()
        for index, bfs_layer in enumerate(self.bfs_layers):
            bfs_layer = [(node_id, self.nn_graph.nodes[str(node_id)]) for node_id in bfs_layer]
            bfs_layer = sorted(bfs_layer, key=lambda x: (x[1]['is_first'], x[1]['is_last'], x[1]['is_custom'], x[1]['has_subgraph'], x[1]['type'], x[1]['domain'], x[1]['in_number'], x[1]['out_number']))
            layer_string = f'[{index}:'
            for i, (node_id, node) in enumerate(bfs_layer):
                layer_string += f'={i}'
                if node['is_custom']:
                    layer_string += f'|in_number={node["in_number"]}'
                    layer_string += f'|out_number={node["out_number"]}'
                else:
                    layer_string += f'|type={node["type"]}'
                    layer_string += f'|domain={node["domain"]}'
                    layer_string += f'|in_number={node["in_number"]}'
                    layer_string += f'|out_number={node["out_number"]}'
                layer_string += f'|quasi_dict={json.dumps(self._nn_nodes[str(node_id)].quasi_dict, sort_keys=True)}'
                layer_string += f'|='
            layer_string = layer_string + ']'
            layer_strings.append(layer_string)
        return hash_strings(layer_strings)

    @property
    def prototype(self) -> Prototype:
        return Prototype(self._nn_graph, self._nn_size, self._is_sub, self._is_fnc)

    @property
    def status(self) -> int:
        status = ((len(self._nn_graph) == 0) << 2) + (self.meta.release << 1) + (self.meta.retired or self._legacy)
        # 0B100 -> It has never been added before.
        # 0BX1X
        # 0BX10 -> Release (New)
        # 0BX11 -> Retired (Old)
        # 0BX0X
        # 0BX00 -> New
        # 0BX01 -> Old
        assert 0B000 <= status and status <= 0B111, f'Invalid status code: {status}.'
        assert (status >> 1) != 0B11, f'Invalid status code: {status}.'
        assert status != 0B101, f'Invalid status code: {status}.'
        return status

    @property
    def is_fresh(self) -> bool:
        return self.status == 0B100

    @property
    def is_release(self) -> bool:
        return (self.status & 0B011) == 0B010

    @property
    def is_retired(self) -> bool:
        return (self.status & 0B011) == 0B011

    @property
    def is_internal(self) -> bool:
        return (self.status & 0B010) == 0B010

    @property
    def is_external(self) -> bool:
        return (self.status & 0B010) == 0B000

    @property
    def is_new(self) -> bool:
        return (self.status & 0B001) == 0B000

    @property
    def is_old(self) -> bool:
        return (self.status & 0B001) == 0B001

    def set_new(self) -> None:
        self._legacy = False
        return

    def set_old(self) -> None:
        self._legacy = True
        return

    def __eq__(self, network: 'Network') -> bool:
        if self.prototype != network.prototype:
            return False
        if self.identifier != network.identifier:
            return False
        return True

    def load(self, network_dirpath: pathlib.Path) -> None:
        assert network_dirpath.is_dir(), f'There is no \"Network\" can be loaded from the specified directory \"{network_dirpath.absolute()}\".'
        meta_filepath = network_dirpath.joinpath(self._meta_filename)
        self._load_meta(meta_filepath)
        prototype_filepath = network_dirpath.joinpath(self._prototype_filename)
        super(Network, self).load(prototype_filepath)
        nn_nodes_filepath = network_dirpath.joinpath(self._nn_nodes_filename)
        self._load_nn_nodes(nn_nodes_filepath)

        stamps_filepath = network_dirpath.joinpath(self._stamps_filename)
        self._load_stamps(stamps_filepath)
        uniques_filepath = network_dirpath.joinpath(self._uniques_filename)
        self._load_uniques(uniques_filepath)
        models_dirpath = network_dirpath.joinpath(self._models_dirname)
        self._load_models(models_dirpath)
        return 

    def save(self, network_dirpath: pathlib.Path) -> None:
        assert not network_dirpath.is_dir(), f'\"Network\" can not be saved into the specified directory \"{network_dirpath.absolute()}\".'
        prototype_filepath = network_dirpath.joinpath(self._prototype_filename)
        super(Network, self).save(prototype_filepath)
        nn_nodes_filepath = network_dirpath.joinpath(self._nn_nodes_filename)
        self._save_nn_nodes(nn_nodes_filepath)
        meta_filepath = network_dirpath.joinpath(self._meta_filename)
        self._save_meta(meta_filepath)

        models_dirpath = network_dirpath.joinpath(self._models_dirname)
        self._save_models(models_dirpath)
        uniques_filepath = network_dirpath.joinpath(self._uniques_filename)
        self._save_uniques(uniques_filepath)
        stamps_filepath = network_dirpath.joinpath(self._stamps_filename)
        self._save_stamps(stamps_filepath)
        return

    def _load_nn_nodes(self, nn_nodes_filepath: pathlib.Path) -> None:
        assert nn_nodes_filepath.is_file(), f'There is no \"nn_nodes\" can be loaded from the specified path \"{nn_nodes_filepath.absolute()}\".'
        nn_nodes = read_json(nn_nodes_filepath)
        self._nn_nodes = dict()
        for nid, nn_node in nn_nodes.items():
            nn_node = Node(**nn_node)
            self._nn_nodes[nid] = nn_node
        return

    def _save_nn_nodes(self, nn_nodes_filepath: pathlib.Path) -> None:
        assert not nn_nodes_filepath.is_file(), f'\"nn_nodes\" can not be saved into the specified path \"{nn_nodes_filepath.absolute()}\".'
        nn_nodes = dict()
        for nid, nn_node in self._nn_nodes.items():
            nn_node = nn_node.dict
            nn_nodes[nid] = nn_node
        write_json(nn_nodes, nn_nodes_filepath)
        return

    def _load_meta(self, meta_filepath: pathlib.Path) -> None:
        assert meta_filepath.is_file(), f'There is no \"Meta\" can be loaded from the specified path \"{meta_filepath.absolute()}\".'
        meta = read_json(meta_filepath)
        self._meta = Meta(**meta)
        return

    def _save_meta(self, meta_filepath: pathlib.Path) -> None:
        assert not meta_filepath.is_file(), f'\"Meta\" can not be saved into the specified path \"{meta_filepath.absolute()}\".'
        meta = self._meta.dict
        write_json(meta, meta_filepath)
        return

    def _load_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert stamps_filepath.is_file(), f'There is no \"Stamp\"s can be loaded from the specified path \"{stamps_filepath.absolute()}\".'
        stamps = read_json(stamps_filepath)
        self._stamps = set()
        for stamp in stamps:
            self.stamps.add(Stamp(**stamp))
        return

    def _save_stamps(self, stamps_filepath: pathlib.Path) -> None:
        assert not stamps_filepath.is_file(), f'\"Stamp\"s can not be saved into the specified path \"{stamps_filepath.absolute()}\".'
        stamps = list()
        for stamp in self.stamps:
            stamps.append(stamp.dict)
        write_json(stamps, stamps_filepath)
        return

    def _load_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert uniques_filepath.is_file(), f'There is no \"Unique\"s can be loaded from the specified path \"{uniques_filepath.absolute()}\".'
        self._uniques = read_json(uniques_filepath)
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        return

    def _save_uniques(self, uniques_filepath: pathlib.Path) -> None:
        assert not uniques_filepath.is_file(), f'\"Unique\"s can not be saved into the specified path \"{uniques_filepath.absolute()}\".'
        assert isinstance(self._uniques, list), f'Wrong type of the \"Unique\"s, should be \"{type(list())}\" instead \"{type(self._uniques)}\"'
        write_json(self._uniques, uniques_filepath)
        return

    def _load_models(self, models_dirpath: pathlib.Path) -> None:
        if len(self._uniques) == 0:
            return
        assert models_dirpath.is_dir(), f'There is no \"Model\" can be loaded from the specified directory \"{models_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            model_dirpath = models_dirpath.joinpath(f'{index}-{identifier}')
            self._models[identifier] = Model()
            self._models[identifier].load(model_dirpath)
            logger.info(f' = [YBD] =   \u2514 No.{index} Model: = {self._models[identifier].name} (opset={self._models[identifier].opset}) = {identifier}')
        return

    def _save_models(self, models_dirpath: pathlib.Path) -> None:
        if len(self._uniques) == 0:
            return
        assert not models_dirpath.is_dir(), f'\"Model\"s can not be saved into the specified directory \"{models_dirpath.absolute()}\".'
        for index, identifier in enumerate(self._uniques):
            model_dirpath = models_dirpath.joinpath(f'{index}-{identifier}')
            model = self._models[identifier]
            model.save(model_dirpath)
            logger.info(f' = [YBD] =   \u2514 No.{index} Model: = {model.name} (opset={model.opset}) = {identifier}')
        return

    def acquire(self, version: semantic_version.Version) -> 'Network':
        if (self.meta.release and self.meta.release_version <= version) and (not self.meta.retired or version < self.meta.retired_version):
            network = self.copy()
            for index, identifier in enumerate(self._uniques):
                model = self._models[identifier].acquire(version)
                if model is not None:
                    network._models[identifier] = model
                    logger.info(f' = [YBD] = Acquired   \u250c No.{index} Model: = {model.name} = {identifier}')
        else:
            network = None
        return network

    def check(self) -> None:
        assert len(self.uniques) == len(self.models), f'The number of \"Model\"s does not match the number of \"Unique\"s.'
        for identifier in self.uniques:
            model = self.models[identifier]
            assert identifier == model.identifier, f'The \"Identifier={model.identifier}\" of \"Model\" does not match \"Unique={identifier}\" '
        return

    def copy(self) -> 'Network':
        network = Network(nn_graph=self._nn_graph, nn_nodes=self._nn_nodes, nn_size=self._nn_size, is_sub=self._is_sub, is_fnc=self._is_fnc)
        return network

    def insert(self, model: Model) -> bool:
        if self.is_fresh:
            return False
        if self.is_new and self.identifier == self.__class__.extract_network(model).identifier:
            new_model = model.copy()
            self._models[new_model.identifier] = new_model
            return True
        return False

    def delete(self, model: Model) -> bool:
        if self.is_fresh:
            return False
        if self.is_new and self.identifier == self.__class__.extract_network(model).identifier:
            old_model = self._models.get(model.identifier, Model())
            old_model.set_old()
            return True
        return False

    def insert_models(self, models: List[Model]) -> int:
        flags = list()
        for model in models:
            flags.append(self.insert(model))
        return sum(flags)

    def delete_models(self, models: List[Model]) -> int:
        flags = list()
        for model in models:
            flags.append(self.insert(model))
        return sum(flags)

    def release(self, version: semantic_version.Version) -> None:
        if self.is_fresh or version == semantic_version.Version('0.0.0'):
            return

        assert self.latest_version < version, (
            f'Version provided less than or equal to the latest version:\n'
            f'Provided: {version}\n'
            f'Latest: {self.latest_version}'
        )

        for identifier, model in self._models.items():
            if model.is_external:
                if model.is_new:
                    self._uniques.append(identifier)
                if model.is_old:
                    self._models.pop(identifier)
            model.release(version)

        stamp = Stamp(
            str(version),
            self.checksum,
        )
        if stamp in self._stamps:
            return
        else:
            self._stamps.add(stamp)

        if self.meta.release:
            if self.is_old:
                self.meta.set_retired(version)
        else:
            if self.is_new:
                self.meta.set_release(version)

        return

    @classmethod
    def extract_network(cls, model: Model) -> 'Network':
        network, _ = cls.extract_networks(model, deep_extract=False, skip_function=True)
        return network

    @classmethod
    def extract_networks(cls, model: Model, deep_extract: bool = False, skip_function: bool = False) -> Tuple['Network', List['Network']]:
        onnx_model = infer_shapes(model.onnx_model)
        deep_networks = list()

        fp_networks = dict()
        for function in onnx_model.functions:
            fp_networks[(function.name, function.domain)] = function

        if not skip_function:
            for function in onnx_model.functions:
                fp_network = fp_networks[(function.name, function.domain)]
                if isinstance(fp_network, onnx.FunctionProto):
                    fp_network, fp_deep_networks = cls.extract_from_fp(function, fp_networks, deep_extract=deep_extract, is_sub=True)
                    fp_networks[(function.name, function.domain)] = fp_network
                    deep_networks.append(fp_network)
                    deep_networks.extend(fp_deep_networks)
                else:
                    assert isinstance(fp_network, Network)
                    deep_networks.append(fp_network)

        network, gp_deep_networks = cls.extract_from_gp(onnx_model.graph, fp_networks, deep_extract=deep_extract, is_sub=False)

        deep_networks.extend(gp_deep_networks)
        return network, deep_networks

    @classmethod
    def extract_from_fp(
        cls,
        fp: onnx.FunctionProto,
        fp_networks: Dict[Tuple[str, str], 'Network'],
        deep_extract: bool = False,
        is_sub: bool = False
    ) -> Tuple['Network', List['Network']]:
        nn_graph = networkx.DiGraph()
        nn_nodes = dict()
        nn_size = 0
        deep_networks = list()

        i2x = dict()
        o2x = dict()
        nid2x = dict()
        for nid, node in enumerate(fp.node):
            nid = str(nid)
            nid2x[nid] = dict()
            if node.domain in ONNXOperatorDomain:
                operator = onnx.defs.get_schema(node.op_type, domain=node.domain)

                has_subgraph = False
                attributes = dict()
                for attribute in node.attribute:
                    if attribute.type == ONNXAttributeType.GRAPH or attribute.type == ONNXAttributeType.GRAPHS:
                        has_subgraph = True
                        all_extracted = list()
                        all_deep_extracted = list()
                        if deep_extract and attribute.type == ONNXAttributeType.GRAPH:
                            extracted, deep_extracted = cls.extract_from_gp(attribute.g, fp_networks, deep_extract=deep_extract, is_sub=True)
                            all_extracted.append(extracted)
                            all_deep_extracted.extend(deep_extracted)
                        if deep_extract and attribute.type == ONNXAttributeType.GRAPHS:
                            for attribute_g in attribute.graphs:
                                extracted, deep_extracted = cls.extract_from_gp(attribute_g, fp_networks, deep_extract=deep_extract, is_sub=True)
                                all_extracted.append(extracted)
                                all_deep_extracted.extend(deep_extracted)
                        attributes[attribute.name] = [extracted.identifier for extracted in all_extracted]
                        deep_networks.extend(all_extracted + all_deep_extracted)
                    else:
                        attributes[attribute.name] = json_format.MessageToDict(attribute)

                operands = dict()
                nid2x[nid]['input'] = dict()
                variadic_index = 0
                for index, input in enumerate(node.input):
                    origin_index = index
                    if index >= len(operator.inputs) and operator.inputs[-1].option == operator.inputs[-1].option.Variadic.value:
                        index = len(operator.inputs) - 1
                        variadic_index += 1
                    operand_name = operator.inputs[index].name + (f'_{variadic_index}' if variadic_index else '')
                    operands[operand_name] = input
                    i2x_this = i2x.get(input, list())
                    i2x_this.append((nid, operand_name, index))
                    i2x[input] = i2x_this
                    nid2x[nid]['input'][origin_index] = operand_name

                results = dict()
                nid2x[nid]['output'] = dict()
                variadic_index = 0
                for index, output in enumerate(node.output):
                    origin_index = index
                    if index >= len(operator.outputs) and operator.outputs[-1].option == operator.outputs[-1].option.Variadic.value:
                        index = len(operator.outputs) - 1
                        variadic_index += 1
                    result_name = operator.outputs[index].name + (f'_{variadic_index}' if variadic_index else '')
                    results[result_name] = output
                    o2x_this = o2x.get(output, list())
                    o2x_this.append((nid, result_name, index))
                    o2x[output] = o2x_this
                    nid2x[nid]['output'][origin_index] = result_name

                node = Node(
                    operator_type=node.op_type,
                    operator_domain=node.domain,
                    attributes=attributes,
                    parameters=dict(),
                    operands=operands,
                    results=results,
                    has_subgraph=has_subgraph,
                )
            else:
                fp_network = fp_networks.get((node.op_type, node.domain), None)
                if fp_network is None:
                    attributes = dict()
                    has_subgraph=False
                else:
                    if isinstance(fp_network, onnx.FunctionProto):
                        fp_network, fp_deep_networks = cls.extract_from_fp(fp_network, fp_networks, deep_extract=deep_extract, is_sub=True)
                        fp_networks[(function.name, function.domain)] = fp_network
                        deep_networks.extend(fp_deep_networks)
                    else:
                        assert isinstance(fp_network, Network)
                    attributes = dict(__YBD_function__=fp_network.identifier)
                    has_subgraph=False

                operands = dict()
                nid2x[nid]['input'] = dict()
                for index, input in enumerate(node.input):
                    operands[input] = input
                    i2x_this = i2x.get(input, list())
                    i2x_this.append((nid, input, index))
                    i2x[input] = i2x_this
                    nid2x[nid]['input'][index] = input

                results = dict()
                nid2x[nid]['output'] = dict()
                for index, output in enumerate(node.output):
                    results[output] = output
                    o2x_this = o2x.get(output, list())
                    o2x_this.append((nid, output, index))
                    o2x[output] = o2x_this
                    nid2x[nid]['output'][index] = output

                node = Node(
                    operator_type=node.op_type,
                    operator_domain=node.domain,
                    attributes=attributes,
                    parameters=dict(),
                    operands=operands,
                    results=results,
                    is_custom=True,
                    has_subgraph = has_subgraph,
                )

            nn_graph.add_node(nid, **node.features)
            nn_nodes[nid] = node
            nn_size += 1

        for nid, node in enumerate(fp.node):
            nid = str(nid)

            for index, input in enumerate(node.input):
                v_nid = nid
                v_opn = nid2x[nid]['input'][index]
                v_opi = index
                if input in o2x:
                    for o2x_this in o2x[input]:
                        u_nid, u_opn, u_opi = o2x_this
                        nn_graph.add_edge(u_nid, v_nid, u_opn=u_opn, v_opn=v_opn, u_opi=u_opi, v_opi=v_opi)

            for index, output in enumerate(node.output):
                u_nid = nid
                u_opn = nid2x[nid]['output'][index]
                u_opi = index
                if output in i2x:
                    for i2x_this in i2x[output]:
                        v_nid, v_opn, v_opi = i2x_this
                        nn_graph.add_edge(u_nid, v_nid, u_opn=u_opn, v_opn=v_opn, u_opi=u_opi, v_opi=v_opi)

        assert networkx.is_directed_acyclic_graph(nn_graph), f'The \"Network\" converted from the \"ONNX Model\" (onnx_model) of Model() is not a Directed Acyclic Graph.'

        network = Network(nn_graph=nn_graph, nn_nodes=nn_nodes, nn_size=nn_size, is_sub=is_sub, is_fnc=True)

        return network, deep_networks

    @classmethod
    def extract_from_gp(
            cls,
            gp: onnx.GraphProto,
            fp_networks: Dict[Tuple[str, str], 'Network'],
            deep_extract: bool = False,
            is_sub: bool = False
    ) -> Tuple['Network', List['Network']]:
        assert isinstance(gp, onnx.GraphProto)
        nn_graph = networkx.DiGraph()
        nn_nodes = dict()
        nn_size = 0
        deep_networks = list()

        # Parameter
        pm_info = dict()
        for initializer in gp.initializer:
            initializer_dict = json_format.MessageToDict(initializer)
            pm_info[initializer.name] = dict(
                dims = ['dims'],
                dataType = initializer_dict['dataType'],
            )

        # Input & Output
        io_info = dict()
        inputs = set()
        outputs = set()
        for input in gp.input:
            io_info[input.name] = json_format.MessageToDict(input.type)
            inputs.add(input.name)
        for value in gp.value_info:
            io_info[value.name] = json_format.MessageToDict(value.type)
        for output in gp.output:
            io_info[output.name] = json_format.MessageToDict(output.type)
            outputs.add(output.name)

        i2x = dict()
        o2x = dict()
        nid2x = dict()
        for nid, node in enumerate(gp.node):
            nid = str(nid)
            nid2x[nid] = dict()
            if node.domain in ONNXOperatorDomain:
                operator = onnx.defs.get_schema(node.op_type, domain=node.domain)

                has_subgraph = False
                attributes = dict()
                for attribute in node.attribute:
                    if attribute.type == ONNXAttributeType.GRAPH or attribute.type == ONNXAttributeType.GRAPHS:
                        has_subgraph = True
                        all_extracted = list()
                        all_deep_extracted = list()
                        if deep_extract and attribute.type == ONNXAttributeType.GRAPH:
                            extracted, deep_extracted = cls.extract_from_gp(attribute.g, fp_networks, deep_extract=deep_extract, is_sub=True)
                            all_extracted.append(extracted)
                            all_deep_extracted.extend(deep_extracted)
                        if deep_extract and attribute.type == ONNXAttributeType.GRAPHS:
                            for attribute_g in attribute.graphs:
                                extracted, deep_extracted = cls.extract_from_gp(attribute_g, fp_networks, deep_extract=deep_extract, is_sub=True)
                                all_extracted.append(extracted)
                                all_deep_extracted.extend(deep_extracted)
                        attributes[attribute.name] = [extracted.identifier for extracted in all_extracted]
                        deep_networks.extend(all_extracted + all_deep_extracted)
                    else:
                        attributes[attribute.name] = json_format.MessageToDict(attribute)

                parameters = dict()
                variadic_index = 0
                for index, input in enumerate(node.input):
                    if index >= len(operator.inputs) and operator.inputs[-1].option == operator.inputs[-1].option.Variadic.value:
                        index = len(operator.inputs) - 1
                        variadic_index += 1
                    parameter_name = operator.inputs[index].name + (f'_{variadic_index}' if variadic_index else '')
                    if input in pm_info:
                        parameters[parameter_name] = pm_info[input]
                    else:
                        parameters[parameter_name] = dict()

                is_first = False
                operands = dict()
                nid2x[nid]['input'] = dict()
                variadic_index = 0
                for index, input in enumerate(node.input):
                    origin_index = index
                    if index >= len(operator.inputs) and operator.inputs[-1].option == operator.inputs[-1].option.Variadic.value:
                        index = len(operator.inputs) - 1
                        variadic_index += 1
                    operand_name = operator.inputs[index].name + (f'_{variadic_index}' if variadic_index else '')
                    if input in io_info:
                        operands[operand_name] = io_info[input]
                    else:
                        operands[operand_name] = dict()
                    i2x_this = i2x.get(input, list())
                    i2x_this.append((nid, operand_name, index))
                    i2x[input] = i2x_this
                    nid2x[nid]['input'][origin_index] = operand_name
                    is_first |= input in inputs

                is_last = False
                results = dict()
                nid2x[nid]['output'] = dict()
                variadic_index = 0
                for index, output in enumerate(node.output):
                    origin_index = index
                    if index >= len(operator.outputs) and operator.outputs[-1].option == operator.outputs[-1].option.Variadic.value:
                        index = len(operator.outputs) - 1
                        variadic_index += 1
                    result_name = operator.outputs[index].name + (f'_{variadic_index}' if variadic_index else '')
                    if output in io_info:
                        results[result_name] = io_info[output]
                    else:
                        results[result_name] = dict()
                    o2x_this = o2x.get(output, list())
                    o2x_this.append((nid, result_name, index))
                    o2x[output] = o2x_this
                    nid2x[nid]['output'][origin_index] = result_name
                    is_last |= output in outputs

                node = Node(
                    operator_type=node.op_type,
                    operator_domain=node.domain,
                    attributes=attributes,
                    parameters=parameters,
                    operands=operands,
                    is_first=is_first,
                    is_last=is_last,
                    results=results,
                    has_subgraph=has_subgraph,
                )
            else:
                fp_network = fp_networks.get((node.op_type, node.domain), None)
                if fp_network is None:
                    attributes = dict()
                    has_subgraph=False
                else:
                    if isinstance(fp_network, Network):
                        attributes = dict(__YBD_function__=fp_network.identifier)
                    else:
                        attributes = dict(__YBD_function__=str())
                    has_subgraph = True

                operands = dict()
                nid2x[nid]['input'] = dict()
                for index, input in enumerate(node.input):
                    if input in io_info:
                        operands[input] = io_info[input]
                    else:
                        operands[input] = dict()
                    i2x_this = i2x.get(input, list())
                    i2x_this.append((nid, input, index))
                    i2x[input] = i2x_this
                    nid2x[nid]['input'][index] = input

                results = dict()
                nid2x[nid]['output'] = dict()
                for index, output in enumerate(node.output):
                    if output in io_info:
                        results[output] = io_info[output]
                    else:
                        results[output] = dict()
                    o2x_this = o2x.get(output, list())
                    o2x_this.append((nid, output, index))
                    o2x[output] = o2x_this
                    nid2x[nid]['output'][index] = output

                node = Node(
                    operator_type=node.op_type,
                    operator_domain=node.domain,
                    attributes=attributes,
                    parameters=dict(),
                    operands=operands,
                    results=results,
                    is_custom=True,
                    has_subgraph = has_subgraph,
                )

            nn_graph.add_node(nid, **node.features)
            nn_nodes[nid] = node
            nn_size += 1

        for nid, node in enumerate(gp.node):
            nid = str(nid)

            for index, input in enumerate(node.input):
                v_nid = nid
                v_opn = nid2x[nid]['input'][index]
                v_opi = index
                if input in o2x:
                    for o2x_this in o2x[input]:
                        u_nid, u_opn, u_opi = o2x_this
                        nn_graph.add_edge(u_nid, v_nid, u_opn=u_opn, v_opn=v_opn, u_opi=u_opi, v_opi=v_opi)

            for index, output in enumerate(node.output):
                u_nid = nid
                u_opn = nid2x[nid]['output'][index]
                u_opi = index
                if output in i2x:
                    for i2x_this in i2x[output]:
                        v_nid, v_opn, v_opi = i2x_this
                        nn_graph.add_edge(u_nid, v_nid, u_opn=u_opn, v_opn=v_opn, u_opi=u_opi, v_opi=v_opi)

        assert networkx.is_directed_acyclic_graph(nn_graph), f'The \"Network\" converted from the \"ONNX Model\" (onnx_model) of Model() is not a Directed Acyclic Graph.'

        network = Network(nn_graph=nn_graph, nn_nodes=nn_nodes, nn_size=nn_size, is_sub=is_sub, is_fnc=False)

        return network, deep_networks