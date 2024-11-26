#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-16 08:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import math
import tqdm
import pathlib
import networkx
import multiprocessing

from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node, get_onnx_opset_version


def update_unique_instance(unique_instance: Instance, purified_instance: Instance):
    new_model_sources = [] if purified_instance.labels['model_source'] is None else [purified_instance.labels['model_source']]
    new_model_name = [] if purified_instance.labels['model_name'] is None else [purified_instance.labels['model_name']]
    new_downloads = [] if purified_instance.labels['download'] is None else [purified_instance.labels['download']]
    new_likes = [] if purified_instance.labels['like'] is None else [purified_instance.labels['like']]
    new_tags = [] if purified_instance.labels['tag'] is None else purified_instance.labels['tag']

    unique_instance.labels['model_sources'] = list(set(unique_instance.labels['model_sources'] + new_model_sources))
    unique_instance.labels['model_name'] = list(set(unique_instance.labels['model_name'] + new_model_name))
    unique_instance.labels['downloads'] = list(set(unique_instance.labels['downloads'] + new_downloads))
    unique_instance.labels['likes'] = list(set(unique_instance.labels['likes'] + new_likes))
    unique_instance.labels['tags'] = list(set(unique_instance.labels['tags'] + new_tags))

    if purified_instance.labels['annotations'] and purified_instance.labels['annotations']['eval_results']:
        for eval_result in purified_instance.labels['annotations']['eval_results']:
            task = eval_result['task']
            dataset_name, dataset_split = eval_result['dataset']
            metric_name, metric_value = eval_result['metric']
            if math.isnan(metric_value):
                continue
            unique_instance.labels['evaluations'].append([task, dataset_name, dataset_split, metric_name, metric_value])


def purify_instance_with_graph_hash(parameter: tuple[str, int]) -> tuple[Instance, str]:
    path, max_inclusive_version, clean = parameter
    instance = Instance()
    instance.load(path)

    standardized_graph = Network.standardize(instance.network.graph)
    for node_index in standardized_graph.nodes():
        operator = standardized_graph.nodes[node_index]['features']['operator']
        attributes = standardized_graph.nodes[node_index]['features']['attributes']
        standardized_graph.nodes[node_index]['features']['attributes'] = get_complete_attributes_of_node(attributes, operator['op_type'], operator['domain'], max_inclusive_version)

    standardized_graph.graph.clear()

    instance.setup_network(Network(standardized_graph))

    if clean:
        cleansed_graph = networkx.DiGraph()
        cleansed_graph.add_nodes_from(instance.network.graph.nodes(data=True))
        cleansed_graph.add_edges_from(instance.network.graph.edges(data=True))
        for node_index in cleansed_graph.nodes():
            cleansed_graph.nodes[node_index]['operator'] = cleansed_graph.nodes[node_index]['features']['operator']
        graph_hash = Network.hash(cleansed_graph, node_attr='operator')
    else:
        graph_hash = Network.hash(instance.network.graph, node_attr='features')

    return (instance, graph_hash)


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, worker_number: int = 4, max_inclusive_version: int | None = None, clean: bool = False):
    if max_inclusive_version:
        logger.info(f'Using ONNX Max Inclusive Version: {max_inclusive_version}')
    else:
        max_inclusive_version = get_onnx_opset_version()
        logger.info(f'Not Specified ONNX Max Inclusive Version. Using Latest Version: {max_inclusive_version}')

    if clean:
        logger.info(f'Filter without Attributes')
    else:
        logger.info(f'Filter with Attributes')

    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters = list()
    for path in dataset_dirpath.iterdir():
        if path.is_dir():
            parameters.append((path, max_inclusive_version, clean))

    logger.info(f'Total Instances To Be Filtered: {len(parameters)}')

    unique_instances: dict[str, Instance] = dict()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Filtering') as progress_bar:
            for index, (purified_instance, graph_hash) in enumerate(pool.imap_unordered(purify_instance_with_graph_hash, parameters), start=1):
                if purified_instance is None:
                    continue
                if graph_hash in unique_instances:
                    unique_instance = unique_instances[graph_hash]
                else:
                    unique_instance = purified_instance.copy()
                    unique_instance.clean_labels()
                    unique_instance.setup_labels(dict(model_sources=[], model_name=[], downloads=[], likes=[], tags=[], evaluations=[], hash=graph_hash))
                update_unique_instance(unique_instance, purified_instance)
                unique_instances[graph_hash] = unique_instance
                progress_bar.update(1)
                graph_num_node = unique_instance.network.graph.number_of_nodes()
                graph_num_edge = unique_instance.network.graph.number_of_edges()
                progress_bar.set_postfix({f'Current Node/Edge': f'{graph_num_node}/{graph_num_edge}', f'Current Unique': f'{len(unique_instances)}'})
    logger.info(f'Total Unique Instances Filtered: {len(unique_instances)}')

    logger.info(f'Saving Unique Instances Into: {save_dirpath}')
    with tqdm.tqdm(total=len(unique_instances), desc='Saving') as progress_bar:
        for graph_hash, unique_instance in unique_instances.items():
            instance_save_dirpath = save_dirpath.joinpath(graph_hash)
            unique_instance.save(instance_save_dirpath)
            progress_bar.update(1)
    logger.info(f'Finished')