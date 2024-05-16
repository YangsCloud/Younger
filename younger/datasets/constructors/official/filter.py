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
import multiprocessing

from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_all_attributes_of_operator


def update_unique_instance(unique_instance: Instance, purified_instance: Instance):
    unique_instance.labels['model_sources'].extend([] if purified_instance.labels['model_source'] is None else [purified_instance.labels['model_source']])
    unique_instance.labels['downloads'].extend([] if purified_instance.labels['download'] is None else [purified_instance.labels['download']])
    unique_instance.labels['likes'].extend([] if purified_instance.labels['like'] is None else [purified_instance.labels['like']])
    unique_instance.labels['tags'].extend([] if purified_instance.labels['tag'] is None else purified_instance.labels['tag'])

    if purified_instance.labels['annotations'] and purified_instance.labels['annotations']['eval_results']:
        for eval_result in purified_instance.labels['annotations']['eval_results']:
            task = eval_result['task']
            dataset_name, dataset_split = eval_result['dataset']
            metric_name, metric_value = eval_result['metric']
            if math.isnan(metric_value):
                continue
            unique_instance.labels['evaluations'].append([task, dataset_name, dataset_split, metric_name, metric_value])


def complete_attributes_of_node(node_features: dict[str, dict[str, str | dict]], max_inclusive_version: int) -> dict[str, dict[str, str | tuple[int, str]]]:
    operator: dict[str, str] = node_features['operator']
    attributes: dict[str, dict] = node_features['attributes']
    all_attributes = get_all_attributes_of_operator(operator['op_type'], max_inclusive_version, domain=operator['domain'])
    if all_attributes is None:
        all_attributes = dict()
        for attribute_name, attribute_proto_dict in attributes.items():
            all_attributes[attribute_name] = (attribute_proto_dict['attr_type'].value, str(attribute_proto_dict['value']))
    else:
        for attribute_name, attribute_proto_dict in attributes.items():
            if attribute_name not in all_attributes:
                continue

            all_attributes[attribute_name] = (all_attributes[attribute_name][0], str(attribute_proto_dict['value']))

    return dict(
        operator = operator,
        attributes = all_attributes
    )


def purify_instance(parameter: tuple[str, int]) -> Instance:
    path, max_inclusive_version = parameter
    instance = Instance()
    try:
        instance.load(path)
    except:
        return None

    standardized_graph = Network.standardize(instance.network.graph)
    for node_index in standardized_graph.nodes():
        standardized_graph.nodes[node_index]['features'] = complete_attributes_of_node(standardized_graph.nodes[node_index]['features'], max_inclusive_version)

    standardized_graph.graph.clear()

    instance.setup_network(Network(standardized_graph))
    return instance


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, max_inclusive_version: int, worker_number: int = 4):
    logger.info(f'Scanning Dataset Directory Path: {dataset_dirpath}')
    parameters = list()
    for path in dataset_dirpath.iterdir():
        if path.is_dir():
            parameters.append((path, max_inclusive_version))

    logger.info(f'Total Instances To Be Filtered: {len(parameters)}')

    unique_instances: dict[str, Instance] = dict()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters)) as progress_bar:
            for index, purified_instance in enumerate(pool.imap_unordered(purify_instance, parameters), start=1):
                if purified_instance is None:
                    continue
                graph_hash = Network.hash(purified_instance.network.graph, node_attr='features')
                if graph_hash in unique_instances:
                    unique_instance = unique_instances[graph_hash]
                else:
                    unique_instance = purified_instance.copy()
                    unique_instance.clean_labels()
                    unique_instance.setup_labels(dict(model_sources=[], downloads=[], likes=[], tags=[], evaluations=[]))
                try:
                    update_unique_instance(unique_instance, purified_instance)
                except:
                    print(purified_instance.labels)
                    continue
                unique_instances[graph_hash] = unique_instance
                progress_bar.update(1)

    logger.info(f'Total Unique Instances Filtered: {len(unique_instances)}')

    logger.info(f'Saving Unique Instances Into: {save_dirpath}')
    for graph_hash, unique_instance in unique_instances.items():
        instance_save_dirpath = save_dirpath.joinpath(graph_hash)
        unique_instance.save(instance_save_dirpath)
    logger.info(f'Finished')