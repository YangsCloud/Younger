#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Luzhou Peng (彭路洲) and Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 17:20
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import time
import onnx
import pathlib
import datetime
import networkx
import itertools
import multiprocessing

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger.datasets.modules import Instance


def statistics_graph(graph: networkx.DiGraph) -> tuple[dict[str, int], int, int]:
    num_operators: dict[str, int] = dict()
    num_node: int = graph.number_of_nodes()
    num_edge: int = graph.number_of_edges()
    num_sub: int = 0

    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        # For Operator Stats
        if node_data['type'] == 'operator':
            operator = str(node_data['operator'])
            num_operators[operator] = num_operators.get(operator, 0) + 1
            for attribute_key, attribute_value in node_data['attributes'].items():
                if attribute_value['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPHS:
                    for sub_graph in attribute_value['value']:
                        sub_num_operators, sub_num_node, sub_num_edge, sub_num_sub = statistics_graph(sub_graph)
                        for sub_operator, sub_num_operator in sub_num_operators.items():
                            num_operators[sub_operator] = num_operators.get(sub_operator, 0) + sub_num_operator
                        num_node = num_node + sub_num_node
                        num_edge = num_edge + sub_num_edge
                        num_sub = num_sub + sub_num_sub + 1
                elif attribute_value['attr_type'] == onnx.defs.OpSchema.AttrType.GRAPH:
                    sub_graph = attribute_value['value']
                    sub_num_operators, sub_num_node, sub_num_edge, sub_num_sub = statistics_graph(sub_graph)
                    for sub_operator, sub_num_operator in sub_num_operators.items():
                        num_operators[sub_operator] = num_operators.get(sub_operator, 0) + sub_num_operator
                    num_node = num_node + sub_num_node
                    num_edge = num_edge + sub_num_edge
                    num_sub = num_sub + sub_num_sub + 1
        else:
            num_node = num_node - 1
            num_edge = num_edge - graph.degree(node_id)

    return num_operators, num_node, num_edge, num_sub

def statistics_instance(parameters: tuple[pathlib.Path, list]) -> tuple[dict[str, list], dict[str, int]]:
    path, combined_filters, has_task_filters, has_dataset_filters, has_split_filters, has_metric_filters = parameters

    occurrence: dict[str, int] = dict()
    statistics: dict[str, list] = dict()
    for combined_filter in combined_filters:
        statistics[str(combined_filter)] = list()

    instance = Instance()
    instance.load(path)

    model_name = path.name
    graph_stats = statistics_graph(instance.network.graph)

    labels = instance.labels.get('labels', None)
    if labels:
        for label in labels:
            task = label['task']
            dataset = label['dataset'][0]
            split = label['dataset'][1]
            metric = label['metric'][0]
            combined_filter_pattern = str((
                task if has_task_filters else '*',
                dataset if has_dataset_filters else '*',
                split if has_split_filters else '*',
                metric if has_metric_filters else '*',
            ))
            metric_value = label['metric'][1]
            if combined_filter_pattern in statistics:
                occurrence[str((task, dataset, split, metric))] = occurrence.get(str((task, dataset, split, metric)), 0) + 1
                statistics[combined_filter_pattern].append(dict(
                    model_name = model_name,
                    graph_stats = graph_stats,
                    metric_value = metric_value
                ))
    else:
        combined_filter_pattern = str(('*', '*', '*', '*'))
        if combined_filter_pattern in statistics:
            statistics[combined_filter_pattern].append(dict(
                model_name = model_name,
                graph_stats = graph_stats
            ))
    return (statistics, occurrence)


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, tasks: list[str], datasets: list[str], splits: list[str], metrics: list[str], worker_number: int = 4):
    tic = time.time()
    has_task_filters = len(tasks) != 0
    has_dataset_filters = len(datasets) != 0
    has_split_filters = len(splits) != 0
    has_metric_filters = len(metrics) != 0

    task_filters = tasks if has_task_filters else ['*']
    dataset_filters = datasets if has_dataset_filters else ['*']
    split_filters = splits if has_split_filters else ['*']
    metric_filters = metrics if has_metric_filters else ['*']

    combined_filters = list()
    for combined_filter in itertools.product(*[task_filters, dataset_filters, split_filters, metric_filters]):
        combined_filters.append(combined_filter)

    occurrence: dict[str, int] = dict()
    statistics: dict[str, list] = dict()
    for combined_filter in combined_filters:
        statistics[str(combined_filter)] = list()

    parameters_iterator = ((path, combined_filters, has_task_filters, has_dataset_filters, has_split_filters, has_metric_filters) for path in dataset_dirpath.iterdir() if path.is_dir())
    with multiprocessing.Pool(worker_number) as pool:
        for index, (instance_statistics, instance_occurrence) in enumerate(pool.imap_unordered(statistics_instance, parameters_iterator), start=1):
            for filter_type_key, filter_type_value in instance_occurrence.items():
                occurrence[filter_type_key] = occurrence.get(filter_type_key, 0) + filter_type_value
            for instance_statistics_key, instance_statistics_value in instance_statistics.items():
                statistics[instance_statistics_key].extend(instance_statistics_value)
            logger.info(f'Processed Total {index} instances')

    toc = time.time()
    finished_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    stats_filepath = save_dirpath.joinpath(f'statistics_{finished_datetime}.json')
    save_json(statistics, stats_filepath, indent=2)

    ocrcs_filepath = save_dirpath.joinpath(f'occurrence_{finished_datetime}.json')
    save_json(sorted(list(occurrence.items()), key=lambda x: x[1], reverse=True), ocrcs_filepath, indent=2)
    logger.info(f'Time Cost: {toc-tic:.2f}')
    logger.info(f'Statistics Saved into: {stats_filepath}')
    logger.info(f'Occurrence Saved into: {ocrcs_filepath}')
