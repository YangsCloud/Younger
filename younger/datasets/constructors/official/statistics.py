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

import math
import tqdm
import time
import pathlib
import datetime
import networkx
import itertools
import multiprocessing

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.detectors.metrics import normalize


def statistics_graph(graph: networkx.DiGraph) -> dict[str, dict[str, int] | int]:
    graph_statistics = dict(
        num_operators = dict(),
        num_node = graph.number_of_nodes(),
        num_edge = graph.number_of_edges(),
    )

    for node_id in graph.nodes:
        node_features = graph.nodes[node_id]['features']
        node_identifier = Network.get_node_identifier_from_features(node_features)
        graph_statistics['num_operators'][node_identifier] = graph_statistics['num_operators'].get(node_identifier, 0) + 1
    return graph_statistics

def statistics_instance(parameter: tuple[pathlib.Path, list, bool, bool, bool, bool]) -> tuple[dict[str, list], dict[str, int]]:
    path, combined_filters, has_task_filters, has_dataset_name_filters, has_dataset_split_filters, has_metric_name_filters = parameter

    occurrence: dict[str, int] = dict()
    statistics: dict[str, list] = dict()
    for combined_filter in combined_filters:
        statistics[str(combined_filter)] = list()

    instance = Instance()
    instance.load(path)

    instance_name = path.name
    graph_stats = statistics_graph(instance.network.graph)

    if instance.labels['evaluations']:
        for task, dataset_name, dataset_split, metric_name, metric_value in instance.labels['evaluations']:
            combined_filter_pattern = str((
                task if has_task_filters else '*',
                dataset_name if has_dataset_name_filters else '*',
                dataset_split if has_dataset_split_filters else '*',
                metric_name if has_metric_name_filters else '*',
            ))
            if combined_filter_pattern in statistics:
                occurrence[str((task, dataset_name, dataset_split, metric_name))] = occurrence.get(str((task, dataset_name, dataset_split, metric_name)), 0) + 1
                statistics[combined_filter_pattern].append(dict(
                    instance_name = instance_name,
                    graph_stats = graph_stats,
                    task = task,
                    dataset_name = dataset_name,
                    dataset_split = dataset_split,
                    metric_name = metric_name,
                    metric_value = metric_value
                ))
    else:
        combined_filter_pattern = str(('*', '*', '*', '*'))
        if combined_filter_pattern in statistics:
            occurrence[combined_filter_pattern] = occurrence.get(combined_filter_pattern, 0) + 1
            statistics[combined_filter_pattern].append(dict(
                instance_name = instance_name,
                graph_stats = graph_stats,
                task = None,
                dataset_name = None,
                dataset_split = None,
                metric_name = None,
                metric_value = None 
            ))
    return (statistics, occurrence)


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, tasks: list[str], dataset_names: list[str], dataset_splits: list[str], metric_names: list[str], worker_number: int = 4):
    tic = time.time()
    has_task_filters = len(tasks) != 0
    has_dataset_name_filters = len(dataset_names) != 0
    has_dataset_split_filters = len(dataset_splits) != 0
    has_metric_name_filters = len(metric_names) != 0

    task_filters = tasks if has_task_filters else ['*']
    dataset_name_filters = dataset_names if has_dataset_name_filters else ['*']
    dataset_split_filters = dataset_splits if has_dataset_split_filters else ['*']
    metric_name_filters = metric_names if has_metric_name_filters else ['*']

    combined_filters = list()
    for combined_filter in itertools.product(*[task_filters, dataset_name_filters, dataset_split_filters, metric_name_filters]):
        combined_filters.append(combined_filter)

    occurrence: dict[str, int] = dict()
    statistics: dict[str, list] = dict()
    mum: dict[str, list[float]] = dict() # Maximum & Minimum
    for combined_filter in combined_filters:
        statistics[str(combined_filter)] = list()

    parameters = [(path, combined_filters, has_task_filters, has_dataset_name_filters, has_dataset_split_filters, has_metric_name_filters) for path in dataset_dirpath.iterdir()]
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Calculating Statistics') as progress_bar:
            for index, (instance_statistics, instance_occurrence) in enumerate(pool.imap_unordered(statistics_instance, parameters), start=1):
                for filter_type_key, filter_type_value in instance_occurrence.items():
                    occurrence[filter_type_key] = occurrence.get(filter_type_key, 0) + filter_type_value
                for instance_statistics_key, instance_statistics_value in instance_statistics.items():
                    statistics[instance_statistics_key].extend(instance_statistics_value)
                progress_bar.update(1)

    for instances_statistics in statistics.values():
        for instance_statistics in instances_statistics:
            metric_name = instance_statistics['metric_name']
            metric_value = instance_statistics['metric_value']

            if metric_name is None or metric_value is None:
                continue
            else:
                minimum, maximum = mum.get(metric_name, (0, 1))
                mum[metric_name] = [min(minimum, metric_value), max(maximum, metric_value)]

    for key in statistics.keys():
        for index, instance_statistics in enumerate(statistics[key]):
            metric_name = instance_statistics['metric_name']
            metric_value = instance_statistics['metric_value']
            if metric_name is None or metric_value is None:
                continue
            else:
                statistics[key][index]['metric_value'] = normalize(metric_name, metric_value, mum[metric_name][0], mum[metric_name][1])

    toc = time.time()
    finished_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    stats_filepath = save_dirpath.joinpath(f'statistics_{finished_datetime}.json')
    save_json(statistics, stats_filepath, indent=2)

    ocrcs_filepath = save_dirpath.joinpath(f'occurrence_{finished_datetime}.json')
    save_json(sorted(list(occurrence.items()), key=lambda x: x[1], reverse=True), ocrcs_filepath, indent=2)
    logger.info(f'Time Cost: {toc-tic:.2f}')
    logger.info(f'Statistics Saved into: {stats_filepath}')
    logger.info(f'Occurrence Saved into: {ocrcs_filepath}')
    mum_filepath = save_dirpath.joinpath(f'mum_{finished_datetime}.json')
    save_json(list(mum.items()), mum_filepath, indent=2)
    logger.info(f'Mum Saved into: {mum_filepath}')
