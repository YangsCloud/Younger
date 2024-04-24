#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 21:13
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import ast
import numpy
import pathlib
import multiprocessing

from typing import Any

from younger.commons.io import load_json, save_json, save_pickle, create_dir, tar_archive
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.detectors.metrics import get_metric_theroy_range


def sample_by_partition(examples: list, partition_number: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[list[int], list[int], list[int]]:
    assert train_ratio + valid_ratio + test_ratio == 1
    # examples should be a sortable list
    sorted_indices = sorted(range(len(examples)), key=lambda x: examples[x])

    partition_size = len(sorted_indices) // partition_number
    remainder_size = len(sorted_indices) %  partition_number

    train_indices = list()
    valid_indices = list()
    test_indices = list()
    for partition_index in range(partition_number):
        this_partition_l = partition_index * partition_size
        this_partition_r = (partition_index + 1) * partition_size + (remainder_size if (partition_index + 1) == partition_number else 0)
        partition = sorted_indices[this_partition_l : this_partition_r]
        test_num = int(test_ratio * len(partition))
        valid_num = int(valid_ratio * len(partition))
        indices = numpy.random.choice(partition, test_num+valid_num, replace=False)
        test_indices.extend(indices[:test_num])
        valid_indices.extend(indices[test_num:])
        train_indices.extend(numpy.setdiff1d(partition, indices))

    return train_indices, valid_indices, test_indices


def save_graph(parameters: tuple[pathlib.Path, pathlib.Path, dict[str, Any]]):
    i_path, o_path, instance_statistics = parameters
    instance = Instance()
    instance.load(i_path)
    graph = Network.simplify(instance.network.graph, preserve_node_attributes=['type', 'operator'])
    graph.graph['task'] = instance_statistics['task']
    graph.graph['dataset'] = instance_statistics['dataset']
    graph.graph['metric'] = instance_statistics['metric']
    graph.graph['metric_value'] = instance_statistics['metric_value']
    save_pickle(graph, o_path)


def save_split(meta: dict[str, Any], instances_statistics: list[dict[str, Any]], dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, worker_number: int):
    version_dirpath = save_dirpath.joinpath(meta['version'])
    split_dirpath = version_dirpath.joinpath(meta['split'])
    archive_filepath = split_dirpath.joinpath(meta['archive'])
    split = meta['split']

    graph_dirpath = split_dirpath.joinpath('graph')
    meta_filepath = split_dirpath.joinpath('meta.json')
    create_dir(graph_dirpath)

    logger.info(f'Saving \'{split}\' Split META {meta_filepath.absolute()} ... ')
    save_json(meta, meta_filepath, indent=2)

    logger.info(f'Saving \'{split}\' Split {graph_dirpath.absolute()} ... ')
    parameters = (
        (
            dataset_dirpath.joinpath(instance_statistics['instance_name']),
            graph_dirpath.joinpath(f'instance-{i}.pkl'),
            instance_statistics,
        ) for i, instance_statistics in enumerate(instances_statistics)
    )
    with multiprocessing.Pool(worker_number) as pool:
        for index, _ in enumerate(pool.imap_unordered(save_graph, parameters), start=1):
            logger.info(f'Saved Total {index} graphs')

    logger.info(f'Saving \'{split}\' Split Tar {archive_filepath.absolute()} ... ')
    tar_archive(
        [graph_dirpath.joinpath(f'instance-{i}.pkl') for i, _ in enumerate(instances_statistics)],
        archive_filepath,
        compress=True
    )


def main(
    statistics_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    version: str,
    train_proportion: int = 80, valid_proportion: int = 10, test_proportion: int = 10,
    partition_number: int = 10,
    clean: bool = False,
    worker_number: int = 4,
):
    # 1. 'statistics_filepath' is the outputs of the statistics.main

    # 2. 'clean' Usually we do not clean instances with theory value range of metric.
    # For example:
    # WER maybe larger than 1 and Word Accuracy maybe smaller than 0 in ASR research area.

    assert train_proportion + valid_proportion + test_proportion == 100

    total_proportion = train_proportion + valid_proportion + test_proportion
    train_ratio = train_proportion / total_proportion
    valid_ratio = valid_proportion / total_proportion
    test_ratio = test_proportion / total_proportion
    statistics = load_json(statistics_filepath)
    logger.info(f'Total Instances To Be Splited: {len(statistics)}')

    tasks = set()
    datasets = set()
    splits = set()
    metrics = set()
    operators = set()

    train_split = list()
    valid_split = list()
    test_split = list()

    for stats_key, stats_value in statistics.items():
        task, dataset, split, metric = ast.literal_eval(stats_key)
        if clean:
            eligible_stats_value = list()
            metric_range = get_metric_theroy_range(ins_stats['metric']) if metric == '*' else metric
            if metric_range in {(-1, 1), (0, 1), (0, 100)}:
                lower_bound, upper_bound = 0, 1
            else:
                lower_bound, upper_bound = metric_range
            for ins_stats in stats_value:
                if 'metric_value' in ins_stats:
                    metric_value = ins_stats['metric_value']
                    if (lower_bound is not None and metric_value < lower_bound) or (upper_bound is not None and upper_bound < metric_value):
                        logger.info(f'Outlier: {metric_value}. Skip The Instance: {ins_stats["instance_name"]}')
                        continue
                eligible_stats_value.append(ins_stats)
        else:
            eligible_stats_value = stats_value
        if len(eligible_stats_value) * min(train_ratio, valid_ratio, test_ratio) < 1:
            continue
        else:
            num_nodes = list()
            for eligible_instance_stats_value in eligible_stats_value:
                num_nodes.append(eligible_instance_stats_value['graph_stats']['num_node'])
                operators.update(eligible_instance_stats_value['graph_stats']['num_operators'].keys())

                tasks.add(eligible_instance_stats_value['task'])
                datasets.add(eligible_instance_stats_value['dataset'])
                splits.add(eligible_instance_stats_value['split'])
                metrics.add(eligible_instance_stats_value['metric'])

            train_indices, valid_indices, test_indices = sample_by_partition(num_nodes, partition_number, train_ratio, valid_ratio, test_ratio)
            train_split.extend([eligible_stats_value[train_index] for train_index in train_indices])
            valid_split.extend([eligible_stats_value[valid_index] for valid_index in valid_indices])
            test_split.extend([eligible_stats_value[test_index] for test_index in test_indices])

    logger.info(f'Split Finished - Train: {len(train_split)}; Valid: {len(valid_split)}; Test: {len(test_split)};')
    meta = dict(
        tasks = list(tasks),
        datasets = list(datasets),
        splits = list(splits),
        metrics = list(metrics),
        operators = list(operators),
    )
    logger.info(f'Details - Tasks: {len(tasks)}; Datasets: {len(datasets)}; Splits: {len(splits)}; Metrics: {len(metrics)}; Operators: {len(operators)};')

    train_split_meta = dict(
        split = f'train',
        archive = f'train.tar.gz',
        version = f'{version}',
        size = len(train_split),
        url = "",
    )
    train_split_meta.update(meta)
    save_split(train_split_meta, train_split, dataset_dirpath, save_dirpath, worker_number)

    valid_split_meta = dict(
        split = f'valid',
        archive = f'valid.tar.gz',
        version = f'{version}',
        size = len(valid_split),
        url = "",
    )
    valid_split_meta.update(meta)
    save_split(valid_split_meta, valid_split, dataset_dirpath, save_dirpath, worker_number)

    test_split_meta = dict(
        split = f'test',
        archive = f'test.tar.gz',
        version = f'{version}',
        size = len(test_split),
        url = "",
    )
    test_split_meta.update(meta)
    save_split(test_split_meta, test_split, dataset_dirpath, save_dirpath, worker_number)
