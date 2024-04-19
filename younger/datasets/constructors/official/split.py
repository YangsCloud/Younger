#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 21:13
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import ast
import numpy
import pathlib

from younger.commons.io import load_json, save_json, tar_archive
from younger.commons.logging import logger

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


def main(
    statistics_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    version: str,
    train_proportion: int = 80, valid_proportion: int = 10, test_proportion: int = 10,
    partition_number: int = 10,
):
    # stats is the outputs of the statistics.main
    assert train_proportion + valid_proportion + test_proportion == 100

    total_proportion = train_proportion + valid_proportion + test_proportion
    train_ratio = train_proportion / total_proportion
    valid_ratio = valid_proportion / total_proportion
    test_ratio = test_proportion / total_proportion
    statistics = load_json(statistics_filepath)

    tasks = set()
    datasets = set()
    splits = set()
    metrics = set()
    operators = set()

    train_set = list()
    valid_set = list()
    test_set = list()
    for stats_key, stats_value in statistics.items():
        task, dataset, split, metric = ast.literal_eval(stats_key)
        metric_range = get_metric_theroy_range(metric)
        if metric_range in {(-1, 1), (0, 1), (0, 100)}:
            lower_bound, upper_bound = 0, 1
        else:
            lower_bound, upper_bound = metric_range
        eligible_stats_value = list()
        for ins_stats in stats_value:
            if 'metric_value' in ins_stats:
                metric_value = ins_stats['metric_value']
                if (lower_bound is not None and metric_value < lower_bound) or (upper_bound is not None and upper_bound < metric_value):
                    logger.info(f'Outlier: {metric_value}. Skip The Instance: {ins_stats["model_name"]}')
                    continue
            eligible_stats_value.append(ins_stats)
        if len(eligible_stats_value) * min(train_ratio, valid_ratio, test_ratio) < 1:
            continue
        else:
            num_nodes = [eligible_instance_stats_value['graph_stats']['num_node'] for eligible_instance_stats_value in eligible_stats_value]
            train_indices, valid_indices, test_indices = sample_by_partition(num_nodes, partition_number, train_ratio, valid_ratio, test_ratio)
            train_set.extend([eligible_stats_value[train_index]['model_name'] for train_index in train_indices])
            valid_set.extend([eligible_stats_value[valid_index]['model_name'] for valid_index in valid_indices])
            test_set.extend([eligible_stats_value[test_index]['model_name'] for test_index in test_indices])

            tasks.add(task)
            datasets.add(dataset)
            splits.add(split)
            metrics.add(metric)
            for eligible_instance_stats_value in eligible_stats_value:
                operators.update(eligible_instance_stats_value['graph_stats']['num_operators'].keys())

    meta = dict(
        tasks = list(tasks),
        datsets = list(datasets),
        splits = list(splits),
        metrics = list(metrics),
        operators = list(operators),
    )
    train_split_meta = dict(
        version = f'{version}',
        archive = f'{version}_train.tar.gz',
        instance_names = train_set,
    )
    train_split_meta.update(meta)
    save_json(train_split_meta, save_dirpath.joinpath(f'{version}_train.json'), indent=2)
    tar_archive(
        [dataset_dirpath.joinpath(instance_name) for instance_name in train_split_meta['instance_names']],
        save_dirpath.joinpath(train_split_meta['archive']),
        compress=True
    )

    valid_split_meta = dict(
        version = f'{version}',
        archive = f'{version}_valid.tar.gz',
        instance_names = valid_set,
    )
    valid_split_meta.update(meta)
    save_json(valid_split_meta, save_dirpath.joinpath(f'{version}_valid.json'), indent=2)
    tar_archive(
        [dataset_dirpath.joinpath(instance_name) for instance_name in valid_split_meta['instance_names']],
        save_dirpath.joinpath(valid_split_meta['archive']),
        compress=True
    )

    test_split_meta = dict(
        version = f'{version}',
        archive = f'{version}_test.tar.gz',
        instance_names = test_set,
    )
    test_split_meta.update(meta)
    save_json(test_split_meta, save_dirpath.joinpath(f'{version}_test.json'), indent=2)
    tar_archive(
        [dataset_dirpath.joinpath(instance_name) for instance_name in test_split_meta['instance_names']],
        save_dirpath.joinpath(test_split_meta['archive']),
        compress=True
    )
