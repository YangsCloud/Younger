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
import tqdm
import numpy
import pathlib
import multiprocessing

from typing import Any

from younger.commons.io import load_json, save_json, save_pickle, create_dir, tar_archive
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network


def save_graph(parameter: tuple[pathlib.Path, pathlib.Path, dict[str, Any]]):
    save_filepath, instance = parameter
    graph = instance.network.graph
    graph.graph.update(instance.labels)
    save_pickle(graph, save_filepath)


def save_split(meta: dict[str, Any], instances: list[Instance], save_dirpath: pathlib.Path, worker_number: int):
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
    parameters = [
        (
            graph_dirpath.joinpath(f'sample-{i}.pkl'),
            instance,
        ) for i, instance in enumerate(instances)
    ]
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters)) as progress_bar:
            for index, _ in enumerate(pool.imap_unordered(save_graph, parameters), start=1):
                progress_bar.update(1)

    logger.info(f'Saving \'{split}\' Split Tar {archive_filepath.absolute()} ... ')
    tar_archive(
        [graph_dirpath.joinpath(f'sample-{i}.pkl') for i, _ in enumerate(instances)],
        archive_filepath,
        compress=True
    )


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


def process_instance(parameter: tuple[pathlib.Path, set[str], str]) -> tuple[Instance, list[int], set[str], set[str]] | None:
    path, tasks, metric_name = parameter
    instance = Instance()
    instance.load(path)
    instance_downloads = instance.labels['downloads']
    instance_likes = instance.labels['likes']
    instance_tasks = set(instance.labels['tags']) if instance.labels['tags'] else set()
    eval_metrics: dict[str, list[float]] = dict()
    for eval_task, eval_dataset_name, eval_dataset_split, eval_metric_name, eval_metric_value in instance.labels['evaluations']:
        instance_tasks.add(eval_task)
        eval_metric_values = eval_metrics.get(eval_metric_name, list())
        eval_metric_values.append(float(eval_metric_value))
        eval_metrics[eval_metric_name] = eval_metric_values

    instance.clean_labels()
    if metric_name and metric_name not in eval_metrics:
        return None
    instance_tasks = list(instance_tasks & tasks)
    instance_metrics = eval_metrics.get(metric_name, list())
    instance.setup_labels(dict(downloads=instance_downloads, likes=instance_likes, tasks=instance_tasks, metrics=instance_metrics))

    instance_size = instance.network.graph.number_of_nodes()

    
    return instance, instance_size


def generate_dict(instance: Instance) -> tuple[set[str], set[str]]:
    instance_tasks: set[str] = set(instance.labels['tasks'])
    instance_nodes: set[str] = set()
    for node_index in instance.network.graph.nodes():
        node_features = instance.network.graph.nodes[node_index]['features']
        node_identifier = Network.standardized_node_identifier(node_features)
        instance_nodes.add(node_identifier)
    return instance_tasks, instance_nodes


def main(
    tasks_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    version: str,
    metric_name: str | None = None,
    train_proportion: int = 80, valid_proportion: int = 10, test_proportion: int = 10,
    partition_number: int = 10,
    worker_number: int = 4,
):
    # 0. Each graph of the dataset MUST be standardized graph
    # 1. Tasks File should be a *.json file, which contains an list of tasks (list[str]) (It can be an empty list)
    # 2. Usually we do not clean instances with theory value range of metric.
    # For example:
    # WER maybe larger than 1 and Word Accuracy maybe smaller than 0 in ASR research area.

    assert train_proportion + valid_proportion + test_proportion == 100

    total_proportion = train_proportion + valid_proportion + test_proportion
    train_ratio = train_proportion / total_proportion
    valid_ratio = valid_proportion / total_proportion
    test_ratio = test_proportion / total_proportion

    tasks: set[str] = set(load_json(tasks_filepath))


    # Instance.labels - {tasks: [task_1, task_2]; metric_values: [value_1, value_2]; metric_name: metric_name}
    logger.info(f'Checking Valid Instances ...')
    instances = list()
    parameters = list()
    for path in dataset_dirpath.iterdir():
        parameters.append((path, tasks, metric_name))
    logger.info(f'Total Instances: {len(parameters)}')

    all_size = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters)) as progress_bar:
            for index, result in enumerate(pool.imap_unordered(process_instance, parameters), start=1):
                if result:
                    (instance, instance_size) = result
                    instances.append(instance)
                    all_size.append(instance_size)
                progress_bar.update(1)

    logger.info(f'Total Valid Instances: {len(instances)}')

    train_indices, valid_indices, test_indices = sample_by_partition(all_size, partition_number, train_ratio, valid_ratio, test_ratio)
    train_split = [instances[train_index] for train_index in train_indices]
    valid_split = [instances[valid_index] for valid_index in valid_indices]
    test_split = [instances[test_index] for test_index in test_indices]

    logger.info(f'Split Finished - Train: {len(train_split)}; Valid: {len(valid_split)}; Test: {len(test_split)};')

    all_tasks = set()
    all_nodes = set()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(train_split)) as progress_bar:
            for index, result in enumerate(pool.imap_unordered(generate_dict, train_split), start=1):
                (instance_tasks, instance_nodes) = result
                all_tasks.update(instance_tasks)
                all_nodes.update(instance_nodes)
                progress_bar.update(1)
    logger.info(f'Details - Tasks: {len(all_tasks)}; Nodes: {len(all_nodes)};')

    meta = dict(
        metric_name = metric_name,
        all_tasks = list(all_tasks),
        all_nodes = list(all_nodes),
    )

    train_split_meta = dict(
        split = f'train',
        archive = f'train.tar.gz',
        version = f'{version}',
        size = len(train_split),
        url = "",
    )
    train_split_meta.update(meta)
    save_split(train_split_meta, train_split, save_dirpath, worker_number)

    valid_split_meta = dict(
        split = f'valid',
        archive = f'valid.tar.gz',
        version = f'{version}',
        size = len(valid_split),
        url = "",
    )
    valid_split_meta.update(meta)
    save_split(valid_split_meta, valid_split, save_dirpath, worker_number)

    test_split_meta = dict(
        split = f'test',
        archive = f'test.tar.gz',
        version = f'{version}',
        size = len(test_split),
        url = "",
    )
    test_split_meta.update(meta)
    save_split(test_split_meta, test_split, save_dirpath, worker_number)
