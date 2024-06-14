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


import tqdm
import numpy
import pathlib
import multiprocessing

from typing import Any

from younger.commons.io import load_json, save_json, save_pickle, create_dir, tar_archive
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_operator_origin


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
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split {graph_dirpath.absolute()} ... ')
    parameters = [
        (
            graph_dirpath.joinpath(f'sample-{i}.pkl'),
            instance,
        ) for i, instance in enumerate(instances)
    ]
    with tqdm.tqdm(total=len(parameters), desc='Saving') as progress_bar:
        for index, parameter in enumerate((parameters), start=1):
            save_graph(parameter)
            progress_bar.update(1)
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split Tar {archive_filepath.absolute()} ... ')
    tar_archive(
        [graph_dirpath.joinpath(f'sample-{i}.pkl') for i, _ in enumerate(instances)],
        archive_filepath,
        compress=True
    )
    logger.info(f'Saved.')


def clean_split(split: list[Instance], task_dict: dict[str, int], node_dict: dict[str, dict[str, int]], operator_dict: dict[str, dict[str, int]]):
    task_dict_keys = set(task_dict.keys())
    node_dict_keys = set(node_dict['onnx'].keys()) | set(node_dict['others'].keys())
    operator_dict_keys = set(operator_dict['onnx'].keys()) | set(operator_dict['others'].keys())
    cleaned_split = []
    with tqdm.tqdm(total=len(split), desc='Cleaning') as progress_bar:
        for index, instance in enumerate(split, start=1):
            instance_tasks, instance_nodes, instance_operators = check_instance(instance)
            instance_task_keys = set(instance_tasks.keys())
            instance_node_keys = set(instance_nodes['onnx'].keys()) | set(instance_nodes['others'].keys())
            instance_operator_keys = set(instance_operators['onnx'].keys()) | set(instance_operators['others'].keys())
            if len(instance_task_keys - task_dict_keys) != 0 or len(instance_node_keys - node_dict_keys) != 0 or len(instance_operator_keys - operator_dict_keys) != 0:
                pass
            else:
                cleaned_split.append(instance)
            progress_bar.update(1)
            progress_bar.set_postfix({f'Current Cleaned': f'{len(cleaned_split)}'})
    return cleaned_split


def update_dict_count(origin_dict: dict[str, int], other_dict: dict[str, int]) -> dict[str, int]:
    for key, value in other_dict.items():
        count = origin_dict.get(key, 0)
        origin_dict[key] = count + value
    
    return origin_dict


def check_instance(instance: Instance) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    instance_tasks: dict[str, int] = {task: 1 for task in instance.labels['tasks']}

    instance_nodes: dict[str, dict[str, int]] = dict()
    instance_nodes['onnx'] = dict()
    instance_nodes['others'] = dict()

    instance_operators: dict[str, dict[str, int]] = dict()
    instance_operators['onnx'] = dict()
    instance_operators['others'] = dict()

    for node_index in instance.network.graph.nodes():
        node_features = instance.network.graph.nodes[node_index]['features']
        node_origin = get_operator_origin(node_features['operator']['op_type'], domain=node_features['operator']['domain'])
        if node_origin != 'onnx':
            node_origin = 'others'

        # TODO: All attribute value of operators in domain 'ai.onnx.ml' are omited default. Because its loooong value may stuck the process.
        node_identifier = Network.get_node_identifier_from_features(node_features, mode='full')
        node_count = instance_nodes[node_origin].get(node_identifier, 0)
        instance_nodes[node_origin][node_identifier] = node_count + 1

        operator_identifier = Network.get_node_identifier_from_features(node_features, mode='type')
        operator_count = instance_operators[node_origin].get(operator_identifier, 0)
        instance_operators[node_origin][operator_identifier] = operator_count + 1

    return instance_tasks, instance_nodes, instance_operators


def extract_dict(split: list[Instance], worker_number: int) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    all_tasks: dict[str, int] = dict()

    all_nodes: dict[str, dict[str, int]] = dict()
    all_nodes['onnx'] = dict()
    all_nodes['others'] = dict()

    all_operators: dict[str, dict[str, int]] = dict()
    all_operators['onnx'] = dict()
    all_operators['others'] = dict()

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(split), desc='Extracting') as progress_bar:
            for index, result in enumerate(pool.imap_unordered(check_instance, split), start=1):
                (instance_tasks, instance_nodes, instance_operators) = result

                all_tasks = update_dict_count(all_tasks, instance_tasks)

                all_nodes['onnx']   = update_dict_count(all_nodes['onnx'],   instance_nodes['onnx'])
                all_nodes['others'] = update_dict_count(all_nodes['others'], instance_nodes['others'])

                all_operators['onnx']   = update_dict_count(all_operators['onnx'],   instance_operators['onnx'])
                all_operators['others'] = update_dict_count(all_operators['others'], instance_operators['others'])

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        f'Task': f'{len(all_tasks)}',
                        f'ONNX Node{{OP}}': f'{len(all_nodes["onnx"])}{{{len(all_operators["onnx"])}}}',
                        f'Others Node{{OP}}': f'{len(all_nodes["others"])}{{{len(all_operators["others"])}}}',
                    }
                )
    return all_tasks, all_nodes, all_operators


def sample_by_partition(examples: list, partition_number: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[list[int], list[int], list[int]]:
    assert train_ratio + valid_ratio + test_ratio == 1
    # examples should be a sortable list
    sorted_indices = sorted(range(len(examples)), key=lambda index: examples[index])

    partition_size = len(sorted_indices) // partition_number
    remainder_size = len(sorted_indices) %  partition_number

    ratios = [train_ratio, valid_ratio, test_ratio]
    splits = ['Train', 'Valid', 'Test']
    if any(partition_size * ratio < 1 for ratio in ratios):
        logger.warn(f'The number of partition is too large - {partition_number}. Please lower it or the split {splits[numpy.argmin(ratios)]} may be 0.')

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

    return sorted(train_indices), sorted(valid_indices), sorted(test_indices)


def select_instance(parameter: tuple[pathlib.Path, set[str], str]) -> tuple[Instance, tuple[int, int]] | None:
    path, tasks, metric_name, node_size_lbound, node_size_ubound, edge_size_lbound, edge_size_ubound = parameter

    instance = Instance()
    instance.load(path)
    instance_downloads = instance.labels['downloads']
    instance_model_names = instance.labels['model_name']
    instance_likes = instance.labels['likes']
    instance_tasks = set(instance.labels['tags'])

    eval_metrics: dict[str, list[float]] = dict()
    for eval_task, eval_dataset_name, eval_dataset_split, eval_metric_name, eval_metric_value in instance.labels['evaluations']:
        instance_tasks.add(eval_task)
        eval_metric_values = eval_metrics.get(eval_metric_name, list())
        eval_metric_values.append(float(eval_metric_value))
        eval_metrics[eval_metric_name] = eval_metric_values

    instance_hash = instance.labels['hash']

    instance.clean_labels()

    # Check Metric Name
    if metric_name and metric_name not in eval_metrics:
        return None

    # Check Metric Name
    instance_size = (instance.network.graph.number_of_nodes(), instance.network.graph.number_of_edges())
    node_size_lbound = instance_size[0] if node_size_lbound is None else node_size_lbound
    node_size_ubound = instance_size[0] if node_size_ubound is None else node_size_ubound

    edge_size_lbound = instance_size[1] if edge_size_lbound is None else edge_size_lbound
    edge_size_ubound = instance_size[1] if edge_size_ubound is None else edge_size_ubound
    if instance_size[0] < node_size_lbound or node_size_ubound < instance_size[0] or instance_size[1] < edge_size_lbound or edge_size_ubound < instance_size[1]:
        return None

    instance_tasks = list(instance_tasks & tasks)
    instance_metrics = eval_metrics.get(metric_name, list())
    instance.setup_labels(dict(model_names = instance_model_names, downloads=instance_downloads, likes=instance_likes, tasks=instance_tasks, metrics=instance_metrics, hash=instance_hash))

    return instance, instance_size


def main(
    tasks_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    version: str,
    silly: bool = True,
    metric_name: str | None = None,
    node_size_lbound: int | None = None, node_size_ubound: int | None = None,
    edge_size_lbound: int | None = None, edge_size_ubound: int | None = None,
    train_proportion: int = 80, valid_proportion: int = 10, test_proportion: int = 10,
    partition_number: int = 10,
    worker_number: int = 4,
    seed: int = 16861,
):
    # 0. Each graph of the dataset MUST be standardized graph
    # 1. Tasks File should be a *.json file, which contains an list of tasks (list[str]) (It can be an empty list)
    # 2. Usually we do not clean instances with theory value range of metric.
    # For example:
    # WER maybe larger than 1 and Word Accuracy maybe smaller than 0 in ASR research area.

    numpy.random.seed(seed)

    assert train_proportion + valid_proportion + test_proportion == 100
    total_proportion = train_proportion + valid_proportion + test_proportion
    train_ratio = train_proportion / total_proportion
    valid_ratio = valid_proportion / total_proportion
    test_ratio = test_proportion / total_proportion
    logger.info(f'Split Ratio - Train/Valid/Test = {train_ratio:.2f} / {valid_ratio:.2f} / {test_ratio:.2f}')

    tasks: set[str] = set(load_json(tasks_filepath))
    logger.info(f'Total Tasks = {len(tasks)}')

    # Instance.labels - {tasks: [task_1, task_2]; metric_values: [value_1, value_2]}
    logger.info(f'Checking Existing Instances ...')
    paths = sorted([path for path in dataset_dirpath.iterdir()])
    parameters = list()
    for path in paths:
        parameters.append((path, tasks, metric_name, node_size_lbound, node_size_ubound, edge_size_lbound, edge_size_ubound))
    logger.info(f'Total Instances: {len(parameters)}')

    logger.info(f'Selecting Instances (Metric Name = {metric_name}; Node/Edge Size Bound = ({node_size_lbound}, {node_size_ubound})/({edge_size_lbound}, {edge_size_ubound}) ...')
    instances = list()
    all_size = list()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Selecting') as progress_bar:
            for index, result in enumerate(pool.imap_unordered(select_instance, parameters), start=1):
                if result:
                    (instance, instance_size) = result
                    instances.append(instance)
                    all_size.append(instance_size)
                progress_bar.update(1)
                progress_bar.set_postfix({f'Current Selected': f'{len(instances)}'})
    sorted_indices = sorted(range(len(instances)), key=lambda index: instances[index].labels['hash'])
    instances = [instances[index] for index in sorted_indices]
    all_size = [all_size[index] for index in sorted_indices]
    logger.info(f'Total Selected Instances: {len(instances)}')

    node_nums = [node_num for node_num, edge_num in all_size]
    edge_nums = [edge_num for node_num, edge_num in all_size]
    logger.info(f'Node Num - Max/Min = {max(node_nums)} / {min(node_nums)}')
    logger.info(f'Edge Num - Max/Min = {max(edge_nums)} / {min(edge_nums)}')

    train_indices, valid_indices, test_indices = sample_by_partition(all_size, partition_number, train_ratio, valid_ratio, test_ratio)
    train_split = [instances[train_index] for train_index in train_indices]
    valid_split = [instances[valid_index] for valid_index in valid_indices]
    test_split = [instances[test_index] for test_index in test_indices]
    logger.info(f'Split Finished - Train: {len(train_split)}; Valid: {len(valid_split)}; Test: {len(test_split)};')
    logger.info(f' - First 10 Train Index: {train_indices[:10]}')
    logger.info(f' - First 10 Valid Index: {valid_indices[:10]}')
    logger.info(f' - First 10 Test  Index: {test_indices[:10]}')

    logger.info(f'Extract Dictionaries From Train Split: Task_Dict & Node_Dict')
    train_task_dict, train_node_dict, train_operator_dict = extract_dict(train_split, worker_number)
    if silly:
        valid_split = clean_split(valid_split, train_task_dict, train_node_dict, train_operator_dict)
        test_split = clean_split(test_split, train_task_dict, train_node_dict, train_operator_dict)
        logger.info(f'-!!!-New-!!!-')
        logger.info(f'Clean Split Finished - Train: {len(train_split)}; Valid: {len(valid_split)}; Test: {len(test_split)};')
        logger.info(f' - First 10 Train Index: {train_indices[:10]}')
        logger.info(f' - First 10 Valid Index: {valid_indices[:10]}')
        logger.info(f' - First 10 Test  Index: {test_indices[:10]}')

    train_task_dict_keys = set(train_task_dict.keys())
    train_node_dict_keys = set(train_node_dict['onnx'].keys()) | set(train_node_dict['others'].keys())
    train_operator_dict_keys = set(train_operator_dict['onnx'].keys()) | set(train_operator_dict['others'].keys())
    logger.info(f'Extracted - Task_Dict: {len(train_task_dict_keys)};')
    logger.info(f'Extracted - ONNX & Others Node_Dict: {len(train_node_dict["onnx"])} & {len(train_node_dict["others"])};')
    logger.info(f'Extracted - ONNX & Others Operator_Dict: {len(train_operator_dict["onnx"])} & {len(train_operator_dict["others"])};')
    logger.info(f'Checking Unknown [Tasks & Nodes & Operators] In Valid & Test Splits.')

    logger.info(f'Valid Split:')
    valid_task_dict, valid_node_dict, valid_operator_dict = extract_dict(valid_split, worker_number)

    valid_task_dict_keys = set(valid_task_dict.keys())
    valid_node_dict_keys = set(valid_node_dict['onnx'].keys()) | set(valid_node_dict['others'].keys())
    valid_operator_dict_keys = set(valid_operator_dict['onnx'].keys()) | set(valid_operator_dict['others'].keys())

    num_valid_unknown_tasks = len(valid_task_dict_keys - train_task_dict_keys)
    num_valid_unknown_nodes = len(valid_node_dict_keys - train_node_dict_keys)
    num_valid_unknown_operators = len(valid_operator_dict_keys - train_operator_dict_keys)

    num_onnx_valid_unknown_nodes = len(set(valid_node_dict['onnx'].keys()) - set(train_node_dict['onnx'].keys()))
    num_others_valid_unknown_nodes = len(set(valid_node_dict['others'].keys()) - set(train_node_dict['others'].keys()))

    num_onnx_valid_unknown_operators = len(set(valid_operator_dict['onnx'].keys()) - set(train_operator_dict['onnx'].keys()))
    num_others_valid_unknown_operators = len(set(valid_operator_dict['others'].keys()) - set(train_operator_dict['others'].keys()))

    percent_tasks_valid = num_valid_unknown_tasks / len(valid_task_dict_keys) * 100 if len(valid_task_dict_keys) != 0 else 0

    percent_nodes_valid = num_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0
    percent_onnx_nodes_valid = num_onnx_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0
    percent_others_nodes_valid = num_others_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0

    percent_operators_valid = num_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0
    percent_onnx_operators_valid = num_onnx_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0
    percent_others_operators_valid = num_others_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0

    logger.info(f'Valid Unknown [ Number / Ratio ]:')
    logger.info(f' - Tasks = [ {num_valid_unknown_tasks} & {percent_tasks_valid:.2f}% ]')
    logger.info(f' - Nodes = [ {num_valid_unknown_nodes} ({num_onnx_valid_unknown_nodes}/{num_others_valid_unknown_nodes}) & {percent_nodes_valid:.2f}% ({percent_onnx_nodes_valid}/{percent_others_nodes_valid}) ]')
    logger.info(f' - Operators = [ {num_valid_unknown_operators} ({num_onnx_valid_unknown_operators}/{num_others_valid_unknown_operators}) & {percent_operators_valid:.2f}% ({percent_onnx_operators_valid}/{percent_others_operators_valid}) ]')

    logger.info(f'Test Split:')
    test_task_dict, test_node_dict, test_operator_dict = extract_dict(test_split, worker_number)

    test_task_dict_keys = set(test_task_dict.keys())
    test_node_dict_keys = set(test_node_dict['onnx'].keys()) | set(test_node_dict['others'].keys())
    test_operator_dict_keys = set(test_operator_dict['onnx'].keys()) | set(test_operator_dict['others'].keys())

    num_test_unknown_tasks = len(test_task_dict_keys - train_task_dict_keys)

    num_test_unknown_nodes = len(test_node_dict_keys - train_node_dict_keys)
    num_onnx_test_unknown_nodes = len(set(test_node_dict['onnx'].keys()) - set(train_node_dict['onnx'].keys()))
    num_others_test_unknown_nodes = len(set(test_node_dict['others'].keys()) - set(train_node_dict['others'].keys()))

    num_test_unknown_operators = len(test_operator_dict_keys - train_operator_dict_keys)
    num_onnx_test_unknown_operators = len(set(test_operator_dict['onnx'].keys()) - set(train_operator_dict['onnx'].keys()))
    num_others_test_unknown_operators = len(set(test_operator_dict['others'].keys()) - set(train_operator_dict['others'].keys()))

    percent_tasks_test = num_test_unknown_tasks / len(test_task_dict_keys) * 100 if len(test_task_dict_keys) != 0 else 0

    percent_nodes_test = num_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0
    percent_onnx_nodes_test = num_onnx_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0
    percent_others_nodes_test = num_others_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0

    percent_operators_test = num_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0
    percent_onnx_operators_test = num_onnx_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0
    percent_others_operators_test = num_others_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0

    logger.info(f'Test Unknown  [ Number / Ratio ]:')
    logger.info(f' - Tasks = [ {num_test_unknown_tasks} & {percent_tasks_test:.2f}% ]')
    logger.info(f' - Nodes = [ {num_test_unknown_nodes} ({num_onnx_test_unknown_nodes}/{num_others_test_unknown_nodes}) & {percent_nodes_test:.2f}% ({percent_onnx_nodes_test}/{percent_others_nodes_test}) ]')
    logger.info(f' - Operators = [ {num_test_unknown_operators} ({num_onnx_test_unknown_operators}/{num_others_test_unknown_operators}) & {percent_operators_test:.2f}% ({percent_onnx_operators_test}/{percent_others_operators_test}) ]')

    meta = dict(
        metric_name = metric_name,
        all_tasks = train_task_dict,
        all_nodes = train_node_dict,
        all_operators = train_operator_dict,
    )

    if len(train_split):
        train_split_meta = dict(
            split = f'train',
            archive = f'train.tar.gz',
            version = f'{version}',
            size = len(train_split),
            url = "",
        )
        train_split_meta.update(meta)
        save_split(train_split_meta, train_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Train Split')

    if len(valid_split):
        valid_split_meta = dict(
            split = f'valid',
            archive = f'valid.tar.gz',
            version = f'{version}',
            size = len(valid_split),
            url = "",
        )
        valid_split_meta.update(meta)
        save_split(valid_split_meta, valid_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Valid Split')

    if len(valid_split):
        test_split_meta = dict(
            split = f'test',
            archive = f'test.tar.gz',
            version = f'{version}',
            size = len(test_split),
            url = "",
        )
        test_split_meta.update(meta)
        save_split(test_split_meta, test_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Test Split')