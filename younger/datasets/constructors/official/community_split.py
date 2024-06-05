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
import random
import pathlib
import networkx
import multiprocessing

from typing import Any, Literal

from younger.commons.io import load_json, save_json, save_pickle, create_dir, tar_archive
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_operator_origin


def save_item(parameter: tuple[pathlib.Path, tuple[str, tuple[networkx.DiGraph, set]]]):
    save_filepath, item = parameter
    save_pickle(item, save_filepath)


def save_split(meta: dict[str, Any], selected_subgraph_with_labels: list[tuple[str, tuple[networkx.DiGraph, set]]], save_dirpath: pathlib.Path, worker_number: int):
    version_dirpath = save_dirpath.joinpath(meta['version'])
    split_dirpath = version_dirpath.joinpath(meta['split'])
    archive_filepath = split_dirpath.joinpath(meta['archive'])
    split = meta['split']

    item_dirpath = split_dirpath.joinpath('item')
    meta_filepath = split_dirpath.joinpath('meta.json')
    create_dir(item_dirpath)

    logger.info(f'Saving \'{split}\' Split META {meta_filepath.absolute()} ... ')
    save_json(meta, meta_filepath, indent=2)
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split {item_dirpath.absolute()} ... ')
    parameters = [
        (
            item_dirpath.joinpath(f'sample-{i}.pkl'),
            item,
        ) for i, item in enumerate(selected_subgraph_with_labels)
    ]
    with tqdm.tqdm(total=len(parameters), desc='Saving') as progress_bar:
        for index, parameter in enumerate((parameters), start=1):
            save_item(parameter)
            progress_bar.update(1)
    logger.info(f'Saved.')

    logger.info(f'Saving \'{split}\' Split Tar {archive_filepath.absolute()} ... ')
    tar_archive(
        [item_dirpath.joinpath(f'sample-{i}.pkl') for i, _ in enumerate(selected_subgraph_with_labels)],
        archive_filepath,
        compress=True
    )
    logger.info(f'Saved.')


def clean_split(split: list[tuple[str, networkx.DiGraph, set]], node_dict: dict[str, dict[str, int]], operator_dict: dict[str, dict[str, int]]):
    node_dict_keys = set(node_dict['onnx'].keys()) | set(node_dict['others'].keys())
    operator_dict_keys = set(operator_dict['onnx'].keys()) | set(operator_dict['others'].keys())
    cleaned_split = []
    with tqdm.tqdm(total=len(split), desc='Cleaning') as progress_bar:
        for index, item in enumerate(split, start=1):
            subgraph_nodes, subgraph_operators = check_subgraph(item)
            subgraph_node_keys = set(subgraph_nodes['onnx'].keys()) | set(subgraph_nodes['others'].keys())
            subgraph_operator_keys = set(subgraph_operators['onnx'].keys()) | set(subgraph_operators['others'].keys())
            if len(subgraph_node_keys - node_dict_keys) != 0 or len(subgraph_operator_keys - operator_dict_keys) != 0:
                pass
            else:
                cleaned_split.append(item)
            progress_bar.update(1)
            progress_bar.set_postfix({f'Current Cleaned': f'{len(cleaned_split)}'})
    return cleaned_split


def update_dict_count(origin_dict: dict[str, int], other_dict: dict[str, int]) -> dict[str, int]:
    for key, value in other_dict.items():
        count = origin_dict.get(key, 0)
        origin_dict[key] = count + value
    
    return origin_dict


def check_subgraph(parameter: tuple[str, networkx.DiGraph, set]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    subgraph_hash, subgraph, boundary = parameter
    subgraph_nodes: dict[str, dict[str, int]] = dict()
    subgraph_nodes['onnx'] = dict()
    subgraph_nodes['others'] = dict()

    subgraph_operators: dict[str, dict[str, int]] = dict()
    subgraph_operators['onnx'] = dict()
    subgraph_operators['others'] = dict()

    for node_index in subgraph.nodes():
        node_features = subgraph.nodes[node_index]['features']
        node_origin = get_operator_origin(node_features['operator']['op_type'], domain=node_features['operator']['domain'])
        if node_origin != 'onnx':
            node_origin = 'others'

        # TODO: All attribute value of operators in domain 'ai.onnx.ml' are omited default. Because its loooong value may stuck the process.
        node_identifier = Network.get_node_identifier_from_features(node_features, mode='full')
        node_count = subgraph_nodes[node_origin].get(node_identifier, 0)
        subgraph_nodes[node_origin][node_identifier] = node_count + 1

        operator_identifier = Network.get_node_identifier_from_features(node_features, mode='type')
        operator_count = subgraph_operators[node_origin].get(operator_identifier, 0)
        subgraph_operators[node_origin][operator_identifier] = operator_count + 1

    return subgraph_nodes, subgraph_operators


def extract_dict(split: list[tuple[str, networkx.DiGraph, set]], worker_number: int) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    all_nodes: dict[str, dict[str, int]] = dict()
    all_nodes['onnx'] = dict()
    all_nodes['others'] = dict()

    all_operators: dict[str, dict[str, int]] = dict()
    all_operators['onnx'] = dict()
    all_operators['others'] = dict()

    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(split), desc='Extracting') as progress_bar:
            for index, result in enumerate(pool.imap_unordered(check_subgraph, split), start=1):
                (subgraph_nodes, subgraph_operators) = result

                all_nodes['onnx']   = update_dict_count(all_nodes['onnx'],   subgraph_nodes['onnx'])
                all_nodes['others'] = update_dict_count(all_nodes['others'], subgraph_nodes['others'])

                all_operators['onnx']   = update_dict_count(all_operators['onnx'],   subgraph_operators['onnx'])
                all_operators['others'] = update_dict_count(all_operators['others'], subgraph_operators['others'])

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        f'ONNX Node{{OP}}': f'{len(all_nodes["onnx"])}{{{len(all_operators["onnx"])}}}',
                        f'Others Node{{OP}}': f'{len(all_nodes["others"])}{{{len(all_operators["others"])}}}',
                    }
                )
    return all_nodes, all_operators


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


def get_communities(graph: networkx.DiGraph) -> list[tuple[networkx.DiGraph, set, str]]:
    communities = list(networkx.community.greedy_modularity_communities(graph, resolution=1, cutoff=1))

    all_subgraph_with_labels = list()
    for community in communities:
        if len(community) == 0:
            continue
        boundary = networkx.node_boundary(graph, community)
        if len(boundary) == 0:
            continue
        subgraph: networkx.DiGraph = networkx.subgraph(graph, community | boundary).copy()

        subgraph_hash = Network.hash(subgraph, node_attr='features')

        # cleansed_subgraph = networkx.DiGraph()
        # cleansed_subgraph.add_nodes_from(subgraph.nodes(data=True))
        # cleansed_subgraph.add_edges_from(subgraph.edges(data=True))
        # for node_index in cleansed_subgraph.nodes():
        #     cleansed_subgraph.nodes[node_index]['operator'] = cleansed_subgraph.nodes[node_index]['features']['operator']
        # subgraph_hash = Network.hash(cleansed_subgraph, node_attr='operator')

        all_subgraph_with_labels.append((subgraph, boundary, subgraph_hash))
    return all_subgraph_with_labels


def select_subgraphs(parameter: tuple[pathlib.Path, set[str], str]) -> tuple[Instance, tuple[int, int], list[tuple[networkx.DiGraph, set, str]]] | None:
    path, node_size_lbound, node_size_ubound, edge_size_lbound, edge_size_ubound = parameter

    instance = Instance()
    instance.load(path)

    if instance.network.graph.number_of_nodes() < 1 or instance.network.graph.number_of_edges() < 1:
        return None

    all_subgraph_with_boundary = get_communities(instance.network.graph)

    valid_all_subgraph_with_boundary = list()
    for subgraph_with_boundary in all_subgraph_with_boundary:
        subgraph = subgraph_with_boundary[0]
        subgraph_size = (subgraph.number_of_nodes(), subgraph.number_of_edges())
        node_size_lbound = subgraph_size[0] if node_size_lbound is None else node_size_lbound
        node_size_ubound = subgraph_size[0] if node_size_ubound is None else node_size_ubound

        edge_size_lbound = subgraph_size[1] if edge_size_lbound is None else edge_size_lbound
        edge_size_ubound = subgraph_size[1] if edge_size_ubound is None else edge_size_ubound
        if subgraph_size[0] < node_size_lbound or node_size_ubound < subgraph_size[0] or subgraph_size[1] < edge_size_lbound or edge_size_ubound < subgraph_size[1]:
            continue
        valid_all_subgraph_with_boundary.append(subgraph_with_boundary)

    return valid_all_subgraph_with_boundary


def main(
    tasks_filepath: pathlib.Path, dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    version: str,
    silly: bool = True,
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
        parameters.append((path, node_size_lbound, node_size_ubound, edge_size_lbound, edge_size_ubound))
    logger.info(f'Total Instances: {len(parameters)}')

    logger.info(f'Selecting Subgraphs (Node/Edge Size Bound = ({node_size_lbound}, {node_size_ubound})/({edge_size_lbound}, {edge_size_ubound}) ...')
    unique_subgraph_with_boundary: dict[str, tuple[networkx.DiGraph, set]] = dict()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Selecting') as progress_bar:
            for index, all_subgraph_with_boundary in enumerate(pool.imap_unordered(select_subgraphs, parameters), start=1):
                if all_subgraph_with_boundary:
                    for subgraph, boundary, subgraph_hash in all_subgraph_with_boundary:
                        if subgraph_hash in unique_subgraph_with_boundary:
                            pass
                        else:
                            unique_subgraph_with_boundary[subgraph_hash] = (subgraph, boundary)
                progress_bar.update(1)
                progress_bar.set_postfix({f'Current Selected': f'{len(unique_subgraph_with_boundary)}'})

    subgraph_sizes = list()
    selected_subgraph_with_boundary = list()
    for subgraph_hash, (subgraph, boundary) in unique_subgraph_with_boundary.items():
        selected_subgraph_with_boundary.append((subgraph_hash, subgraph, boundary))
        subgraph_sizes.append((subgraph.number_of_nodes(), subgraph.number_of_edges()))
    sorted_indices = sorted(range(len(selected_subgraph_with_boundary)), key=lambda index: selected_subgraph_with_boundary[index][0])
    selected_subgraph_with_boundary = [selected_subgraph_with_boundary[index] for index in sorted_indices]
    subgraph_sizes = [subgraph_sizes[index] for index in sorted_indices]
    logger.info(f'Total Selected Subgraphs: {len(selected_subgraph_with_boundary)}')

    subgraph_train_indices, subgraph_valid_indices, subgraph_test_indices = sample_by_partition(subgraph_sizes, partition_number, train_ratio, valid_ratio, test_ratio)
    subgraph_train_split = [selected_subgraph_with_boundary[subgraph_train_index] for subgraph_train_index in subgraph_train_indices]
    subgraph_valid_split = [selected_subgraph_with_boundary[subgraph_valid_index] for subgraph_valid_index in subgraph_valid_indices]
    subgraph_test_split = [selected_subgraph_with_boundary[subgraph_test_index] for subgraph_test_index in subgraph_test_indices]
    logger.info(f'Subgraph Split Finished - Subgraph_Train: {len(subgraph_train_split)}; Subgraph_Valid: {len(subgraph_valid_split)}; Subgraph_Test: {len(subgraph_test_split)};')
    logger.info(f' - First 10 Train Index: {subgraph_train_indices[:10]}')
    logger.info(f' - First 10 Valid Index: {subgraph_valid_indices[:10]}')
    logger.info(f' - First 10 Test  Index: {subgraph_test_indices[:10]}')

    node_nums = [node_num for node_num, edge_num in subgraph_sizes]
    edge_nums = [edge_num for node_num, edge_num in subgraph_sizes]
    logger.info(f'Node Num - Max/Min = {max(node_nums)} / {min(node_nums)}')
    logger.info(f'Edge Num - Max/Min = {max(edge_nums)} / {min(edge_nums)}')

    logger.info(f'Extract Dictionaries From Train Split: Task_Dict & Node_Dict')
    train_node_dict, train_operator_dict = extract_dict(subgraph_train_split, worker_number)
    if silly:
        subgraph_valid_split = clean_split(subgraph_valid_split, train_node_dict, train_operator_dict)
        subgraph_test_split = clean_split(subgraph_test_split, train_node_dict, train_operator_dict)
        logger.info(f'-!!!-New-!!!-')
        logger.info(f'Clean Split Finished - Subgraph_Train: {len(subgraph_train_split)}; Subgraph_Valid: {len(subgraph_valid_split)}; Subgraph_Test: {len(subgraph_test_split)};')
        logger.info(f' - First 10 Train Index: {subgraph_train_indices[:10]}')
        logger.info(f' - First 10 Valid Index: {subgraph_valid_indices[:10]}')
        logger.info(f' - First 10 Test  Index: {subgraph_test_indices[:10]}')

    train_node_dict_keys = set(train_node_dict['onnx'].keys()) | set(train_node_dict['others'].keys())
    train_operator_dict_keys = set(train_operator_dict['onnx'].keys()) | set(train_operator_dict['others'].keys())
    logger.info(f'Extracted - ONNX & Others Node_Dict: {len(train_node_dict["onnx"])} & {len(train_node_dict["others"])};')
    logger.info(f'Extracted - ONNX & Others Operator_Dict: {len(train_operator_dict["onnx"])} & {len(train_operator_dict["others"])};')
    logger.info(f'Checking Unknown [Tasks & Nodes & Operators] In Valid & Test Splits.')

    logger.info(f'Valid Split:')
    valid_node_dict, valid_operator_dict = extract_dict(subgraph_valid_split, worker_number)

    valid_node_dict_keys = set(valid_node_dict['onnx'].keys()) | set(valid_node_dict['others'].keys())
    valid_operator_dict_keys = set(valid_operator_dict['onnx'].keys()) | set(valid_operator_dict['others'].keys())

    num_valid_unknown_nodes = len(valid_node_dict_keys - train_node_dict_keys)
    num_valid_unknown_operators = len(valid_operator_dict_keys - train_operator_dict_keys)

    num_onnx_valid_unknown_nodes = len(set(valid_node_dict['onnx'].keys()) - set(train_node_dict['onnx'].keys()))
    num_others_valid_unknown_nodes = len(set(valid_node_dict['others'].keys()) - set(train_node_dict['others'].keys()))

    num_onnx_valid_unknown_operators = len(set(valid_operator_dict['onnx'].keys()) - set(train_operator_dict['onnx'].keys()))
    num_others_valid_unknown_operators = len(set(valid_operator_dict['others'].keys()) - set(train_operator_dict['others'].keys()))

    percent_nodes_valid = num_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0
    percent_onnx_nodes_valid = num_onnx_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0
    percent_others_nodes_valid = num_others_valid_unknown_nodes / len(valid_node_dict_keys) * 100 if len(valid_node_dict_keys) != 0 else 0

    percent_operators_valid = num_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0
    percent_onnx_operators_valid = num_onnx_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0
    percent_others_operators_valid = num_others_valid_unknown_operators / len(valid_operator_dict_keys) * 100 if len(valid_operator_dict_keys) != 0 else 0

    logger.info(f'Valid Unknown [ Number / Ratio ]:')
    logger.info(f' - Nodes = [ {num_valid_unknown_nodes} ({num_onnx_valid_unknown_nodes}/{num_others_valid_unknown_nodes}) & {percent_nodes_valid:.2f}% ({percent_onnx_nodes_valid}/{percent_others_nodes_valid}) ]')
    logger.info(f' - Operators = [ {num_valid_unknown_operators} ({num_onnx_valid_unknown_operators}/{num_others_valid_unknown_operators}) & {percent_operators_valid:.2f}% ({percent_onnx_operators_valid}/{percent_others_operators_valid}) ]')

    logger.info(f'Test Split:')
    test_node_dict, test_operator_dict = extract_dict(subgraph_test_split, worker_number)

    test_node_dict_keys = set(test_node_dict['onnx'].keys()) | set(test_node_dict['others'].keys())
    test_operator_dict_keys = set(test_operator_dict['onnx'].keys()) | set(test_operator_dict['others'].keys())

    num_test_unknown_nodes = len(test_node_dict_keys - train_node_dict_keys)
    num_onnx_test_unknown_nodes = len(set(test_node_dict['onnx'].keys()) - set(train_node_dict['onnx'].keys()))
    num_others_test_unknown_nodes = len(set(test_node_dict['others'].keys()) - set(train_node_dict['others'].keys()))

    num_test_unknown_operators = len(test_operator_dict_keys - train_operator_dict_keys)
    num_onnx_test_unknown_operators = len(set(test_operator_dict['onnx'].keys()) - set(train_operator_dict['onnx'].keys()))
    num_others_test_unknown_operators = len(set(test_operator_dict['others'].keys()) - set(train_operator_dict['others'].keys()))

    percent_nodes_test = num_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0
    percent_onnx_nodes_test = num_onnx_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0
    percent_others_nodes_test = num_others_test_unknown_nodes / len(test_node_dict_keys) * 100 if len(test_node_dict_keys) != 0 else 0

    percent_operators_test = num_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0
    percent_onnx_operators_test = num_onnx_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0
    percent_others_operators_test = num_others_test_unknown_operators / len(test_operator_dict_keys) * 100 if len(test_operator_dict_keys) != 0 else 0

    logger.info(f'Test Unknown  [ Number / Ratio ]:')
    logger.info(f' - Nodes = [ {num_test_unknown_nodes} ({num_onnx_test_unknown_nodes}/{num_others_test_unknown_nodes}) & {percent_nodes_test:.2f}% ({percent_onnx_nodes_test}/{percent_others_nodes_test}) ]')
    logger.info(f' - Operators = [ {num_test_unknown_operators} ({num_onnx_test_unknown_operators}/{num_others_test_unknown_operators}) & {percent_operators_test:.2f}% ({percent_onnx_operators_test}/{percent_others_operators_test}) ]')

    meta = dict(
        all_nodes = train_node_dict,
        all_operators = train_operator_dict,
    )

    if len(subgraph_train_split):
        train_split_meta = dict(
            split = f'train',
            archive = f'train.tar.gz',
            version = f'{version}',
            size = len(subgraph_train_split),
            url = "",
        )
        train_split_meta.update(meta)
        save_split(train_split_meta, subgraph_train_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Train Split')

    if len(subgraph_valid_split):
        valid_split_meta = dict(
            split = f'valid',
            archive = f'valid.tar.gz',
            version = f'{version}',
            size = len(subgraph_valid_split),
            url = "",
        )
        valid_split_meta.update(meta)
        save_split(valid_split_meta, subgraph_valid_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Valid Split')

    if len(subgraph_valid_split):
        test_split_meta = dict(
            split = f'test',
            archive = f'test.tar.gz',
            version = f'{version}',
            size = len(subgraph_test_split),
            url = "",
        )
        test_split_meta.update(meta)
        save_split(test_split_meta, subgraph_test_split, save_dirpath, worker_number)
    else:
        logger.info(f'No Instances in Test Split')