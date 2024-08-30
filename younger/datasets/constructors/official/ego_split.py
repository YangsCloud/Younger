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
import time
import tqdm
import numpy
import pandas
import pathlib
import networkx
import multiprocessing

from younger.commons.io import load_json, save_json, save_pickle, create_dir, tar_archive
from younger.commons.hash import hash_string
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_operator_origin


def find_all_operators_of_instance(parameter: tuple[pathlib.Path]) -> dict[str, int]:
    (path, _, _) = parameter

    instance = Instance()
    instance.load(path)

    graph = instance.network.graph

    all_operators_of_instance: dict[str, int] = dict()
    all_operator_positions_of_instance: dict[str, list[tuple[str, pathlib.Path]]] = dict()

    for node_index in graph.nodes():
        operator_identifier = Network.get_node_identifier_from_features(graph.nodes[node_index]['features'], mode='type')

        all_operators_of_instance[operator_identifier] = all_operators_of_instance.get(operator_identifier, 0) + 1
        all_operator_positions_of_instance[operator_identifier] = all_operator_positions_of_instance.get(operator_identifier, list())
        all_operator_positions_of_instance[operator_identifier].append((node_index, path))

    return all_operators_of_instance, all_operator_positions_of_instance


def match_nodes_of_instance(parameter: tuple[pathlib.Path, str]) -> tuple[list[str], pathlib.Path]:
    (path, ox_operator_id) = parameter
    instance = Instance()
    instance.load(path)

    graph = instance.network.graph

    matched_nodes = list()
    for node_index in graph.nodes():
        operator_identifier = Network.get_node_identifier_from_features(graph.nodes[node_index]['features'], mode='type')
        if operator_identifier == ox_operator_id:
            matched_nodes.append(node_index)
    return matched_nodes, path


def get_all_egos_of_instance(parameter) -> list[tuple[networkx.DiGraph, str, networkx.DiGraph, str]]:
    (node_index, path, radius, ox_operator_id) = parameter
    instance = Instance()
    instance.load(path)
    graph = instance.network.graph
    # tic = time.time()
    # print(path, 1)
    l_ego = networkx.ego_graph(graph, node_index, radius=radius, center=True)
    r_ego = networkx.ego_graph(graph.reverse(), node_index, radius=radius, center=True)
    # toc = time.time()
    # print(path, 2, toc-tic)
    l_ego_hash = hash_string(f'{networkx.weisfeiler_lehman_graph_hash(l_ego, node_attr="operator", iterations = 3, digest_size = 16)}-{ox_operator_id}', hash_algorithm='blake2b', digest_size=16)
    r_ego_hash = hash_string(f'{networkx.weisfeiler_lehman_graph_hash(r_ego, node_attr="operator", iterations = 3, digest_size = 16)}-{ox_operator_id}', hash_algorithm='blake2b', digest_size=16)
    # tocc = time.time()
    # print(path, 3, tocc-toc)
    return l_ego, l_ego_hash, r_ego, r_ego_hash

def find_all_egos_focus_ox_operator(matched_nodes: list[str, pathlib.Path], ox_operator_id: str, range_left: int, range_right: int, sample_frequency: int, worker_number: int) -> dict[str, int]:
    all_egos_of_instance: list[str, networkx.DiGraph, str] = list() # [(focus, ego, ego_hash)]
    # print(len(matched_nodes))
    for radius in range(range_left, range_right+1):
        indices = numpy.random.choice(list(range(len(matched_nodes))), size=min(sample_frequency, len(matched_nodes)), replace=False)

        parameters = list()
        for index in indices:
            (node_index, path) = matched_nodes[index]
            parameters.append((node_index, path, radius, ox_operator_id))

        with tqdm.tqdm(total=len(parameters), desc=f'Finding Egos Focus {ox_operator_id}') as progress_bar:
            with multiprocessing.Pool(worker_number) as pool:
                for index, (l_ego, l_ego_hash, r_ego, r_ego_hash) in enumerate(pool.imap_unordered(get_all_egos_of_instance, parameters)):
                    # if ox_operator_id == "('NonMaxSuppression', '')":
                    #     print(index)
                    all_egos_of_instance.append((node_index, l_ego, l_ego_hash))
                    all_egos_of_instance.append((node_index, r_ego, r_ego_hash))
                    progress_bar.update(1)

    return all_egos_of_instance


def main(
    dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path,
    range_left: int = 3, range_right: int = 6,
    sample_frequency: int = 100,
    worker_number: int = 4,
    seed: int = 16861,
):
    # 0. Each graph of the dataset MUST be standardized graph
    # 1. Tasks File should be a *.json file, which contains an list of tasks (list[str]) (It can be an empty list)
    # 2. Usually we do not clean instances with theory value range of metric.
    # For example:
    # WER maybe larger than 1 and Word Accuracy maybe smaller than 0 in ASR research area.

    numpy.random.seed(seed)

    assert range_left <= range_right
    logger.info(f'Ego Radius Range = [ {range_left}, {range_right} ]')

    logger.info(f'Checking Existing Instances ...')
    paths = sorted([path for path in dataset_dirpath.iterdir()])
    parameters = list()
    for path in paths:
        parameters.append((path, range_left, range_right))
    logger.info(f'Total Instances: {len(parameters)}')

    logger.info(f'Finding All Operators ...')
    all_operators: dict[str, int] = dict()
    all_operator_positions: dict[str, list[(str, pathlib.Path)]] = dict()
    with multiprocessing.Pool(worker_number) as pool:
        with tqdm.tqdm(total=len(parameters), desc='Finding') as progress_bar:
            for index, (all_operators_of_instance, all_operator_positions_of_instance) in enumerate(pool.imap_unordered(find_all_operators_of_instance, parameters), start=1):
                for operator, count in all_operators_of_instance.items():
                    all_operators[operator] = all_operators.get(operator, 0) + count
                    all_operator_positions[operator] = all_operator_positions.get(operator, list())
                    all_operator_positions[operator].extend(all_operator_positions_of_instance.get(operator, list()))
                progress_bar.update(1)
    all_operators: list[tuple[str, int]] = sorted(list(all_operators.items()), key=lambda x: (x[1], x[0]))[::-1]
    ox_ao = 0
    ot_ao = 0
    ox_all_operators: list[tuple[str, int]] = list()
    ot_all_operators: list[tuple[str, int]] = list()
    for operator_id, count in all_operators:
        op_data = ast.literal_eval(str(operator_id))
        origin = get_operator_origin(op_data[0], op_data[1])
        if origin == 'onnx':
            ox_ao += count
            ox_all_operators.append((operator_id, count))
        else:
            ot_ao += count
            ot_all_operators.append((operator_id, count))
    all_operators: tuple[list[tuple[str, int]], list[tuple[str, int]]] = (ox_all_operators, ot_all_operators)
    logger.info(f'Total = {len(ox_all_operators) + len(ot_all_operators)};')
    logger.info(f'ONNX Total = {len(ox_all_operators)};')
    logger.info(f'Others Total = {len(ot_all_operators)};')

    all_operators_filepath = save_dirpath.joinpath('all_operators.json')
    logger.info(f'Saving Count of All Operators: {all_operators_filepath.absolute()} ... ')
    save_json(all_operators, all_operators_filepath, indent=2)
    ox_ao_data = {
        'OP_Name': [operator_id for operator_id, _ in ox_all_operators],
        'Count':  [count for _, count in ox_all_operators],
    }

    data_frame = pandas.DataFrame(ox_ao_data)

    excel_filepath = save_dirpath.joinpath('ox_all_operators.xlsx')
    data_frame.to_excel(excel_filepath, index=False)
    logger.info(f'ONNX Operator Statistics Saved in Excel: {excel_filepath.absolute()} ... ')
    logger.info(f'Saved.')

    all_egos_dirpath = save_dirpath.joinpath('all_egos')
    create_dir(all_egos_dirpath)
    logger.info(f'Saving Ego Into {all_egos_dirpath.absolute()} ... ')

    logger.info(f'Finding Subgraphs ...')
    ego_hash_table: set[str] = set() # {ego_hash: (focus, ego)} ego_hash = hash((focus, ego))
    all_egos: dict[str, set[str]] = dict() # {Operator_ID: set[ego_hash]}
    for index, (ox_operator_id, _) in enumerate(ox_all_operators):
        op_type = ast.literal_eval(str(ox_operator_id))[0]
        all_egos_focus_ox_operator = find_all_egos_focus_ox_operator(all_operator_positions[ox_operator_id], ox_operator_id, range_left, range_right, sample_frequency, worker_number)
        logger.info({f'[No. {index}] Current Focus': f'{op_type}', f'Current Found': f'{len(all_egos_focus_ox_operator)}'})
        with tqdm.tqdm(total=len(all_egos_focus_ox_operator), desc=f'Saving Egos Focus {op_type}') as progress_bar:
            for focus, ego, ego_hash in all_egos_focus_ox_operator:
                if ego_hash not in ego_hash_table:
                    ego_hash_table.add(ego_hash)
                    ego_filepath = all_egos_dirpath.joinpath(f'{ego_hash}.pkl')
                    if not ego_filepath.is_file():
                        save_pickle([focus, ego, ego_hash], ego_filepath)
                    progress_bar.set_postfix({f'Current Saving': f'{ego_hash}'})
                else:
                    progress_bar.set_postfix({f'Current Saving': f'Encountered No Save'})
                all_egos[ox_operator_id] = all_egos.get(ox_operator_id, set())
                all_egos[ox_operator_id].add(ego_hash)
                progress_bar.update(1)

    all_egos: list[tuple[str, list[str]]] = [(operator_id, list(ego_hash_set)) for operator_id, ego_hash_set in all_egos.items()]
    all_egos = sorted(all_egos, key=lambda x: len(x[1]))
    logger.info(f'Total = {sum([len(ego_hash_set) for _, ego_hash_set in all_egos])};')

    all_egos_filepath = save_dirpath.joinpath('all_egos.json')
    logger.info(f'Saving All Egos: {all_egos_filepath.absolute()} ... ')
    save_json(all_egos, all_egos_filepath, indent=2)
    logger.info(f'Saved.')