#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-07-15 15:46
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import ast
import pathlib
import xlsxwriter

from typing import Literal

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger

from younger.datasets.modules import Dataset, Network
from younger.datasets.utils.translation import get_operator_origin


def statistically_analyze(dataset_name: str, dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path) -> dict[str, int | dict[str, tuple[int, float]]]:
    logger.info(f' v Now statistically analyzing {dataset_name} ...')

    statistics = dict()

    total_ops = 0
    op_type_frequency = dict()
    op_origin_frequency = dict()
    unknown_op_type_frequency = dict()
    for instance in Dataset.load_instances(dataset_dirpath):
        try:
            graph = Network.standardize(instance.network.graph)
        except:
            # Already cleansed.
            graph = instance.network.graph
        total_ops += graph.number_of_nodes()
        for node_index in graph.nodes():
            op_type = Network.get_node_identifier_from_features(graph.nodes[node_index], mode='type')
            op_origin = get_operator_origin(graph.nodes[node_index]['operator']['op_type'], graph.nodes[node_index]['operator']['domain'])
            if op_origin != 'unknown':
                op_type_frequency[op_type] = op_type_frequency.get(op_type, 0) + 1
                op_origin_frequency[op_origin] = op_origin_frequency.get(op_origin, 0) + 1
            else:
                unknown_op_type_frequency[op_type] = unknown_op_type_frequency.get(op_type, 0) + 1
    statistics['op_type_frequency'] = op_type_frequency
    statistics['op_origin_frequency'] = op_origin_frequency
    statistics['unknown_op_type_frequency'] = unknown_op_type_frequency
    logger.info(f'   Total operators = {total_ops}')
    logger.info(f'   Total different operator types = {len(op_type_frequency)}')
    logger.info(f'   Total different operator origins = {len(op_origin_frequency)}')

    statistics['total_ops'] = total_ops
    for op_type, frequency in statistics['op_type_frequency'].items():
        statistics['op_type_frequency'][op_type] = (frequency, frequency/total_ops)

    for op_origin, frequency in statistics['op_origin_frequency'].items():
        statistics['op_origin_frequency'][op_origin] = (frequency, frequency/total_ops)

    for unknown_op_type, frequency in statistics['unknown_op_type_frequency'].items():
        statistics['unknown_op_type_frequency'][unknown_op_type] = (frequency, frequency/total_ops)

    # v =================================== Save To File =================================== v
    # Save Statistics JSON
    json_filepath = statistics_dirpath.joinpath(f'statistics_{dataset_name}.json')
    save_json(statistics, json_filepath, indent=2)
    logger.info(f'   {dataset_name}\'s statistics results (JSON format) saved into: {json_filepath}')

    # Save Statistics XLSX
    xlsx_filepath = statistics_dirpath.joinpath(f'statistics_{dataset_name}.xlsx')
    workbook = xlsxwriter.Workbook(xlsx_filepath)

    # op type frequency
    worksheet = workbook.add_worksheet('op_type_frequency')

    worksheet.write(0, 0, 'OP_Name')
    worksheet.write(0, 1, 'OP_Domain')
    worksheet.write(0, 2, 'Frequency')
    worksheet.write(0, 3, 'Ratio')

    for index, (op_type, (frequency, ratio)) in enumerate(statistics['op_type_frequency'].items(), start=1):
        op_name, op_domain = ast.literal_eval(op_type)
        worksheet.write(index, 0, op_name)
        worksheet.write(index, 1, op_domain)
        worksheet.write(index, 2, frequency)
        worksheet.write(index, 3, ratio)

    # op origin frequency
    worksheet = workbook.add_worksheet('op_origin_frequency')

    worksheet.write(0, 0, 'OP_Origin')
    worksheet.write(0, 1, 'Frequency')
    worksheet.write(0, 2, 'Ratio')

    for index, (op_origin, (frequency, ratio)) in enumerate(statistics['op_origin_frequency'].items(), start=1):
        worksheet.write(index, 0, op_origin)
        worksheet.write(index, 1, frequency)
        worksheet.write(index, 2, ratio)

    workbook.close()
    logger.info(f'   {dataset_name}\'s statistics results (XLSX format) saved into: {xlsx_filepath}')
    # ^ =================================== Save To File =================================== ^

    logger.info(f' ^ Done')
    return statistics


def statistical_analysis(younger_dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None):
    younger_dataset_statistics = statistically_analyze('younger', younger_dataset_dirpath, statistics_dirpath)

    if other_dataset_indices_filepath is not None:
        other_dataset_statistics = dict()
        with open(other_dataset_indices_filepath, 'r') as f:
            for line in f:
                other_dataset_name, other_dataset_dirpath = line.split(':')[0].strip(), line.split(':')[1].strip()
                other_dataset_statistics[other_dataset_name] = statistically_analyze(other_dataset_name, other_dataset_dirpath, statistics_dirpath)

        if len(other_dataset_statistics) != 0:
            logger.info(f' v Analyzing Younger Compare To Other Datasets ...')
            for dataset_name, dataset_statistics in other_dataset_statistics.items():
                op_type_cover_ratios = list() # Other Cover Younger
                uncovered_op_types = list() # Other Uncovered By Younger
                for op_type, (frequency, ratio) in dataset_statistics['op_type_frequency'].items():
                    if op_type in younger_dataset_statistics['op_type_frequency']:
                        op_type_cover_ratios.append((op_type, frequency / younger_dataset_statistics['op_type_frequency'][op_type][0]))
                    else:
                        uncovered_op_types.append(op_type)

                op_origin_cover_ratios = list() # Other Cover Younger
                uncovered_op_origins = list() # Other Uncovered By Younger
                for op_origin, (frequency, ratio) in dataset_statistics['op_origin_frequency'].items():
                    if op_origin in younger_dataset_statistics['op_origin_frequency']:
                        op_origin_cover_ratios.append((op_origin, frequency / younger_dataset_statistics['op_origin_frequency'][op_origin][0]))
                    else:
                        uncovered_op_origins.append(op_origin)

                compare_statistics = dict(
                    op_type_cover_ratios = op_type_cover_ratios,
                    uncovered_op_types = uncovered_op_types,
                    op_origin_cover_ratios = op_origin_cover_ratios,
                    uncovered_op_origins = uncovered_op_origins
                )

                json_filepath = statistics_dirpath.joinpath(f'statistics_compare_{dataset_name}.json')
                save_json(compare_statistics, json_filepath, indent=2)
                logger.info(f'   {dataset_name}\'s statistics results (JSON format) compared to Younger saved into: {json_filepath}')

            logger.info(f' ^ Done')


def structurally_analyze(dataset_name: str, dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path) -> dict[str, int | dict[str, tuple[int, float]]]:
    logger.info(f' v Now statistically analyzing {dataset_name} ...')

    statistics = dict()

    total_ops = 0
    op_type_frequency = dict()
    op_origin_frequency = dict()
    unknown_op_type_frequency = dict()
    for instance in Dataset.load_instances(dataset_dirpath):
        try:
            graph = Network.standardize(instance.network.graph)
        except:
            # Already cleansed.
            graph = instance.network.graph
        total_ops += graph.number_of_nodes()
        for node_index in graph.nodes():
            op_type = Network.get_node_identifier_from_features(graph.nodes[node_index], mode='type')
            op_origin = get_operator_origin(graph.nodes[node_index]['operator']['op_type'], graph.nodes[node_index]['operator']['domain'])
            if op_origin != 'unknown':
                op_type_frequency[op_type] = op_type_frequency.get(op_type, 0) + 1
                op_origin_frequency[op_origin] = op_origin_frequency.get(op_origin, 0) + 1
            else:
                unknown_op_type_frequency[op_type] = unknown_op_type_frequency.get(op_type, 0) + 1
    statistics['op_type_frequency'] = op_type_frequency
    statistics['op_origin_frequency'] = op_origin_frequency
    statistics['unknown_op_type_frequency'] = unknown_op_type_frequency
    logger.info(f'   Total operators = {total_ops}')
    logger.info(f'   Total different operator types = {len(op_type_frequency)}')
    logger.info(f'   Total different operator origins = {len(op_origin_frequency)}')

    statistics['total_ops'] = total_ops
    for op_type, frequency in statistics['op_type_frequency'].items():
        statistics['op_type_frequency'][op_type] = (frequency, frequency/total_ops)

    for op_origin, frequency in statistics['op_origin_frequency'].items():
        statistics['op_origin_frequency'][op_origin] = (frequency, frequency/total_ops)

    for unknown_op_type, frequency in statistics['unknown_op_type_frequency'].items():
        statistics['unknown_op_type_frequency'][unknown_op_type] = (frequency, frequency/total_ops)

    # v =================================== Save To File =================================== v
    # Save Statistics JSON
    json_filepath = statistics_dirpath.joinpath(f'statistics_{dataset_name}.json')
    save_json(statistics, json_filepath, indent=2)
    logger.info(f'   {dataset_name}\'s statistics results (JSON format) saved into: {json_filepath}')

    # Save Statistics XLSX
    xlsx_filepath = statistics_dirpath.joinpath(f'statistics_{dataset_name}.xlsx')
    workbook = xlsxwriter.Workbook(xlsx_filepath)

    # op type frequency
    worksheet = workbook.add_worksheet('op_type_frequency')

    worksheet.write(0, 0, 'OP_Name')
    worksheet.write(0, 1, 'OP_Domain')
    worksheet.write(0, 2, 'Frequency')
    worksheet.write(0, 3, 'Ratio')

    for index, (op_type, (frequency, ratio)) in enumerate(statistics['op_type_frequency'].items(), start=1):
        op_name, op_domain = ast.literal_eval(op_type)
        worksheet.write(index, 0, op_name)
        worksheet.write(index, 1, op_domain)
        worksheet.write(index, 2, frequency)
        worksheet.write(index, 3, ratio)

    # op origin frequency
    worksheet = workbook.add_worksheet('op_origin_frequency')

    worksheet.write(0, 0, 'OP_Origin')
    worksheet.write(0, 1, 'Frequency')
    worksheet.write(0, 2, 'Ratio')

    for index, (op_origin, (frequency, ratio)) in enumerate(statistics['op_origin_frequency'].items(), start=1):
        worksheet.write(index, 0, op_origin)
        worksheet.write(index, 1, frequency)
        worksheet.write(index, 2, ratio)

    workbook.close()
    logger.info(f'   {dataset_name}\'s statistics results (XLSX format) saved into: {xlsx_filepath}')
    # ^ =================================== Save To File =================================== ^

    logger.info(f' ^ Done')
    return statistics


def structural_analysis(younger_dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None, operator_embedding_dirpath: pathlib.Path | None = None):
    pass


def main(younger_dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path, other_dataset_indices_filepath: pathlib.Path | None = None, operator_embedding_dirpath: pathlib.Path | None = None, mode: Literal['sts', 'stc', 'both'] = 'sts'):
    assert mode in {'sts', 'stc', 'both'}
    analyzed = False
    if mode in {'sts', 'both'}:
        statistical_analysis(younger_dataset_dirpath, statistics_dirpath, other_dataset_indices_filepath)
        analyzed = True

    if mode in {'stc', 'both'}:
        structural_analysis(younger_dataset_dirpath, statistics_dirpath, other_dataset_indices_filepath, operator_embedding_dirpath)
        analyzed = True

    if analyzed:
        logger.info(f' = Analyzed Younger and Other Datasets.')