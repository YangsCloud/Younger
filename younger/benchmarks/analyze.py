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


import pathlib
import xlsxwriter

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger.datasets.modules import Network

from younger.benchmarks.utils import get_instances, get_op_string


def main(dataset_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path):
    logger.info(f' v Now analyzing ...')
    statistics = dict()

    op_frequency = dict()
    instances = get_instances(dataset_dirpath)
    for instance in instances:
        graph = Network.standardize(instance.network.graph)
        for node_index in graph.nodes():
            op = get_op_string(graph.nodes[node_index]['operator']['op_type'], graph.nodes[node_index]['operator']['domain'])
            op_frequency[op] = op_frequency.get(op, 0) + 1
    statistics['op_frequency'] = op_frequency
    logger.info(f'   Total different operators = {len(op_frequency)}')

    # Save Statistics JSON

    statistics_json_filepath = statistics_dirpath.joinpath('statistics.json')
    save_json(statistics, statistics_json_filepath, indent=2)
    logger.info(f'   JSON Statistics results saved into: {statistics_json_filepath}')

    # Save Statistics XLSX
    statistics_xlsx_filepath = statistics_dirpath.joinpath('statistics.xlsx')
    workbook = xlsxwriter.Workbook(statistics_xlsx_filepath)

    # op frequency
    worksheet = workbook.add_worksheet('op_frequency')

    worksheet.write(0, 0, 'OP_Name')
    worksheet.write(0, 1, 'Phoronix')

    for index, (op, frequency) in enumerate(op_frequency.items(), start=1):
        worksheet.write(index, 0, op)
        worksheet.write(index, 1, frequency)

    workbook.close()
    logger.info(f'   XLSX Statistics results saved into: {statistics_xlsx_filepath}')
    logger.info(f' ^ Done')