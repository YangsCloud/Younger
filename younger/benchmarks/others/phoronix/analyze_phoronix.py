#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-08-30 14:49
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import xlsxwriter

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.constants import ONNXOperator


def analyze_phoronix(phoronix_dir: pathlib.Path, analysis_dir: pathlib.Path):
    instances: list[Instance] = list()
    for model_dir in phoronix_dir.iterdir():
        model_name = model_dir.name
        logger.info(f' v Now processing model: {model_name}')
        model_filepaths = [model_filepath for model_filepath in model_dir.rglob('*.onnx')]
        assert len(model_filepaths) == 1

        model_filepath = model_filepaths[0]
        instance_dirpath = analysis_dir.joinpath(model_name)
        if instance_dirpath.is_dir():
            logger.info(f'   Instance Alreadly Exists: {instance_dirpath}')
            instance = Instance()
            instance.load(instance_dirpath)
            instances.append(instance)
        else:
            logger.info(f'   Model Filepath: {model_filepath}')
            logger.info(f'   Extracting Instance ...')
            instance = Instance(model_filepath)
            instances.append(instance)
            logger.info(f'   Extracted')
            instance.save(instance_dirpath)
            logger.info(f'   Instance saved into {instance_dirpath}')
        logger.info(f' ^ Done')

    logger.info(f' v Now analyzing ...')
    op_count = dict()
    for instance in instances:
        graph = Network.standardize(instance.network.graph)
        for node_index in graph.nodes():
            op = str((graph.nodes[node_index]['operator']['op_type'], graph.nodes[node_index]['operator']['domain']))
            op_count[op] = op_count.get(op, 0) + 1
    logger.info(f'   Total Different Operators = {len(op_count)}')
    analysis_json_filepath = analysis_dir.joinpath('op_count.json')
    save_json(op_count, analysis_json_filepath, indent=2)
    logger.info(f'   Analytical Results Saved into: {analysis_json_filepath}')

    analysis_xlsx_filepath = analysis_dir.joinpath('op_count.xlsx')
    workbook = xlsxwriter.Workbook(analysis_xlsx_filepath)
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'OP_Name')
    worksheet.write(0, 1, 'Phoronix')

    for index, operator_type in enumerate(sorted(ONNXOperator.TYPES), start=1):
        op = str(operator_type)
        count = op_count.get(op, 0)
        worksheet.write(index, 0, op)
        worksheet.write(index, 1, count)

    workbook.close()
    logger.info(f'   Excel Analytical Results Saved into: {analysis_xlsx_filepath}')
    logger.info(f' ^ Done')