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

from younger.commons.io import tar_extract, load_json, save_json
from younger.commons.logging import logger
from younger.commons.download import download

from younger.benchmarks.utils import get_instances, get_op_string


YOUNGER_V_PAPER_FILTER_DETAIL_URL="https://datasets.yangs.cloud/public/assets/622387f1-dee7-4019-b538-34f341eb2914?download="
YOUNGER_V_PAPER_FILTER_DETAIL_NAME = 'detailed_filter_series_without_attributes_paper'


def main(real_dirpath: pathlib.Path, product_dirpath: pathlib.Path, statistics_dirpath: pathlib.Path | None = None):
    logger.info(f' v Downloading Younger ...')
    tar_filepath = download(YOUNGER_V_PAPER_FILTER_DETAIL_URL, real_dirpath, force=False)
    logger.info(f' ^ Done')

    logger.info(f' v Uncompressing ...')
    tar_extract(tar_filepath, real_dirpath)
    logger.info(f' ^ Done')

    logger.info(f' v Loading all Instances in Younger ...')
    younger_dirpath = real_dirpath.joinpath(YOUNGER_V_PAPER_FILTER_DETAIL_NAME)
    instances = get_instances(younger_dirpath)
    logger.info(f' ^ Done')

    if statistics_dirpath is None:
        logger.info(f' - Statistics are not specified, skip ...')
        statistics = None
    else:
        logger.info(f' - Statistics are specified')
        logger.info(f' v Loading Statistics ...')
        statistics = load_json(statistics_dirpath.joinpath('statistics.json'))
        logger.info(f' ^ Done')

    logger.info(f' v Analyzing Younger ...')
    younger_statistics = dict()
    younger_op_frequency = dict()
    for instance in instances:
        for node_index in instance.network.graph.nodes():
            op = get_op_string(instance.network.graph.nodes[node_index]['operator']['op_type'], instance.network.graph.nodes[node_index]['operator']['domain'])
            younger_op_frequency[op] = younger_op_frequency.get(op, 0) + 1

    younger_statistics['op_frequency'] = younger_op_frequency
    younger_statistics_filepath = product_dirpath.joinpath('statistics.json')
    save_json(younger_statistics, younger_statistics_filepath)
    logger.info(f'   Younger\'s JSON Statistics results saved into: {younger_statistics_filepath}')
    logger.info(f' ^ Done')

    statistics['op_frequency']