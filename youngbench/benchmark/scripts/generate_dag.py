#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-15 14:45
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy
import random
import pathlib
import argparse
import networkx
import semantic_version

from youngbench.dataset.modules import Dataset
from youngbench.dataset.utils.management import check_dataset

from youngbench.benchmark.generator.prototype import get_prototype

from youngbench.logging import set_logger, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Neural Architecture Based on the Statistics of Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    parser.add_argument('-n', '--nodes-number', type=int, default=100)

    parser.add_argument('-s', '--save-dirpath', type=str, default='')
    parser.add_argument('-l', '--logging-path', type=str, default='')

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    save_dirpath = pathlib.Path(args.save_dirpath)
    assert save_dirpath.is_dir(), f'Directory does not exist at the specified \"Save DirPath\": {save_dirpath}.'

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    set_logger(path=args.logging_path)
    random.seed(1234)
    numpy.random.seed(1234)

    dataset = Dataset()
    logger.info(f' v Loading Dataset ... ')
    dataset.load(dataset_path)
    logger.info(f' ^ Loaded. ')

    # logger.info(f' v Checking Dataset {version}... ')
    # check_dataset(dataset, whole_check=False)
    # logger.info(f' ^ Checked. ')

    # logger.info(f' v Getting Version {version} Dataset ... ')
    # dataset = dataset.acquire(version)
    # logger.info(f' ^ Got. ')

    prototype = get_prototype(args.nodes_number, dataset)

    networkx.write_gml(prototype, save_dirpath.joinpath('test_prototype.gml'))

    print(prototype)