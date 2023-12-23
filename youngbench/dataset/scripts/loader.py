#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-09 10:57
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import argparse
import semantic_version

from youngbench.dataset.modules import Dataset
from youngbench.logging import set_logger, logger
from youngbench.constants import InstanceLabelName


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    parser.add_argument('-l', '--logging-path', type=str, default='')

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    set_logger(path=args.logging_path)

    dataset = Dataset()
    logger.info(f' v Loading Dataset ... ')
    dataset.load(dataset_path)
    logger.info(f' ^ Loaded. ')

    logger.info(f' v Checking Dataset {version}... ')
    dataset.check()
    logger.info(f' ^ Checked. ')

    logger.info(f' v Getting Version {version} Dataset ... ')
    dataset = dataset.acquire(version)
    logger.info(f' ^ Got. ')

    logger.info(f' = Below are Details of Acquired Dataset ({version}) = ')
    total_model = 0
    total_network = 0
    for index, (instance_identifier, instance) in enumerate(dataset.instances.items()):
        logger.info(f' . No.{index} Instance: {instance_identifier}')
        logger.info(f' . \u2514 Network: {instance.network.hash}')
        logger.info(f' .   \u2514 Simplified Graph: {len(instance.network.simplified_graph)}')
        logger.info(f' . \u2514 Model: Name - {instance.labels[InstanceLabelName.Name]}')
        total_model += 1
        total_network += len(instance.network.simplified_graph)

    logger.info(f' - Total Models: {total_model}')
    logger.info(f' - Total Networks: {total_network}')