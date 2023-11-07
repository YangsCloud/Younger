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
from youngbench.dataset.utils.management import check_dataset
from youngbench.logging import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    dataset = Dataset()
    logger.info(f' v Loading Dataset ... ')
    dataset.load(dataset_path)
    logger.info(f' ^ Loaded. ')

    logger.info(f' v Checking Dataset {version}... ')
    check_dataset(dataset, whole_check=False)
    logger.info(f' ^ Checked. ')

    logger.info(f' v Getting Version {version} Dataset ... ')
    dataset = dataset.acquire(version)
    logger.info(f' ^ Got. ')

    logger.info(f' = Below are Details of Acquired Dataset ({version}) = ')
    total_model = 0
    total_network = 0
    for index, (instance_identifier, instance) in enumerate(dataset.instances.items()):
        logger.info(f' . No.{index} Instance: {instance_identifier}')
        for network_index, (network_identifier, network) in enumerate(instance.networks.items()):
            total_network += 1
            logger.info(f' . \u2514 No.{network_index} Network: {network_identifier}')
            logger.info(f' .   \u2514 - {network.nn_graph}')
            for model_index, (model_identifier, model) in enumerate(network.models.items()):
                total_model += 1
                logger.info(f' .   \u2514 No.{model_index} Model: [= {model.name} (opset={model.opset}) =] {model_identifier}')

    logger.info(f' - Total Models: {total_model}')
    logger.info(f' - Total Networks: {total_network}')