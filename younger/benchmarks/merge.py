#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-09-08 19:48
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import tqdm
import pathlib

from younger.commons.logging import logger
from younger.datasets.modules import Dataset, Network
from younger.datasets.utils.translation import get_onnx_opset_version
from younger.datasets.constructors.official.filter import update_unique_instance, purify_instance_with_graph_hash

def main(real_dirpath: pathlib.Path, save_dirpath: pathlib.Path, other_dirpaths: list[pathlib.Path]):
    unique_instances = dict()

    max_inclusive_version = get_onnx_opset_version()

    for younger_instance in Dataset.load_instances(real_dirpath):
        younger_identifier = Network.hash(younger_instance.network.graph, node_attr='operator')
        unique_instances[younger_identifier] = younger_instance

    logger.info(f'Before Update: Total Unique Instances - {len(unique_instances)}')

    other_instance_paths = list()
    for other_dirpath in other_dirpaths:
        other_instance_paths += list(other_dirpath.iterdir())

    for other_instance_path in tqdm.tqdm(other_instance_paths, desc='Update'):
        parameter = (other_instance_path, max_inclusive_version, True)
        other_instance, other_instance_identifier = purify_instance_with_graph_hash(parameter)
        if other_instance is None:
            continue
        if other_instance_identifier in unique_instances:
            unique_instance = unique_instances[other_instance_identifier]
        else:
            unique_instance = other_instance.copy()
            unique_instance.clean_labels()
            unique_instance.setup_labels(dict(model_sources=[], model_name=[], downloads=[], likes=[], tags=[], evaluations=[], hash=other_instance_identifier))
        update_unique_instance(unique_instance, other_instance)
        unique_instances[other_instance_identifier] = unique_instance
    logger.info(f'After Update: Total Unique Instances - {len(unique_instances)}')

    logger.info(f'Saving Unique Instances Into: {save_dirpath}')
    with tqdm.tqdm(total=len(unique_instances), desc='Saving') as progress_bar:
        for graph_hash, unique_instance in unique_instances.items():
            instance_save_dirpath = save_dirpath.joinpath(graph_hash)
            unique_instance.save(instance_save_dirpath)
            progress_bar.update(1)
    logger.info(f'Finished')