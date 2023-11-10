#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-10 09:58
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import json
import pathlib
import argparse
import semantic_version

from youngbench.benchmark.analyzer import get_opstats_of_dataset, get_opstats_per_model

from youngbench.dataset.modules import Dataset
from youngbench.dataset.utils.management import check_dataset
from youngbench.logging import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get The Statistics of Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    parser.add_argument('-n', '--save-dirpath', type=str, default='')

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    save_dirpath = pathlib.Path(args.save_dirpath)
    opstats_of_dataset_json = save_dirpath.joinpath('opstats_dataset.json')
    opstats_per_model_json = save_dirpath.joinpath('opstats_per_model.json')

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

    opstats_of_dataset = get_opstats_of_dataset(dataset)
    opstats_of_dataset = sorted(list(opstats_of_dataset.items()), key=lambda x: (~x[1]['cus'], x[1]['num']))
    stats_str = str()
    for op, opstat_of_dataset in opstats_of_dataset:
        stats_str += f'{str(op):<36} \t {str(opstat_of_dataset["num"]):<10} \t {str(opstat_of_dataset["cus"]):<10}\n'
    logger.info(f'Below is statistics of Datset:\n{stats_str}')

    with open(opstats_of_dataset_json, 'w') as f:
        to_dump = dict()
        for k, v in opstats_of_dataset:
            to_dump[str(k)] = v
        json.dump(to_dump, f, indent=2)

    opstats_per_model = get_opstats_per_model(dataset)
    stats_str = str()
    for model_id, opstat_per_model in opstats_per_model.items():
        stats_str += f'{model_id}:\n'
        opstat_per_model = sorted(list(opstat_per_model.items()), key=lambda x: (~x[1]['cus'], x[1]['num']))
        for op, opstat in opstat_per_model:
            stats_str += f'{str(op):<36} \t {str(opstat["num"]):<10} \t {str(opstat["cus"]):<10}\n'
    logger.info(f'Below is statistics per Model:\n{stats_str}')

    with open(opstats_per_model_json, 'w') as f:
        to_dump = dict()
        for k, v in opstats_per_model.items():
            toto_dump = dict()
            for ki, vi in v.items():
                toto_dump[str(ki)] = vi
            to_dump[k] = toto_dump
        json.dump(to_dump, f, indent=2)