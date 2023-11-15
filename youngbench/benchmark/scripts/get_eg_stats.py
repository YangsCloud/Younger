#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-14 16:44
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import json
import pickle
import pathlib
import argparse
import semantic_version

from youngbench.benchmark.analyzer import get_egstats_of_dataset

from youngbench.dataset.modules import Dataset
from youngbench.dataset.utils.management import check_dataset
from youngbench.logging import set_logger, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get The Edge Statistics of Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    parser.add_argument('-n', '--save-dirpath', type=str, default='')
    parser.add_argument('-l', '--logging-path', type=str, default='')

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    save_dirpath = pathlib.Path(args.save_dirpath)
    egstats_of_dataset_json = save_dirpath.joinpath('egstats_dataset.json')
    egstats_of_dataset_pkl = save_dirpath.joinpath('egstats_dataset.pkl')

    assert semantic_version.validate(args.version), f'The version provided must follow the SemVer 2.0.0 Specification.'
    version = semantic_version.Version(args.version)

    set_logger(path=args.logging_path)

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

    egstats_of_dataset = get_egstats_of_dataset(dataset)
    with open(egstats_of_dataset_pkl, 'wb') as f:
        pickle.dump(egstats_of_dataset, f)

    egstats_of_dataset = sorted(list(egstats_of_dataset.items()), key=lambda x: x[1])
    stats_str = str()
    for (u_op, v_op), egstat_of_dataset in egstats_of_dataset:
        eg = f'{u_op} -> {v_op}'
        stats_str += f'{eg:<66} \t {str(egstat_of_dataset):<10}\n'
    logger.info(f'Below is edge statistics of Datset:\n{stats_str}')

    with open(egstats_of_dataset_json, 'w') as f:
        f.writelines(stats_str)