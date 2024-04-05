#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-17 15:42
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



import json
import pickle
import networkx
import pathlib
import argparse
import semantic_version

import matplotlib.pyplot as plt

from youngbench.benchmark.analyzer import save_bkstats_of_dataset, load_bkstats_of_dataset, get_bkstats_of_dataset

from youngbench.dataset.modules import Dataset
from youngbench.dataset.utils.management import check_dataset
from youngbench.logging import set_logger, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get The Block Statistics of Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    parser.add_argument('-s', '--save-dirpath', type=str, default='')
    parser.add_argument('-l', '--logging-path', type=str, default='')
    parser.add_argument('-b', '--block-sizes', nargs='+', type=int, default=[3,])

    # Dataset Release Version.
    parser.add_argument('--version', type=str, default='')

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    save_dirpath = pathlib.Path(args.save_dirpath)
    bkstats_of_dataset_json = save_dirpath.joinpath('bkstats_dataset.txt')
    bkstats_of_dataset_dir = save_dirpath.joinpath('bkstats_dataset')

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


    bkstats_of_dataset = get_bkstats_of_dataset(dataset, args.block_sizes)
    save_bkstats_of_dataset(bkstats_of_dataset, bkstats_of_dataset_dir)

    overall_stats_str = str()
    for block_size, bkstats_of_dataset_at_size in sorted(bkstats_of_dataset.items(), key=lambda x: x[0]):
        bkstats_of_dataset_at_size = sorted(list(bkstats_of_dataset_at_size.items()), key=lambda x: x[1]['num'])
        stats_str = str()
        for wl_hash, bkstat in bkstats_of_dataset_at_size:
            stats_str += f'{str(block_size):<6} {wl_hash:<38} {str(bkstat["num"])}\n'
            for (u_nid, v_nid) in bkstat['absbk'].edges:
                stats_str += f'    {bkstat["absbk"].nodes[u_nid]["op"]} -> {bkstat["absbk"].nodes[v_nid]["op"]}\n'
        overall_stats_str += stats_str
    logger.info(f'Below is block statistics of Datset:\n{overall_stats_str}')

    with open(bkstats_of_dataset_json, 'w') as f:
        f.writelines(overall_stats_str)