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
    dataset.load(dataset_path)
    dataset = dataset.acquire(version)
    total_nn = 0
    for net_index, network in enumerate(dataset.networks):
        print(f'Net {net_index}: {network.identifier}')
        for ins_index, instance in enumerate(network.instances):
            total_nn += 1
            print(f' -> Ins {ins_index}: {instance.identifier}')

    print(f'Total NN: {total_nn}')