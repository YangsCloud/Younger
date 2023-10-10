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

import pathlib
import argparse
import semantic_version

from youngbench.dataset.modules import Dataset, Prototype
from youngbench.benchmark.analyzer import get_dataset_ops, get_op_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    # Dataset Save/Load Path.
    parser.add_argument('-p', '--dataset-path', type=str, required=True)

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_path)
    assert dataset_path.is_dir(), f'Directory does not exist at the specified \"Dataset Path\": {dataset_path}.'

    dataset = Dataset()
    dataset.load(dataset_path)

    prototypes = list()
    for network in dataset.networks:
        prototypes.append(network.prototype)


    op_types = get_dataset_ops(dataset)
    op_types = sorted(list(op_types.items()), key=lambda x: x[1])
    for op_type, num in op_types:
        print(f'{op_type}: {num}')

    op = ('Conv', 'ai.onnx')
    op_per_network = get_op_stats(dataset, op_type='Conv', op_domain='ai.onnx')
    for net_id, op_num in op_per_network.items():
        print(op_num)