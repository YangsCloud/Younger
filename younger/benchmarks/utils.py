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


import tqdm
import pathlib

from younger.datasets.modules import Instance


def get_instances(dataset_dirpath: pathlib.Path, remove_tiny: int | None = None) -> list[Instance]:
    instances = list()
    instance_dirpaths = list(dataset_dirpath.iterdir())
    for instance_dirpath in tqdm.tqdm(instance_dirpaths, desc='Loading Instance'):
        instance = Instance()
        try:
            instance.load(instance_dirpath)
            if remove_tiny and len(instance.network.graph) < remove_tiny:
                continue
            else:
                instances.append(instance)
        except:
            continue

    return instances


def get_op_string(op_type: str, domain: str) -> str:
    return str((op_type, domain))