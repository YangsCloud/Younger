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

from typing import Generator
from younger.datasets.modules import Instance


def get_instances(dataset_dirpath: pathlib.Path | str, remove_tiny: int | None = None) -> Generator[Instance, None, None]:
    dataset_dirpath = pathlib.Path(dataset_dirpath) if isinstance(dataset_dirpath, str) else dataset_dirpath

    instance_dirpaths = list(dataset_dirpath.iterdir())
    with tqdm.tqdm(total=len(instance_dirpaths), desc='Processing Instance') as progress_bar:
        for instance_dirpath in instance_dirpaths:
            instance = Instance()
            try:
                instance.load(instance_dirpath)
                if remove_tiny and len(instance.network.graph) < remove_tiny:
                    continue
                else:
                    yield instance
            except:
                continue
            progress_bar.update(1)


def get_op_string(op_type: str, domain: str) -> str:
    return str((op_type, domain))