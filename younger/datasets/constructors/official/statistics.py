#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Luzhou Peng (彭路洲) and Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 17:20
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, tasks: list[str], datasets: list[str], splits: list[str], metrics: list[str]):
    print(dataset_dirpath)
    print(save_dirpath)
    print(tasks)
    print(datasets)
    print(splits)
    print(metrics)