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

from younger.commons.logging import logger


def main(dataset_dirpath: pathlib.Path, save_dirpath: pathlib.Path, tasks: list[str], datasets: list[str], splits: list[str], metrics: list[str]):
    logger.info(f'{dataset_dirpath.absolute()}')
    logger.info(f'{save_dirpath.name}')
    print(tasks)
    print(datasets)
    print(splits)
    print(metrics)