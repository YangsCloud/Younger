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


import toml
import pathlib


def main(benchmark_dirpath: pathlib.Path, configuration_filepath: pathlib.Path):
    configuration = toml.load(configuration_filepath)
    user_requires = configuration['user_requires']
    embs_filepath = configuration['embs_filepath']