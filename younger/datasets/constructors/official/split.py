#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-17 21:13
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

from younger.commons.io import load_json


def main(statistics_dirpath: pathlib.Path, save_dirpath: pathlib.Path, train_proportion: int = 80, valid_proportion: int = 10, test_proportion: int = 10):
    # stats is the outputs of the statistics.main
    assert train_proportion + valid_proportion + test_proportion == 100

    statistics = load_json(statistics_dirpath)
