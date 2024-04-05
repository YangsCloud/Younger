#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-19 11:16
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib

from younger.commons.constants import YoungerHandle


cache_root: pathlib.Path = pathlib.Path(os.getcwd()).joinpath(f'{YoungerHandle.MainName}/{YoungerHandle.DatasetsName}')


def set_cache_root(dirpath: pathlib.Path) -> None:
    assert isinstance(dirpath, pathlib.Path)
    global cache_root
    cache_root = dirpath
    return


def get_cache_root() -> pathlib.Path:
    return cache_root