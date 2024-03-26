#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-19 11:16
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib
import tempfile

from youngbench.constants import YoungBenchHandle


cache_root: pathlib.Path = pathlib.Path(tempfile.gettempdir()).joinpath(f'{YoungBenchHandle.Name}/{YoungBenchHandle.DatasetName}')


def set_cache_root(dirpath: pathlib.Path) -> None:
    assert isinstance(dirpath, pathlib.Path)
    global cache_root
    cache_root = dirpath
    return


def get_cache_root() -> pathlib.Path:
    return cache_root