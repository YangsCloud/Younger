#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-10-06 06:51
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

from typing import Optional

from youngbench.constants import YoungBenchHandle

from yoolkit.logging import setup_logger, logging_level


logger = setup_logger(f'{YoungBenchHandle.Name}', logging_level=logging_level['INFO'])

def set_logger(name: str = YoungBenchHandle.Name, level: str = 'INFO', path: Optional[str] = ''):
    global logger
    print(f'Logger will be reset ...')
    print(f'Name:{name}; Level: {level}; Path: {path if path else "System Default"}')
    logger = setup_logger(name, logging_level=logging_level[level], logging_path=path)

def get_logger():
    return logger