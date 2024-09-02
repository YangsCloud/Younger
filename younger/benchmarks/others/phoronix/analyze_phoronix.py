#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-08-30 14:49
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pathlib

from younger.commons.io import tar_extract
from younger.commons.logging import logger
from younger.commons.download import download

from younger.datasets.modules import Instance


def analyze_phoronix(phoronix_dir: pathlib.Path, analysis_dir: pathlib.Path):
    for onnx_filepath in phoronix_dir.rglob('*.onnx'):
        instance = Instance(onnx_filepath)
        instance.save(analysis_dir)
        print(onnx_filepath)
    print(analysis_dir)
