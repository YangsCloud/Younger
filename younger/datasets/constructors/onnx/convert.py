#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-05 01:34
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

from onnx import hub

from younger.datasets.modules import Instance

from younger.commons.io import delete_dir
from younger.commons.logging import logger

from younger.datasets.constructors.utils import get_instance_dirname
from younger.datasets.constructors.onnx.utils import get_onnx_model_infos


def main(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path):
    hub.set_dir(str(cache_dirpath.absolute()))
    logger.info(f'ONNX Hub cache location is set to: {hub.get_dir()}')

    model_infos = get_onnx_model_infos()

    logger.info(f'-> Instances Creating ...')
    for index, model_info in enumerate(model_infos, start=1):
        logger.info(f' # No.{index}/{len(model_infos)}: Now processing the model: {model_info.model} (ONNX opset={model_info.opset} ...')
        model_name = model_info.model
        onnx_model_filename = pathlib.Path(model_info.model_path).name
        onnx_model = hub.load(model=model_info.model, opset=model_info.opset)
        instance = Instance(model=onnx_model, labels=dict(model_source='ONNX', model_name=model_name, onnx_model_filename=onnx_model_filename))
        instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_name, 'ONNX', onnx_model_filename))
        instance.save(instance_save_dirpath)
        delete_dir(cache_dirpath)
    logger.info(f'-> Instances Created.')
