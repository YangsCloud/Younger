#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-12 08:56
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import json
import torch
import pathlib
import torchvision

from younger.datasets.modules import Instance

from younger.commons.io import load_json, create_dir, delete_dir
from younger.commons.logging import logger

from younger.datasets.constructors.utils import get_instance_dirname
from younger.datasets.constructors.torchvision.utils import get_torchvision_model_info, get_torchvision_model_input, get_torchvision_model_module
from younger.datasets.constructors.torchvision.annotations import get_heuristic_annotations


def save_status(status_filepath: pathlib.Path, status: dict[str, str]):
    with open(status_filepath, 'a') as status_file:
        status = json.dumps(status)
        status_file.write(f'{status}\n')


def main(
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, model_ids_filepath: pathlib.Path, status_filepath: pathlib.Path,
):
    model_ids: set[str] = set(load_json(model_ids_filepath))

    logger.info(f'-> Checking Existing Instances ...')
    for index, instance_dirpath in enumerate(save_dirpath.iterdir(), start=1):
        if len(model_ids) == 0:
            logger.info(f'-> Finished. All Models Have Been Already Converted.')
            break
        instance = Instance()
        instance.load(instance_dirpath)
        if instance.labels['model_source'] == 'TorchVision':
            logger.info(f' . Converted. Skip Total {index} - {instance.labels["model_name"]}')
            model_ids = model_ids - {instance.labels['model_name']}

    if status_filepath.is_file():
        logger.info(f'-> Found Existing Status File')
        logger.info(f'-> Now Checking Status File ...')
        with open(status_filepath, 'r') as status_file:
            for index, line in enumerate(status_file, start=1):
                try:
                    status = json.loads(line)
                except:
                    logger.warn(f' . Skip No.{index}. Parse Error: Line in Status File: {line}')
                    continue

                if status['model_name'] not in model_ids:
                    logger.info(f' . Skip No.{index}. Not In Model ID List.')
                    continue

                logger.info(f' . Skip No.{index}. This Model Converted Before With Status: \"{status["flag"]}\".')
                model_ids = model_ids - {status['model_name']}
    else:
        logger.info(f'-> Not Found Existing Status Files')

    convert_cache_dirpath = cache_dirpath.joinpath('Convert')
    create_dir(convert_cache_dirpath)
    onnx_model_filepath = convert_cache_dirpath.joinpath('model.onnx')

    logger.info(f'-> Instances Creating ...')
    for index, model_id in enumerate(model_ids, start=1):
        logger.info(f' # No.{index} Model ID = {model_id}: Now Converting ...') 
        logger.info(f'   v Converting TorchVision Model into ONNX:')
        model_input = get_torchvision_model_input(model_id)
        if model_input is None:
            flag = f'unknown_input-model_type:{get_torchvision_model_module(model_id)}'
            logger.warn(f'   - Conversion Not Success - Flag: {flag}.')
            save_status(status_filepath, dict(model_name=model_id, flag=flag))
            continue
        else:
            model = torchvision.models.get_model(model_id, weights=None)
            torch.onnx.export(model, model_input, str(onnx_model_filepath), verbose=True)
        logger.info(f'   ^ Finished.')

        model_info = get_torchvision_model_info(model_id)
        annotations = get_heuristic_annotations(model_id, model_info['metrics'])

        logger.info(f'   v Converting ONNX Model into NetworkX ...')
        try:
            instance = Instance(
                model=onnx_model_filepath,
                labels=dict(
                    model_source='TorchVision',
                    model_name=model_id,
                    onnx_model_filename=onnx_model_filepath.name,
                    download=None,
                    like=None,
                    tag=None,
                    readme=None,
                    annotations=annotations
                )
            )
            instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace(' ', '_').replace('/', '--TV--'), 'TorchVision', onnx_model_filepath.stem))
            instance.save(instance_save_dirpath)
            logger.info(f'     ┌ No.0 Converted')
            logger.info(f'     | From: {onnx_model_filepath}')
            logger.info(f'     └ Save: {instance_save_dirpath}')
            flag = 'success'
        except Exception as exception:
            logger.info(f'     ┌ No.0 Error')
            logger.error(f'    └ [ONNX -> NetworkX Error] OR [Instance Saving Error] - {exception}')
            flag = 'fail'
        logger.info(f'   ^ Converted.')
        save_status(status_filepath, dict(model_name=model_id, flag=flag))
        delete_dir(convert_cache_dirpath, only_clean=True)

    logger.info(f'-> Instances Created.')
