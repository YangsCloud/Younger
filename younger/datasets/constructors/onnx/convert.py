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


import json
import pathlib

from onnx import hub

from younger.datasets.modules import Instance

from younger.commons.io import load_json, create_dir, delete_dir
from younger.commons.logging import logger

from younger.datasets.constructors.utils import get_instance_dirname
from younger.datasets.constructors.onnx.utils import get_onnx_model_info


def save_status(status_filepath: pathlib.Path, status: dict[str, str]):
    with open(status_filepath, 'a') as status_file:
        status = json.dumps(status)
        status_file.write(f'{status}\n')


def main(
    save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, model_ids_filepath: pathlib.Path, status_filepath: pathlib.Path,
):
    hub.set_dir(str(cache_dirpath.absolute()))
    logger.info(f'ONNX Hub cache location is set to: {hub.get_dir()}')

    model_ids: set[str] = set(load_json(model_ids_filepath))

    logger.info(f'-> Checking Existing Instances ...')
    for index, instance_dirpath in enumerate(save_dirpath.iterdir(), start=1):
        if len(model_ids) == 0:
            logger.info(f'-> Finished. All Models Have Been Already Converted.')
            break
        instance = Instance()
        instance.load(instance_dirpath)
        if instance.labels['model_source'] == 'ONNX':
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

    onnx_cache_dirpath = cache_dirpath.joinpath('ONNX')
    create_dir(onnx_cache_dirpath)
    hub.set_dir(onnx_cache_dirpath)

    logger.info(f'-> Instances Creating ...')
    for index, model_id in enumerate(model_ids, start=1):
        logger.info(f' # No.{index} Model ID = {model_id}: Now Converting ...') 
        model_info = get_onnx_model_info(model_id)

        logger.info(f'   v Converting ONNX Model into NetworkX ...')
        for convert_index, variation in enumerate(model_info['variations'], start=1):
            onnx_model = hub.load(model=model_id, opset=variation['opset'])
            onnx_model_filepath = onnx_cache_dirpath.joinpath(variation['path'])
            try:
                instance = Instance(
                    model=onnx_model,
                    labels=dict(
                        model_source='ONNX',
                        model_name=model_id,
                        onnx_model_filename=onnx_model_filepath.name,
                        download=None,
                        like=None,
                        tag=variation['tags'],
                        readme=None,
                        annotations=None
                    )
                )
                instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace(' ', '_').replace('/', '--TV--'), 'ONNX', f'{onnx_model_filepath.stem}-{convert_index}'))
                instance.save(instance_save_dirpath)
                logger.info(f'     ┌ No.{convert_index} (Opset={variation["opset"]}) Converted')
                logger.info(f'     | From: {onnx_model_filepath}')
                logger.info(f'     └ Save: {instance_save_dirpath}')
                flag = 'success'
            except Exception as exception:
                logger.info(f'     ┌ No.{convert_index} (Opset={variation["opset"]}) Error')
                logger.error(f'    └ [ONNX -> NetworkX Error] OR [Instance Saving Error] - {exception}')
                flag = 'fail'
        logger.info(f'   ^ Converted.')
        save_status(status_filepath, dict(model_name=model_id, flag=flag))
        delete_dir(onnx_cache_dirpath, only_clean=True)

    logger.info(f'-> Instances Created.')
