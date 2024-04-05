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

from typing import Literal

from optimum.exporters.onnx import main_export
from huggingface_hub.utils._errors import RepositoryNotFoundError

from younger.datasets.modules import Instance

from younger.datasets.utils.logging import logger

from younger.datasets.constructors.utils import convert_bytes, get_instance_dirname
from younger.datasets.constructors.huggingface.utils import get_huggingface_model_infos, infer_model_size, clean_default_cache_repo, clean_specify_cache_repo


def main(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, library: str, device: Literal['cpu', 'cuda'] = 'cpu', threshold: int | None = None):
    assert device in {'cpu', 'cuda'}

    huggingface_cache_dirpath = cache_dirpath.joinpath('HuggingFace')
    convert_cache_dirpath = cache_dirpath.joinpath('Convert')

    model_infos = get_huggingface_model_infos(filter_list=[library], full=True, limit=1000, config=True)

    logger.info(f'-> Instances Creating ...')
    for index, model_info in enumerate(model_infos, start=1):
        model_id = model_info['id']
        infered_model_size = infer_model_size(model_id)
        if threshold is None:
            pass
        else:
            if infered_model_size > threshold:
                logger.warn(f'Model Size: {convert_bytes(infered_model_size)} Larger Than Threshold! Skip.')
                continue

        logger.info(f' # No.{index}: Now processing the model: {model_id} ...')
        logger.info(f' v - Converting HuggingFace Model into ONNX:')
        try:
            main_export(model_id, convert_cache_dirpath, device=device, cache_dir=huggingface_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
        except MemoryError as error:
            logger.error(f'Model ID = {model_id}: Skip! Maybe OOM - {error}')
        except RepositoryNotFoundError as error:
            logger.error(f'Model ID = {model_id}: Skip! Maybe Deleted By Author - {error}')
        except Exception as error:
            logger.error(f'Model ID = {model_id}: Conversion Error - {error} ')

        logger.info(f'     Infered Repo Size = {convert_bytes(infered_model_size)}')

        onnx_model_filenames = list()
        for filename in convert_cache_dirpath.iterdir():
            if filename.suffix == '.onnx':
                onnx_model_filenames.append(filename)
        logger.info(f' ^ - Converted To ONNX: Got {len(onnx_model_filenames)} ONNX Models.')

        for convert_index, onnx_model_filename in enumerate(onnx_model_filenames, start=1):
            onnx_model_filepath = convert_cache_dirpath.joinpath(onnx_model_filename)
            logger.info(f'      > Converting ONNX -> NetworkX: ONNX Filepath - {onnx_model_filepath}')
            try:
                instance = Instance(model=onnx_model_filepath, labels=dict(model_source='HuggingFace', model_name=model_id, onnx_model_filename=onnx_model_filename))
                instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace('/', '--HF--'), 'HuggingFace', onnx_model_filename))
                instance.save(instance_save_dirpath)
                logger.info(f'        No.{convert_index} Instance Saved: {instance_save_dirpath}')
            except Exception as error:
                logger.error(f'Error! [ONNX -> NetworkX Error] OR [Dataset Insertion Error] - {error}')
                pass
            logger.info(f'      > Converted.')
        
        clean_default_cache_repo(model_id)
        clean_specify_cache_repo(model_id, convert_cache_dirpath)

    logger.info(f'-> Instances Created.')