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

from typing import Literal

from optimum.exporters.onnx import main_export
from huggingface_hub import HfFileSystem, login
from huggingface_hub.utils._errors import RepositoryNotFoundError

from younger.commons.io import load_json
from younger.commons.logging import logger

from younger.datasets.modules import Instance

from younger.datasets.constructors.utils import convert_bytes, get_instance_dirname
from younger.datasets.constructors.huggingface.utils import infer_model_size, clean_default_cache_repo, clean_specify_cache_repo, get_huggingface_model_readme, get_huggingface_model_card_data_from_readme
from younger.datasets.constructors.huggingface.annotations import get_heuristic_annotations


def main(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, model_ids_filepath: pathlib.Path, status_filepath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', threshold: int | None = None, huggingface_token: str | None = None):
    assert device in {'cpu', 'cuda'}

    if huggingface_token is not None:
        login(huggingface_token)

    hf_file_system = HfFileSystem()
    huggingface_cache_dirpath = cache_dirpath.joinpath('HuggingFace')
    convert_cache_dirpath = cache_dirpath.joinpath('Convert')

    model_ids = set(load_json(model_ids_filepath))

    logger.info(f'-> Checking Existing Instances ...')
    for index, instance_dirpath in enumerate(save_dirpath.iterdir()):
        if len(model_ids) == 0:
            logger.info(f'Finished. All Models Have Been Already Converted.')
            break
        instance = Instance()
        instance.load(instance_dirpath)
        if instance.labels['model_source'] == 'HuggingFace':
            logger.info(f'Skip Total {index} - {instance.labels["model_name"]}')
            model_ids = model_ids - {instance.labels['model_name']}

    logger.info(f'-> Instances Creating ...')
    for index, model_id in enumerate(model_ids, start=1):
        infered_model_size = infer_model_size(model_id)
        if threshold is None:
            pass
        else:
            if infered_model_size > threshold:
                logger.warn(f'Model Size: {convert_bytes(infered_model_size)} Larger Than Threshold! Skip.')
                with open(status_filepath, 'a') as status_file:
                    status = json.dumps(dict(model_name=model_id, status='oom'))
                    status_file.write(f'{status}\n')
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
            logger.error(f'Model ID = {model_id}: Conversion Error - {error}')

        logger.info(f'     Infered Repo Size = {convert_bytes(infered_model_size)}')

        onnx_model_filenames: list[str] = list()
        for filepath in convert_cache_dirpath.iterdir():
            if filepath.suffix == '.onnx':
                onnx_model_filenames.append(filepath.name)
        logger.info(f' ^ - Converted To ONNX: Got {len(onnx_model_filenames)} ONNX Models.')

        readme = None
        labels = None
        try:
            readme = get_huggingface_model_readme(model_id, hf_file_system)
            card_data = get_huggingface_model_card_data_from_readme(readme)
            labels = get_heuristic_annotations(model_id, card_data)
        except Exception as error:
            logger.error(f'Skip Label For Model ID = {model_id}: Error Occur While Extracting Model Card - {error}')

        for convert_index, onnx_model_filename in enumerate(onnx_model_filenames, start=1):
            onnx_model_filepath = convert_cache_dirpath.joinpath(onnx_model_filename)
            logger.info(f'      > Converting ONNX -> NetworkX: ONNX Filepath - {onnx_model_filepath}')
            try:
                instance = Instance(model=onnx_model_filepath, labels=dict(model_source='HuggingFace', model_name=model_id, onnx_model_filename=onnx_model_filename, readme=readme, labels=labels))
                instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace('/', '--HF--'), 'HuggingFace', onnx_model_filename))
                instance.save(instance_save_dirpath)
                logger.info(f'        No.{convert_index} Instance Saved: {instance_save_dirpath}')
                with open(status_filepath, 'a') as status_file:
                    status = json.dumps(dict(model_name=model_id, status='succ'))
                    status_file.write(f'{status}\n')
            except Exception as error:
                logger.error(f'Error! [ONNX -> NetworkX Error] OR [Instance Saving Error] - {error}')
                with open(status_filepath, 'a') as status_file:
                    status = json.dumps(dict(model_name=model_id, status='fail'))
                    status_file.write(f'{status}\n')
            logger.info(f'      > Converted.')
        clean_default_cache_repo(model_id)
        clean_specify_cache_repo(model_id, convert_cache_dirpath)
        clean_specify_cache_repo(model_id, huggingface_cache_dirpath)

    logger.info(f'-> Instances Created.')