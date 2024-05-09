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


import re
import json
import pathlib
import multiprocessing

from typing import Literal

from optimum.exporters.onnx import main_export
from huggingface_hub import HfFileSystem, login, hf_hub_download
from huggingface_hub.utils._errors import RepositoryNotFoundError

from younger.commons.io import load_json, create_dir, delete_dir
from younger.commons.logging import logger

from younger.datasets.modules import Instance

from younger.datasets.constructors.utils import convert_bytes, get_instance_dirname
from younger.datasets.constructors.huggingface.utils import infer_model_size, clean_default_cache_repo, clean_specify_cache_repo, get_huggingface_model_readme, get_huggingface_model_card_data_from_readme, get_huggingface_model_info
from younger.datasets.constructors.huggingface.annotations import get_heuristic_annotations


def save_status(status_filepath: pathlib.Path, status: dict[str, str]):
    with open(status_filepath, 'a') as status_file:
        status = json.dumps(status)
        status_file.write(f'{status}\n')


def clean_all_cache(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path):
    clean_default_cache_repo(model_id)
    delete_dir(convert_cache_dirpath, only_clean=True)
    clean_specify_cache_repo(model_id, huggingface_cache_dirpath)


def safe_optimum_export(model_id: str, convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, device: str, flag_queue: multiprocessing.Queue):
    try:
        main_export(model_id, convert_cache_dirpath, device=device, cache_dir=huggingface_cache_dirpath, monolith=True, do_validation=False, trust_remote_code=True, no_post_process=True)
        flag_queue.put('success')
    except MemoryError as error:
        logger.error(f'Model ID = {model_id}: Skip! Maybe OOM - {error}')
        flag_queue.put('memory_error')
    except RepositoryNotFoundError as error:
        logger.error(f'Model ID = {model_id}: Skip! Maybe Deleted By Author - {error}')
        flag_queue.put('access_error')
    except Exception as error:
        logger.error(f'Model ID = {model_id}: Conversion Error - {error}')
        flag_queue.put('convert_error')


def convert_optimum(model_id: set[str], convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    assert device in {'cpu', 'cuda'}

    flag_queue = multiprocessing.Queue()
    subprocess = multiprocessing.Process(target=safe_optimum_export, args=(model_id, convert_cache_dirpath, huggingface_cache_dirpath, device, flag_queue))
    subprocess.start()
    subprocess.join()
    if flag_queue.empty():
        logger.warn(f'Export Process May Be Killed By System! Skip.')
        flag = 'system_kill'
    else:
        flag = flag_queue.get()

    onnx_model_filepaths: list[pathlib.Path] = list()
    for filepath in convert_cache_dirpath.iterdir():
        if filepath.suffix == '.onnx':
            onnx_model_filepaths.append(filepath)
    if len(onnx_model_filepaths) == 0:
        flag = 'convert_nothing'
    return flag, onnx_model_filepaths


def convert_onnx(model_id: set[str], convert_cache_dirpath: pathlib.Path, huggingface_cache_dirpath: pathlib.Path, hf_file_system: HfFileSystem, device: Literal['cpu', 'cuda'] = 'cpu') -> tuple[str, list[pathlib.Path]]:
    flag = 'success'
    onnx_model_filepaths: list[pathlib.Path] = list()
    remote_onnx_model_filepaths = hf_file_system.glob(model_id + '/' + '**.onnx')
    for remote_onnx_model_filepath in remote_onnx_model_filepaths:
        onnx_model_filepath = hf_hub_download(model_id, remote_onnx_model_filepath[len(model_id):], cache_dir=huggingface_cache_dirpath, local_dir=convert_cache_dirpath)
        onnx_model_filepaths.append(pathlib.Path(onnx_model_filepath))
    if len(onnx_model_filepaths) == 0:
        flag = 'convert_nothing'
    return flag, onnx_model_filepaths


def main(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, model_ids_filepath: pathlib.Path, status_filepath: pathlib.Path, device: Literal['cpu', 'cuda'] = 'cpu', model_size_threshold: int | None = None, huggingface_token: str | None = None, mode: Literal['optimum', 'onnx'] = 'optimum'):

    support_convert_method = dict(
        optimum = convert_optimum,
        onnx = convert_onnx,
    )

    assert mode in {'optimum', 'onnx', 'keras'}

    if huggingface_token is not None:
        login(huggingface_token)

    model_ids: set[str] = set(load_json(model_ids_filepath))

    logger.info(f'-> Checking Existing Instances ...')
    for index, instance_dirpath in enumerate(save_dirpath.iterdir(), start=1):
        if len(model_ids) == 0:
            logger.info(f'-> Finished. All Models Have Been Already Converted.')
            break
        instance = Instance()
        instance.load(instance_dirpath)
        if instance.labels['model_source'] == 'HuggingFace':
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

                if re.fullmatch(f'model_size_threshold_(\d+)', status['flag']):
                    origin_repo_size_threshold = int(re.fullmatch(f'model_size_threshold_(\d+)', status['flag']).group(1))
                    if model_size_threshold and model_size_threshold <= origin_repo_size_threshold:
                        model_ids = model_ids - {status['model_name']}
                        logger.info(f' . Skip No.{index}. This Model Converted Before, But Exceed The Threshold.')
                else:
                    logger.info(f' . Skip No.{index}. This Model Converted Before With Status: \"{status["flag"]}\".')
                    model_ids = model_ids - {status['model_name']}
    else:
        logger.info(f'-> Not Found Existing Status Files')

    hf_file_system = HfFileSystem()

    convert_cache_dirpath = cache_dirpath.joinpath('Convert' + mode.title())
    create_dir(convert_cache_dirpath)

    huggingface_cache_dirpath = cache_dirpath.joinpath('HuggingFace')
    create_dir(huggingface_cache_dirpath)

    logger.info(f'-> Instances Creating ...')
    for index, model_id in enumerate(sorted(model_ids), start=1):
        try:
            infered_model_size = infer_model_size(model_id)
        except Exception as error:
            logger.error(f' # No.{index} Model ID = {model_id}: Cannot Get The Model. Access Maybe Requested - {error}')
            save_status(status_filepath, dict(model_name=model_id, flag='access_error'))
            continue

        if model_size_threshold is None:
            pass
        else:
            if model_size_threshold < infered_model_size:
                logger.warn(f' # No.{index} Model ID = {model_id}: Model Size {convert_bytes(infered_model_size)} Larger Than Threshold! Skip.')
                save_status(status_filepath, dict(model_name=model_id, flag=f'model_size_threshold_{model_size_threshold}'))
                continue

        logger.info(f'-> Infered Model Size = {convert_bytes(infered_model_size)}')

        logger.info(f' # No.{index} Model ID = {model_id}: Now Converting ...') 
        logger.info(f'   v Converting HuggingFace Model into ONNX:')
        flag, onnx_model_filepaths = support_convert_method[mode](model_id, convert_cache_dirpath, huggingface_cache_dirpath, hf_file_system, device)
        logger.info(f'   ^ Finish With Flag - \"{flag}\".')

        if flag == 'success':
            pass
        else:
            logger.warn(f'   - Conversion Not Success - Flag: {flag}.')
            save_status(status_filepath, dict(model_name=model_id, flag=flag))
            continue

        model_info = None
        readme = None
        card_data = None
        annotations = None
        try:
            model_info = get_huggingface_model_info(model_id)
            readme = get_huggingface_model_readme(model_id, hf_file_system)
            card_data = get_huggingface_model_card_data_from_readme(readme)
            annotations = get_heuristic_annotations(model_id, card_data)
        except Exception as error:
            logger.warn(f'   -> No Heuristic Annotations: {error}')

        logger.info(f'   v Converting ONNX Model into NetworkX ...')
        for convert_index, onnx_model_filepath in enumerate(onnx_model_filepaths, start=1):
            try:
                instance = Instance(
                    model=onnx_model_filepath,
                    labels=dict(
                        model_source='HuggingFace',
                        model_name=model_id,
                        onnx_model_filename=onnx_model_filepath.name,
                        download=model_info['downloads'],
                        like=model_info['likes'],
                        tag=model_info['tags'],
                        readme=readme,
                        annotations=annotations
                    )
                )
                instance_save_dirpath = save_dirpath.joinpath(get_instance_dirname(model_id.replace('/', '--HF--'), 'HuggingFace', onnx_model_filepath.name))
                instance.save(instance_save_dirpath)
                logger.info(f'     ┌ No.{convert_index} Converted')
                logger.info(f'     | From: {onnx_model_filepath}')
                logger.info(f'     └ Save: {instance_save_dirpath}')
                flag = 'success'
            except Exception as error:
                logger.info(f'     ┌ No.{convert_index} Error')
                logger.error(f'    └ [ONNX -> NetworkX Error] OR [Instance Saving Error] - {error}')
                flag = 'fail'
        logger.info(f'   ^ Converted.')
        save_status(status_filepath, dict(model_name=model_id, flag=flag))

        clean_all_cache(model_id, convert_cache_dirpath, huggingface_cache_dirpath)

    logger.info(f'-> Instances Created.')