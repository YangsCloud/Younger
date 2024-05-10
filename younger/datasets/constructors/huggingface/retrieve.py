#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-04-05 09:41
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import json
import tqdm
import pandas
import pathlib
import multiprocessing

from typing import Literal
from huggingface_hub import login, list_metrics, HfFileSystem

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger

from younger.datasets.constructors.huggingface.utils import get_huggingface_model_infos, get_huggingface_model_ids, get_huggingface_tasks, check_huggingface_model_eval_results


def save_huggingface_models(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, json_indent: int | None):
    pass


def get_huggingface_model_label_status(model_id: str) -> tuple[str, bool]:
    return (model_id, check_huggingface_model_eval_results(model_id, HfFileSystem()))


def save_huggingface_model_infos(save_dirpath: pathlib.Path, json_indent: int | None, library: str | None = None, label: bool | None = True, token: str | None = None, force_reload: bool | None = None, worker_number: int = 10):
    login(token=token)
    suffix = f'_{library}.json' if library else '.json'
    save_filepath = save_dirpath.joinpath(f'model_infos{suffix}')
    if save_filepath.is_file() and not force_reload:
        model_infos = load_json(save_filepath)
        logger.info(f' -> Already Retrieved. Total {len(model_infos)} Model Infos{f" (Library - {library})" if library else ""}. Results From: \'{save_filepath}\'.')
    else:
        filter_list = [library] if library else None
        model_infos = list(get_huggingface_model_infos(filter_list=filter_list, full=True, config=True, token=token))
        logger.info(f' -> Total {len(model_infos)} Model Infos{f" (Library - {library})" if library else ""}.')
        logger.info(f' v Saving Results Into {save_filepath} ...')
        save_json(model_infos, save_filepath, indent=json_indent)
        logger.info(f' ^ Saved.')

    logger.info(f' -> Retrieve Label Status: {"Yes" if label is True else "No"}')
    if label:
        json_miwls_suffix = f'_{library}-with_label_status.json' if library else '-with_label_status.json'
        json_miwls_save_filepath = save_dirpath.joinpath(f'model_ids{json_miwls_suffix}')
        temp_miwls_suffix = f'_{library}-with_label_status.temp' if library else '-with_label_status.temp'
        temp_miwls_save_filepath = save_dirpath.joinpath(f'model_ids{temp_miwls_suffix}')
        if json_miwls_save_filepath.is_file() and not force_reload:
            model_ids_with_label_status = load_json(json_miwls_save_filepath)
            logger.info(f' -> Already Retrieved. Total {len(model_ids_with_label_status)} Model Ids With Label Status{f" (Library - {library})" if library else ""}. Results Saved In: \'{json_miwls_save_filepath}\'.')
        else:
            model_ids = set([model_info['id'] for model_info in model_infos])
            model_ids_with_label_status = dict()

            if temp_miwls_save_filepath.is_file():
                if force_reload:
                    temp_miwls_save_filepath.unlink()
                else:
                    with open(temp_miwls_save_filepath, 'r') as temp_miwls_save_file:
                        for line in temp_miwls_save_file:
                            model_id, label_status = json.loads(line)
                            model_ids_with_label_status[model_id] = label_status
                            model_ids.remove(model_id)
            else:
                temp_miwls_save_filepath.touch()

            logger.info(f' v Retrieving ...')
            with multiprocessing.Pool(worker_number) as pool:
                with tqdm.tqdm(total=len(model_ids)) as progress_bar:
                    for index, (model_id, label_status) in enumerate(pool.imap_unordered(get_huggingface_model_label_status, model_ids), start=1):
                        model_ids_with_label_status[model_id] = label_status
                        with open(temp_miwls_save_filepath, 'a') as temp_miwls_save_file:
                            model_id_with_label_status = json.dumps([model_id, label_status])
                            temp_miwls_save_file.write(f'{model_id_with_label_status}\n')
                        progress_bar.update()
            logger.info(f' ^ Retrieved.')
            logger.info(f' v Saving Results Into {json_miwls_save_filepath} ...')
            save_json(model_ids_with_label_status, json_miwls_save_filepath, indent=json_indent)
            logger.info(f' ^ Saved.')
    logger.info(f' => Finished')


def save_huggingface_model_ids(save_dirpath: pathlib.Path, json_indent: int | None, library: str | None = None, token: str | None = None):
    model_ids = list(get_huggingface_model_ids(library, token=token))
    suffix = f'_{library}.json' if library else '.json'
    save_filepath = save_dirpath.joinpath(f'model_ids{suffix}')
    save_json(model_ids, save_filepath, indent=json_indent)
    logger.info(f'Total {len(model_ids)} Model IDs{f" (Library - {library})" if library else ""}. Results Saved In: \'{save_filepath}\'.')


def save_huggingface_metrics(save_dirpath: pathlib.Path):
    metrics = list_metrics()
    ids = [metric.id for metric in metrics]
    descriptions = [metric.description for metric in metrics]
    data_frame = pandas.DataFrame({'Metric Names': ids, 'Descriptions': descriptions})
    save_filepath = save_dirpath.joinpath('huggingface_metrics.xlsx')
    data_frame.to_excel(save_filepath, index=False)
    logger.info(f'Total {len(metrics)} Metrics. Results Saved In: \'{save_filepath}\'.')


def save_huggingface_tasks(save_dirpath: pathlib.Path, json_indent: int | None):
    tasks = get_huggingface_tasks()
    save_filepath = save_dirpath.joinpath('huggingface_tasks.json')
    save_json(tasks, save_filepath, indent=json_indent)
    logger.info(f'Total {len(tasks)} Tasks. Results Saved In: \'{save_filepath}\'.')


def main(mode: Literal['Models', 'Model_Infos', 'Model_IDs', 'Metrics', 'Tasks'], save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, min_json: bool, **kwargs):
    assert mode in {'Models', 'Model_Infos', 'Model_IDs', 'Metrics', 'Tasks'}

    json_indent = None if min_json else 2

    if mode == 'Models':
        save_huggingface_models(save_dirpath, cache_dirpath, json_indent)
        return

    if mode == 'Model_Infos':
        save_huggingface_model_infos(save_dirpath, json_indent, library=kwargs['library'], label=kwargs['label'], token=kwargs['token'], force_reload=kwargs['force_reload'], worker_number=kwargs['worker_number'])
        return

    if mode == 'Model_IDs':
        save_huggingface_model_ids(save_dirpath, json_indent, library=kwargs['library'], token=kwargs['token'])
        return

    if mode == 'Metrics':
        save_huggingface_metrics(save_dirpath)
        return

    if mode == 'Tasks':
        save_huggingface_tasks(save_dirpath, json_indent)
        return