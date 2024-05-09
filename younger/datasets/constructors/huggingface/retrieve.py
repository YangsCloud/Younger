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


import tqdm
import pandas
import pathlib
import multiprocessing

from typing import Literal, Any
from huggingface_hub import login, list_metrics, HfFileSystem

from younger.commons.io import save_json
from younger.commons.logging import logger

from younger.datasets.constructors.huggingface.utils import get_huggingface_model_infos, get_huggingface_model_ids, get_huggingface_tasks, check_huggingface_model_eval_results


def get_huggingface_model_label_status(model_info: dict[str, Any]) -> bool:
    return check_huggingface_model_eval_results(model_info['id'], HfFileSystem())


def save_huggingface_models(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, json_indent: int | None):
    pass


def save_huggingface_model_infos(save_dirpath: pathlib.Path, json_indent: int | None, library: str | None = None, label: bool | None = True, token: str | None = None, worker_number: int = 10):
    login(token=token)

    filter_list = [library] if library else None
    model_infos = list(get_huggingface_model_infos(filter_list=filter_list, full=True, config=True, token=token))

    logger.info(f' -> Retrieve Label Status: {"Yes" if label is True else "No"}')
    if label:
        suffix = f'_{library}-without_label_status.json' if library else '-without_label_status.json'
        save_filepath = save_dirpath.joinpath(f'model_infos{suffix}')
        save_json(model_infos, save_filepath, indent=json_indent)
        logger.info(f'   Results Without Label Status Saved In: \'{save_filepath}\'.')
        logger.info(f' v Retrieving ...')
        all_label_status = list()
        with multiprocessing.Pool(worker_number) as pool:
            with tqdm.tqdm(total=len(model_infos)) as progress_bar:
                for label_status in pool.imap_unordered(get_huggingface_model_label_status, model_infos):
                    all_label_status.append(label_status)
                    progress_bar.update()
        model_infos = [dict(model_info=model_info, label_status=label_status) for model_info, label_status in zip(model_infos, all_label_status)]
        logger.info(f' ^ Retrieved')

    suffix = f'_{library}-with_label_status.json' if library else '-with_label_status.json'
    save_filepath = save_dirpath.joinpath(f'model_infos{suffix}')
    save_json(model_infos, save_filepath, indent=json_indent)
    logger.info(f'Total {len(model_infos)} Model Infos{f" (Library - {library})" if library else ""}. Results Saved In: \'{save_filepath}\'.')


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
        save_huggingface_model_infos(save_dirpath, json_indent, library=kwargs['library'], label=kwargs['label'], token=kwargs['token'], worker_number=kwargs['worker_number'])
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