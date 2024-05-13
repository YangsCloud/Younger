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


import pathlib

from typing import Literal

from younger.commons.io import load_json, save_json
from younger.commons.logging import logger

from younger.datasets.constructors.torchvision.utils import get_torchvision_model_infos, get_torchvision_model_ids


def save_torchvision_models(save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, json_indent: int | None):
    pass


def save_torchvision_model_infos(save_dirpath: pathlib.Path, json_indent: int | None, force_reload: bool | None = None):
    save_filepath = save_dirpath.joinpath(f'model_infos.json')
    if save_filepath.is_file() and not force_reload:
        model_infos = load_json(save_filepath)
        logger.info(f' -> Already Retrieved. Total {len(model_infos)} Model Infos. Results From: \'{save_filepath}\'.')
    else:
        model_infos = get_torchvision_model_infos()
        logger.info(f' -> Total {len(model_infos)} Model Infos.')
        logger.info(f' v Saving Results Into {save_filepath} ...')
        save_json(model_infos, save_filepath, indent=json_indent)
        logger.info(f' ^ Saved.')
    logger.info(f' => Finished')


def save_torchvision_model_ids(save_dirpath: pathlib.Path, json_indent: int | None):
    model_ids = get_torchvision_model_ids()
    save_filepath = save_dirpath.joinpath(f'model_ids.json')
    save_json(model_ids, save_filepath, indent=json_indent)
    logger.info(f'Total {len(model_ids)} Model IDs. Results Saved In: \'{save_filepath}\'.')


def main(mode: Literal['Models', 'Model_Infos', 'Model_IDs', 'Model_Inputs'], save_dirpath: pathlib.Path, cache_dirpath: pathlib.Path, min_json: bool, **kwargs):
    assert mode in {'Models', 'Model_Infos', 'Model_IDs'}

    json_indent = None if min_json else 2

    if mode == 'Models':
        save_torchvision_models(save_dirpath, cache_dirpath, json_indent)
        return

    if mode == 'Model_Infos':
        save_torchvision_model_infos(save_dirpath, json_indent, force_reload=kwargs['force_reload'])
        return

    if mode == 'Model_IDs':
        save_torchvision_model_ids(save_dirpath, json_indent)
        return