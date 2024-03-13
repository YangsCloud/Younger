#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-11 20:59
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib
import tempfile
import argparse

from huggingface_hub import login

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.get_info import get_hf_model_infos
from youngbench.dataset.construct.utils.schema import Model
from youngbench.dataset.construct.utils.action import create_model_item, read_model_items_manually, create_model_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    parser.add_argument('--tag', type=str, required=True)

    parser.add_argument('--token', type=str, required=True)

    parser.add_argument('--number', type=int, default=1)

    parser.add_argument('--report', type=int, default=100)

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    args = parser.parse_args()

    assert args.number > 0

    assert args.report > 0

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    logger.info(f' = Fetching Models\' Info ( Tag: {args.tag} ) ...')
    model_infos = get_hf_model_infos(filter_list=[args.tag], token=args.hf_token, full=True)

    index = 0
    exist_models = set()
    logger.info("Checking Exist Models... ")
    tempdir = tempfile.mkdtemp()
    export_filepath = pathlib.Path(tempdir).joinpath('eduhf')
    export_filepath.parent.mkdir(parents=True, exist_ok=True)
    exact_filepaths = list()
    logger.info(f'Retrieving All Exist Items Into Temp Files ...')
    for exact_filepath in read_model_items_manually(args.token, limit=1000, fields=['model_id'], export_filepath=export_filepath):
        logger.info(f'ID of Exist Items Saved Into {exact_filepath}')
        exact_filepaths.append(exact_filepath)

    logger.info(f'Recording All Exist Items Saved In Temp Files ...')
    for exact_filepath in exact_filepaths:
        with open(exact_filepath, 'r') as f:
            model_items = json.load(f)
            for model_item in model_items:
                index += 1
                exist_models.add(model_item['model_id'])
                if index % 1000 == 0:
                    logger.info(f" Total {index} Found ... ")
    logger.info(f"Finish Record. Total {index}.")

    index = 0
    success = 0
    failure = 0
    skip = 0

    batch = list()
    for model_info in model_infos:

        index += 1
        if model_info['id'] in exist_models:
            skip += 1
            logger.info(f' - No.{index} Item Exists - Skip - Model ID: {model_info["id"]}')

        else:
            model = Model(model_id=model_info['id'], model_source='HuggingFace', model_likes=model_info['likes'], model_downloads=model_info['downloads'])
            if len(batch) < args.number:
                batch.append(model)

            if len(batch) == args.number:
                def one_by_one(model):
                    global index, success, args, failure, logger
                    result = create_model_item(model, args.token)
                    if result is None:
                        failure += 1
                        logger.info(f' - No.{index} Item Creation Error - Model ID: {model.model_id}')
                    else:
                        exist_models.add(model.model_id)
                        success += 1

                if len(batch) == 1:
                    one_by_one(batch[0])
                else:
                    models = create_model_items(batch, args.token)
                    if len(models) == 0:
                        logger.info(f' - No. {index-len(batch)+1}-{index} Items Creation Has Error. Now Create Items 1-By-1!')
                        for model in batch:
                            one_by_one(model)
                    else:
                        success += len(models)
                        for model in models:
                            exist_models.add(model.model_id)

                batch = list()

        if index % args.report == 0:
            logger.info(f' - [Index: {index}] Success/Failure/Skip/OnRoud:{success}/{failure}/{skip}/{len(batch)}')

    if len(batch) > 0:
        models = create_model_items(batch, args.token)
        if len(models) == 0:
            logger.info(f' - No. {index-len(batch)+1}-{index} Items Creation Has Error. Now Create Items 1-By-1!')
            for model in batch:
                result = create_model_item(model, args.token)
                if result is None:
                    failure += 1
                    logger.info(f' - No.{index} Item Creation Error - Model ID: {model.model_id}')
                else:
                    exist_models.add(model.model_id)
                    success += 1
        else:
            success += len(models)
            batch = list()
    
    logger.info(f' = END [Index: {index}] Success/Failure/Skip/OnRoud:{success}/{failure}/{skip}/{len(batch)}')