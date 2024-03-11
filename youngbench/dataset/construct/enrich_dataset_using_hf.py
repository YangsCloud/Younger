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

import argparse

from huggingface_hub import login

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.get_info import get_hf_model_infos
from youngbench.dataset.construct.utils.schema import Model
from youngbench.dataset.construct.utils.action import create_model_item, read_model_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    parser.add_argument('--tag', type=str, required=True)

    parser.add_argument('--token', type=str, required=True)

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    args = parser.parse_args()

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    logger.info(f' = Fetching Models\' Info ( Tag: {args.tag} ) ...')
    model_infos = get_hf_model_infos(filter_list=[args.tag], token=args.hf_token, full=True)

    exist_models = set()
    for exist_model in read_model_items(args.token):
        exist_models.add(exist_model.model_id)

    index = 0
    success = 0
    failure = 0
    skip = 0
    for model_info in model_infos:
        index += 1
        if model_info['id'] in exist_models:
            skip += 1
            logger.info(f' - No.{index} Item Exists - Skip - Model ID: {model_info["id"]}')

        else:
            model = Model(model_id=model_info['id'], model_source='HuggingFace', model_likes=model_info['likes'], model_downloads=model_info['downloads'])
            # Optimize
            # exist_model = read_model_item_by_model_id(model_id=model.model_id, token=args.token)

            model = create_model_item(model, args.token)
            if model is None:
                failure += 1
                logger.info(f' - No.{index} Item Creation Error - Model ID: {model_info["id"]}')
            else:
                exist_models.add(model.model_id)
                success += 1

        if index % 100 == 0:
            logger.info(f' - [Index: {index}] Success/Failure/Skip:{success}/{failure}/{skip}')