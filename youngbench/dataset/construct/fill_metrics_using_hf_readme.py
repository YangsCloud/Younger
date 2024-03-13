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

from huggingface_hub import login, HfFileSystem

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.schema import Model
from youngbench.dataset.construct.utils.action import update_model_item_by_model_id, read_limit_model_items
from youngbench.dataset.construct.utils.extract import extract_all_metrics, filter_readme_filepaths, realtime_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enrich The Young Neural Network Architecture Dataset (YoungBench - Dataset).")

    parser.add_argument('--token', type=str, required=True)

    parser.add_argument('--number', type=int, default=1)

    parser.add_argument('--report', type=int, default=100)

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--save-dirpath', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    parser.add_argument('--only-download', action='store_true')

    args = parser.parse_args()

    assert args.number > 0

    assert args.report > 0

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    fs = HfFileSystem()

    logger.info(f' = Fetching *README* & Filling *Metrics* For All Models ...')

    # YBDM.Maintaining = False: Basic Fetched;
    # YBDM.Maintaining = True: README Checked;
    # All Items Must Be Inserted Into HFI **BEFORE** *README* and *Metrics* Processed;
    # Items in HFI Can Be Downloaded Then;
    index = 0
    success = 0
    failure = 0
    filter = {
        'maintaining': {
            '_eq': False
        },
        'model_source': {
            '_eq': 'HuggingFace'
        }
    }
    batch = list()
    models = read_limit_model_items(args.token, limit=args.number, fields=['model_id'], filter=filter)
    while models:
        if models:
            if not args.only_download:
                logger.info(f' - Retrieved {len(models)} Model Items. Now Extract All Metrics For These Models.')
            for model in models:
                index += 1
                filepaths = fs.glob(f'{model.model_id}/**')
                all_metrics = extract_all_metrics(filter_readme_filepaths(filepaths), fs, save_dirpath=args.save_dirpath, only_download=args.only_download)
                if args.only_download:
                    if index % args.report == 0:
                        logger.info(f' - Only Download - [Index: {index}]')
                    continue
                new_model = Model(all_metrics=all_metrics, maintaining=True)

                if len(batch) < args.number:
                    batch.append((new_model, model.model_id))
                    realtime_write('.')

                if len(batch) == args.number:
                    realtime_write('\n')
                    logger.info(f' Updating Records:')
                    for new_model, model_id in batch:
                        result = update_model_item_by_model_id(model_id, new_model, args.token)
                        realtime_write('.')
                        if result is None:
                            failure += 1
                            logger.info(f' - No.{index} Item Creation Error - Model ID: {model_id}')
                        else:
                            success += 1
                    batch = list()

                if index % args.report == 0:
                    realtime_write('\n')
                    logger.info(f' - [Index: {index}] Success/Failure/OnRoud:{success}/{failure}/{len(batch)}')
        models = read_limit_model_items(args.token, limit=args.number, fields=['model_id'], filter=filter)

    if not args.only_download:
        if len(batch) > 0:
            for new_model, model_id in batch:
                result = update_model_item_by_model_id(model_id, new_model, args.token)
                if result is None:
                    failure += 1
                    logger.info(f' - No.{index} Item Creation Error - Model ID: {model_id}')
                else:
                    success += 1
            batch = list()
        logger.info(f' = END [Index: {index}] Success/Failure/OnRoud:{success}/{failure}/{len(batch)}')
    else:
        logger.info(f' = END = Only Download - [Index: {index}]')