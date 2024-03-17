#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-03-17 21:14
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json
import psutil
import pathlib
import argparse

from huggingface_hub import login

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.get_model import cache_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HuggingFace Models ['Timm', 'Diffusers', 'Transformers', 'Sentence Transformers'] to ONNX format.")

    parser.add_argument('--cache-dirpath', type=str, default='./')
    parser.add_argument('--fails-flag-path', type=str, default='./fails.flg')
    parser.add_argument('--cache-flag-path', type=str, default='./cache.flg')
    parser.add_argument('--model-ids-filepath', type=str, default='./model_ids.json')

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    parser.add_argument('--ignore', action='store_true')

    parser.add_argument('--yes', action='store_true')

    args = parser.parse_args()

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    fails_flag_path = pathlib.Path(args.fails_flag_path)
    cache_flag_path = pathlib.Path(args.cache_flag_path)
    flags = set()
    if cache_flag_path.is_file():
        with open(cache_flag_path, 'r') as f:
            for line in f:
                model_args_json = line.strip()
                if len(model_args_json) == 0:
                    continue
                model_args_dict: dict = json.loads(model_args_json)
                model_id = model_args_dict.pop('model_id')
                assert model_id not in flags
                flags.add(model_id)

    model_ids_filepath = pathlib.Path(args.model_ids_filepath)
    logger.info(f"Loading Model IDs (To Be Cached): {model_ids_filepath.absolute()} ...")
    assert model_ids_filepath.is_file()
    model_ids = list()
    with open(model_ids_filepath, 'r') as midsf:
        model_ids = json.load(midsf)
    logger.info(f"Load Total {len(model_ids)} Model IDs.")

    cache_dirpath = pathlib.Path(args.cache_dirpath)
    logger.info(f"User specified cache folder: {cache_dirpath.absolute()}")

    index = 0
    for model_id in model_ids:
        if model_id not in flags:
            # try:
            logger.info(f'= v. Finished/Total ({len(flags)}/{len(model_ids)}) - Now Cache {model_id}')
            try:
                cached_args_dict = cache_model(model_id, cache_dir=str(cache_dirpath), monolith=False)
            except Exception as e:
                logger.info(f' - Model ID:{model_id} - Cache Failed Finished.')
                with open(fails_flag_path, 'a') as f:
                    f.write(f'{model_id}\n')
                logger.error(f'E: {e}')
                logger.error(f'There is an error occurred during cache onnx model, please re-run the script or stop the process.')
                if not args.ignore:
                    if args.yes:
                        print("Continuing with the process...")
                        continue
                    user_input = input("Do you want to continue? (yes/no): ")
                    if user_input.lower() == 'yes':
                        print("Continuing with the process...")
                        continue
                    else:
                        print("Process aborted.")
                        sys.exit(1)
            assert model_id == cached_args_dict['model_id']
            flags.add(model_id)
            with open(cache_flag_path, 'a') as f:
                cached_args_json = json.dumps(cached_args_dict)
                f.write(f'{cached_args_json}\n')
            logger.info(f'= ^. Finished/Total ({len(flags)}/{len(model_ids)}) - \'{model_id}\' Finished.')
            # except Exception as e:
            #     logger.error(f'E: {e}')
            #     logger.error(f'There is an error occurred during cache onnx model, please re-run the script or stop the process.')
            #     if not args.ignore:
            #         user_input = input("Do you want to continue? (yes/no): ")
            #         if user_input.lower() == 'yes':
            #             print("Continuing with the process...")
            #         else:
            #             print("Process aborted.")
            #             sys.exit(1)
        else:
            continue

    logger.info("All Done!")
