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

import os
import sys
import json
import psutil
import tarfile
import pathlib
import argparse

from huggingface_hub import login

from youngbench.logging import set_logger, logger
from youngbench.dataset.construct.utils.get_model import cache_model


def get_free(path: str | pathlib.Path) -> int:
    usage = psutil.disk_usage(path)
    return usage.free


def archive_cache(cache_dirpath: str | pathlib.Path, cache_savepath: str | pathlib.Path, only_tar: bool = True):
    if only_tar:
        with tarfile.open(cache_savepath, mode='w', dereference=False) as tar:
            tar.add(cache_dirpath, arcname="")
    else:
        with tarfile.open(cache_savepath, mode='w:gz', dereference=False) as tar:
            tar.add(cache_dirpath, arcname=os.path.basename(cache_dirpath))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HuggingFace Models ['Timm', 'Diffusers', 'Transformers', 'Sentence Transformers'] to ONNX format.")

    parser.add_argument('--disk-usage-percentage', type=float, default=0.9)
    parser.add_argument('--cache-dirpath', type=str, default='./Cache')
    parser.add_argument('--cache-tar-dirpath', type=str, default='./Cache_Tar')
    parser.add_argument('--fails-flag-path', type=str, default='./fails.flg')
    parser.add_argument('--cache-flag-path', type=str, default='./cache.flg')
    parser.add_argument('--model-ids-filepath', type=str, default='./model_ids.json')

    parser.add_argument('--hf-token', type=str, default=None)

    parser.add_argument('--logging-path', type=str, default=None)

    parser.add_argument('--ignore', action='store_true')

    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--compress', action='store_true')

    args = parser.parse_args()

    if args.hf_token is not None:
        login(token=args.hf_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    du_per = args.disk_usage_percentage
    assert 0 < du_per and du_per <= 1

    cache_flags = set()
    cache_flag_path = pathlib.Path(args.cache_flag_path)
    if cache_flag_path.is_file():
        with open(cache_flag_path, 'r') as f:
            for line in f:
                model_args_json = line.strip()
                if len(model_args_json) == 0:
                    continue
                model_args_dict: dict = json.loads(model_args_json)
                model_id = model_args_dict.pop('model_id')
                assert model_id not in cache_flags
                cache_flags.add(model_id)
    logger.info(f'Already Cached: {len(cache_flags)}')

    fails_flags = set()
    fails_flag_path = pathlib.Path(args.fails_flag_path)
    if fails_flag_path.is_file():
        with open(fails_flag_path, 'r') as f:
            for line in f:
                model_id = line.strip()
                if len(model_id) == 0:
                    continue
                assert model_id not in fails_flags
                fails_flags.add(model_id)
    logger.info(f'Previous Failed: {len(fails_flags)}')
 
    model_ids_filepath = pathlib.Path(args.model_ids_filepath)
    logger.info(f"Loading Model IDs (To Be Cached): {model_ids_filepath.absolute()} ...")
    assert model_ids_filepath.is_file()
    model_ids = list()
    with open(model_ids_filepath, 'r') as midsf:
        model_ids = json.load(midsf)
    logger.info(f"Load Total {len(model_ids)} Model IDs.")

    cache_dirpath = pathlib.Path(args.cache_dirpath)
    cache_dirpath.mkdir(parents=True, exist_ok=True)
    logger.info(f"User specified cache folder: {cache_dirpath.absolute()}")
    cache_tar_dirpath = pathlib.Path(args.cache_tar_dirpath)
    cache_tar_dirpath.mkdir(parents=True, exist_ok=True)
    logger.info(f"User specified cache Tar folder: {cache_tar_dirpath.absolute()}")

    total_free = get_free(cache_dirpath)

    tobeuse_disk = total_free * du_per
    logger.info(f'Current Disk Space Left: {total_free / 1024 / 1024 / 1024:.3f} GB. {du_per*100:.2f}% = {tobeuse_disk / 1024 / 1024 / 1024:.3f} GB will be used.')

    flags = set()
    flags.update(cache_flags)
    flags.update(fails_flags)
    for index, model_id in enumerate(model_ids):
        if get_free(cache_dirpath) < total_free - tobeuse_disk:
            logger.info(f'- Stop! Reach Half of The Maximum Disk Usage!')
            break
        if model_id not in flags:
            # try:
            logger.info(f'= v. Now Cache {model_id}')
            try:
                this_cache_dirpath = cache_dirpath.joinpath(f'No-{index}')
                this_cache_dirpath.mkdir(parents=True, exist_ok=True)
                cached_args_dict = cache_model(model_id, cache_dir=str(this_cache_dirpath), monolith=False)
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
            this_cache_tar_dirpath = cache_tar_dirpath.joinpath(f'No-{index}.tar.gz')
            logger.info(f'... Begin Tar Cache: {this_cache_dirpath} -> To {this_cache_tar_dirpath}.')
            archive_cache(this_cache_dirpath, str(this_cache_tar_dirpath), not args.compress)
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
