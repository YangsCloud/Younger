#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2023-11-10 15:24
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json
import pathlib
import argparse

from huggingface_hub import login, snapshot_download
from optimum.exporters.onnx import main_export

from youngbench.logging import set_logger, logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HuggingFace Models ['Timm', 'Diffusers', 'Transformers', 'Sentence Transformers'] to ONNX format.")

    # Model Info Dir
    parser.add_argument('--key', type=str, default=None)

    parser.add_argument('--flag-path', type=str, default='./cho.flg')
    parser.add_argument('--info-dirpath', type=str, default='./')
    parser.add_argument('--save-dirpath', type=str, default='./')

    parser.add_argument('--logging-path', type=str, default=None)

    parser.add_argument('--api-token', type=str, default=None)

    parser.add_argument('--ignore', action='store_true')

    args = parser.parse_args()

    if args.api_token is not None:
        login(token=args.api_token)

    if args.logging_path is not None:
        set_logger(path=args.logging_path)

    if args.key is not None:
        load_key = args.key

    flag_path = pathlib.Path(args.flag_path)
    flags = set()
    if flag_path.is_file():
        with open(flag_path, 'r') as f:
            for line in f:
                model_id = line.strip()
                flags.add(model_id)

    info_dirpath = pathlib.Path(args.info_dirpath).joinpath('model_infos')
    def model_infos():
        part_index = 0
        while True:
            part_info_filepath = info_dirpath.joinpath(f'part_{part_index}-{load_key}.json')
            part_index += 1
            if part_info_filepath.is_file():
                part_info = list()
                with open(part_info_filepath, 'r') as f:
                    part_info = json.load(f)
                for model_info in part_info:
                    yield model_info
            else:
                break

    save_dirpath = pathlib.Path(args.save_dirpath)
    save_dirpath = save_dirpath.joinpath('onnx_models')
    save_dirpath.mkdir(parents=True, exist_ok=True)

    index = 0
    for model_info in model_infos():
        model_id = model_info['id']
        if model_id not in flags:
            onnx_model_save_path = save_dirpath.joinpath(model_id)
            try:
                logger.info(f'= v. Fin/To={len(flags)}/{index} - Now export {model_id}')
                convert_args = snapshot_download(model_id, repo_type='model', resume_download=True)
                main_export()
                flags.add(model_id)
                with open(flag_path, 'a') as f:
                    f.write(f'{model_id}\n')
                logger.info(f'= ^. Fin/To={len(flags)}/{index} - Finish Exporting')
            except Exception as e:
                logger.error(f'E: {e}')
                logger.error(f'There is an error occurred during exporting onnx model, please re-run the script or stop the process.')
                if not args.ignore:
                    user_input = input("Do you want to continue? (yes/no): ")
                    if user_input.lower() == 'yes':
                        print("Continuing with the process...")
                    else:
                        print("Process aborted.")
                        sys.exit(1)
        else:
            continue