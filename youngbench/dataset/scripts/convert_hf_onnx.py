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

from typing import Dict
from huggingface_hub import login
from optimum.exporters.tasks import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.exporters.onnx.convert import onnx_export_from_model
from optimum.utils.save_utils import maybe_load_preprocessors

from youngbench.logging import set_logger, logger


def mine_export(model_id, output, cache_dir, model_args_dict):

    task=model_args_dict['task']
    subfolder=model_args_dict['subfolder']
    revision=model_args_dict['revision']
    use_auth_token=model_args_dict['use_auth_token']
    local_files_only=model_args_dict['local_files_only']
    force_download=model_args_dict['force_download']
    trust_remote_code=model_args_dict['trust_remote_code']
    framework=model_args_dict['framework']
    torch_dtype=model_args_dict['torch_dtype']
    device=model_args_dict['device']
    library_name=model_args_dict['library_name']
    loading_kwargs=model_args_dict['loading_kwargs']
    opset=model_args_dict['opset']
    optimize=model_args_dict['optimize']
    monolith=model_args_dict['monolith']
    no_post_process=model_args_dict['no_post_process']
    atol=model_args_dict['atol']
    do_validation=model_args_dict['do_validation']
    model_kwargs=model_args_dict['model_kwargs']
    custom_onnx_configs=model_args_dict['custom_onnx_configs']
    fn_get_submodels=model_args_dict['fn_get_submodels']
    use_subprocess=model_args_dict['use_subprocess']
    _variant=model_args_dict['_variant']
    legacy=model_args_dict['legacy']
    trust_remote_code=model_args_dict['trust_remote_code']
    no_dynamic_axes=model_args_dict['no_dynamic_axes']
    do_constant_folding=model_args_dict['do_constant_folding']

    model = TasksManager.get_model_from_task(
        task,
        model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
        library_name=library_name,
        **loading_kwargs,
    )

    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = model_args_dict.get(input_name, DEFAULT_DUMMY_SHAPES[input_name])

    # The preprocessors are loaded as they may be useful to export the model. Notably, some of the static input shapes may be stored in the
    # preprocessors config.
    preprocessors = maybe_load_preprocessors(
        model_id, subfolder=subfolder, trust_remote_code=trust_remote_code
    )

    onnx_export_from_model(
        model=model,
        output=output,
        opset=opset,
        optimize=optimize,
        monolith=monolith,
        no_post_process=no_post_process,
        atol=atol,
        do_validation=do_validation,
        model_kwargs=model_kwargs,
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=fn_get_submodels,
        _variant=_variant,
        legacy=legacy,
        preprocessors=preprocessors,
        device=device,
        no_dynamic_axes=no_dynamic_axes,
        task=task,
        use_subprocess=use_subprocess,
        do_constant_folding=do_constant_folding,
        **input_shapes,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HuggingFace Models ['Timm', 'Diffusers', 'Transformers', 'Sentence Transformers'] to ONNX format.")

    # Model Info Dir
    parser.add_argument('--key', type=str, default=None)

    parser.add_argument('--info-dirpath', type=str, default='./')
    parser.add_argument('--cache-dirpath', type=str, default='./')
    parser.add_argument('--cache-flag-path', type=str, default='./cache.flg')
    parser.add_argument('--save-dirpath', type=str, default='./')
    parser.add_argument('--save-flag-path', type=str, default='./save.flg')

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

    cache_flag_path = pathlib.Path(args.cache_flag_path)
    all_cache_args = dict()
    if cache_flag_path.is_file():
        with open(cache_flag_path, 'r') as f:
            for line in f:
                model_args_json = line.strip()
                model_args_dict: Dict = json.loads(model_args_json)
                model_id = model_args_dict.pop('model_id')
                assert model_id not in all_cache_args
                all_cache_args[model_id] = model_args_dict

    save_flag_path = pathlib.Path(args.save_flag_path)
    flags = set()
    if save_flag_path.is_file():
        with open(save_flag_path, 'r') as f:
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

    cache_dirpath = pathlib.Path(args.cache_dirpath)
    logger.info(f"User specified cache folder: {cache_dirpath.absolute()}")
    if not cache_dirpath.is_dir():
        user_input = input("Cache diretory does not exist, do you want to continue? (yes/no): ")
        if user_input.lower() == 'yes':
            print("Continuing with the process under empty cache directory...")
        else:
            print("Process aborted.")
            sys.exit(1)

    index = 0
    for model_info in model_infos():
        model_id = model_info['id']
        if model_id not in flags:
            if model_id not in all_cache_args:
                continue
            onnx_model_save_path = save_dirpath.joinpath(model_id)
            try:
                logger.info(f'= v. Fin/To={len(flags)}/{index} - Now export {model_id}')
                mine_export(model_id, onnx_model_save_path, str(cache_dirpath), all_cache_args[model_id])
                flags.add(model_id)
                with open(save_flag_path, 'a') as f:
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