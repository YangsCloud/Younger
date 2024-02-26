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
import torch
import pathlib
import argparse

from typing import Dict
from packaging import version
from huggingface_hub import login
from optimum.exporters.tasks import TasksManager
from optimum.configuration_utils import _transformers_version
from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED

from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer

from youngbench.logging import set_logger, logger


def cache_model(model_id, cache_dir, monolith) -> Dict:
    subfolder = ""
    revision = None
    use_auth_token = None
    local_files_only = False
    force_download = False
    trust_remote_code = False
    device = 'cpu'
    framework = None


    dtype = "fp32"
    task = "auto"
    original_task = task
    task = TasksManager.map_from_synonym(task)
    framework = TasksManager.determine_framework(model_id, subfolder=subfolder, framework=framework)
    library_name = TasksManager.infer_library_from_model(model_id)
    torch_dtype = None
    if framework == "pt":
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_id)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    
    custom_architecture = False
    loading_kwargs = {}
    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        model_type = config.model_type.replace("_", "-")

        if model_type not in TasksManager._SUPPORTED_MODEL_TYPE:
            custom_architecture = True
        elif task not in TasksManager.get_supported_tasks_for_model_type(
            model_type, "onnx", library_name=library_name
        ):
            if original_task == "auto":
                autodetected_message = " (auto-detected)"
            else:
                autodetected_message = ""
            model_tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type, exporter="onnx", library_name=library_name
            )
            raise ValueError(
                f"Asked to export a {model_type} model for the task {task}{autodetected_message}, but the Optimum ONNX exporter only supports the tasks {', '.join(model_tasks.keys())} for {model_type}. Please use a supported task. Please open an issue at https://github.com/huggingface/optimum/issues if you would like the task {task} to be supported in the ONNX export for {model_type}."
            )

        # TODO: Fix in Transformers so that SdpaAttention class can be exported to ONNX. `attn_implementation` is introduced in Transformers 4.36.
        if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED and _transformers_version >= version.parse("4.35.99"):
            loading_kwargs["attn_implementation"] = "eager"

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


    needs_pad_token_id = task == "text-classification" and getattr(model.config, "pad_token_id", None) is None

    if needs_pad_token_id:
        if pad_token_id is not None:
            model.config.pad_token_id = pad_token_id
        else:
            tok = AutoTokenizer.from_pretrained(model_id)
            pad_token_id = getattr(tok, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                )
            model.config.pad_token_id = pad_token_id

    if "stable-diffusion" in task:
        model_type = "stable-diffusion"
    elif hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type.replace("_", "-")
    else:
        model_type = model.config.model_type.replace("_", "-")

    if (
        not custom_architecture
        and library_name != "diffusers"
        and task + "-with-past"
        in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
    ):
        # Make -with-past the default if --task was not explicitely specified
        if original_task == "auto" and not monolith:
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )
            model.config.use_cache = False
    
    if original_task == "auto":
        synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
        if synonyms_for_task:
            synonyms_for_task = ", ".join(synonyms_for_task)
            possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
        else:
            possible_synonyms = ""
        logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

    cache_args_dict = dict(
        task=task,
        model_id=model_id,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        framework=framework,
        torch_dtype=torch_dtype,
        device=device,
        library_name=library_name,
        loading_kwargs=loading_kwargs,
        opset=None,
        optimize=None,
        monolith=monolith,
        no_post_process=False,
        atol=None,
        do_validation=True,
        model_kwargs=None,
        custom_onnx_configs=None,
        fn_get_submodels=None,
        use_subprocess=False,
        _variant="default",
        legacy=False,
        trust_remote_code=trust_remote_code,
        no_dynamic_axes=False,
        do_constant_folding=True,
    )

    return cache_args_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert HuggingFace Models ['Timm', 'Diffusers', 'Transformers', 'Sentence Transformers'] to ONNX format.")

    # Model Info Dir
    parser.add_argument('--key', type=str, default=None)

    parser.add_argument('--info-dirpath', type=str, default='./')
    parser.add_argument('--cache-dirpath', type=str, default='./')
    parser.add_argument('--cache-flag-path', type=str, default='./cache.flg')

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
    flags = set()
    if cache_flag_path.is_file():
        with open(cache_flag_path, 'r') as f:
            for line in f:
                model_args_json = line.strip()
                model_args_dict: Dict = json.loads(model_args_json)
                model_id = model_args_dict.pop('model_id')
                assert model_id not in flags
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

    cache_dirpath = pathlib.Path(args.cache_dirpath)
    logger.info(f"User specified cache folder: {cache_dirpath.absolute()}")

    index = 0
    for model_info in model_infos():
        model_id = model_info['id']
        if model_id not in flags:
            try:
                logger.info(f'= v. Fin/To={len(flags)}/{index} - Now Cache {model_id}')
                cached_args_dict = cache_model(model_id, cache_dir=str(cache_dirpath), monolith=False)
                assert model_id == cached_args_dict['model_id']
                flags.add(model_id)
                with open(cache_flag_path, 'a') as f:
                    cached_args_json = json.dumps(cached_args_dict)
                    f.write(f'{cached_args_json}\n')
                logger.info(f'= ^. Fin/To={len(flags)}/{index} - \'{model_id}\' Finished.')
            except Exception as e:
                logger.error(f'E: {e}')
                logger.error(f'There is an error occurred during cache onnx model, please re-run the script or stop the process.')
                if not args.ignore:
                    user_input = input("Do you want to continue? (yes/no): ")
                    if user_input.lower() == 'yes':
                        print("Continuing with the process...")
                    else:
                        print("Process aborted.")
                        sys.exit(1)
        else:
            continue