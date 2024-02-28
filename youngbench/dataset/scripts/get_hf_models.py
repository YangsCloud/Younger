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

import os
import sys
import json
import torch
import pathlib
import argparse
import importlib
import dataclasses

from functools import partial
from typing import List, Dict, Type, Tuple, Optional
from packaging import version
from urllib.parse import urlsplit
from requests.exceptions import ConnectionError as RequestsConnectionError

from huggingface_hub import login, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from optimum.exporters.tasks import TasksManager
from optimum.configuration_utils import _transformers_version
from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED

from timm import __version__
from timm.models import load_model_config_from_hf, is_model
from timm.layers import set_layer_config
from timm.models._pretrained import PretrainedCfg
from timm.models._builder import _resolve_pretrained_source
from timm.models._hub import _get_safe_alternatives

from transformers import AutoConfig

from youngbench.logging import set_logger, logger

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False


HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl


def timm_hf_split(hf_id: str):
    # FIXME I may change @ -> # and be parsed as fragment in a URI model name scheme
    rev_split = hf_id.split('@')
    assert 0 < len(rev_split) <= 2, 'hf_hub id should only contain one @ character to identify revision.'
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def timm_load_state_dict_from_hf(model_id: str, filename: str = HF_WEIGHTS_NAME):
    # Modified from sources: timm.load_state_dict_from_hf
    hf_model_id, hf_revision = timm_hf_split(model_id)

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                hf_hub_download = partial(hf_hub_download, library_name="timm", library_version=__version__)
                cached_safe_file = hf_hub_download(repo_id=hf_model_id, filename=safe_filename, revision=hf_revision)
                logger.info(
                    f"[{model_id}] Safe alternative available for '{filename}' "
                    f"(as '{safe_filename}'). Loading weights using safetensors.")
                return safetensors.torch.load_file(cached_safe_file, device="cpu")
            except EntryNotFoundError:
                pass

    # Otherwise, load using pytorch.load
    hf_hub_download(hf_model_id, filename=filename, revision=hf_revision)
    logger.debug(f"[{model_id}] Safe alternative not found for '{filename}'. Loading weights using default pytorch.")
    return


def timm_parse_model_name(model_name: str):
    # Modified from sources: timm.parse_model_name
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme == 'hf-hub'
    return (parsed.scheme, parsed.path)


def timm_cache_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg = None,
    pretrained_cfg_overlay = None,
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    **kwargs,
):
    # Modified from sources: timm.create_model

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = timm_parse_model_name(model_name)
    assert model_source == 'hf-hub'
    assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
    # For model names specified in the form `hf-hub:path/architecture_name@revision`,
    # load model weights + pretrained_cfg from Hugging Face hub.
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        assert isinstance(pretrained_cfg, dict)
        pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        pretrained_cfg_overlay = pretrained_cfg_overlay or {}
        assert pretrained_cfg.architecture
        pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)
        pretrained_cfg = pretrained_cfg.to_dict()
        load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
        assert load_from == 'hf-hub'
        logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            timm_load_state_dict_from_hf(*pretrained_loc)
        else:
            timm_load_state_dict_from_hf(pretrained_loc)

    return


def get_model_class_for_task(
    task: str,
    framework: str = "pt",
    model_type: Optional[str] = None,
    model_class_name: Optional[str] = None,
    library: str = "transformers",
) -> Tuple[List[str], Type]:
    # Modified from sources: TasksManager.get_model_class_from_task
    """
    Attempts to retrieve an AutoModel class from a task name.

    Args:
        task (`str`):
            The task required.
        framework (`str`, defaults to `"pt"`):
            The framework to use for the export.
        model_type (`Optional[str]`, defaults to `None`):
            The model type to retrieve the model class for. Some architectures need a custom class to be loaded,
            and can not be loaded from auto class.
        model_class_name (`Optional[str]`, defaults to `None`):
            A model class name, allowing to override the default class that would be detected for the task. This
            parameter is useful for example for "automatic-speech-recognition", that may map to
            AutoModelForSpeechSeq2Seq or to AutoModelForCTC.
        library (`str`, defaults to `transformers`):
                The library name of the model.

    Returns:
        The AutoModel class corresponding to the task.
    """
    task = task.replace("-with-past", "")
    task = TasksManager.map_from_synonym(task)

    TasksManager._validate_framework_choice(framework)

    if (framework, model_type, task) in TasksManager._CUSTOM_CLASSES:
        library, class_name = TasksManager._CUSTOM_CLASSES[(framework, model_type, task)]
        loaded_library = importlib.import_module(library)

        return getattr(loaded_library, class_name)
    else:
        if framework == "pt":
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library]
        else:
            tasks_to_model_loader = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[library]

        loaded_library = importlib.import_module(library)

        model_class_names = list()
        if model_class_name is None:
            if task not in tasks_to_model_loader:
                raise KeyError(
                    f"Unknown task: {task}. Possible values are: "
                    + ", ".join([f"`{key}` for {tasks_to_model_loader[key]}" for key in tasks_to_model_loader])
                )

            if isinstance(tasks_to_model_loader[task], str):
                model_class_name = tasks_to_model_loader[task]
                model_class_names.append(model_class_name)
            else:
                # automatic-speech-recognition case, which may map to several auto class
                if library == "transformers":
                    if model_type is None:
                        logger.warning(
                            f"No model type passed for the task {task}, that may be mapped to several loading"
                            f" classes ({tasks_to_model_loader[task]}). Defaulting to {tasks_to_model_loader[task][0]}"
                            " to load the model."
                        )
                        model_class_names.extend(list(tasks_to_model_loader[task]))
                    else:
                        for autoclass_name in tasks_to_model_loader[task]:
                            module = getattr(loaded_library, autoclass_name)
                            # TODO: we must really get rid of this - and _ mess
                            if (
                                model_type in module._model_mapping._model_mapping
                                or model_type.replace("-", "_") in module._model_mapping._model_mapping
                            ):
                                model_class_names.append(autoclass_name)

                        if len(model_class_names) == 0:
                            raise ValueError(
                                f"Unrecognized configuration classes {tasks_to_model_loader[task]} do not match"
                                f" with the model type {model_type} and task {task}."
                            )
                else:
                    raise NotImplementedError(
                        "For library other than transformers, the _TASKS_TO_MODEL_LOADER mapping should be one to one."
                    )
        else:
            model_class_names.append(model_class_name)

        return model_class_names, loaded_library

def cache_model_from_task(
    task,
    model_id,
    subfolder,
    revision,
    cache_dir,
    framework,
    torch_dtype,
    device,
    library_name,
    **model_kwargs,
):
    # Sources from TasksManager.get_model_from_task

    framework = TasksManager.determine_framework(model_id, subfolder=subfolder, framework=framework)

    original_task = task
    if task == "auto":
        task = TasksManager.infer_task_from_model(model_id, subfolder=subfolder, revision=revision)

    library_name = TasksManager.infer_library_from_model(
        model_id, subfolder, revision, cache_dir, library_name
    )

    model_type = None
    model_class_name = None
    kwargs = {"subfolder": subfolder, "revision": revision, "cache_dir": cache_dir, **model_kwargs}

    if library_name == "transformers":
        config = AutoConfig.from_pretrained(model_id, **kwargs)
        model_type = config.model_type.replace("_", "-")
        # TODO: if automatic-speech-recognition is passed as task, it may map to several
        # different auto class (AutoModelForSpeechSeq2Seq or AutoModelForCTC),
        # depending on the model type
        # if original_task in ["auto", "automatic-speech-recognition"]:
        if original_task == "automatic-speech-recognition" or task == "automatic-speech-recognition":
            if original_task == "auto" and config.architectures is not None:
                model_class_name = config.architectures[0]

    model_class_names, loaded_library = get_model_class_for_task(
        task, framework, model_type=model_type, model_class_name=model_class_name, library=library_name
    )

    if library_name == "timm":
        assert len(model_class_names) == 1
        assert model_class_names[0] == 'create_model'
        timm_cache_model(f"hf_hub:{model_id}", pretrained=True, exportable=True)
    elif library_name == "sentence_transformers":
        assert len(model_class_names) == 1
        assert model_class_names[0] == 'SentenceTransformer'
        cache_folder = model_kwargs.pop("cache_folder", None)
        use_auth_token = model_kwargs.pop("use_auth_token", None)
        model = model_class(
            model_id, device=device, cache_folder=cache_folder, use_auth_token=use_auth_token
        )
    else:
        try:
            if framework == "pt":
                kwargs["torch_dtype"] = torch_dtype

                if isinstance(device, str):
                    device = torch.device(device)
                elif device is None:
                    device = torch.device("cpu")

                # TODO : fix EulerDiscreteScheduler loading to enable for SD models
                if version.parse(torch.__version__) >= version.parse("2.0") and library_name != "diffusers":
                    with device:
                        # Initialize directly in the requested device, to save allocation time. Especially useful for large
                        # models to initialize on cuda device.
                        model = model_class.from_pretrained(model_id, **kwargs)
                else:
                    model = model_class.from_pretrained(model_id, **kwargs).to(device)
            else:
                model = model_class.from_pretrained(model_id, **kwargs)
        except OSError:
            if framework == "pt":
                logger.info("Loading TensorFlow model in PyTorch before exporting.")
                kwargs["from_tf"] = True
                model = model_class.from_pretrained(model_id, **kwargs)
            else:
                logger.info("Loading PyTorch model in TensorFlow before exporting.")
                kwargs["from_pt"] = True
                model = model_class.from_pretrained(model_id, **kwargs)

    TasksManager.standardize_model_attributes(
        model_id, model, subfolder, revision, cache_dir, library_name
    )


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

    cache_model_from_task(
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
        custom_architecture=custom_architecture,
        original_task=original_task
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